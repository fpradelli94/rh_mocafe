"""
Contains methods and classes to run and resume simulations in 2D and 3D
"""
import fenics
import time
import json
import logging
from pathlib import Path
import sys
import pandas as pd
from itertools import product
from pint import Quantity, UnitRegistry
from tqdm import tqdm
import numpy as np
import shutil
from typing import Dict, List
from petsc4py import PETSc
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansim
import mocafe.angie.forms
from mocafe.fenut.parameters import Parameters
from mocafe.angie.tipcells import TipCellManager, load_tip_cells_from_json
from mocafe.fenut.solvers import PETScProblem, PETScNewtonSolver
import src.forms
from src.ioutils import (read_parameters, write_parameters, dump_json, load_json, move_files_once_per_node,
                         rmtree_if_exists_once_per_node)
from src.expressions import get_growing_RH_expression, BWImageExpression
from src.simulation2d import compute_2d_mesh_for_patient
from src.simulation3d import get_3d_mesh_for_patient, get_3d_c0


# MPI variables
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()

# get logger
logger = logging.getLogger(__name__)


def get_ureg_with_arbitrary_units(sim_parameters: Parameters):
    """
            Generate a UnitRegistry with the arbitrary units as defined in the Simulation Parameters
            """
    # initialize unit registry
    local_ureg = UnitRegistry()

    # get parameters dataframe
    parameters_df = sim_parameters.as_dataframe()

    # set name of arbitrary units as defined in the parameters
    sau_name = "Space Arbitrary Unit"
    tau_name = "Time Arbitrary Unit"
    afau_name = "AFs Arbitrary Unit"

    # define sau, tau and afau according to dataframe
    local_ureg.define(f"{sau_name} = "
                      f"{parameters_df.loc[sau_name, 'real_value']} * {parameters_df.loc[sau_name, 'real_um']} = "
                      f"sau")
    local_ureg.define(f"{tau_name} = "
                      f"{parameters_df.loc[tau_name, 'real_value']} * {parameters_df.loc[tau_name, 'real_um']} = "
                      f"tau")
    local_ureg.define(f"af concentration arbitrary unit = "
                      f"{parameters_df.loc[afau_name, 'real_value']} * {parameters_df.loc[afau_name, 'real_um']} = "
                      f"afau")

    return local_ureg


class GradientEvaluator:
    """
    Class to efficiently compute the af gradient. Defined to avoid the use of ``fenics.project(grad(af), V)``.
    This class keep the solver for the gradient saved, so to reuse it every time it is required.
    """
    def __init__(self):
        self.solver = None

    def compute_gradient(self,
                         f: fenics.Function,
                         V: fenics.FunctionSpace,
                         sol: fenics.Function,
                         options: Dict = None):
        """
        Efficiently compute the gradient of a function.

        :param f: function to be derived.
        :param V: f's function space.
        :param sol: function to store the gradient.
        :param options: options for the solver to be used.
        """
        # manage none dict
        if options is None:
            options = {}
        # define test function
        v = fenics.TestFunction(V)
        # define variational form
        if self.solver is None:
            # define trial function
            u = fenics.TrialFunction(V)
            # define operator
            A = fenics.PETScMatrix()
            a = fenics.dot(u, v) * fenics.dx
            fenics.assemble(a, tensor=A)
            # define solver
            petsc_options = PETSc.Options()
            for option, value in options.items():
                petsc_options.setValue(option, value)
            ksp = PETSc.KSP().create()
            ksp.setFromOptions()
            ksp.setOperators(fenics.as_backend_type(A).mat())
            # set solver
            self.solver = ksp

        # define b
        L = fenics.inner(fenics.grad(f), v) * fenics.dx
        b = fenics.PETScVector()
        fenics.assemble(L, tensor=b)
        b = fenics.as_backend_type(b).vec()
        # solve
        self.solver.solve(b, sol.vector().vec())
        sol.vector().update_ghost_values()


class RHSimulation:
    def __init__(self,
                 spatial_dimension: int,
                 sim_parameters: Parameters,
                 patient_parameters: Dict,
                 out_folder_name: str = mansim.default_data_folder_name,
                 out_folder_mode: str = None,
                 sim_rationale: str = "No comment",
                 slurm_job_id: int = None,
                 load_from_cache: bool = False,
                 write_checkpoints: bool = True):
        """
        Initialize a Simulation Object.

        :param spatial_dimension: 2 == 2D simulation; 3 == 3D simulation.
        :param sim_parameters: simulation parameters.
        :param patient_parameters: patient-specific parameters.
        :param out_folder_name: folder where the simulation will be saved:
        :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in
        out_folder_name; if ``datetime``, the output folder name will be also followed by a string containing the date
        and the time of the simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``).
        The latter is recommended to run multiple simulations of the same kind. Default is ```None``.
        :param sim_rationale: provide a rationale for running this simulation (e.g. checking the effect of this
        parameter). It will be added to the simulation report in ``saved_sim/sim_name/sim_info/sim_info.html``. Default
        is "No comment".
        :param slurm_job_id: slurm job ID assigned to the simulation, if performed with slurm. It is used to generate a
        pbar stored in ``slurm/<slurm job ID>.pbar``.
        :param load_from_cache: load mesh and some function from the cache. Default is False.
        """
        # parameters
        self.sim_parameters: Parameters = sim_parameters  # simulation parameters
        self.patient_parameters: Dict = patient_parameters  # patient parameters
        self.lsp = {
            "ksp_type": "gmres",
            "pc_type": "gamg",
            "ksp_monitor": None}  # linear solver parameters

        # proprieties
        self.spatial_dimension: int = spatial_dimension  # spatial dimension
        self.init_time: float = time.perf_counter()  # initial simulation time
        self.ureg: UnitRegistry = get_ureg_with_arbitrary_units(sim_parameters)  # unit registry
        self.sim_rationale: str = sim_rationale  # sim description
        self.slurm_job_id: int = slurm_job_id  # slurm job id (if available)
        self.error_msg = None  # error message in case of simulation errors
        self.out_folder_name = out_folder_name

        # flags
        self.recompute_mesh = (not load_from_cache)  # mesh should be recomputed
        self.recompute_c0 = (not load_from_cache)  # set if c0 should be recomputed
        self.runtime_error_occurred = False  # flag to activate in case of sim errors
        self.write_checkpoints: bool = write_checkpoints  # if True, write simulation with checkpoints

        # Folders
        self.data_folder: Path = mansim.setup_data_folder(folder_path=f"saved_sim/{out_folder_name}",
                                                          auto_enumerate=out_folder_mode)
        self.report_folder: Path = self.data_folder / Path("sim_info")
        self.reproduce_folder: Path = self.data_folder / Path("0_reproduce")
        cache_folder = Path(".sim_cache")
        self.cache_mesh_folder: Path = cache_folder / Path("mesh")
        self.cache_c0_folder: Path = cache_folder / Path("c0")
        self.slurm_folder = Path("slurm")

        # XDMF files
        self.cache_mesh_xdmf = self.cache_mesh_folder / Path("mesh.xdmf")
        self.cache_c0_xdmf = self.cache_c0_folder / Path("c0.xdmf")

        # Other files
        self.mesh_parameters_file: Path = self.cache_mesh_folder / Path("mesh_parameters.csv")

        # Pbar file
        if (self.slurm_job_id is not None) and (rank == 0):
            self.pbar_file = open(f"slurm/{self.slurm_job_id}pbar.o", 'w')
        else:
            self.pbar_file = sys.stdout

    def _check_simulation_properties(self):
        # check spatial dimension
        assert (self.spatial_dimension == 2 or self.spatial_dimension == 3), \
            f"Cannot run simulation for dimension {self.spatial_dimension}"

    def _sim_mkdir_list(self, *folders_list: Path):
        if rank == 0:
            # generate simulation folder
            for folder in folders_list:
                folder.mkdir(exist_ok=True, parents=True)

            # if slurm id is provided, generate slurm folder
            if self.slurm_job_id is not None:
                self.slurm_folder.mkdir(exist_ok=True)

        comm_world.Barrier()

    def _fill_reproduce_folder(self):
        if rank == 0:
            # patterns ignored when copying code after simulation
            ignored_patterns = ["README.md",
                                "saved_sim*",
                                "*.ipynb_checkpoints*",
                                "sif",
                                "visualization",
                                "*pycache*",
                                ".thumbs",
                                ".sim_cache/*",
                                "jobids.txt",
                                "slurm/*"]

            # if reproduce folder exists, remove it
            if self.reproduce_folder.exists():
                shutil.rmtree(str(self.reproduce_folder.resolve()))
            # copy all the code contained in reproduce folder
            shutil.copytree(src=str(Path(__file__).parent.parent.resolve()),
                            dst=str(self.reproduce_folder.resolve()),
                            ignore=shutil.ignore_patterns(*ignored_patterns))

        comm_world.Barrier()

    def _generate_mesh(self):
        if self.spatial_dimension == 2:
            mesh, mesh_parameters = compute_2d_mesh_for_patient(self.sim_parameters.as_dataframe(),
                                                                self.ureg,
                                                                self.patient_parameters)
        else:
            mesh, mesh_parameters = get_3d_mesh_for_patient(self.sim_parameters.as_dataframe(),
                                                            self.ureg,
                                                            self.patient_parameters,
                                                            self.cache_mesh_xdmf,
                                                            self.mesh_parameters_file,
                                                            self.recompute_mesh)

        # store as class properties
        self.mesh = mesh
        self.mesh_parameters = mesh_parameters

    def _spatial_discretization(self):
        self.V = fu.get_mixed_function_space(self.mesh, 3)
        self.vec_V = fenics.VectorFunctionSpace(self.mesh, "CG", 1)  # for grad_af

    def _generate_sim_parameters_independent_initial_conditions(self):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # capillaries
        if self.spatial_dimension == 2:
            c_old_expression = BWImageExpression(self.patient_parameters["pic2d"], [-1, 1], self.ureg)
            self.c_old = fenics.interpolate(c_old_expression, self.V.sub(0).collapse())
        else:
            # define c0 function
            self.c_old = fenics.Function(self.V.sub(0).collapse())
            # get 3d c0
            get_3d_c0(self.c_old, self.patient_parameters, self.mesh_parameters, self.cache_c0_xdmf,
                      self.recompute_c0, self.write_checkpoints)
        # name c_old
        self.c_old.rename("c", "capillaries")

        # auxiliary fun for capillaries, initially set to 0
        logger.info(f"{self.out_folder_name}:Computing mu0...")
        self.mu_old = fenics.interpolate(fenics.Constant(0.), self.V.sub(0).collapse())

        # t_c_f_function (dynamic tip cell position)
        logger.info(f"{self.out_folder_name}:Computing t_c_f_function...")
        self.t_c_f_function = fenics.interpolate(fenics.Constant(0.), self.V.sub(0).collapse())
        self.t_c_f_function.rename("tcf", "tip cells function")

        # define initial time
        self.t0 = 0

    def _generate_sim_parameters_dependent_initial_conditions(self, sim_parameters):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # define initial condition for tumor
        logger.info(f"{self.out_folder_name}:Computing phi0...")
        # initial semiaxes of the tumor
        self.phi_expression = get_growing_RH_expression(sim_parameters,
                                                        self.patient_parameters,
                                                        self.mesh_parameters,
                                                        self.ureg,
                                                        0,
                                                        self.spatial_dimension)
        self.phi = fenics.interpolate(self.phi_expression, self.V.sub(0).collapse())
        self.phi.rename("phi", "retinal hemangioblastoma")

        # af
        logger.info(f"{self.out_folder_name}:Computing af0...")
        self.af_old = fenics.Function(self.V.sub(0).collapse())
        self.__compute_af_0(sim_parameters)
        self.af_old.rename("af", "angiogenic factor")

        # af gradient
        logger.info(f"{self.out_folder_name}:Computing grad_af0...")
        self.ge = GradientEvaluator()
        self.grad_af_old = fenics.Function(self.vec_V)
        self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
        self.grad_af_old.rename("grad_af", "angiogenic factor gradient")

        # define tip cell manager
        self.tip_cell_manager = TipCellManager(self.mesh, self.sim_parameters)

    def __compute_af_0(self, sim_parameters: Parameters, options: Dict = None):
        """
        Solve equilibrium system for af considering the initial values of phi and c.

        Basically, this function is used to generate the initial condition for af assuming that af is at equilibrium
        at the beginning of the simulation.
        """
        # manage none dict
        if options is None:
            options = self.lsp
        # get af variable
        af = fenics.TrialFunction(self.V.sub(0).collapse())
        # get test function
        v = fenics.TestFunction(self.V.sub(0).collapse())
        # built equilibrium form for af
        af_form = src.forms.angiogenic_factors_form_eq(af, self.phi, self.c_old, v, sim_parameters)
        af_form_a = fenics.lhs(af_form)
        af_form_L = fenics.rhs(af_form)
        # define operator
        A = fenics.PETScMatrix()
        fenics.assemble(af_form_a, tensor=A)
        # define solver
        petsc_options = PETSc.Options()
        for option, value in options.items():
            petsc_options.setValue(option, value)
        ksp = PETSc.KSP().create()
        ksp.setFromOptions()
        ksp.setOperators(fenics.as_backend_type(A).mat())
        # define b
        b = fenics.PETScVector()
        fenics.assemble(af_form_L, tensor=b)
        b = fenics.as_backend_type(b).vec()
        # solve
        ksp.solve(b, self.af_old.vector().vec())
        self.af_old.vector().update_ghost_values()

    def _set_pbar(self, total: int):
        self.pbar = tqdm(total=total,
                         ncols=100,
                         desc=self.out_folder_name,
                         file=self.pbar_file,
                         disable=True if rank != 0 else False)

    def _end_simulation(self):
        logger.info("Ending simulation... ")

        # close pbar file
        if (rank == 0) and (self.slurm_job_id is not None):
            self.pbar_file.close()

        # save sim report
        if self.runtime_error_occurred:
            sim_description = "RUNTIME ERROR OCCURRED. \n" + self.sim_rationale
        else:
            sim_description = self.sim_rationale
        sim_time = time.perf_counter() - self.init_time
        mansim.save_sim_info(data_folder=self.report_folder,
                             parameters={"Sim parameters:": self.sim_parameters,
                                         "Mesh parameters": self.mesh_parameters},
                             execution_time=sim_time,
                             sim_name=self.out_folder_name,
                             sim_description=sim_description,
                             error_msg=self.error_msg)

        # save parameters
        write_parameters(self.sim_parameters, self.report_folder / Path("sim_parameters.csv"))
        write_parameters(self.mesh_parameters, self.report_folder / Path("mesh_parameters.csv"))
        dump_json(self.patient_parameters, self.report_folder / Path("patient_parameters.json"))


class RHTimeSimulation(RHSimulation):
    def __init__(self,
                 spatial_dimension: int,
                 sim_parameters: Parameters,
                 patient_parameters: Dict,
                 steps: int,
                 save_rate: int = 1,
                 out_folder_name: str = mansim.default_data_folder_name,
                 out_folder_mode: str = None,
                 sim_rationale: str = "No comment",
                 slurm_job_id: int = None,
                 load_from_cache: bool = False,
                 write_checkpoints: bool = True,
                 save_distributed_files_to: str or None = None):
        """
        Initialize a Simulation Object.

        :param spatial_dimension: 2 == 2D simulation; 3 == 3D simulation.
        :param sim_parameters: simulation parameters.
        :param patient_parameters: patient-specific parameters.
        :param steps: simulation steps in time.
        :param save_rate: frequency of simulation state saves in terms of steps. E.g., if ``save_rate=10``, the
        simulation will be saved every 10 steps. By default, the last simulation step is always saved.
        :param out_folder_name: folder where the simulation will be saved:
        :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in
        out_folder_name; if ``datetime``, the output folder name will be also followed by a string containing the date
        and the time of the simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``).
        The latter is recommended to run multiple simulations of the same kind. Default is ```None``.
        :param sim_rationale: provide a rationale for running this simulation (e.g. checking the effect of this
        parameter). It will be added to the simulation report in ``saved_sim/sim_name/sim_info/sim_info.html``. Default
        is "No comment".
        :param slurm_job_id: slurm job ID assigned to the simulation, if performed with slurm. It is used to generate a
        pbar stored in ``slurm/<slurm job ID>.pbar``.
        :param load_from_cache: load mesh and some function from the cache. Default is False.
        """
        # init super class
        super().__init__(spatial_dimension=spatial_dimension,
                         sim_parameters=sim_parameters,
                         patient_parameters=patient_parameters,
                         out_folder_name=out_folder_name,
                         out_folder_mode=out_folder_mode,
                         sim_rationale=sim_rationale,
                         slurm_job_id=slurm_job_id,
                         load_from_cache=load_from_cache,
                         write_checkpoints=write_checkpoints)

        # specific properties
        self.steps: int = steps  # simulations steps
        self.save_rate: int = save_rate  # set how often writes the output

        # specific flags
        self.__resumed: bool = False  # secret flag to check if simulation has been resumed
        self.save_distributed_files = (save_distributed_files_to is not None)  # Use PVD files instead of XDMF

        # specific folders
        self.resume_folder: Path = self.data_folder / Path("resume")
        if self.save_distributed_files:
            distributed_data_folder = Path(save_distributed_files_to) / Path(self.data_folder.name)
            pvd_folder = self.data_folder / Path("pvd")
            # clean out folders if they exist
            rmtree_if_exists_once_per_node(distributed_data_folder)
            if rank == 0:
                if pvd_folder.exists():
                    shutil.rmtree(pvd_folder)
            self.distributed_data_folder = distributed_data_folder
            self.pvd_folder = pvd_folder

        # specific files
        file_names = ["c", "af", "grad_af", "tipcells", "phi"]
        if self.save_distributed_files:
            # specific PVD files
            self.c_pvd, self.af_pvd, self.grad_af_pvd, self.tipcells_pvd, self.phi_pvd = \
                [fenics.File(str(self.distributed_data_folder / Path(f"{fn}.pvd"))) for fn in file_names]
        else:
            # specific XDMF files
            self.c_xdmf, self.af_xdmf, self.grad_af_xdmf, self.tipcells_xdmf, self.phi_xdmf = fu.setup_xdmf_files(
                file_names,
                self.data_folder,
                {"flush_output": True, "rewrite_function_mesh": False})

        self.__resume_files_dict: Dict or None = None  # dictionary containing files to resume

    def run(self) -> bool:
        """
        Run simulation. Return True if a runtime error occurred, False otherwise.
        """
        self._check_simulation_properties()  # Check class proprieties. Return error if something does not work.

        self._sim_mkdir()  # create all simulation folders

        self._fill_reproduce_folder()  # store current script in reproduce folder to keep track of the code

        if self.__resumed:
            self._resume_mesh()  # load mesh
        else:
            self._generate_mesh()  # generate mesh

        self._spatial_discretization()  # initialize function space

        if self.__resumed:
            self._resume_initial_conditions()  # resume initial conditions
        else:
            self._generate_initial_conditions()  # generate initial condition

        # write initial conditions
        self._write_files(0)

        # define solver before time iteration
        self.solver = PETScNewtonSolver(self.lsp, self.mesh.mpi_comm())

        self._time_iteration()  # run simulation in time

        self._end_simulation()  # conclude time iteration

        return self.runtime_error_occurred

    def _check_simulation_properties(self):
        # check base conditions
        super()._check_simulation_properties()

        # check if steps is positive
        assert self.steps >= 0, "Simulation should have a positive number of steps"

    def _sim_mkdir(self):
        super()._sim_mkdir_list(self.data_folder, self.report_folder, self.reproduce_folder,
                                self.resume_folder, self.cache_mesh_folder, self.cache_c0_folder)
        if self.save_distributed_files:
            # make directories for distributed save
            super()._sim_mkdir_list(self.distributed_data_folder, self.pvd_folder)

    def _resume_mesh(self):
        assert self.__resumed, "Trying to resume mesh but simulation was not resumed"
        logger.info(f"{self.out_folder_name}:Loading Mesh... ")
        mesh = fenics.Mesh()
        with fenics.XDMFFile(str(self.__resume_files_dict["mesh.xdmf"])) as infile:
            infile.read(mesh)
        self.mesh = mesh
        self.mesh_parameters = read_parameters(self.__resume_files_dict["mesh_parameters.csv"])

    def _generate_initial_conditions(self):
        # generate initial conditions independent of sim parameters
        super()._generate_sim_parameters_independent_initial_conditions()
        # generate initial conditions dependent from sim parameters
        super()._generate_sim_parameters_dependent_initial_conditions(self.sim_parameters)

    def _resume_initial_conditions(self):
        # load incremental tip cells
        input_itc = load_json(self.__resume_files_dict["incremental_tip_cells.json"])

        # get last stem of the resumed folder
        last_step = max([int(step.replace("step_", "")) for step in input_itc])

        # capillaries
        logger.info(f"{self.out_folder_name}:Loading c0... ")
        self.c_old = fenics.Function(self.V.sub(0).collapse())
        resume_c_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["c.xdmf"]))
        with resume_c_xdmf as infile:
            infile.read_checkpoint(self.c_old, "c", 0)
        self.c_old.rename("c", "capillaries")

        # mu
        logger.info(f"{self.out_folder_name}:Loading mu... ")
        self.mu_old = fenics.Function(self.V.sub(0).collapse())
        resume_mu_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["mu.xdmf"]))
        with resume_mu_xdmf as infile:
            infile.read_checkpoint(self.mu_old, "mu", 0)

        # phi
        logger.info(f"{self.out_folder_name}:Generating phi... ")
        self.phi_expression = get_growing_RH_expression(self.sim_parameters,
                                                        self.patient_parameters,
                                                        self.mesh_parameters,
                                                        self.ureg,
                                                        last_step,
                                                        self.spatial_dimension)
        self.phi = fenics.interpolate(self.phi_expression, self.V.sub(0).collapse())
        self.phi.rename("phi", "retinal hemangioblastoma")

        # tcf function
        self.t_c_f_function = fenics.interpolate(fenics.Constant(0.), self.V.sub(0).collapse())
        self.t_c_f_function.rename("tcf", "tip cells function")

        # af
        logger.info(f"{self.out_folder_name}:Loading af... ")
        self.af_old = fenics.Function(self.V.sub(0).collapse())
        resume_af_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["af.xdmf"]))
        with resume_af_xdmf as infile:
            infile.read_checkpoint(self.af_old, "af", 0)
        self.af_old.rename("af", "angiogenic factor")

        # af gradient
        logger.info(f"{self.out_folder_name}:Loading af_grad... ")
        self.ge = GradientEvaluator()
        self.grad_af_old = fenics.Function(self.vec_V)
        resume_grad_af_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["grad_af.xdmf"]))
        with resume_grad_af_xdmf as infile:
            infile.read_checkpoint(self.grad_af_old, "grad_af", 0)
        self.grad_af_old.rename("grad_af", "angiogenic factor gradient")

        # define tip cell manager
        if rank == 0:
            initial_tcs = load_tip_cells_from_json(str(self.__resume_files_dict["tipcells.json"]))
        else:
            initial_tcs = None
        initial_tcs = comm_world.bcast(initial_tcs, 0)
        self.tip_cell_manager = TipCellManager(self.mesh,
                                               self.sim_parameters,
                                               initial_tcs=initial_tcs)

        # definie initial time
        self.t0 = last_step

    def _write_files(self, t: int):
        # write files
        if self.save_distributed_files:
            logger.info(f"Starting writing of PVD files ")
            logger.info(f"Writing af_pvd... ")
            self.af_pvd << (self.af_old, t)
            logger.info(f"Writing c_pvd... ")
            self.c_pvd << (self.c_old, t)
            logger.info(f"Writing grad_af_pvd... ")
            self.grad_af_pvd << (self.grad_af_old, t)
            logger.info(f"Writing phi_pvd... ")
            self.phi_pvd << (self.phi, t)
            logger.info(f"Writing tipcells_pvd... ")
            self.tipcells_pvd << (self.t_c_f_function, t)
        else:
            logger.info(f"Starting writing of XDMF files ")
            logger.info(f"Writing af_xdmf... ")
            self.af_xdmf.write(self.af_old, t)
            logger.info(f"Writing c_xdmf... ")
            self.c_xdmf.write(self.c_old, t)
            logger.info(f"Writing grad_af_xdmf... ")
            self.grad_af_xdmf.write(self.grad_af_old, t)
            logger.info(f"Writing phi_xdmf... ")
            self.phi_xdmf.write(self.phi, t)
            # save tip cells current position
            logger.info(f"Writing tipcells_xdmf... ")
            self.tipcells_xdmf.write(self.t_c_f_function, t)

    def _solve(self, problem: PETScProblem, u: fenics.Function):
        try:
            self.solver.solve(problem, u.vector())
        except RuntimeError as e:
            # store error info
            self.runtime_error_occurred = True
            self.error_msg = str(e)
            logger.error(str(e))

    def _time_iteration(self):
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        u = fenics.Function(self.V)
        # assign functions to u
        fenics.assign(u, [self.af_old, self.c_old, self.mu_old])
        af, c, mu = fenics.split(u)
        # define test functions
        v1, v2, v3 = fenics.TestFunctions(self.V)
        # build total form
        af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters)
        capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters)
        form = af_form + capillaries_form

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        # define Jacobian
        J = fenics.derivative(form, u)
        problem = PETScProblem(J, form, [])

        # init time iteration
        t = self.t0
        dt = self.sim_parameters.get_value("dt")

        super()._set_pbar(total=self.steps)

        # log
        logger.info(f"{self.out_folder_name}:Starting time iteration...")
        # iterate in time
        for step in range(1, self.steps + 1):
            # update time
            t += dt

            # activate tip cells
            self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, step)

            # revert tip cells
            self.tip_cell_manager.revert_tip_cells(self.af_old, self.grad_af_old)

            # move tip cells
            self.tip_cell_manager.move_tip_cells(self.c_old, self.af_old, self.grad_af_old)

            # store tip cells in fenics function and json file
            self.t_c_f_function.assign(self.tip_cell_manager.get_latest_tip_cell_function())
            self.tip_cell_manager.save_incremental_tip_cells(f"{self.report_folder}/incremental_tipcells.json", step)

            # solve
            self._solve(problem, u)

            # if error occurred, stop iteration
            if self.runtime_error_occurred:
                break

            # assign to old
            fenics.assign([self.af_old, self.c_old, self.mu_old], u)
            # assign new value to grad_af_old
            self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
            # assign new value to phi
            self.phi_expression.t = t  # update time
            self.phi.assign(fenics.interpolate(self.phi_expression, self.V.sub(0).collapse()))

            # save
            if (step % self.save_rate == 0) or (step == self.steps) or self.runtime_error_occurred:
                self._write_files(step)

            # update progress bar
            self.pbar.update(1)

    def _end_simulation(self):
        # save resume info
        if self.write_checkpoints:
            self._save_resume_info()

        # mv distributed files to data folder
        if self.save_distributed_files:
            move_files_once_per_node(src_folder=self.distributed_data_folder, dst=self.pvd_folder)
            rmtree_if_exists_once_per_node(self.distributed_data_folder)

        super()._end_simulation()

    def _save_resume_info(self):
        """
        Save the mesh and/or the fenics Functions in a resumable format (i.e. using the FEniCS function
        `write_checkpoint`).
        """
        logger.info("Starting checkpoint write")

        # init list of saved file
        saved_files = {}

        # copy cached mesh in resume folder
        for mf in self.cache_mesh_folder.glob("mesh.*"):
            shutil.copy(mf, self.resume_folder)

        # write functions
        fnc_dict = {"af": self.af_old, "c": self.c_old, "mu": self.mu_old, "phi": self.phi, "grad_af": self.grad_af_old}
        for name, fnc in fnc_dict.items():
            file_name = f"{self.resume_folder}/{name}.xdmf"
            with fenics.XDMFFile(file_name) as outfile:
                logger.info(f"Checkpoint writing of {name}....")
                outfile.write_checkpoint(fnc, name, 0, fenics.XDMFFile.Encoding.HDF5, False)
            saved_files[name] = str(Path(file_name).resolve())

        # store tip cells position
        file_name = f"{self.resume_folder}/tipcells.json"
        self.tip_cell_manager.save_tip_cells(file_name)
        saved_files["tipcells"] = str(Path(file_name).resolve())

    @classmethod
    def resume(cls,
               resume_from: Path,
               steps: int,
               save_rate: int,
               out_folder_name: str = mansim.default_data_folder_name,
               out_folder_mode: str = None,
               sim_rationale: str = "No comment",
               slurm_job_id: int = None,
               write_checkpoints: bool = True):
        """
        Resume a simulation stored in a given folder.

        :param resume_from: folder containing the simulation data to be resumed. It must contain a ``resume`` folder
        (e.g. if ``resume_from=/home/user/my_sim``, there must be a ``/home/user/my_sim/resume`` folder).
        :param steps: number of simulation steps.
        :param save_rate: specify how often the simulation status will be stored (e.g. if ``save_rate=10``, it will be
        stored every 10 steps). The last simulation step is always saved.
        :param out_folder_name: name for the simulation. It will be used to name the folder containing the output of the
        simulation (e.g. ``saved_sim/sim_name``).
        :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in
        out_folder_name; if ``datetime``, the output folder name will be also followed by a string containing the date
        and the time of the simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``). The latter is
        recommended to run multiple simulations of the same kind. Default is ```None``.
        :param sim_rationale: provide a rationale for running this simulation (e.g. checking the effect of this
        parameter). It will be added to the simulation report contained in
        ``saved_sim/sim_name/sim_info/sim_info.html``. Default is "No comment".
        :param slurm_job_id: slurm job ID assigned to the simulation, if performed with slurm. It is used to generate a
        pbar stored in ``slurm/<slurm job ID>.pbar``.
        :param write_checkpoints: set to True to save simulation with checkpoints
        """
        # ----------------------------------------------------------------------------------------------------------- #
        # 1. Check resume folder consistency
        # ----------------------------------------------------------------------------------------------------------- #
        logger.info(f"{out_folder_name}: Checking if it's possible to resume simulation ... ")

        # init error msg
        error_msg_preamble = "Not enough info to resume."

        # check if all relevant files exist
        input_report_folder = resume_from / Path("sim_info")
        sim_parameters_csv = input_report_folder / Path("sim_parameters.csv")
        mesh_parameters_csv = input_report_folder / Path("mesh_parameters.csv")
        input_incremental_tip_cells = input_report_folder / Path("incremental_tipcells.json")
        patient_parameters_file = input_report_folder / Path("patient_parameters.json")
        input_resume_folder = resume_from / Path("resume")
        resume_files = ["mesh.xdmf", "af.xdmf", "c.xdmf", "grad_af.xdmf", "mu.xdmf", "phi.xdmf", "tipcells.json"]
        resume_files_dict = {file_name: input_resume_folder / Path(file_name)
                             for file_name in resume_files}
        resume_files_dict["incremental_tip_cells.json"] = input_incremental_tip_cells
        resume_files_dict["mesh_parameters.csv"] = mesh_parameters_csv
        for f in [sim_parameters_csv, input_resume_folder, patient_parameters_file, *resume_files_dict.values()]:
            if not f.exists():
                raise RuntimeError(f"{error_msg_preamble} {f} is missing.")

        # ----------------------------------------------------------------------------------------------------------- #
        # 2. Generate simulation from resumed data
        # ----------------------------------------------------------------------------------------------------------- #
        # compute spatial dimension
        mesh_parameters = read_parameters(mesh_parameters_csv)
        if mesh_parameters.is_value_present("Lz"):
            spatial_dimension = 3
        else:
            spatial_dimension = 2

        # load simulation parameters
        sim_parameters = read_parameters(sim_parameters_csv)

        # load patient parameters
        patient_parameters = load_json(patient_parameters_file)

        # init simulation obj
        simulation = cls(spatial_dimension=spatial_dimension,
                         sim_parameters=sim_parameters,
                         patient_parameters=patient_parameters,
                         steps=steps,
                         save_rate=save_rate,
                         out_folder_name=out_folder_name,
                         out_folder_mode=out_folder_mode,
                         sim_rationale=sim_rationale,
                         slurm_job_id=slurm_job_id,
                         write_checkpoints=write_checkpoints)
        simulation.__resumed = True
        simulation.__resume_files_dict = resume_files_dict
        return simulation


class RHTimeAdaptiveSimulation(RHTimeSimulation):
    def __init__(self,
                 spatial_dimension: int,
                 sim_parameters: Parameters,
                 patient_parameters: Dict,
                 steps: int,
                 delta_steps: int,
                 trigger_adaptive_tc_activation_after_steps: int,
                 save_rate: int = 1,
                 out_folder_name: str = mansim.default_data_folder_name,
                 out_folder_mode: str = None,
                 sim_rationale: str = "No comment",
                 slurm_job_id: int = None,
                 load_from_cache: bool = False,
                 write_checkpoints: bool = True,
                 save_distributed_files_to: str or None = None):
        super().__init__(spatial_dimension,
                         sim_parameters,
                         patient_parameters,
                         steps,
                         save_rate,
                         out_folder_name,
                         out_folder_mode,
                         sim_rationale,
                         slurm_job_id,
                         load_from_cache,
                         write_checkpoints,
                         save_distributed_files_to)
        # init time window
        self.time_jump: float = sim_parameters.get_value("dt") * delta_steps
        # set after how many steps the adaptive solver is activated
        self.trigger_delta_step: int = trigger_adaptive_tc_activation_after_steps

    def __build_weak_form(self, u: fenics.Function,
                          phi_temp: fenics.Function or None = None,
                          **weak_form_kwargs) -> fenics.Form:
        # assign functions to u
        # fenics.assign(u, [self.af_old, self.c_old, self.mu_old])
        af, c, mu = fenics.split(u)
        # define test functions
        v1, v2, v3 = fenics.TestFunctions(self.V)
        # build total form
        if phi_temp is None:
            af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters,
                                                           **weak_form_kwargs)
        else:
            af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, phi_temp, c, v1, self.sim_parameters,
                                                           **weak_form_kwargs)
        capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters, **weak_form_kwargs
        )
        form = af_form + capillaries_form
        # return form
        return form

    def _test_tip_cell_activation_after_delta_step(self,
                                                   u_temp: fenics.Function,
                                                   phi_temp: fenics.Function,
                                                   phi_exp_temp: fenics.Expression,
                                                   grad_af_temp: fenics.Function,
                                                   t: float,
                                                   time_jump: float):
        # define bigger dt
        bigger_dt = time_jump
        # update tumor dimension
        phi_exp_temp.t = t + time_jump
        phi_temp.assign(fenics.interpolate(phi_exp_temp, self.V.sub(0).collapse()))
        # build weak from for dt = time_jump * dt
        form = self.__build_weak_form(u_temp, phi_temp=phi_temp, dt=bigger_dt)
        # get Jacobian
        J = fenics.derivative(form, u_temp)
        # define problem
        problem = PETScProblem(J, form, [])
        # solve
        self._solve(problem, u_temp)
        # check if runtime error occurred
        if self.runtime_error_occurred:
            return False
        # define functions
        af_temp, c_temp, mu_temp = u_temp.split()
        self.ge.compute_gradient(af_temp, self.vec_V, grad_af_temp, self.lsp)
        # check tc_activation
        tc_activate_after_delta_step = self.tip_cell_manager.test_tip_cell_activation(c_temp,
                                                                                      af_temp,
                                                                                      grad_af_temp)

        return tc_activate_after_delta_step

    def _find_next_activation_time(self, t: float,
                                   putative_next_time: float,
                                   u_temp: fenics.Function,
                                   phi_temp: fenics.Function,
                                   phi_exp_temp: fenics.Expression,
                                   grad_af_temp: fenics.Function) -> float:
        if abs(putative_next_time - t) <= self.sim_parameters.get_value("dt"):
            logger.info(f"{self.out_folder_name}:Found next time: {putative_next_time}")
            return putative_next_time
        else:
            # define mid step
            mid_time = (putative_next_time + t) / 2
            # define the delta between the current step and the actual step
            delta_mid_time = mid_time - t

            logger.info(f"{self.out_folder_name}:Checking tip cell activation at mid point: {mid_time}")
            # solve problem after delta_mid_step
            tc_activate_at_mid_step = self._test_tip_cell_activation_after_delta_step(u_temp, phi_temp, phi_exp_temp,
                                                                                      grad_af_temp, t, delta_mid_time)
            if tc_activate_at_mid_step:
                return self._find_next_activation_time(t, mid_time, u_temp, phi_temp, phi_exp_temp, grad_af_temp)
            else:
                return self._find_next_activation_time(mid_time, putative_next_time, u_temp, phi_temp, phi_exp_temp,
                                                       grad_af_temp)

    def _time_iteration(self):
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        u = fenics.Function(self.V)
        form = self.__build_weak_form(u)

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        # define Jacobian
        J = fenics.derivative(form, u)
        # define problem
        problem = PETScProblem(J, form, [])

        # init time iteration
        dt = self.sim_parameters.get_value("dt")
        self.t = dt
        step = 1

        # define temp phi exp
        phi_exp_temp = get_growing_RH_expression(self.sim_parameters,
                                                 self.patient_parameters,
                                                 self.mesh_parameters,
                                                 self.ureg,
                                                 initial_t=self.t,
                                                 spatial_dimension=3)

        # init count of tip cells
        tip_cell_count = np.zeros(self.steps)
        # setup pbar
        super()._set_pbar(total=self.steps)

        # log
        logger.info(f"{self.out_folder_name}:Starting time iteration...")
        # iterate in time
        while step < self.steps + 1:
            if (step < 4) or (step > 100):
                # activate tip cells
                self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, step)

                # revert tip cells
                self.tip_cell_manager.revert_tip_cells(self.af_old, self.grad_af_old)

                # move tip cells
                self.tip_cell_manager.move_tip_cells(self.c_old, self.af_old, self.grad_af_old)

                # store tip cells in fenics function and json file
                self.t_c_f_function.assign(self.tip_cell_manager.get_latest_tip_cell_function())
                self.tip_cell_manager.save_incremental_tip_cells(f"{self.report_folder}/incremental_tipcells.json", step)

                # store tip cell count in array
                current_tip_cell_count = len(self.tip_cell_manager.get_global_tip_cells_list())
                tip_cell_count[step - 1] = current_tip_cell_count

            if step > self.trigger_delta_step and \
               np.all(tip_cell_count[step+1-self.trigger_delta_step:step+1] == 0):
                logger.info(f"{self.out_folder_name}:Entered adaptive time iteration")

                # 1. Proceed solving the problem after the time_jump
                putative_next_time = self.t + self.time_jump             # init putative next time
                u_temp = fenics.Function(self.V)                         # init u_temp
                grad_af_temp = fenics.Function(self.vec_V)               # init grad_af_temp
                phi_temp = fenics.Function(self.V.sub(0).collapse())     # init phi_temp
                phi_exp_temp = get_growing_RH_expression(self.sim_parameters,
                                                         self.patient_parameters,
                                                         self.mesh_parameters,
                                                         self.ureg,
                                                         initial_t=self.t,
                                                         spatial_dimension=3)
                # 2. Check if tc activate after time_jump
                logger.info(f"{self.out_folder_name}:Checking tip cell activation after time jump "
                            f"(sim time: {self.t})")

                tc_activate_at_putative_next_time = self._test_tip_cell_activation_after_delta_step(
                    u_temp, phi_temp, phi_exp_temp, grad_af_temp, step, self.time_jump
                )
                logger.info(f"{self.out_folder_name}:Checking tip cell activation after time jump: "
                            f"(sim time: {self.t}): {tc_activate_at_putative_next_time}")
                # if runtime error occurred, exit
                if self.runtime_error_occurred:
                    return 1

                # 3. If tc activated at next step, find the actual next step. Else, set the putative next step as
                # the next step
                if tc_activate_at_putative_next_time:
                    next_time = self._find_next_activation_time(self.t, putative_next_time, u_temp, phi_temp,
                                                                phi_exp_temp, grad_af_temp)
                else:
                    next_time = putative_next_time
                # if runtime error occurred, exit
                if self.runtime_error_occurred:
                    return 1

                # 4. Update status
                logger.info(f"{self.out_folder_name}:Updating simulation status after adaptive time stepping")
                # compute delta in terms of steps
                delta_step = int(np.round(next_time - self.t))
                # update time
                self.t = next_time
                # update phi
                self.phi.assign(phi_temp)
                # update all functions
                fenics.assign([self.af_old, self.c_old, self.mu_old], u_temp)
                fenics.assign(u, [self.af_old, self.c_old, self.mu_old])
                fenics.assign(self.grad_af_old, grad_af_temp)
                # update pbar
                self.pbar.update(delta_step)
                # Update sim step
                step += delta_step
                # write functions
                if ((step % self.save_rate == 0)
                        or (step == self.steps)
                        or self.runtime_error_occurred
                        or tc_activate_at_putative_next_time):
                    self._write_files(step)

                # 5. update tip cells incremental file
                with open(f"{self.report_folder}/incremental_tipcells.json", "r+") as itc_json:
                    itc_dict = json.load(itc_json)
                    last_saved_step = list(itc_dict.keys())[-1]
                    last_saved_step = int(last_saved_step.replace("step_", ""))
                    for missing_step in range(last_saved_step, step):
                        itc_dict[f"step_{missing_step}"] = {}
                    json.dump(itc_dict, itc_json)

                logger.info(f"{self.out_folder_name}:Exiting adaptive time stepping")
            else:
                # solve
                self._solve(problem, u)

                # if error occurred, stop iteration
                if self.runtime_error_occurred:
                    return 1

                # assign to old
                fenics.assign([self.af_old, self.c_old, self.mu_old], u)
                # assign new value to grad_af_old
                self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
                # assign new value to phi
                self.phi_expression.t = self.t  # update time
                self.phi.assign(fenics.interpolate(self.phi_expression, self.V.sub(0).collapse()))

                # save
                if (step % self.save_rate == 0) or (step == self.steps) or self.runtime_error_occurred:
                    self._write_files(step)

                # update progress bar
                self.pbar.update(1)

                # update time
                self.t += dt
                step += 1

        return 0


class RHTestTipCellActivation(RHSimulation):
    def __init__(self,
                 spatial_dimension: int,
                 standard_params: Parameters,
                 patient_parameters: Dict,
                 out_folder_name: str = mansim.default_data_folder_name,
                 out_folder_mode: str = None,
                 sim_rationale: str = "No comment",
                 slurm_job_id: int = None,
                 load_from_cache: bool = False,
                 write_checkpoints: bool = True):
        # init super class
        super().__init__(spatial_dimension=spatial_dimension,
                         sim_parameters=standard_params,
                         patient_parameters=patient_parameters,
                         out_folder_name=out_folder_name,
                         out_folder_mode=out_folder_mode,
                         sim_rationale=sim_rationale,
                         slurm_job_id=slurm_job_id,
                         load_from_cache=load_from_cache,
                         write_checkpoints=write_checkpoints)

        self.df_standard_params: pd.DataFrame = self.sim_parameters.as_dataframe()

    def run(self, append_result_to_csv: str or None = None, **kwargs) -> bool:
        """
        Run simulation for tip cell activation. Return True if a runtime error occurred, False otherwise.
        """
        self._check_simulation_properties()  # Check class proprieties. Return error if something does not work.

        self._sim_mkdir()  # create all simulation folders

        self._fill_reproduce_folder()  # store current script in reproduce folder to keep track of the code

        self._generate_mesh()  # generate mesh

        self._spatial_discretization()  # initialize function space

        self._generate_sim_parameters_independent_initial_conditions()

        # get params dictionary
        self._generate_test_parameters(**kwargs)

        # generate column name
        columns_name = ["tip_cell_activated",
                        *[f"{k} (range: [{np.amin(r)}, {np.amax(r)}])" for k, r in self.params_dictionary.items()]]

        tip_cell_activation_df = self._setup_output_dataframe(columns_name, append_result_to_csv)

        # init pbar
        self._set_pbar(total=len(list(product(*self.params_dictionary.values()))))

        # iterate on parameters
        for param_values in product(*self.params_dictionary.values()):
            # set parameters value
            current_sim_parameters = Parameters(self.df_standard_params)
            for param_name, param_value in zip(self.params_dictionary.keys(), param_values):
                current_sim_parameters.set_value(param_name, param_value)

            # generate sim parameters dependent initial conditions
            self._generate_sim_parameters_dependent_initial_conditions(sim_parameters=current_sim_parameters)

            # call activate tip cell
            self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, 1)

            # check if tip cell has been activated
            tipcell_activated = (len(self.tip_cell_manager.get_global_tip_cells_list()) > 0)

            # store result in dataframe
            tip_cell_activation_df = tip_cell_activation_df.append(
                {col: val for col, val in zip(columns_name, [tipcell_activated, *param_values])},
                ignore_index=True
            )

            if rank == 0:
                # write at each step
                tip_cell_activation_df.to_csv(f"{self.data_folder}/tipcell_activation.csv")
                # update pbar
                self.pbar.update(1)

            # wait for all procs
            comm_world.Barrier()

        self._end_simulation()  # conclude time iteration

        return self.runtime_error_occurred

    def _sim_mkdir(self):
        super()._sim_mkdir_list(self.data_folder, self.report_folder, self.reproduce_folder,
                                self.cache_mesh_folder, self.cache_c0_folder)

    def _generate_test_parameters(self, **kwargs):
        """ Generate dictionary with ranges of parameters to test """

        # if kwargs is not empty, set it as param dictionary (user directly inputs parameters to test)
        # else, use a set of default conditions to test
        if kwargs:
            params_dictionary = kwargs
        else:
            # else create default params dictionary
            tdf_i_range = np.linspace(start=0.2, stop=1.0, num=5, endpoint=True)
            D_af_range = [float(self.df_standard_params.loc["D_af", "sim_range_min"]),
                          float(self.df_standard_params.loc["D_af", "sim_value"] / 10),
                          float(self.df_standard_params.loc["D_af", "sim_value"]),
                          float(self.df_standard_params.loc["D_af", "sim_range_max"])]
            V_pT_af_range = np.logspace(np.log10(float(self.df_standard_params.loc["V_pT_af", "sim_range_min"])),
                                        np.log10(float(self.df_standard_params.loc["V_pT_af", "sim_range_max"])),
                                        num=5,
                                        endpoint=True)
            V_uc_af_range = np.logspace(np.log10(2 * float(self.df_standard_params.loc["V_d_af", "sim_range_min"])),
                                        np.log10(10 * float(self.df_standard_params.loc["V_uc_af", "sim_value"])),
                                        num=5,
                                        endpoint=True)
            params_dictionary = {"tdf_i": tdf_i_range,
                                 "D_af": D_af_range,
                                 "V_pT_af": V_pT_af_range,
                                 "V_uc_af": V_uc_af_range}

        self.params_dictionary = params_dictionary

    def _setup_output_dataframe(self, columns_name: List[str], append_to: str or None):
        # if append_to is None, generate output dataframe from columns
        # else, generate output dataframe from append_to
        if append_to is None:
            tip_cell_activation_df = pd.DataFrame(
                columns=columns_name
            )
        else:
            if rank == 0:
                append_to_content = pd.read_csv(append_to)
                append_to_content = append_to_content.to_dict('records')
            else:
                append_to_content = None
            append_to_content = comm_world.bcast(append_to_content, 0)
            tip_cell_activation_df = pd.read_csv(append_to_content)

        return tip_cell_activation_df

    def _end_simulation(self):
        # add param dictionary to sim description
        self.sim_rationale = (f"{self.sim_rationale}\n"
                              f"The following parameters where changed to the values in brakets\n")
        for p, l in self.params_dictionary.items():
            self.sim_rationale += f"- {p}: {l}\n"

        super()._end_simulation()


def run_simulation(spatial_dimension: int,
                   sim_parameters: Parameters,
                   patient_parameters: Dict,
                   steps: int,
                   save_rate: int = 1,
                   out_folder_name: str = mansim.default_data_folder_name,
                   out_folder_mode: str = None,
                   sim_rationale: str = "No comment",
                   slurm_job_id: int = None,
                   recompute_mesh: bool = False,
                   recompute_c0: bool = False,
                   write_checkpoints: bool = True,
                   save_distributed_files_to: str or None = None):
    """
    Run a simulation and store the result

    :param spatial_dimension: specify if the simulation will be in 2D or 3D.
    :param sim_parameters: parameters to be used in the simulation.
    :param patient_parameters:
    :param steps: number of simulation steps.
    :param save_rate: specify how often the simulation status will be stored (e.g. if ``save_rate=10``, it will be
    stored every 10 steps). The last simulation step is always saved.
    :param out_folder_name: name for the folder containing the output of the simulation (e.g. if
    ``out_folder_name`` is ``my_sim``, the output folder will be ``saved_sim/my_sim``).
    :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in out_folder_name; if
    ``datetime``, the output folder name will be also followed by a string containing the date and the time of the
    simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``). The latter is recommended to run multiple
    simulations of the same kind. Default is ```None``.
    :param sim_rationale: provide a rationale for running this simulation (e.g. checking the effect of this parameter).
    It will be added to the simulation report contained in ``saved_sim/sim_name/sim_info/sim_info.html``. Default is
    "No comment".
    :param slurm_job_id: slurm job ID assigned to the simulation, if performed with slurm. It is used to generate a pbar
    stored in ``slurm/<slurm job ID>.pbar``.
    to choose between two similar versions of the vessel images (see ``notebooks/vessels_image_processing.ipynb``). No
    difference in the simulations results was found using one image or another.
    :param recompute_mesh: recompute the mesh for the simulation
    :param recompute_c0: recompute c0 for the simulation. If False, the most recent c0 is used. Default is False.
    :param write_checkpoints: set to True to save simulation with checkpoints
    :param save_distributed_files_to: set where the distributed files should be saved on the local node file system.
    """
    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                 Init Simulation
    # ---------------------------------------------------------------------------------------------------------------- #

    # init simulation object
    simulation = RHTimeSimulation(
        spatial_dimension=spatial_dimension,
        sim_parameters=sim_parameters,
        patient_parameters=patient_parameters,
        steps=steps,
        save_rate=save_rate,
        out_folder_name=out_folder_name,
        out_folder_mode=out_folder_mode,
        sim_rationale=sim_rationale,
        slurm_job_id=slurm_job_id,
        load_from_cache=(not recompute_mesh) or (not recompute_c0),
        write_checkpoints=write_checkpoints,
        save_distributed_files_to=save_distributed_files_to
    )
    # run simulation
    simulation.run()

    return simulation.runtime_error_occurred


def resume_simulation(resume_from: str,
                      steps: int,
                      save_rate: int = 1,
                      out_folder_name: str = mansim.default_data_folder_name,
                      out_folder_mode: str = None,
                      sim_rationale: str = "No comment",
                      slurm_job_id: int = None,
                      write_checkpoints: bool = True):
    """
    Resume a simulation stored in a given folder.

    :param resume_from: folder containing the simulation data to be resumed. It must contain a ``resume`` folder
    (e.g. if ``resume_from=/home/user/my_sim``, there must be a ``/home/user/my_sim/resume`` folder).
    :param steps: number of simulation steps.
    :param save_rate: specify how often the simulation status will be stored (e.g. if ``save_rate=10``, it will be
    stored every 10 steps). The last simulation step is always saved.
    :param out_folder_name: name for the simulation. It will be used to name the folder containing the output of the
    simulation (e.g. ``saved_sim/sim_name``).
    :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in out_folder_name; if
    ``datetime``, the output folder name will be also followed by a string containing the date and the time of the
    simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``). The latter is recommended to run multiple
    simulations of the same kind. Default is ```None``.
    :param sim_rationale: provide a rationale for running this simulation (e.g. checking the effect of this parameter).
    It will be added to the simulation report contained in ``saved_sim/sim_name/sim_info/sim_info.html``. Default is
    "No comment".
    :param slurm_job_id: slurm job ID assigned to the simulation, if performed with slurm. It is used to generate a pbar
    stored in ``slurm/<slurm job ID>.pbar``.
    :param write_checkpoints: set to True to save simulation with checkpoints
    """
    simulation = RHTimeSimulation.resume(resume_from=Path(resume_from),
                                         steps=steps,
                                         save_rate=save_rate,
                                         out_folder_name=out_folder_name,
                                         out_folder_mode=out_folder_mode,
                                         sim_rationale=sim_rationale,
                                         slurm_job_id=slurm_job_id,
                                         write_checkpoints=write_checkpoints)
    # run simulation
    simulation.run()

    return simulation.runtime_error_occurred


def test_tip_cell_activation(spatial_dimension: int,
                             standard_sim_parameters: Parameters,
                             patient_parameters: Dict,
                             out_folder_name: str = mansim.default_data_folder_name,
                             out_folder_mode: str = None,
                             sim_rationale: str = "No comment",
                             slurm_job_id: int = None,
                             results_df: str = None,
                             recompute_mesh: bool = False,
                             recompute_c0: bool = False,
                             write_checkpoints: bool = True,
                             **kwargs):
    simulation = RHTestTipCellActivation(spatial_dimension,
                                         standard_params=standard_sim_parameters,
                                         patient_parameters=patient_parameters,
                                         out_folder_name=out_folder_name,
                                         out_folder_mode=out_folder_mode,
                                         sim_rationale=sim_rationale,
                                         slurm_job_id=slurm_job_id,
                                         load_from_cache=(not recompute_mesh) or (not recompute_c0),
                                         write_checkpoints=write_checkpoints)

    return simulation.run(results_df, **kwargs)
