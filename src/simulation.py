"""
Contains methods and classes to run and resume simulations in 2D and 3D
"""
import json
import fenics
import time
import logging
from pathlib import Path
import sys
import pandas as pd
from itertools import product
from pint import Quantity, UnitRegistry
from tqdm import tqdm
import numpy as np
import shutil
from typing import Dict
from petsc4py import PETSc
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansim
import mocafe.angie.forms
from mocafe.fenut.parameters import Parameters
from mocafe.angie.tipcells import TipCellManager, load_tip_cells_from_json
from mocafe.fenut.solvers import PETScProblem, PETScNewtonSolver
from mocafe.fenut.log import confgure_root_logger_with_standard_settings
import src.forms
from src.expressions import get_growing_RH_expression, BWImageExpression
from src.simulation2d import compute_2d_mesh_for_patient
from src.simulation3d import get_3d_mesh_for_patient, get_3d_c0


# MPI variables
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()

# Patterns ignored when copying code after simulation
ignored_patterns = ["README.md",
                    "saved_sim*",
                    "*.ipynb_checkpoints*",
                    "sif",
                    "visualization",
                    "*pycache*",
                    ".thumbs"]

# get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RHSimulation:
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
                 activate_logger: bool = False,
                 load_from_cache: bool = False):
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
        :param activate_logger: activate the standard logger to save detailed information on the simulation activity.
        :param load_from_cache: load mesh and some function from the cache. Default is False.
        """
        # parameters
        self.sim_parameters: Parameters = sim_parameters  # simulation parameters
        self.patient_parameters: Dict = patient_parameters  # patient parameters
        self.lsp = {"ksp_type": "gmres", "pc_type": "gamg", "ksp_monitor": None}  # linear solver parameters

        # proprieties
        self.spatial_dimension: int = spatial_dimension  # spatial dimension
        self.init_time: float = time.perf_counter()  # initial simulation time
        self.ureg: UnitRegistry = self._get_ureg()  # unit registry
        self.steps: int = steps  # simulations steps
        self.save_rate: int = save_rate  # set how often writes the output
        self.sim_rationale: str = sim_rationale  # sim description
        self.slurm_job_id: int = slurm_job_id  # slurm job id (if available)
        self.error_msg = None  # error message in case of simulation errors
        self.out_folder_name = out_folder_name

        # flags
        self.recompute_mesh = load_from_cache  # set if mesh should be recomputed
        self.recompute_c0 = load_from_cache  # set if c0 should be recomputed# set up error flag
        self.runtime_error_occurred = False  # flag to activate in case of sim errors
        self.__resumed = False  # secret flag to check if simulation has been resumed

        # folders
        self.data_folder: Path = mansim.setup_data_folder(folder_path=f"saved_sim/{out_folder_name}",
                                                          auto_enumerate=out_folder_mode)
        self.report_folder: Path = self.data_folder / Path("sim_info")
        self.reproduce_folder: Path = self.data_folder / Path("0_reproduce")
        self.resume_folder: Path = self.data_folder / Path("resume")
        cache_folder = Path(".sim_cache")
        self.cache_mesh_folder: Path = cache_folder / Path("mesh")
        self.cache_c0_folder: Path = cache_folder / Path("c0")
        self.slurm_folder = Path("slurm")

        # XDMF files
        self.c_xdmf, self.af_xdmf, self.grad_af_xdmf, self.tipcells_xdmf, self.phi_xdmf = fu.setup_xdmf_files(
            ["c", "af", "grad_af", "tipcells", "phi"],
            self.data_folder,
            {"flush_output": True, "rewrite_function_mesh": False})
        self.cache_mesh_xdmf = self.cache_mesh_folder / Path("mesh.xdmf")
        self.cache_c0_xdmf = self.cache_c0_folder / Path("c0.xdmf")
        self.__resume_files_dict: Dict or None = None  # dictionary containing files to resume

        # Other files
        self.mesh_parameters_file: Path = self.cache_mesh_folder / Path("mesh_parameters.csv")

        # Pbar file
        if (self.slurm_job_id is not None) and (rank == 0):
            self.pbar_file = open(f"slurm/{self.slurm_job_id}pbar.o", 'w')
        else:
            self.pbar_file = sys.stdout

        # setup Mocafe Logger
        if activate_logger:
            confgure_root_logger_with_standard_settings(self.data_folder)

    def _get_ureg(self) -> UnitRegistry:
        """
        Generate a UnitRegistry with the arbitrary units as defined in the Simulation Parameters
        """
        # initialize unit registry
        local_ureg = UnitRegistry()

        # get parameters dataframe
        parameters_df = self.sim_parameters.as_dataframe()

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

        self._time_iteration()  # run simulation in time

        self._end_simulation()  # conclude time iteration

        return self.runtime_error_occurred

    def _check_simulation_properties(self):
        # check spatial dimension
        assert (self.spatial_dimension == 2 or self.spatial_dimension == 3), \
            f"Cannot run simulation for dimension {self.spatial_dimension}"

        # check if steps is positive
        assert self.steps > 0, "Simulation should have a positive number of steps"

    def _sim_mkdir(self):
        if rank == 0:
            # generate simulation folder
            for folder in [self.data_folder, self.report_folder, self.reproduce_folder, self.resume_folder,
                           self.cache_mesh_folder, self.cache_c0_folder]:
                folder.mkdir(exist_ok=True, parents=True)

            # if slurm id is provided, generate slurm folder
            if self.slurm_job_id is not None:
                self.slurm_folder.mkdir(exist_ok=True)

        comm_world.Barrier()

    def _fill_reproduce_folder(self):
        if rank == 0:
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

    def _resume_mesh(self):
        assert self.__resumed, "Trying to resume mesh but simulation was not resumed"
        logger.info("Loading Mesh... ")
        mesh = fenics.Mesh()
        with fenics.XDMFFile(str(self.__resume_files_dict["mesh.xdmf"])) as infile:
            infile.read(mesh)
        self.mesh = mesh
        self.mesh_parameters = Parameters(pd.read_csv(self.__resume_files_dict["mesh_parameters.csv"]))

    def _spatial_discretization(self):
        self.V = fu.get_mixed_function_space(self.mesh, 3)
        self.vec_V = fenics.VectorFunctionSpace(self.mesh, "CG", 1)  # for grad_af

    def _generate_initial_conditions(self):
        # capillaries
        if self.spatial_dimension == 2:
            c_old_expression = BWImageExpression(self.patient_parameters["pic2d"], [-1, 1], self.ureg)
            self.c_old = fenics.interpolate(c_old_expression, self.V.sub(0).collapse())
        else:
            # define c0 function
            self.c_old = fenics.Function(self.V.sub(0).collapse())
            # get 3d c0
            get_3d_c0(self.c_old, self.patient_parameters, self.mesh_parameters, self.cache_c0_xdmf, self.recompute_c0)
        # name c_old
        self.c_old.rename("c", "capillaries")

        # auxiliary fun for capillaries, initially set to 0
        logger.info("Computing mu0...")
        self.mu_old = fenics.interpolate(fenics.Constant(0.), self.V.sub(0).collapse())

        # define initial condition for tumor
        logger.info("Computing phi0...")
        # initial semiaxes of the tumor
        self.phi_expression = get_growing_RH_expression(self.sim_parameters,
                                                        self.patient_parameters,
                                                        self.mesh_parameters,
                                                        self.ureg,
                                                        0,
                                                        self.spatial_dimension)
        self.phi = fenics.interpolate(self.phi_expression, self.V.sub(0).collapse())
        self.phi.rename("phi", "retinal hemangioblastoma")

        # t_c_f_function (dynamic tip cell position)
        logger.info("Computing t_c_f_function...")
        self.t_c_f_function = fenics.interpolate(fenics.Constant(0.), self.V.sub(0).collapse())
        self.t_c_f_function.rename("tcf", "tip cells function")

        # af
        logger.info("Computing af0...")
        self.af_old = fenics.Function(self.V.sub(0).collapse())
        self.__compute_af_0()
        self.af_old.rename("af", "angiogenic factor")

        # af gradient
        logger.info("Computing grad_af0...")
        self.ge = GradientEvaluator()
        self.grad_af_old = fenics.Function(self.vec_V)
        self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
        self.grad_af_old.rename("grad_af", "angiogenic factor gradient")

        # define tip cell manager
        self.tip_cell_manager = TipCellManager(self.mesh, self.sim_parameters)

        # define initial time
        self.t0 = 0

        # write initial conditions
        self.__write_xdmf_files(0)

    def _resume_initial_conditions(self):
        # load incremental tip cells
        with open(str(self.__resume_files_dict["incremental_tip_cells.json"]), "r") as infile:
            input_itc = json.load(infile)

        # get last stem of the resumed folder
        last_step = max([int(step.replace("step_", "")) for step in input_itc])

        # capillaries
        logger.info("Loading c0... ")
        self.c_old = fenics.Function(self.V.sub(0).collapse())
        resume_c_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["c.xdmf"]))
        with resume_c_xdmf as infile:
            infile.read_checkpoint(self.c_old, "c", 0)
        self.c_old.rename("c", "capillaries")

        # mu
        logger.info("Loading mu... ")
        self.mu_old = fenics.Function(self.V.sub(0).collapse())
        resume_mu_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["mu.xdmf"]))
        with resume_mu_xdmf as infile:
            infile.read_checkpoint(self.mu_old, "mu", 0)

        # phi
        logger.info("Generating phi... ")
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
        logger.info("Loading af... ")
        self.af_old = fenics.Function(self.V.sub(0).collapse())
        resume_af_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["af.xdmf"]))
        with resume_af_xdmf as infile:
            infile.read_checkpoint(self.af_old, "af", 0)
        self.af_old.rename("af", "angiogenic factor")

        # af gradient
        logger.info("Loading af_grad... ")
        self.ge = GradientEvaluator()
        self.grad_af_old = fenics.Function(self.vec_V)
        resume_grad_af_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["grad_af.xdmf"]))
        with resume_grad_af_xdmf as infile:
            infile.read_checkpoint(self.grad_af_old, "grad_af", 0)
        self.grad_af_old.rename("grad_af", "angiogenic factor gradient")

        # define tip cell manager
        initial_tcs = load_tip_cells_from_json(str(self.__resume_files_dict["tipcells.json"]))
        self.tip_cell_manager = TipCellManager(self.mesh,
                                               self.sim_parameters,
                                               initial_tcs=initial_tcs)

        # definie initial time
        self.t0 = last_step

        # write initial conditions
        self.__write_xdmf_files(last_step)

    def __compute_af_0(self, options: Dict = None):
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
        af_form = src.forms.angiogenic_factors_form_eq(af, self.phi, self.c_old, v, self.sim_parameters)
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

    def _time_iteration(self):
        logger.info("Defining weak form...")

        # define variable functions
        u = fenics.Function(self.V)
        fenics.assign(u, [self.af_old, self.c_old, self.mu_old])
        af, c, mu = fenics.split(u)
        # define test functions
        v1, v2, v3 = fenics.TestFunctions(self.V)
        # build total form
        af_form = src.forms.angiogenic_factors_form_eq(af, self.phi, c, v1, self.sim_parameters)
        capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(c, self.c_old, mu, self.mu_old, v2, v3,
                                                                                 self.sim_parameters)
        form = af_form + capillaries_form

        logger.info("Defining problem...")

        # define Jacobian
        J = fenics.derivative(form, u)
        # define problem
        problem = PETScProblem(J, form, [])
        solver = PETScNewtonSolver(self.lsp, self.mesh.mpi_comm())

        # init time iteration
        t = self.t0
        dt = self.sim_parameters.get_value("dt")

        pbar = tqdm(total=self.steps,
                    ncols=100,
                    desc=self.out_folder_name,
                    file=self.pbar_file,
                    disable=True if rank != 0 else False)

        # log
        logger.info("Starting time iteration...")
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
            try:
                solver.solve(problem, u.vector())
            except RuntimeError as e:
                # store error info
                self.runtime_error_occurred = True
                self.error_msg = str(e)
                logger.error(self.error_msg)
                # stop simulation
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
                self.__write_xdmf_files(t)

            # update progress bar
            pbar.update(1)

    def _end_simulation(self):
        # close pbar file
        if (rank == 0) and (self.slurm_job_id is not None):
            self.pbar_file.close()

        # save resume info
        self.__save_resume_info()

        # save sim report
        if self.runtime_error_occurred:
            sim_description = "RUNTIME ERROR OCCURRED. \n" + self.sim_rationale
        else:
            sim_description = self.sim_rationale
        mansim.save_sim_info(data_folder=self.report_folder,
                             parameters={"Sim parameters:": self.sim_parameters,
                                         "Mesh parameters": self.mesh_parameters},
                             execution_time=time.perf_counter() - self.init_time,
                             sim_name=self.out_folder_name,
                             sim_description=sim_description,
                             error_msg=self.error_msg)

        # save parameters
        self.sim_parameters.as_dataframe().to_csv(self.report_folder / Path("sim_parameters.csv"))
        self.mesh_parameters.as_dataframe().to_csv(self.report_folder / Path("mesh_parameters.csv"))
        with open(self.report_folder / Path("patient_parameters.json"), "w") as outfile:
            json.dump(self.patient_parameters, outfile)

        comm_world.Barrier()  # wait for all processes

    def __save_resume_info(self):
        """
        Save the mesh and/or the fenics Functions in a resumable format (i.e. using the FEniCS function
        `write_checkpoint`).
        """
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
                outfile.write_checkpoint(fnc, name, 0, fenics.XDMFFile.Encoding.HDF5, False)
            saved_files[name] = str(Path(file_name).resolve())

        # store tip cells position
        file_name = f"{self.resume_folder}/tipcells.json"
        self.tip_cell_manager.save_tip_cells(file_name)
        saved_files["tipcells"] = str(Path(file_name).resolve())

    def __write_xdmf_files(self, t: int):
        # write files
        self.af_xdmf.write(self.af_old, t)
        self.c_xdmf.write(self.c_old, t)
        self.grad_af_xdmf.write(self.grad_af_old, t)
        self.phi_xdmf.write(self.phi, t)
        # save tip cells current position
        self.tipcells_xdmf.write(self.t_c_f_function, t)

    @classmethod
    def resume(cls,
               resume_from: Path,
               steps: int,
               save_rate: int,
               out_folder_name: str = mansim.default_data_folder_name,
               out_folder_mode: str = None,
               sim_rationale: str = "No comment",
               slurm_job_id: int = None,
               activate_logger: bool = False):
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
        :param activate_logger: activate the standard logger to save detailed information on the simulation activity.
        """
        # ----------------------------------------------------------------------------------------------------------- #
        # 1. Check resume folder consistency
        # ----------------------------------------------------------------------------------------------------------- #
        logger.info("Checking if it's possible to resume simulation ... ")

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
        mesh_parameters = Parameters(pd.read_csv(mesh_parameters_csv))
        if mesh_parameters.is_value_present("Lz"):
            spatial_dimension = 3
        else:
            spatial_dimension = 2

        # load simulation parameters
        sim_parameters = Parameters(pd.read_csv(sim_parameters_csv))

        # load patient parameters
        with open(patient_parameters_file, "r") as infile:
            patient_parameters = json.load(infile)

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
                         activate_logger=activate_logger)
        simulation.__resumed = True
        simulation.__resume_files_dict = resume_files_dict
        return simulation


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


def run_simulation(spatial_dimension: int,
                   sim_parameters: Parameters,
                   patient_parameters: Dict,
                   steps: int,
                   save_rate: int = 1,
                   out_folder_name: str = mansim.default_data_folder_name,
                   out_folder_mode: str = None,
                   sim_rationale: str = "No comment",
                   slurm_job_id: int = None,
                   activate_logger: bool = False,
                   recompute_mesh: bool = False,
                   recompute_c0: bool = False):
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
    :param activate_logger: activate the standard logger to save detailed information on the simulation activity.
    :param recompute_mesh: recompute the mesh for the simulation
    :param recompute_c0: recompute c0 for the simulation. If False, the most recent c0 is used. Default is False.
    """
    # init simulation object
    simulation = RHSimulation(
        spatial_dimension=spatial_dimension,
        sim_parameters=sim_parameters,
        patient_parameters=patient_parameters,
        steps=steps,
        save_rate=save_rate,
        out_folder_name=out_folder_name,
        out_folder_mode=out_folder_mode,
        sim_rationale=sim_rationale,
        slurm_job_id=slurm_job_id,
        activate_logger=activate_logger,
        load_from_cache=(not recompute_mesh) or (not recompute_c0)
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
                      activate_logger: bool = False):
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
    :param activate_logger: activate the standard logger to save detailed information on the simulation activity.
    """
    simulation = RHSimulation.resume(resume_from=Path(resume_from),
                                     steps=steps,
                                     save_rate=save_rate,
                                     out_folder_name=out_folder_name,
                                     out_folder_mode=out_folder_mode,
                                     sim_rationale=sim_rationale,
                                     slurm_job_id=slurm_job_id,
                                     activate_logger=activate_logger)
    # run simulation
    simulation.run()

    return simulation.runtime_error_occurred


# def test_tip_cell_activation(spatial_dimension: int,
#                              df_standard_params: pd.DataFrame,
#                              patient_parameters: Dict,
#                              out_folder_name: str = mansim.default_data_folder_name,
#                              out_folder_mode: str = None,
#                              slurm_job_id: int = None,
#                              results_df: str = None,
#                              recompute_mesh: bool = False,
#                              recompute_c0: bool = False,
#                              **kwargs):
#     """
#     Test if tip cells activate at first simulation step for a given set of parameters. The default ranges are those
#     used for Fig. 3 of the manuscript. To test a different range or another parameters, use ``kwargs``. For instance,
#     you can do:
#
#         test_tip_cell_activation(spatial_dimension=3,
#                                  df_standard_params=sim_parameters,
#                                  sim_name="test",
#                                  min_tumor_diameter=[0.3, 0.4, 0.5])
#
#     :param spatial_dimension: 2 if 2D, 3 if 3D.
#     :param df_standard_params: dataframe containing the parameter set to be used as base for all simulations. The
#     parameters will be changed according to the given ranges or to the default ranges.
#     :param out_folder_name: name of the output folder.
#     :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in out_folder_name; if
#     ``datetime``, the output folder name will be also followed by a string containing the date and the time of the
#     simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``). The latter is recommended to run multiple
#     simulations of the same kind. Default is ```None``.
#     :param slurm_job_id: slurm job id, if available. Default is None.
#     :param vessels_no_discontinuities: use the vessels image without discontinuities
#     :param results_df: give the dataframe where to save the tip cells activation results as input. Default is None.
#     If None, a new dataframe will be created.
#     :param recompute_mesh: recompute the mesh for the 3d simulation. Default is False
#     :param recompute_c0: recompute the initial condition for the capillaries. Default is False.
#     :param kwargs: give a parameter name and a range as input to test whether tip cells activation occurs in the given
#     parameters range. Each parameter must be associated with a list (even if a single value is given).
#     """
#     # -- Initial setup
#     local_ureg = UnitRegistry()
#
#     # set up folders
#     data_folder = mansim.setup_data_folder(folder_path=f"{Path('saved_sim')}/{out_folder_name}",
#                                            auto_enumerate=out_folder_mode)
#
#     # create 'mesh' folder to store the mesh
#     mesh_folder: Path = Path("./mesh")
#     if rank == 0:
#         mesh_folder.mkdir(exist_ok=True)
#
#     # create c0 folder to store the capillaries initial condition
#     c0_folder: Path = Path("./c0")
#     if rank == 0:
#         c0_folder.mkdir(exist_ok=True)
#
#     # load arbitrary units
#     local_ureg = load_arbitrary_units(local_ureg,
#                                       df_standard_params,
#                                       sau_name="Space Arbitrary Unit",
#                                       tau_name="Time Arbitrary Unit",
#                                       afau_name="AFs Arbitrary Unit")
#
#     # --- Define parameters sets to test
#     if kwargs:
#         # if kwargs is not empty, set it as param dictionary
#         if all([isinstance(v, list) for v in kwargs.values()]):
#             params_dictionary = kwargs
#         else:
#             raise TypeError("Every parameter must be associated with a list.")
#
#     else:
#         # else create default params dictionary
#         tumor_diameter_range = np.linspace(float(df_standard_params.loc["min_tumor_diameter", "sim_value"]) / 10,
#                                            float(df_standard_params.loc["min_tumor_diameter", "sim_value"]),
#                                            num=5,
#                                            endpoint=True)
#         D_af_range = [float(df_standard_params.loc["D_af", "sim_range_min"]),
#                       float(df_standard_params.loc["D_af", "sim_value"] / 10),
#                       float(df_standard_params.loc["D_af", "sim_value"]),
#                       float(df_standard_params.loc["D_af", "sim_range_max"])]
#         V_pT_af_range = np.logspace(np.log10(float(df_standard_params.loc["V_pT_af", "sim_range_min"])),
#                                     np.log10(float(df_standard_params.loc["V_pT_af", "sim_range_max"])),
#                                     num=5,
#                                     endpoint=True)
#         V_uc_af_range = np.logspace(np.log10(2 * float(df_standard_params.loc["V_d_af", "sim_range_min"])),
#                                     np.log10(10 * float(df_standard_params.loc["V_uc_af", "sim_value"])),
#                                     num=5,
#                                     endpoint=True)
#         params_dictionary = {"min_tumor_diameter": tumor_diameter_range,
#                              "D_af": D_af_range,
#                              "V_pT_af": V_pT_af_range,
#                              "V_uc_af": V_uc_af_range}
#
#     # generate dataframe
#     columns_name = ["tip_cell_activated",
#                     *[f"{k} (range: [{np.amin(r)}, {np.amax(r)}])" for k, r in params_dictionary.items()]]
#     if results_df is None:
#         tip_cell_activation_df = pd.DataFrame(
#             columns=columns_name
#         )
#     else:
#         tip_cell_activation_df = pd.read_csv(results_df)
#
#     # --- Define linear solver parameters
#     lsp = {"ksp_type": "gmres", "pc_type": "gamg"}
#
#     # ---------------------------------------------------------------------------------------------------------------- #
#     #                                                 Mesh Definition
#     # ---------------------------------------------------------------------------------------------------------------- #
#     # define mesh_file
#     mesh_xdmf: Path = mesh_folder / Path("mesh.xdmf")
#     # define mesh_parameters file
#     mesh_parameters_file: Path = mesh_folder / Path("mesh_parameters.csv")
#
#     if spatial_dimension == 2:
#         mesh, mesh_parameters = compute_2d_mesh_for_patient(df_standard_params,
#                                                             local_ureg,
#                                                             patient_parameters)
#     else:
#         mesh, mesh_parameters = get_3d_mesh_for_patient(df_standard_params,
#                                                         local_ureg,
#                                                         patient_parameters,
#                                                         mesh_xdmf,
#                                                         mesh_parameters_file,
#                                                         recompute_mesh)
#
#     # --- Spatial discretization
#     logger.info("Starting spatial discretization")
#     V = fu.get_mixed_function_space(mesh, 3)
#     # define fun space for grad_af
#     vec_V = fenics.VectorFunctionSpace(mesh, "CG", 1)
#
#     # --- Define intial condition for c
#     if spatial_dimension == 2:
#         cb = "notebooks/out/RH_vessels_binary_ND.png"
#         c_old_expression = BWImageExpression(cb, [-1, 1], local_ureg)
#         c_old = fenics.interpolate(c_old_expression, V.sub(0).collapse())
#     else:
#         c_old = fenics.Function(V.sub(0).collapse())
#         # define c0 file
#         c0_xdmf: Path = c0_folder / Path("c0.xdmf")
#         # get 3d c0
#         get_3d_c0(c_old, patient_parameters, mesh_parameters, c0_xdmf, recompute_c0)
#
#     # -- Define gradient evaluator to compute grad_af
#     ge = GradientEvaluator()
#
#     # define pbar
#     if rank == 0:
#         if slurm_job_id is not None:
#             pbar_file = open(f"slurm/{slurm_job_id}pbar.o", 'w')
#             pbar = tqdm(total=len(list(product(*params_dictionary.values()))),
#                         ncols=100, desc=out_folder_name, file=pbar_file)
#         else:
#             pbar_file = None
#             pbar = tqdm(total=len(list(product(*params_dictionary.values()))),
#                         ncols=100, desc=out_folder_name)
#     else:
#         pbar_file = None
#         pbar = None
#     comm_world.Barrier()
#
#     # start iteration
#     for param_values in product(*params_dictionary.values()):
#         # set parameters value
#         sim_parameters = Parameters(df_standard_params)
#         for param_name, param_value in zip(params_dictionary.keys(), param_values):
#             sim_parameters.set_value(param_name, param_value)
#
#         # --- Initial conditions
#         # phi
#         if spatial_dimension == 3:
#             logger.info("Computing phi0...")
#         phi_expression = get_growing_RH_expression(sim_parameters,
#                                                    patient_parameters,
#                                                    mesh_parameters,
#                                                    local_ureg,
#                                                    0,
#                                                    spatial_dimension)
#         phi = fenics.interpolate(phi_expression, V.sub(0).collapse())
#         phi.rename("phi", "retinal hemangioblastoma")
#
#         # af
#         if spatial_dimension == 3:
#             logger.info("Computing af0...")
#         af_old = fenics.Function(V.sub(0).collapse())
#         compute_af_0(af_old, V.sub(0).collapse(), phi, c_old, sim_parameters, lsp)
#         af_old.rename("af", "angiogenic factor")
#
#         # af gradient
#         if spatial_dimension == 3:
#             logger.info("Computing grad_af0...")
#         grad_af_old = fenics.Function(vec_V)
#         ge.compute_gradient(af_old, vec_V, grad_af_old, lsp)
#         grad_af_old.rename("grad_af", "angiogenic factor gradient")
#
#         # define tip cell manager
#         tip_cell_manager = TipCellManager(mesh, sim_parameters)
#
#         # call activate tip cell
#         tip_cell_manager.activate_tip_cell(c_old, af_old, grad_af_old, 1)
#
#         # check if tip cell has been activated
#         tipcell_activated = (len(tip_cell_manager.get_global_tip_cells_list()) == 1)
#
#         # store result in dataframe
#         tip_cell_activation_df = tip_cell_activation_df.append(
#             {col: val for col, val in zip(columns_name, [tipcell_activated, *param_values])},
#             ignore_index=True
#         )
#
#         if rank == 0:
#             # write at each step
#             tip_cell_activation_df.to_csv(f"{data_folder}/tipcell_activation.csv")
#             # update pbar
#             pbar.update(1)
#
#         comm_world.Barrier()
#         break
#
#     if rank == 0 and (slurm_job_id is not None):
#         pbar_file.close()
