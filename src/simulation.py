"""
Contains methods and classes to run and resume simulations in 2D and 3D
"""
import sys
import time
import shutil
import logging
import skimage
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import product
from typing import Dict, List
from pint import Quantity, UnitRegistry
import ufl
import dolfinx
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
import mocafe.fenut.fenut as fu
import mocafe.fenut.mansimdata as mansim
import mocafe.angie.forms
from mocafe.fenut.parameters import Parameters
from mocafe.angie.tipcells import TipCellManager, load_tip_cells_from_json
import src.forms
from src.ioutils import (read_parameters, write_parameters, dump_json, load_json, move_files_once_per_node,
                         rmtree_if_exists_once_per_node)
from src.expressions import RHEllipsoid, VesselReconstruction
from src.simulation2d import compute_2d_mesh_for_patient
from src.simulation3d import get_3d_mesh_for_patient


# MPI variables
comm_world = MPI.COMM_WORLD
rank = comm_world.rank

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
    sau_name = "SpaceArbitraryUnit"
    tau_name = "TimeArbitraryUnit"
    afau_name = "AFsArbitraryUnit"

    # define sau, tau and afau according to dataframe
    local_ureg.define(f"{sau_name} = "
                      f"{parameters_df.loc[sau_name, 'real_value']} * {parameters_df.loc[sau_name, 'real_um']} = "
                      f"sau")
    local_ureg.define(f"{tau_name} = "
                      f"{parameters_df.loc[tau_name, 'real_value']} * {parameters_df.loc[tau_name, 'real_um']} = "
                      f"tau")
    local_ureg.define(f"{afau_name} = "
                      f"{parameters_df.loc[afau_name, 'real_value']} * {parameters_df.loc[afau_name, 'real_um']} = "
                      f"afau")

    return local_ureg


def get_c0(spatial_dimension: int,
           c_old: dolfinx.fem.Function,
           patient_parameters: Dict,
           mesh_parameters: Parameters,
           c0_xdmf: Path,
           recompute_c0: bool,
           write_checkpoints: bool) -> None:
    """
    Get the initial condition for c0. If c0_xdmf is empty, it computes the initial condition and saves it in the
    file for future reuse.

    :param spatial_dimension: 2D or 3D
    :param c_old: FEniCS function to store initial condition
    :param patient_parameters: patient-specific parameters Dict
    :param mesh_parameters: mesh parameters (used in the computation)
    :param c0_xdmf: file containing c0 (if already computed once)
    :param write_checkpoints: set to True if the latest computed c0 should be stored as XDMF file
    :param recompute_c0: force the function to recompute c0 even if c0_xdmf is not empty.
    """

    # # define c0_label in xdmf file
    # c0_label = "c0"

    # if c0 file exists and it is not necessary to recompute it, load it. Else compute it.
    if c0_xdmf.exists() and (recompute_c0 is False):
        # logger.info(f"Found existing c0 in {c0_xdmf}. Loading...")
        # with fenics.XDMFFile(str(c0_xdmf.resolve())) as infile:
        #     infile.read_checkpoint(c_old, c0_label, 0)
        raise NotImplementedError(f"Mesh loading not implemented yet")
    else:
        # compute c0
        logger.info("Computing c0...")
        _compute_c_0(spatial_dimension, c_old, patient_parameters, mesh_parameters)
        # store c0
        if write_checkpoints:
            # with fenics.XDMFFile(str(c0_xdmf.resolve())) as outfile:
            #     outfile.write_checkpoint(c_old, c0_label, 0, fenics.XDMFFile.Encoding.HDF5, False)
            raise NotImplementedError(f"Checkpointing not implemented yet")


def _compute_c_0(spatial_dimension: int,
                 c_old: dolfinx.fem.Function,
                 patient_parameters: Dict,
                 mesh_parameters: Parameters):
    """
    Computes c initial condition in 3D
    """
    # load binary image
    pic2d_file = patient_parameters["pic2d"]
    # load distance transform
    pic2d_dt_file = pic2d_file.replace("pic2d.png", "pic2d_dt.npy")
    binary = skimage.io.imread(pic2d_file)
    pic2d_distance_transform = np.load(pic2d_dt_file)
    # get half depth
    half_depth = patient_parameters["half_depth"]
    # get z0 according to spatial dimension
    if spatial_dimension == 2:
        z0 = 0
    else:
        z0 = mesh_parameters.get_value("Lz") - half_depth
    c_old_expression = VesselReconstruction(z_0=z0,
                                            binary_array=binary,
                                            distance_transform_array=pic2d_distance_transform,
                                            mesh_Lx=mesh_parameters.get_value("Lx"),
                                            mesh_Ly=mesh_parameters.get_value("Ly"),
                                            half_depth=half_depth)
    del binary, pic2d_distance_transform

    c_old.interpolate(c_old_expression.eval)
    c_old.x.scatter_forward()


class GradientEvaluator:
    """
    Class to efficiently compute the af gradient. Defined to avoid the use of ``fenics.project(grad(af), V)``.
    This class keep the solver for the gradient saved, so to reuse it every time it is required.
    """
    def __init__(self):
        self.solver = None
        self.a = None

    def compute_gradient(self,
                         f: dolfinx.fem.Function,
                         V: dolfinx.fem.FunctionSpace,
                         sol: dolfinx.fem.Function,
                         options: Dict = None,
                         destroy: bool = False):
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
        v = ufl.TestFunction(V)

        # define lhs variational form (time independent)
        if self.solver is None:
            # define trial function
            u = ufl.TrialFunction(V)
            # define lhs for problem of finding the gradient
            a = dolfinx.fem.form(ufl.dot(u, v) * ufl.dx)
            self.a = a
            # define operator
            A = dolfinx.fem.petsc.assemble_matrix(a, [])
            A.assemble()
            # init solver
            ksp = PETSc.KSP().create(comm_world)
            # set solver options
            opts = PETSc.Options()
            for opt, opt_value in options.items():
                opts[opt] = opt_value
            ksp.setFromOptions()
            # set operator
            ksp.setOperators(A)
            # set solver
            self.solver = ksp
            # destroy A
            A.destroy()

        # define b
        L = dolfinx.fem.form(ufl.inner(ufl.grad(f), v) * ufl.dx)
        b = dolfinx.fem.petsc.assemble_vector(L)
        dolfinx.fem.petsc.apply_lifting(b, [self.a], [[]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # solve
        self.solver.solve(b, sol.vector)
        sol.x.scatter_forward()

        # destroy
        b.destroy()
        if destroy:
            self.solver.destroy()


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
        self.vec_V = dolfinx.fem.FunctionSpace(self.mesh, ufl.VectorElement("P", self.mesh.ufl_cell(), 1))

    def _generate_sim_parameters_independent_initial_conditions(self):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # get collapsed subspace
        subV_collapsed, _ = self.V.sub(0).collapse()

        # capillaries
        self.c_old = dolfinx.fem.Function(subV_collapsed)
        get_c0(self.spatial_dimension, self.c_old, self.patient_parameters, self.mesh_parameters, self.cache_c0_xdmf,
               self.recompute_c0, self.write_checkpoints)
        # name c_old
        self.c_old.name = "c"

        # auxiliary fun for capillaries, initially set to 0
        logger.info(f"{self.out_folder_name}:Computing mu0...")
        self.mu_old = dolfinx.fem.Function(subV_collapsed)

        # t_c_f_function (dynamic tip cell position)
        logger.info(f"{self.out_folder_name}:Computing t_c_f_function...")
        self.t_c_f_function = dolfinx.fem.Function(subV_collapsed)
        self.t_c_f_function.name = "tcf"

        # define initial time
        self.t0 = 0

    def _generate_sim_parameters_dependent_initial_conditions(self, sim_parameters):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # get collapsed subspace
        subV_collapsed, _ = self.V.sub(0).collapse()

        # define initial condition for tumor
        logger.info(f"{self.out_folder_name}:Computing phi0...")
        # initial semiaxes of the tumor
        self.phi_expression = RHEllipsoid(sim_parameters,
                                          self.patient_parameters,
                                          self.mesh_parameters,
                                          self.ureg,
                                          0,
                                          self.spatial_dimension)
        self.phi = dolfinx.fem.Function(subV_collapsed)
        self.phi.interpolate(self.phi_expression.eval)
        self.phi.x.scatter_forward()
        self.phi.name = "phi"

        # af
        logger.info(f"{self.out_folder_name}:Computing af0...")
        self.af_old = dolfinx.fem.Function(subV_collapsed)
        self.__compute_af_0(sim_parameters)
        self.af_old.name = "af"

        # af gradient
        logger.info(f"{self.out_folder_name}:Computing grad_af0...")
        self.ge = GradientEvaluator()
        self.grad_af_old = dolfinx.fem.Function(self.vec_V)
        self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
        self.grad_af_old.name = "grad_af"

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
        subV_collapsed, _ = self.V.sub(0).collapse()
        af = ufl.TrialFunction(subV_collapsed)
        # get test function
        v = ufl.TestFunction(subV_collapsed)
        # built equilibrium form for af
        af_form = src.forms.angiogenic_factors_form_eq(af, self.phi, self.c_old, v, sim_parameters)
        af_form_a = dolfinx.fem.form(ufl.lhs(af_form))
        af_form_L = dolfinx.fem.form(ufl.rhs(af_form))
        # define operator
        A = dolfinx.fem.petsc.assemble_matrix(af_form_a, bcs=[])
        A.assemble()
        # init solver
        ksp = PETSc.KSP().create(comm_world)
        # set solver options
        opts = PETSc.Options()
        for o, v in options.items():
            opts[o] = v
        ksp.setFromOptions()
        # set operator
        ksp.setOperators(A)
        # define b
        b = dolfinx.fem.petsc.assemble_vector(af_form_L)
        dolfinx.fem.petsc.apply_lifting(b, [af_form_a], [[]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # solve
        ksp.solve(b, self.af_old.vector)
        self.af_old.x.scatter_forward()

        # destroy object (workaround for issue: https://github.com/FEniCS/dolfinx/issues/2559)
        A.destroy()
        b.destroy()
        ksp.destroy()

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

        # destroy solver of gradient evaluator
        self.ge.solver.destroy()

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
            self.c_file, self.af_file, self.grad_af_file, self.tipcells_file, self.phi_file = \
                [dolfinx.io.VTKFile(comm_world, str(self.distributed_data_folder / Path(f"{fn}.pvd")), "w")
                 for fn in file_names]
        else:
            # specific XDMF files
            self.c_file, self.af_file, self.grad_af_file, self.tipcells_file, self.phi_file = \
                [dolfinx.io.XDMFile(comm_world, str(self.data_folder / Path(f"{fn}.xdmf")), "w")
                 for fn in file_names]

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
        raise NotImplementedError("Mesh reading not implemented yet")
        # assert self.__resumed, "Trying to resume mesh but simulation was not resumed"
        # logger.info(f"{self.out_folder_name}:Loading Mesh... ")
        # mesh = fenics.Mesh()
        # with fenics.XDMFFile(str(self.__resume_files_dict["mesh.xdmf"])) as infile:
        #     infile.read(mesh)
        # self.mesh = mesh
        # self.mesh_parameters = read_parameters(self.__resume_files_dict["mesh_parameters.csv"])

    def _generate_initial_conditions(self):
        # generate initial conditions independent of sim parameters
        super()._generate_sim_parameters_independent_initial_conditions()
        # generate initial conditions dependent from sim parameters
        super()._generate_sim_parameters_dependent_initial_conditions(self.sim_parameters)

    def _resume_initial_conditions(self):
        raise NotImplementedError(f"Resuming not implmented yet")
        # # load incremental tip cells
        # input_itc = load_json(self.__resume_files_dict["incremental_tip_cells.json"])
        #
        # # get last stem of the resumed folder
        # last_step = max([int(step.replace("step_", "")) for step in input_itc])
        #
        # # capillaries
        # logger.info(f"{self.out_folder_name}:Loading c0... ")
        # self.c_old = fenics.Function(self.V.sub(0).collapse())
        # resume_c_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["c.xdmf"]))
        # with resume_c_xdmf as infile:
        #     infile.read_checkpoint(self.c_old, "c", 0)
        # self.c_old.rename("c", "capillaries")
        #
        # # mu
        # logger.info(f"{self.out_folder_name}:Loading mu... ")
        # self.mu_old = fenics.Function(self.V.sub(0).collapse())
        # resume_mu_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["mu.xdmf"]))
        # with resume_mu_xdmf as infile:
        #     infile.read_checkpoint(self.mu_old, "mu", 0)
        #
        # # phi
        # logger.info(f"{self.out_folder_name}:Generating phi... ")
        # self.phi_expression = get_growing_RH_expression(self.sim_parameters,
        #                                                 self.patient_parameters,
        #                                                 self.mesh_parameters,
        #                                                 self.ureg,
        #                                                 last_step,
        #                                                 self.spatial_dimension)
        # self.phi = fenics.interpolate(self.phi_expression, self.V.sub(0).collapse())
        # self.phi.rename("phi", "retinal hemangioblastoma")
        #
        # # tcf function
        # self.t_c_f_function = fenics.interpolate(fenics.Constant(0.), self.V.sub(0).collapse())
        # self.t_c_f_function.rename("tcf", "tip cells function")
        #
        # # af
        # logger.info(f"{self.out_folder_name}:Loading af... ")
        # self.af_old = fenics.Function(self.V.sub(0).collapse())
        # resume_af_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["af.xdmf"]))
        # with resume_af_xdmf as infile:
        #     infile.read_checkpoint(self.af_old, "af", 0)
        # self.af_old.rename("af", "angiogenic factor")
        #
        # # af gradient
        # logger.info(f"{self.out_folder_name}:Loading af_grad... ")
        # self.ge = GradientEvaluator()
        # self.grad_af_old = fenics.Function(self.vec_V)
        # resume_grad_af_xdmf = fenics.XDMFFile(str(self.__resume_files_dict["grad_af.xdmf"]))
        # with resume_grad_af_xdmf as infile:
        #     infile.read_checkpoint(self.grad_af_old, "grad_af", 0)
        # self.grad_af_old.rename("grad_af", "angiogenic factor gradient")
        #
        # # define tip cell manager
        # if rank == 0:
        #     initial_tcs = load_tip_cells_from_json(str(self.__resume_files_dict["tipcells.json"]))
        # else:
        #     initial_tcs = None
        # initial_tcs = comm_world.bcast(initial_tcs, 0)
        # self.tip_cell_manager = TipCellManager(self.mesh,
        #                                        self.sim_parameters,
        #                                        initial_tcs=initial_tcs)
        #
        # # definie initial time
        # self.t0 = last_step

    def _write_files(self, t: int, write_mesh: bool = False):
        # write mesh
        if write_mesh:
            logger.info(f"Starting writing Mesh")
            logger.info(f"Writing af_pvd... ")
            self.af_file.write_mesh(self.mesh)
            logger.info(f"Writing c_pvd... ")
            self.c_file.write_mesh(self.mesh)
            logger.info(f"Writing grad_af_pvd... ")
            self.grad_af_file.write_mesh(self.mesh)
            logger.info(f"Writing phi_pvd... ")
            self.phi_file.write_mesh(self.mesh)
            logger.info(f"Writing tipcells_pvd... ")
            self.tipcells_file.write_mesh(self.mesh)
        # write functions
        logger.info(f"Starting writing of PVD files ")
        logger.info(f"Writing af_pvd... ")
        self.af_file.write_function(self.af_old, t)
        logger.info(f"Writing c_pvd... ")
        self.c_file.write_function(self.c_old, t)
        logger.info(f"Writing grad_af_pvd... ")
        self.grad_af_file.write_function(self.grad_af_old, t)
        logger.info(f"Writing phi_pvd... ")
        self.phi_file.write_function(self.phi, t)
        logger.info(f"Writing tipcells_pvd... ")
        self.tipcells_file.write_function(self.t_c_f_function, t)

    def _solve(self, u: dolfinx.fem.Function):
        try:
            self.solver.solve(u)
        except RuntimeError as e:
            # store error info
            self.runtime_error_occurred = True
            self.error_msg = str(e)
            logger.error(str(e))

    def _time_iteration(self):
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        u = dolfinx.fem.Function(self.V)
        # split u
        af, c, mu = ufl.split(u)
        # define test functions
        v1, v2, v3 = ufl.TestFunctions(self.V)
        # build total form
        af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters)
        capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters)
        form = af_form + capillaries_form

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        problem = dolfinx.fem.petsc.NonlinearProblem(form, u)

        # define solver
        self.solver = NewtonSolver(comm_world, problem)
        if "ksp_monitor" in self.lsp.keys():
            self.solver.report = True  # report iterations
        # set options for krylov solver
        ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        for o, v in self.lsp.items():
            opts[f"{option_prefix}{o}"] = v
        ksp.setFromOptions()

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
            self._solve(u)

            # if error occurred, stop iteration
            if self.runtime_error_occurred:
                break

            # assign to old
            new_af, new_c, new_mu = u.split()
            self.af_old.interpolate(new_af)
            self.af_old.x.scatter_forward()
            self.c_old.interpolate(new_c)
            self.c_old.x.scatter_forward()
            self.af_old.interpolate(new_mu)
            self.af_old.x.scatter_forward()
            # assign new value to grad_af_old
            self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
            # assign new value to phi
            self.phi_expression.update_time(t)
            self.phi.interpolate(self.phi_expression.eval)
            self.phi.x.scatter_forward()

            # save
            if (step % self.save_rate == 0) or (step == self.steps) or self.runtime_error_occurred:
                self._write_files(step)

            # update progress bar
            self.pbar.update(1)

    def _end_simulation(self):
        # save resume info
        if self.write_checkpoints:
            self._save_resume_info()

        # close files
        self.c_file.close()
        self.af_file.close()
        self.grad_af_file.close()
        self.tipcells_file.close()
        self.phi_file.close()

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
        raise NotImplementedError("Resuming not implemented yet")
        # logger.info("Starting checkpoint write")
        #
        # # init list of saved file
        # saved_files = {}
        #
        # # copy cached mesh in resume folder
        # for mf in self.cache_mesh_folder.glob("mesh.*"):
        #     shutil.copy(mf, self.resume_folder)
        #
        # # write functions
        # fnc_dict = {"af": self.af_old, "c": self.c_old, "mu": self.mu_old, "phi": self.phi, "grad_af": self.grad_af_old}
        # for name, fnc in fnc_dict.items():
        #     file_name = f"{self.resume_folder}/{name}.xdmf"
        #     with fenics.XDMFFile(file_name) as outfile:
        #         logger.info(f"Checkpoint writing of {name}....")
        #         outfile.write_checkpoint(fnc, name, 0, fenics.XDMFFile.Encoding.HDF5, False)
        #     saved_files[name] = str(Path(file_name).resolve())
        #
        # # store tip cells position
        # file_name = f"{self.resume_folder}/tipcells.json"
        # self.tip_cell_manager.save_tip_cells(file_name)
        # saved_files["tipcells"] = str(Path(file_name).resolve())

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
        self.delta_steps = delta_steps
        self.time_jump: float = sim_parameters.get_value("dt") * delta_steps
        # set after how many steps the adaptive solver is activated
        self.trigger_time_jump: int = trigger_adaptive_tc_activation_after_steps

    def __build_weak_form(self, u: dolfinx.fem.Function,
                          af_dt: bool = True,
                          **weak_form_kwargs) -> ufl.Form:
        # assign functions to u
        # fenics.assign(u, [self.af_old, self.c_old, self.mu_old])
        af, c, mu = ufl.split(u)
        # define test functions
        v1, v2, v3 = ufl.TestFunctions(self.V)
        # build total form
        if af_dt:
            af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters,
                                                           **weak_form_kwargs)
        else:
            af_form = src.forms.angiogenic_factors_form_eq(af, self.phi, c, v1, self.sim_parameters,
                                                           **weak_form_kwargs)
        capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters, **weak_form_kwargs
        )
        form = af_form + capillaries_form
        # return form
        return form

    # def _test_tip_cell_activation_after_delta_step(self,
    #                                                u_temp: fenics.Function,
    #                                                phi_temp: fenics.Function,
    #                                                phi_exp_temp: fenics.Expression,
    #                                                grad_af_temp: fenics.Function,
    #                                                t: float,
    #                                                time_jump: float):
    #     # define bigger dt
    #     bigger_dt = time_jump
    #     # update tumor dimension
    #     phi_exp_temp.t = t + time_jump
    #     phi_temp.assign(fenics.interpolate(phi_exp_temp, self.V.sub(0).collapse()))
    #     # build weak from for dt = time_jump * dt
    #     form = self.__build_weak_form(u_temp, phi_temp=phi_temp, dt=bigger_dt)
    #     # get Jacobian
    #     J = fenics.derivative(form, u_temp)
    #     # define problem
    #     problem = PETScProblem(J, form, [])
    #     # solve
    #     self._solve(problem, u_temp)
    #     # check if runtime error occurred
    #     if self.runtime_error_occurred:
    #         return False
    #     # define functions
    #     af_temp, c_temp, mu_temp = u_temp.split()
    #     self.ge.compute_gradient(af_temp, self.vec_V, grad_af_temp, self.lsp)
    #     # check tc_activation
    #     tc_activate_after_delta_step = self.tip_cell_manager.test_tip_cell_activation(c_temp,
    #                                                                                   af_temp,
    #                                                                                   grad_af_temp)
    #
    #     return tc_activate_after_delta_step

    # def _find_next_activation_time(self, t: float,
    #                                putative_next_time: float,
    #                                u_temp: fenics.Function,
    #                                phi_temp: fenics.Function,
    #                                phi_exp_temp: fenics.Expression,
    #                                grad_af_temp: fenics.Function) -> float:
    #     if abs(putative_next_time - t) <= self.sim_parameters.get_value("dt"):
    #         logger.info(f"{self.out_folder_name}:Found next time: {putative_next_time}")
    #         return putative_next_time
    #     else:
    #         # define mid step
    #         mid_time = (putative_next_time + t) / 2
    #         # define the delta between the current step and the actual step
    #         delta_mid_time = mid_time - t
    #
    #         logger.info(f"{self.out_folder_name}:Checking tip cell activation at mid point: {mid_time}")
    #         # solve problem after delta_mid_step
    #         tc_activate_at_mid_step = self._test_tip_cell_activation_after_delta_step(u_temp, phi_temp, phi_exp_temp,
    #                                                                                   grad_af_temp, t, delta_mid_time)
    #         if tc_activate_at_mid_step:
    #             return self._find_next_activation_time(t, mid_time, u_temp, phi_temp, phi_exp_temp, grad_af_temp)
    #         else:
    #             return self._find_next_activation_time(mid_time, putative_next_time, u_temp, phi_temp, phi_exp_temp,
    #                                                    grad_af_temp)
    def _solve_using_solver(self, solver, u):
        try:
            solver.solve(u)
        except RuntimeError as e:
            # store error info
            self.runtime_error_occurred = True
            self.error_msg = str(e)
            logger.error(str(e))

    def _time_iteration(self):
        # define fine_dt and coarse_dt
        fine_dt = self.sim_parameters.get_value("dt")
        fine_delta_step = 1
        coarse_dt = self.time_jump
        coarse_delta_step = self.delta_steps

        # define u
        u = dolfinx.fem.Function(self.V)

        # define fine problem
        logger.info(f"{self.out_folder_name}:Defining fine problem...")
        fine_form = self.__build_weak_form(u)                            # weak form
        fine_problem = dolfinx.fem.petsc.NonlinearProblem(fine_form, u)  # problem

        # define coarse problem
        logger.info(f"{self.out_folder_name}:Defining coarse problem...")
        coarse_form = self.__build_weak_form(u, af_dt=False, dt=coarse_dt)    # weak form
        coarse_problem = dolfinx.fem.petsc.NonlinearProblem(coarse_form, u)     # problem

        # define a solver for each problem
        fine_problem_solver = NewtonSolver(comm_world, fine_problem)  # set solver
        coarse_problem_solver = NewtonSolver(comm_world, coarse_problem)
        for solver in [fine_problem_solver, coarse_problem_solver]:
            if "ksp_monitor" in self.lsp.keys():
                solver.report = True  # report iterations
            # set options for krylov solver
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            for o, v in self.lsp.items():
                opts[f"{option_prefix}{o}"] = v
            ksp.setFromOptions()

        # init time iteration
        step = 0                               # time step
        t = self.t0 + (step * fine_dt)         # time
        tip_cell_count = np.zeros(self.steps)  # tip cell number in time
        super()._set_pbar(total=self.steps)    # setup pbar

        # log
        logger.info(f"{self.out_folder_name}:Starting time iteration...")
        # iterate in time
        while step < self.steps + 1:
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
            tip_cell_count[step] = current_tip_cell_count

            # Check if in all of the latest "trigger_time_jump" steps the number of tip cells is 0
            # If yes, proceed faster in time with coarse problem.
            # Else, go to traditional problem
            if step > self.trigger_time_jump and \
                    np.all(tip_cell_count[step + 1 - self.trigger_time_jump:step + 1] == 0):
                logger.info(f"{self.out_folder_name}: (step {step}) Solving coarse problem...")
                self._solve_using_solver(coarse_problem_solver, u)
                # update time
                t += coarse_dt                          # time
                step += coarse_delta_step               # step
                latest_delta_step = coarse_delta_step   # delta step (to update pbar)
            else:
                logger.info(f"{self.out_folder_name}: (step {step}) Solving fine problem...")
                self._solve_using_solver(fine_problem_solver, u)
                # update time
                t += fine_dt  # time
                step += fine_delta_step  # step
                latest_delta_step = fine_delta_step  # delta step (to update pbar)

            # assign to old
            new_af, new_c, new_mu = u.split()
            self.af_old.interpolate(new_af)
            self.af_old.x.scatter_forward()
            self.c_old.interpolate(new_c)
            self.c_old.x.scatter_forward()
            self.af_old.interpolate(new_mu)
            self.af_old.x.scatter_forward()
            # assign new value to grad_af_old
            self.ge.compute_gradient(self.af_old, self.vec_V, self.grad_af_old, self.lsp)
            # assign new value to phi
            self.phi_expression.update_time(t)
            self.phi.interpolate(self.phi_expression.eval)
            self.phi.x.scatter_forward()

            # save
            if (step % self.save_rate == 0) or (step == self.steps) or self.runtime_error_occurred:
                self._write_files(step)

            # if runtime error occurred exit
            if self.runtime_error_occurred:
                return 1

            # update progress bar
            self.pbar.update(latest_delta_step)

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
