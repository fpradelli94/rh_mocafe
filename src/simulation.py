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
from mocafe.math import project
from mocafe.fenut.parameters import Parameters
from mocafe.angie.tipcells import TipCellManager, load_tip_cells_from_json
from mocafe.refine import nmm_interpolate
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


def compute_mesh(spatial_dimension,
                 sim_parameters: Parameters,
                 patient_parameters: Dict,
                 local_ureg: UnitRegistry,
                 n_factor: float = None):

    # load Lx and Ly form patient_parameters
    Lx_real: Quantity = float(patient_parameters["simulation_box"]["Lx"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Lx"]["mu"])
    Ly_real: Quantity = float(patient_parameters["simulation_box"]["Ly"]["value"]) \
                             * local_ureg(patient_parameters["simulation_box"]["Ly"]["mu"])
    if spatial_dimension == 3:
        Lz_real: Quantity = float(patient_parameters["simulation_box"]["Lz"]["value"]) \
                                 * local_ureg(patient_parameters["simulation_box"]["Lz"]["mu"])
    else:
        Lz_real = 0. * local_ureg(patient_parameters["simulation_box"]["Lx"]["mu"])

    # convert Lx and Ly to sau
    Lx: float = Lx_real.to("sau").magnitude
    Ly: float = Ly_real.to("sau").magnitude
    Lz: float = Lz_real.to("sau").magnitude

    # compute nx and ny based on R_c size
    R_c: float = sim_parameters.get_value('R_c')
    nx: int = int(np.floor(Lx / (R_c * 0.7)))
    ny: int = int(np.floor(Ly / (R_c * 0.7)))
    if spatial_dimension == 3:
        nz: int = int(np.floor(Lz / (R_c * 0.7)))
        n = [nx, ny, nz]
    else:
        nz = 0
        n = [nx, ny]
    if n_factor is not None:
        n = np.array(n) * n_factor
        n = list(n.astype(int))


    # define mesh
    if spatial_dimension == 2:
        mesh = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                                             points=[[0., 0.], [Lx, Ly]],
                                             n=n)
        n.append(nz)  # add nz for mesh parameters
    else:
        mesh = dolfinx.mesh.create_box(comm_world,
                                       points=[[0., 0., 0.], [Lx, Ly, Lz]],
                                       n=n)

    # define mesh parameters
    mesh_parameters: Dict = {
        "name": ["Lx", "Ly", "Lz", "nx", "ny", "nz"],
        "info": [f"Mesh x length based on input image (estimated from tumor diameter)",
                 f"Mesh y length based on input image (estimated from tumor diameter)",
                 f"Mesh z length based on input image (estimated from tumor diameter)",
                 "n elements along the x axes (each element is about R_c / 2 long)",
                 "n elements along the y axes (each element is about R_c / 2 long)",
                 "n elements along the z axes (each element is about R_c / 2 long)"],
        "real_value": [Lx_real.magnitude, Ly_real.magnitude, Lz_real.magnitude, None, None, None],
        "real_um": [Lx_real.units, Ly_real.units, Lz_real.units, None, None, None],
        "sim_value": [Lx, Ly, Lz, n[0], n[1], n[2]],
        "sim_um": ["sau", "sau", "sau", None, None, None],
        "references": [None, None, None, None, None, None]
    }

    # return both
    return mesh, Parameters(pd.DataFrame(mesh_parameters))


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
        self.lsp = {                                      # linear solver parameters
            "ksp_type": "gmres",
            "pc_type": "asm",
            "ksp_monitor": None
        }

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
        self.mesh, self.mesh_parameters = compute_mesh(self.spatial_dimension,
                                                       self.sim_parameters,
                                                       self.patient_parameters,
                                                       self.ureg)

    def _spatial_discretization(self, mesh: dolfinx.mesh.Mesh = None):
        logger.info(f"Generating spatial discretization")
        if mesh is None:
            self.V = fu.get_mixed_function_space(self.mesh, 3)
            self.vec_V = dolfinx.fem.FunctionSpace(self.mesh, ufl.VectorElement("P", self.mesh.ufl_cell(), 1))
        else:
            self.V = fu.get_mixed_function_space(mesh, 3)
            self.vec_V = dolfinx.fem.FunctionSpace(mesh, ufl.VectorElement("P", self.mesh.ufl_cell(), 1))

        self.subV0_collapsed, self.collapsedV0_to_V = self.V.sub(0).collapse()

    def _generate_u_old(self):
        logger.info(f"Initializing u old... ")
        self.u_old = dolfinx.fem.Function(self.V)
        self.af_old, self.c_old, self.mu_old = self.u_old.split()

    def _generate_sim_parameters_independent_initial_conditions(self):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # capillaries
        get_c0(self.spatial_dimension, self.c_old, self.patient_parameters, self.mesh_parameters, self.cache_c0_xdmf,
               self.recompute_c0, self.write_checkpoints)

        # t_c_f_function (dynamic tip cell position)
        logger.info(f"{self.out_folder_name}:Computing t_c_f_function...")
        self.t_c_f_function = dolfinx.fem.Function(self.subV0_collapsed)

        # define initial time
        self.t0 = 0

    def _generate_sim_parameters_dependent_initial_conditions(self, sim_parameters):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # define initial condition for tumor
        logger.info(f"{self.out_folder_name}:Computing phi0...")
        # initial semi-axes of the tumor
        self.phi_expression = RHEllipsoid(sim_parameters,
                                          self.patient_parameters,
                                          self.mesh_parameters,
                                          self.ureg,
                                          0,
                                          self.spatial_dimension)
        self.phi = dolfinx.fem.Function(self.subV0_collapsed)
        self.phi.interpolate(self.phi_expression.eval)
        self.phi.x.scatter_forward()

        # af
        logger.info(f"{self.out_folder_name}:Computing af0...")
        self.__compute_af_0(sim_parameters)

        # af gradient
        logger.info(f"{self.out_folder_name}:Computing grad_af0...")
        self.grad_af_old = dolfinx.fem.Function(self.vec_V)
        project(ufl.grad(self.af_old), target_func=self.grad_af_old)

        # define tip cell manager
        self.tip_cell_manager = TipCellManager(self.mesh, self.sim_parameters, n_checkpoints=8)

    def __compute_af_0(self, sim_parameters: Parameters, options: Dict = None):
        """
        Solve equilibrium system for af considering the initial values of phi and c.

        Basically, this function is used to generate the initial condition for af assuming that af is at equilibrium
        at the beginning of the simulation.
        """
        # manage none dict
        if options is None:
            options = self.lsp
        # get trial function
        af = ufl.TrialFunction(self.subV0_collapsed)
        # get test function
        v = ufl.TestFunction(self.subV0_collapsed)
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
        sol = dolfinx.fem.Function(self.subV0_collapsed)
        ksp.solve(b, sol.vector)
        # interpolate solution on af_old
        self.af_old.interpolate(sol)
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
                [dolfinx.io.XDMFFile(comm_world, str(self.data_folder / Path(f"{fn}.xdmf")), "w")
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
        self._write_files(0, write_mesh=True)

        self._time_iteration()  # run simulation in time

        self._end_simulation()  # conclude time iteration

        return self.runtime_error_occurred

    def setup_convergence_test(self):
        self._check_simulation_properties()  # Check class proprieties. Return error if something does not work.

        self._sim_mkdir()  # create all simulation folders

        self._fill_reproduce_folder()  # store current script in reproduce folder to keep track of the code

        self._generate_mesh()  # generate mesh

        self._spatial_discretization()  # initialize function space

        self._generate_initial_conditions()  # generate initial condition

        # write initial conditions
        # self._write_files(0)

    def test_convergence(self, lsp: Dict = None):
        if lsp is not None:
            self.lsp = lsp

        self._time_iteration(test_convergence=True)

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
        # generate u_old
        super()._generate_u_old()
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

        for fun, name, f_file in zip([self.af_old, self.c_old, self.grad_af_old, self.phi, self.t_c_f_function],
                                     ["af", "c", "grad_af", "phi", "tipcells"],
                                     [self.af_file, self.c_file, self.grad_af_file, self.phi_file, self.tipcells_file]):
            # log
            logger.info(f"Writing {name} file...")
            # if necessary, write mesh
            if write_mesh and (not self.save_distributed_files):
                f_file.write_mesh(self.mesh)
            # write function
            fun.name = name
            print(fun.function_space is self.vec_V)
            if self.save_distributed_files and (not (fun.function_space in [self.subV0_collapsed, self.vec_V])):
                # for pvd files, save collapsed function (if the function is a sub-function of u)
                collapsed_function = fun.collapse()
                collapsed_function.name = name
                f_file.write_function(collapsed_function, t)
            else:
                f_file.write_function(fun, t)

    def _time_iteration(self, test_convergence: bool = False):
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        u = dolfinx.fem.Function(self.V)
        u.x.array[:] = self.u_old.x.array
        # assign u_old to u
        af, c, mu = ufl.split(u)
        # define test functions
        v1, v2, v3 = ufl.TestFunctions(self.V)
        # build total form
        af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters)
        capillaries_form = src.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters)
        form = af_form + capillaries_form

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        problem = dolfinx.fem.petsc.NonlinearProblem(form, u)

        # define solver
        self.solver = NewtonSolver(comm_world, problem)
        self.solver.report = True  # report iterations
        # set options for krylov solver
        opts = PETSc.Options()
        option_prefix = self.solver.krylov_solver.getOptionsPrefix()
        for o, v in self.lsp.items():
            opts[f"{option_prefix}{o}"] = v
        self.solver.krylov_solver.setFromOptions()

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
            logger.debug(f"Saving incremental tip cells")
            self.t_c_f_function = self.tip_cell_manager.get_latest_tip_cell_function()
            self.t_c_f_function.x.scatter_forward()
            self.tip_cell_manager.save_incremental_tip_cells(f"{self.report_folder}/incremental_tipcells.json", step)

            # solve
            logger.debug(f"Solving problem...")
            try:
                self.solver.solve(u)
            except RuntimeError as e:
                # store error info
                self.runtime_error_occurred = True
                self.error_msg = str(e)
                logger.error(str(e))

            # if error occurred, stop iteration
            if self.runtime_error_occurred or test_convergence:
                break

            # assign to old
            logger.debug(f"Updating u_old")
            self.u_old.x.array[:] = u.x.array
            self.u_old.x.scatter_forward()
            self.af_old, self.c_old, self.mu_old = self.u_old.split()
            # assign new value to grad_af_old
            project(ufl.grad(self.af_old), target_func=self.grad_af_old)
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


class RHAdaptiveSimulation(RHTimeSimulation):
    def _solve_problem(self):
        logger.debug(f"Solving problem...")
        try:
            self.solver.solve(self.u)
        except RuntimeError as e:
            # store error info
            self.runtime_error_occurred = True
            self.error_msg = str(e)
            logger.error(str(e))
        logger.debug(f"Ended solution")

    def _check_tc_activation_at_dt(self, putative_dt):
        """
        Check if, after dt, the conditions for tc activation hold
        """
        # log
        logger.info(f"Checking tc activation at dt={putative_dt}")

        self.dt_constant.value = putative_dt
        # update phi
        self.phi_expression.update_time(self.t + putative_dt)
        self.phi.interpolate(self.phi_expression.eval)
        self.phi.x.scatter_forward()
        # compute af0 with new phi

        # solve
        self._solve_problem()

        # check if runtime error occurred
        if self.runtime_error_occurred:
            return False

        # if no error, check for tc activation
        # extract af, c, and mu
        putative_af, putative_c, putative_mu = self.u.split()
        # compute grad_af
        project(ufl.grad(putative_af), target_func=self.grad_af_old)
        # check
        tc_can_activate = self.tip_cell_manager.test_tip_cell_activation(putative_c,
                                                                         putative_af,
                                                                         self.grad_af_old)

        return tc_can_activate

    def _find_next_activation_dt(self,
                                 dt0: float,
                                 dt1: float) -> int:
        # check if dt0 < dt1
        assert dt0 < dt1, "dt0 should be always less than dt1"

        # log
        logger.info(f"Looking for activation dt between {dt0} and {dt1}")

        # If the difference between dt0 and dt1 is less than min_dt, return dt1
        # Else, find the dt value between dt0 and dt1 which lead to the activation of the tcs
        if abs(dt1 - dt0) <= self.min_dt:
            next_activation_dt = int(np.round(dt1))
            logger.info(f"{self.out_folder_name}:Found dt for the next activation: {next_activation_dt}")
            return next_activation_dt
        else:
            # compute mid_dt
            mid_dt = (dt1 + dt0) / 2
            # check tc_activation at mid_dt
            tc_can_activate = self._check_tc_activation_at_dt(mid_dt)

            # if tc can activate, it means that the minimal dt leading to tc activation is between dt0 and mid_dt
            # else, if means that the minimal dt is between mid_dt and dt1
            if tc_can_activate:
                return self._find_next_activation_dt(dt0, mid_dt)
            else:
                return self._find_next_activation_dt(mid_dt, dt1)

    def _active_tc_routine(self):
        # set dt value to min
        self.dt_constant.value = self.min_dt

        # manage tip cells
        self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, self.t)
        self.tip_cell_manager.revert_tip_cells(self.af_old, self.grad_af_old)
        self.tip_cell_manager.move_tip_cells(self.c_old, self.af_old, self.grad_af_old)

        # store tip cells in fenics function and json file
        self.t_c_f_function = self.tip_cell_manager.get_latest_tip_cell_function()
        self.t_c_f_function.x.scatter_forward()
        self.tip_cell_manager.save_incremental_tip_cells(f"{self.report_folder}/incremental_tipcells.json", self.t)

        # solve
        self._solve_problem()

        # update time
        self.t += self.dt_constant.value

        # assign new value to phi
        self.phi_expression.update_time(self.t)
        self.phi.interpolate(self.phi_expression.eval)
        self.phi.x.scatter_forward()

    def _time_iteration(self, test_convergence: bool = False):
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        self.u = dolfinx.fem.Function(self.V)
        self.u.x.array[:] = self.u_old.x.array
        # assign u_old to u
        af, c, mu = ufl.split(self.u)
        # define test functions
        v1, v2, v3 = ufl.TestFunctions(self.V)

        # init dt values
        self.min_dt = int(np.round(self.sim_parameters.get_value("dt")))
        self.max_dt = 100 * self.min_dt
        self.dt_constant = dolfinx.fem.Constant(
            self.mesh, dolfinx.default_scalar_type(self.min_dt)
        )

        # build form
        af_eq_form = src.forms.angiogenic_factors_form_eq(af, self.phi, c, v1, self.sim_parameters)
        capillaries_form = src.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters, dt=self.dt_constant)
        form_af_eq = af_eq_form + capillaries_form

        # free energy form for time stepping
        ch_free_energy_form = dolfinx.fem.form(src.forms.chan_hillard_free_enery(self.mu_old))

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        problem = dolfinx.fem.petsc.NonlinearProblem(form_af_eq, self.u)

        # define solver
        self.solver = NewtonSolver(comm_world, problem)
        self.solver.report = True  # report iterations
        # set options for krylov solver
        opts = PETSc.Options()
        option_prefix = self.solver.krylov_solver.getOptionsPrefix()
        for o, v in self.lsp.items():
            opts[f"{option_prefix}{o}"] = v
        self.solver.krylov_solver.setFromOptions()

        # init time iteration
        self.t = int(np.round(self.t0))
        self.last_writing_time = 0  # last time of writing
        super()._set_pbar(total=self.steps)

        # log
        logger.info(f"{self.out_folder_name}:Starting time iteration...")

        # iterate in time
        while self.t < (self.steps + 1):
            # compute current n_tc
            current_n_tc = len(self.tip_cell_manager.get_global_tip_cells_list())
            # check if the current conditions lead to activation
            tc_can_activate = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)

            # If there are no active tc and tcs cannot activate (simulation becomes purely PDE-based), we fast-forward
            # the simulation.
            if (current_n_tc == 0) and (not tc_can_activate):
                local_CH_energy = dolfinx.fem.assemble_scalar(ch_free_energy_form)
                global_CH_energy = comm_world.allreduce(local_CH_energy, op=MPI.SUM)
                putative_dt = self.max_dt / np.sqrt(1 + (50 * (global_CH_energy ** 2)))
                putative_dt = max(self.min_dt, putative_dt)
                logger.info(f"Initial putative dt: {putative_dt}")

                # check if time dt is enough to lead to activation
                tc_activate_after_putative_dt = self._check_tc_activation_at_dt(putative_dt)

                if tc_activate_after_putative_dt:
                    # find dt for the next tc activation
                    self._find_next_activation_dt(0, putative_dt)
                else:
                    # log
                    logger.info(f"No tc activated after dt={self.dt_constant.value}")

                # store the current empty list of tcs for each time point between t and dt
                for intermediate_time in range(int(self.dt_constant.value)):
                    self.tip_cell_manager.save_incremental_tip_cells(
                        f"{self.report_folder}/incremental_tipcells.json", self.t + intermediate_time)

                # update time
                self.t += self.dt_constant.value
                logger.info(f"Updated time to={self.t}")
            else:
                self._active_tc_routine()

            # assign to old
            self.u_old.x.array[:] = self.u.x.array
            self.u_old.x.scatter_forward()
            self.af_old, self.c_old, self.mu_old = self.u_old.split()
            # assign new value to grad_af_old
            project(ufl.grad(self.af_old), target_func=self.grad_af_old)

            # save
            if ((self.t - self.last_writing_time) < self.save_rate) or (self.t == self.steps) or self.runtime_error_occurred:
                self._write_files(self.t)
                self.last_writing_time = self.t

            # if error occurred, stop iteration
            if self.runtime_error_occurred:
                break

            # update progress bar
            self.pbar.update(self.dt_constant.value)


class RHMeshAdaptiveSimulation(RHAdaptiveSimulation):
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
                 save_distributed_files_to: str or None = None,
                 refine_rate: int = 5):
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
        self.refine_rate = refine_rate
        self.refine_counter = 0
        self.R_c = self.sim_parameters.get_value("R_c")
        self.stop_adapting = False

    def _spatial_discretization(self, mesh: dolfinx.mesh.Mesh = None):
        # call spatial discretization
        super()._spatial_discretization(mesh)

        # add other collapsed subspace
        self.subV1_collapsed, self.collapsedV1_to_V = self.V.sub(1).collapse()
        self.subV2_collapsed, self.collapsedV2_to_V = self.V.sub(2).collapse()

    def _get_tip_cells_marker(self, h: float):
        def marker(x):
            # compute distance of each point x from each center
            distance_from_centers = np.ones(len(x[0])) * np.infty
            for tc in self.tip_cell_manager.get_global_tip_cells_list():
                tcp = tc.get_position()
                current_d = np.sqrt((((x[0] - tcp[0]) ** 2) + ((x[1] - tcp[1]) ** 2) + ((x[2] - tcp[2]) ** 2)))
                distance_from_centers[current_d < distance_from_centers] = current_d[current_d < distance_from_centers]
            # get is in border
            is_in_border = ((self.R_c + h) < distance_from_centers) & (distance_from_centers < (self.R_c + h))
            return is_in_border

        return marker

    def _adapt_mesh(self):
        logger.info(f"Generating coarse mesh")
        adapted_mesh, _ = compute_mesh(self.spatial_dimension,
                                       self.sim_parameters,
                                       self.patient_parameters,
                                       self.ureg,
                                       n_factor=0.1)

        len_high_res_edges = 0  # init high_res_edges
        high_res_edges = []

        logger.info(f"Starting refinement")
        for refinement_cycle in range(self.max_mesh_refinement_cycles):
            logging.info(f"time {self.t} | refinement cycle {refinement_cycle}")

            # define new function spaces
            self._spatial_discretization(adapted_mesh)

            # interpolate c
            a_c_old = nmm_interpolate(dolfinx.fem.Function(self.subV1_collapsed), self.c_old.collapse())

            # -------------------------------------------------------------------------------------------------------- #
            # Refinement based on af residual
            # -------------------------------------------------------------------------------------------------------- #
            # if the residual from last step was too high, do refinement
            if (refinement_cycle == 0) or (len_high_res_edges > 0):
                logger.debug(f"AF based refinement")
                # init dt constant
                dt_constant = dolfinx.fem.Constant(adapted_mesh, dolfinx.default_scalar_type(self.dt_constant.value))

                # generate collapsed functions for af_old and c_old
                a_af_old = nmm_interpolate(dolfinx.fem.Function(self.subV0_collapsed), self.af_old.collapse())

                # interpolate phi from the last step
                a_phi = nmm_interpolate(dolfinx.fem.Function(self.subV0_collapsed), self.phi)

                # define weak form for af
                res_af = dolfinx.fem.Function(self.subV0_collapsed)
                res_v = ufl.TestFunction(self.subV0_collapsed)
                af_form = src.forms.angiogenic_factors_form_dt(res_af, a_af_old, a_phi, a_c_old, res_v,
                                                               self.sim_parameters, dt=dt_constant)

                problem = dolfinx.fem.petsc.NonlinearProblem(af_form, res_af)

                # define solver
                solver = NewtonSolver(comm_world, problem)
                solver.report = True  # report iterations
                # set options for krylov solver
                opts = PETSc.Options()
                option_prefix = solver.krylov_solver.getOptionsPrefix()
                for o, v in self.lsp.items():
                    opts[f"{option_prefix}{o}"] = v
                solver.krylov_solver.setFromOptions()

                # solve
                try:
                    solver.solve(res_af)
                except RuntimeError as e:
                    # store error info
                    self.runtime_error_occurred = True
                    self.error_msg = str(e)
                    logger.error(str(e))
                    break

                # compute af residual for refinement
                af_residual = src.forms.angiogenic_factors_form_dt(res_af, res_af, a_phi, a_c_old, res_v,
                                                                   self.sim_parameters,
                                                                   dt=dt_constant)
                R_af = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(af_residual))
                R_af.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                # create connectivity between vertices and edges
                adapted_mesh.topology.create_connectivity(0, 1)

                # get vertices where the residual of af is higher
                high_res_vertices, = np.nonzero(np.abs(R_af.array) > self.mesh_refinement_tol)  # check higher
                # get edges connected to vertices where the residual is higher
                high_res_edges = dolfinx.mesh.compute_incident_entities(adapted_mesh.topology,
                                                                        high_res_vertices.astype(np.int32),
                                                                        0,
                                                                        1)
                # compute global len of high_res_edges
                local_len_high_res_edges = len(high_res_edges)
                len_high_res_edges = comm_world.allreduce(local_len_high_res_edges, op=MPI.SUM)

            # -------------------------------------------------------------------------------------------------------- #
            # Check min h value; if reached the target, stop refining
            # -------------------------------------------------------------------------------------------------------- #
            logger.debug(f"Refinement h test")
            mesh_cells_index_map = adapted_mesh.topology.index_map(adapted_mesh.topology.dim)
            local_h = adapted_mesh.h(adapted_mesh.topology.dim, range(mesh_cells_index_map.size_local))
            local_h_min = np.amin(local_h)
            global_h_min = comm_world.allreduce(local_h_min, MPI.MIN)
            order_of_magnitude_global_h_min = 10 ** np.floor(np.log10(global_h_min))

            # check h min. If h_min reached the target, stop refining. Else, continue
            if order_of_magnitude_global_h_min < self.mesh_refinement_epsilon:
                logger.info(
                    f"O(h) [O({global_h_min})] < epsilon ({self.mesh_refinement_epsilon}): "
                    f"exit refinement"
                )
                break
            else:
                logger.info(
                    f"O(h) [O({global_h_min})] > epsilon ({self.mesh_refinement_epsilon}): "
                    f"keep refining"
                )

            # -------------------------------------------------------------------------------------------------------- #
            # Refinement based on c gradient
            # -------------------------------------------------------------------------------------------------------- #
            logger.debug(f"C gradient based refinement")
            # create connectivity between cells and edges
            adapted_mesh.topology.create_connectivity(adapted_mesh.topology.dim, 1)

            # compute c vertices far from equilibrium (which are not 1 or -1)
            c_DG = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(adapted_mesh, ("DG", 0)))
            c_DG.interpolate(a_c_old)
            c_DG.x.scatter_forward()
            c_DG.x.array[:] = np.sqrt(c_DG.x.array ** 2)
            c_DG.x.scatter_forward()
            c_DG_infty_norm = self.mesh.comm.allreduce(np.amax(c_DG.x.array), op=MPI.MAX)
            far_from_eq_cells, = np.nonzero((c_DG.x.array / c_DG_infty_norm) < 0.35)
            far_from_eq_edges = dolfinx.mesh.compute_incident_entities(adapted_mesh.topology,
                                                                       far_from_eq_cells.astype(np.int32),
                                                                       adapted_mesh.topology.dim,
                                                                       1)

            # -------------------------------------------------------------------------------------------------------- #
            # Refinement based on TC position
            # -------------------------------------------------------------------------------------------------------- #
            logger.debug(f"TC position based refinement")
            tip_cells_edges = dolfinx.mesh.locate_entities(adapted_mesh,
                                                           1,
                                                           self._get_tip_cells_marker(global_h_min/2))

            # merge edges
            logging.debug(f"af high res edges to refine: {len(high_res_edges)}")
            logging.debug(f"far from eq edges to refine: {len(far_from_eq_edges)}")
            logging.debug(f"tip cell edges to refine: {len(tip_cells_edges)}")
            edges = np.append(high_res_edges, far_from_eq_edges)
            edges = np.append(edges, tip_cells_edges)
            edges = np.unique(edges.astype(np.int32))

            # refine mesh
            logger.debug(f"Mesh refinement")
            adapted_mesh = dolfinx.mesh.refine(adapted_mesh, edges, redistribute=False)

        # set new mesh
        self.mesh = adapted_mesh

    def _set_up_adaptive_problem(self):
        """
        Solve problem using adaptive mesh refinement based on residual
        """
        # if self.refine_counter % self.refine_rate == 0:
        # adapt mesh
        self._adapt_mesh()

        # update counter
        self.refine_counter += 1

        # set dt constant
        self.dt_constant = dolfinx.fem.Constant(self.mesh, dolfinx.default_scalar_type(self.dt_constant.value))

        # interpolate old state on new mesh
        a_af_old = nmm_interpolate(dolfinx.fem.Function(self.subV0_collapsed), self.af_old.collapse())
        a_c_old = nmm_interpolate(dolfinx.fem.Function(self.subV1_collapsed), self.c_old.collapse())
        a_mu_old = nmm_interpolate(dolfinx.fem.Function(self.subV2_collapsed), self.mu_old.collapse())

        # create u_old to keep all the functions
        self.u_old = dolfinx.fem.Function(self.V)
        self.u_old.x.array[self.collapsedV0_to_V] = a_af_old.x.array
        self.u_old.x.array[self.collapsedV1_to_V] = a_c_old.x.array
        self.u_old.x.array[self.collapsedV2_to_V] = a_mu_old.x.array
        self.u_old.x.scatter_forward()

        # overwrite previous state
        self.af_old, self.c_old, self.mu_old = self.u_old.split()

        # interpolate phi on the new mesh
        self.phi = nmm_interpolate(dolfinx.fem.Function(self.subV0_collapsed), self.phi)

        # create u
        self.u = dolfinx.fem.Function(self.V)
        self.u.x.array[:] = self.u_old.x.array
        self.u.x.scatter_forward()

        # generate weak form
        af, c, mu = ufl.split(self.u)
        v1, v2, v3 = ufl.TestFunctions(self.V)
        af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters,
                                                       dt=self.dt_constant)
        capillaries_form = src.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters, dt=self.dt_constant)
        form = af_form + capillaries_form

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        problem = dolfinx.fem.petsc.NonlinearProblem(form, self.u)

        # define solver
        self.solver = NewtonSolver(comm_world, problem)
        self.solver.report = True  # report iterations
        # set options for krylov solver
        opts = PETSc.Options()
        option_prefix = self.solver.krylov_solver.getOptionsPrefix()
        for o, v in self.lsp.items():
            opts[f"{option_prefix}{o}"] = v
        self.solver.krylov_solver.setFromOptions()

    def _solve_problem(self):
        if not self.stop_adapting:
            logger.info(f"Set up refined mesh")
            self._set_up_adaptive_problem()
        super()._solve_problem()

    def _time_iteration(self, test_convergence: bool = False):
        # set tolerance for spatial-refinement
        self.mesh_refinement_tol = 1e-5
        self.max_mesh_refinement_cycles = 15
        self.mesh_refinement_epsilon = np.sqrt(self.sim_parameters.get_value("epsilon"))

        # get dt values for time-adaptive solution
        self.min_dt = int(np.round(self.sim_parameters.get_value("dt")))
        self.max_dt = 100 * self.min_dt
        self.dt_constant = dolfinx.fem.Constant(self.mesh, dolfinx.default_scalar_type(self.min_dt))

        # init time iteration
        self.t = int(np.round(self.t0))
        super()._set_pbar(total=self.steps)

        # log
        logger.info(f"{self.out_folder_name}:Starting time iteration...")

        # start time iteration
        while self.t < (self.steps + 1):
            # compute current n_tc
            current_n_tc = len(self.tip_cell_manager.get_global_tip_cells_list())
            # check if the current conditions lead to activation
            tc_can_activate = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)

            # compute ch free energy
            ch_free_energy_form = dolfinx.fem.form(src.forms.chan_hillard_free_enery(self.mu_old))

            # If there are no active tc and tcs cannot activate (simulation becomes purely PDE-based), we fast-forward
            # the simulation.
            if (current_n_tc == 0) and (not tc_can_activate):
                local_CH_energy = dolfinx.fem.assemble_scalar(ch_free_energy_form)
                global_CH_energy = comm_world.allreduce(local_CH_energy, op=MPI.SUM)
                putative_dt = self.max_dt / np.sqrt(1 + (50 * (global_CH_energy ** 2)))
                putative_dt = max(self.min_dt, putative_dt)
                logger.info(f"Initial putative dt: {putative_dt}")

                # check if time dt is enough to lead to activation
                tc_activate_after_putative_dt = self._check_tc_activation_at_dt(putative_dt)

                if tc_activate_after_putative_dt:
                    # prevent mesh adapting
                    self.stop_adapting = True
                    # find dt for the next tc activation
                    self._find_next_activation_dt(0, putative_dt)
                    self.stop_adapting = False
                else:
                    # log
                    logger.info(f"No tc activated after dt={self.dt_constant.value}")

                # store the current empty list of tcs for each time point between t and dt
                for intermediate_time in range(int(self.dt_constant.value)):
                    self.tip_cell_manager.save_incremental_tip_cells(
                        f"{self.report_folder}/incremental_tipcells.json", self.t + intermediate_time)

                # update time
                self.t += self.dt_constant.value
                logger.info(f"Updated time to={self.t}")
            else:
                self._active_tc_routine()

            # if error occurred, stop iteration
            if self.runtime_error_occurred:
                break

            # assign to old
            logger.debug(f"Updating u_old")
            self.u_old = dolfinx.fem.Function(self.V)
            self.u_old.x.array[:] = self.u.x.array
            self.af_old, self.c_old, self.mu_old = self.u_old.split()
            # assign new value to grad_af_old
            self.grad_af_old = dolfinx.fem.Function(self.vec_V)
            project(ufl.grad(self.af_old), target_func=self.grad_af_old)

            # update mesh of tipcell manager
            self.tip_cell_manager.update_mesh(self.mesh)

            # save
            if (self.t % self.save_rate == 0) or (self.t == self.steps) or self.runtime_error_occurred:
                self._write_files(self.t, write_mesh=self.mesh)

            # update progress bar
            self.pbar.update(self.dt_constant.value)


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

        self._generate_u_old()

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
            tipcell_activated = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)

            # store result in dataframe
            tip_cell_activation_df = tip_cell_activation_df.append(
                {col: val for col, val in zip(columns_name, [tipcell_activated, *param_values])},
                ignore_index=True
            )

            # update pbar
            self.pbar.update(1)

        # write dataframe to csv
        if rank == 0:
            tip_cell_activation_df.to_csv(f"{self.data_folder}/tipcell_activation.csv")

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
