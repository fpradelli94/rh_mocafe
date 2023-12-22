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
from mocafe.math import project
from mocafe.fenut.parameters import Parameters
from mocafe.angie.tipcells import TipCellManager, load_tip_cells_from_json
from mocafe.refine import nmm_interpolate
import src.forms
from src.ioutils import write_parameters, dump_json, move_files_once_per_node, rmtree_if_exists_once_per_node
from src.expressions import RHEllipsoid, VesselReconstruction

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


def compute_c0(spatial_dimension: int,
               c_old: dolfinx.fem.Function,
               patient_parameters: Dict,
               mesh_parameters: Parameters) -> None:
    """
    Get the initial condition for c0.

    :param spatial_dimension: 2D or 3D
    :param c_old: FEniCS function to store initial condition
    :param patient_parameters: patient-specific parameters Dict
    :param mesh_parameters: mesh parameters (used in the computation)
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
    """
    Compute the mesh in 2D or 3D given the patient data.
    :param spatial_dimension: 2 for 2D, 3 for 3D
    :param sim_parameters: simulation parameters
    :param patient_parameters: patient parameters
    :param local_ureg: UnitRegistry containing the measure units
    :param n_factor: reduce nx and ny of the given factor (used for generating coarse meshes)
    """

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
                 slurm_job_id: int = None):
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
        self.runtime_error_occurred = False  # flag to activate in case of sim errors

        # Folders
        self.data_folder: Path = mansim.setup_data_folder(folder_path=f"saved_sim/{out_folder_name}",
                                                          auto_enumerate=out_folder_mode)
        self.report_folder: Path = self.data_folder / Path("sim_info")
        self.reproduce_folder: Path = self.data_folder / Path("0_reproduce")
        self.slurm_folder = Path("slurm")

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
                                ".sim_cache",
                                "jobids.txt",
                                "slurm",
                                "misc"]

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
        logger.debug(f"DOFS: {self.V.dofmap.index_map.size_global}")

    def _generate_u_old(self):
        logger.info(f"Initializing u old... ")
        self.u_old = dolfinx.fem.Function(self.V)
        self.af_old, self.c_old, self.mu_old = self.u_old.split()

    def _generate_sim_parameters_independent_initial_conditions(self):
        """
        Generate initial conditions not depending on from the simulaion parameters
        """
        # capillaries
        compute_c0(self.spatial_dimension, self.c_old, self.patient_parameters, self.mesh_parameters)

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
        self.__compute_af0(sim_parameters)

        # af gradient
        logger.info(f"{self.out_folder_name}:Computing grad_af0...")
        self.grad_af_old = dolfinx.fem.Function(self.vec_V)
        project(ufl.grad(self.af_old), target_func=self.grad_af_old)

        # define tip cell manager
        self.tip_cell_manager = TipCellManager(self.mesh, self.sim_parameters)

    def __compute_af0(self, sim_parameters: Parameters, options: Dict = None):
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
                 save_distributed_files_to: str or None = None,
                 stop_with_zero_tc=True):
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
        """
        # init super class
        super().__init__(spatial_dimension=spatial_dimension,
                         sim_parameters=sim_parameters,
                         patient_parameters=patient_parameters,
                         out_folder_name=out_folder_name,
                         out_folder_mode=out_folder_mode,
                         sim_rationale=sim_rationale,
                         slurm_job_id=slurm_job_id)

        # specific properties
        self.steps: int = steps  # simulations steps
        self.save_rate: int = save_rate  # set how often writes the output

        # specific flags
        self.__resumed: bool = False  # secret flag to check if simulation has been resumed
        self.save_distributed_files = (save_distributed_files_to is not None)  # Use PVD files instead of XDMF
        self.stop_with_zero_tc = stop_with_zero_tc

        # specific folders
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

    def run(self) -> bool:
        """
        Run simulation. Return True if a runtime error occurred, False otherwise.
        """
        self._check_simulation_properties()  # Check class proprieties. Return error if something does not work.

        self._sim_mkdir()  # create all simulation folders

        self._fill_reproduce_folder()  # store current script in reproduce folder to keep track of the code

        self._generate_mesh()  # generate mesh

        self._spatial_discretization()  # initialize function space

        self._generate_initial_conditions()  # generate initial condition

        self._write_files(0, write_mesh=True)  # write initial conditions

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

        self._set_up_problem()

    def test_convergence(self, lsp: Dict = None):
        if lsp is not None:
            self.lsp = lsp

        self._one_step_solve()

    def _check_simulation_properties(self):
        # check base conditions
        super()._check_simulation_properties()

        # check if steps is positive
        assert self.steps >= 0, "Simulation should have a positive number of steps"

    def _sim_mkdir(self):
        super()._sim_mkdir_list(self.data_folder, self.report_folder, self.reproduce_folder)
        if self.save_distributed_files:
            # make directories for distributed save
            super()._sim_mkdir_list(self.distributed_data_folder, self.pvd_folder)

    def _generate_initial_conditions(self):
        # generate u_old
        super()._generate_u_old()
        # generate initial conditions independent of sim parameters
        super()._generate_sim_parameters_independent_initial_conditions()
        # generate initial conditions dependent from sim parameters
        super()._generate_sim_parameters_dependent_initial_conditions(self.sim_parameters)

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
            if self.save_distributed_files and (not (fun.function_space in [self.subV0_collapsed, self.vec_V])):
                # for pvd files, save collapsed function (if the function is a sub-function of u)
                collapsed_function = fun.collapse()
                collapsed_function.name = name
                f_file.write_function(collapsed_function, t)
            else:
                f_file.write_function(fun, t)

    def _set_up_problem(self):
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        self.u = dolfinx.fem.Function(self.V)
        self.u.x.array[:] = self.u_old.x.array
        self.u.x.scatter_forward()

        # assign u_old to u
        af, c, mu = ufl.split(self.u)
        # define test functions
        v1, v2, v3 = ufl.TestFunctions(self.V)
        # build total form
        af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters)
        capillaries_form = src.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters)
        form = af_form + capillaries_form

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        self.problem = dolfinx.fem.petsc.NonlinearProblem(form, self.u)

        # activate tip cells
        self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, 0)

        # revert tip cells
        self.tip_cell_manager.revert_tip_cells(self.af_old, self.grad_af_old)

        # move tip cells
        self.tip_cell_manager.move_tip_cells(self.c_old, self.af_old, self.grad_af_old)

        # measure n_tcs
        n_tcs = len(self.tip_cell_manager.get_global_tip_cells_list())
        logger.info(f"Step {0} | n tc = {n_tcs}")

    def _one_step_solve(self):
        # define solver
        self.solver = NewtonSolver(comm_world, self.problem)
        # Set Newton solver options
        self.solver.atol = 1e-6
        self.solver.rtol = 1e-6
        self.solver.convergence_criterion = "incremental"
        self.solver.max_it = 100
        self.solver.report = True  # report iterations

        # set options for krylov solver
        opts = PETSc.Options()
        option_prefix = self.solver.krylov_solver.getOptionsPrefix()
        for o, v in self.lsp.items():
            opts[f"{option_prefix}{o}"] = v
        self.solver.krylov_solver.setFromOptions()

        # solve
        logger.debug(f"Solving problem...")
        try:
            self.solver.solve(self.u)
        except RuntimeError as e:
            # store error info
            self.runtime_error_occurred = True
            self.error_msg = str(e)
            logger.error(str(e))

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
            logger.info(f"Starting step {step}")
            # update time
            t += dt

            # activate tip cells
            self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, step)

            # revert tip cells
            self.tip_cell_manager.revert_tip_cells(self.af_old, self.grad_af_old)

            # move tip cells
            self.tip_cell_manager.move_tip_cells(self.c_old, self.af_old, self.grad_af_old)

            # measure n_tcs
            n_tcs = len(self.tip_cell_manager.get_global_tip_cells_list())
            logger.info(f"Step {step} | n tc = {n_tcs}")

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
            if ((step % self.save_rate == 0) or
                    (step == self.steps) or
                    self.runtime_error_occurred or
                    (self.stop_with_zero_tc and (n_tcs == 0))):
                self._write_files(step)

            # update progress bar
            self.pbar.update(1)

            # if error occurred, stop iteration
            if self.runtime_error_occurred or test_convergence or (self.stop_with_zero_tc and (n_tcs == 0)):
                break

    def _end_simulation(self):
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
        self.max_dt = 30 * self.min_dt
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
            logger.info(f"Starting iteration at time {self.t}")
            # compute current n_tc
            current_n_tc = len(self.tip_cell_manager.get_global_tip_cells_list())
            logger.info(f"Time {self.t} | n tc = {current_n_tc}")
            # check if the current conditions lead to activation
            tc_can_activate = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)

            # If there are no active tc and tcs cannot activate (simulation becomes purely PDE-based), we fast-forward
            # the simulation.
            if (current_n_tc == 0) and (not tc_can_activate) and (not np.isclose(self.t, 0.)):
                local_CH_energy = dolfinx.fem.assemble_scalar(ch_free_energy_form)
                global_CH_energy = comm_world.allreduce(local_CH_energy, op=MPI.SUM)
                putative_dt = self.max_dt / np.sqrt(1 + (100 * (global_CH_energy ** 2)))
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

            # set adaptive save rate
            if current_n_tc == 0:
                current_save_rate = self.save_rate * 100
            else:
                current_save_rate = self.save_rate

            # save
            if (((self.t - self.last_writing_time) > current_save_rate)
                    or (self.t == self.steps)
                    or self.runtime_error_occurred):
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
                 save_distributed_files_to=None,
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
                         save_distributed_files_to)
        # Define refinement-specific constants
        self.refine_rate = refine_rate  # set how frequently the mesh should be updated
        self.refine_counter = 0  # set a counter to keep track of the number of refinements
        self.stop_adapting = False  # set a flag to stop refining
        self.af_residual_tolerance = 1e-5  # set tolerance for af residual
        self.c_gradient_tolerance = 0.9  # set tolerance for c gradient
        self.max_mesh_refinement_cycles = 15  # set maximum refinement cycles
        self.mesh_refinement_epsilon = 4 * np.sqrt(self.sim_parameters.get_value("epsilon"))  # set max h

        self.Tc = self.sim_parameters.get_value("T_c")
        self.Gm = self.sim_parameters.get_value("G_m")
        tc_max_speed = float(self.sim_parameters.get_value("G_M")) * float(self.sim_parameters.get_value("chi"))
        tc_min_speed = self.Gm * float(self.sim_parameters.get_value("chi"))
        self.tc_mean_speed = (tc_max_speed + tc_min_speed) / 2

    def _spatial_discretization(self, mesh: dolfinx.mesh.Mesh = None):
        # call spatial discretization
        super()._spatial_discretization(mesh)

        # add other collapsed subspace
        self.subV1_collapsed, self.collapsedV1_to_V = self.V.sub(1).collapse()
        self.subV2_collapsed, self.collapsedV2_to_V = self.V.sub(2).collapse()

    def _generate_coarse_mesh(self):
        logger.info(f"Generating coarse mesh")
        self.coarse_mesh, _ = compute_mesh(self.spatial_dimension,
                                           self.sim_parameters,
                                           self.patient_parameters,
                                           self.ureg,
                                           n_factor=0.1)
        self.coarse_mesh.topology.create_entities(1)  # create entities for refinement

    def _get_high_af_residual_edges(self,
                                    adapted_mesh: dolfinx.mesh.Mesh):
        logger.debug(f"AF based refinement")
        # init dt constant
        dt_constant = dolfinx.fem.Constant(adapted_mesh, dolfinx.default_scalar_type(self.dt_constant.value))

        # interpolate function on adapted function space
        a_af_old = nmm_interpolate(dolfinx.fem.Function(self.subV0_collapsed), self.af_old_collapsed)
        a_c_old = nmm_interpolate(dolfinx.fem.Function(self.subV1_collapsed), self.c_old_collapsed)
        a_phi = dolfinx.fem.Function(self.subV0_collapsed)
        a_phi.interpolate(self.phi_expression.eval)
        a_phi.x.scatter_forward()

        # solve af problem on the adapted mesh
        res_v = ufl.TestFunction(self.subV0_collapsed)
        af_residual = src.forms.angiogenic_factors_form_dt(a_af_old, a_af_old, a_phi, a_c_old, res_v,
                                                           self.sim_parameters,
                                                           dt=dt_constant)
        # assemble residual form
        R_af = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(af_residual))
        R_af.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # create connectivity between vertices and edges
        adapted_mesh.topology.create_connectivity(0, 1)

        # get vertices where the residual of af is higher
        high_res_vertices, = np.nonzero(np.abs(R_af.array) > self.af_residual_tolerance)  # check higher
        # get edges connected to vertices where the residual is higher
        high_res_edges = dolfinx.mesh.compute_incident_entities(adapted_mesh.topology,
                                                                high_res_vertices.astype(np.int32),
                                                                0,
                                                                1)
        return high_res_edges

    def _get_high_c_gradient_marker(self):
        """
        Return a function that marks the edges where the gradient of c is high and, thus, require a refinement.
        """
        # compute c_old values at local midpoints in the old mesh
        n_local_cells_mesh = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        local_cells_midpoints_mesh = dolfinx.mesh.compute_midpoints(self.mesh, self.mesh.topology.dim,
                                                                    range(n_local_cells_mesh))
        local_c_values_mesh = self.c_old.eval(local_cells_midpoints_mesh,
                                              np.arange(n_local_cells_mesh)).reshape((n_local_cells_mesh,))

        # broadcast midpoints and c_values over all processes
        global_cells_midpoints_mesh = comm_world.gather(local_cells_midpoints_mesh, 0)
        global_c_values_mesh = comm_world.gather(local_c_values_mesh, 0)
        if rank == 0:
            global_cells_midpoints_mesh = np.vstack(global_cells_midpoints_mesh)
            global_c_values_mesh = np.concatenate(global_c_values_mesh)
        else:
            global_cells_midpoints_mesh = None
        global_cells_midpoints_mesh = comm_world.bcast(global_cells_midpoints_mesh, 0)
        global_c_values_mesh = comm_world.bcast(global_c_values_mesh, 0)

        def marker(adapted_mesh: dolfinx.mesh.Mesh):
            # get the number of local cells on the adapted mesh
            n_cells_adapted_mesh = adapted_mesh.topology.index_map(adapted_mesh.topology.dim).size_local

            # get cells colliding with each global midpoint
            adapted_mesh_bbt = dolfinx.geometry.bb_tree(adapted_mesh, adapted_mesh.topology.dim)
            cell_candidates = dolfinx.geometry.compute_collisions_points(adapted_mesh_bbt, global_cells_midpoints_mesh)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(adapted_mesh, cell_candidates,
                                                                       global_cells_midpoints_mesh)

            # create dictionary of the c_values for each colliding cell
            c_values_for_cell = {i: [] for i in range(n_cells_adapted_mesh)}
            for i, c_val in enumerate(global_c_values_mesh):
                if len(colliding_cells.links(i)) > 0:
                    c_values_for_cell[colliding_cells.links(i)[0]].append(c_val)

            # get mean c value for each cell
            mean_c_for_cell = np.array([abs(np.mean(vals_list)) for vals_list in c_values_for_cell.values()])

            # get cells where mean is under or over tolerance
            far_from_eq_cells, = np.nonzero(
                (mean_c_for_cell < self.c_gradient_tolerance) | (mean_c_for_cell > (2 - self.c_gradient_tolerance))
            )

            # get edges connected to vertices where the residual is higher
            adapted_mesh.topology.create_entities(1)
            far_from_eq_edges = dolfinx.mesh.compute_incident_entities(adapted_mesh.topology,
                                                                       far_from_eq_cells.astype(np.int32),
                                                                       adapted_mesh.topology.dim,
                                                                       1)

            return far_from_eq_edges

        return marker

    def _get_future_tip_cell_poitions_edges(self, adapted_mesh: dolfinx.mesh.Mesh, mesh_bbt):
        # get global sampling from TipCellManager
        n_global_samples = self.tip_cell_manager.mesh_sampling.get_n_global_samples()
        global_sampling = np.array(self.tip_cell_manager.mesh_sampling.global_sampling)

        # find which global samples are on the current processor
        cell_candidates = dolfinx.geometry.compute_collisions_points(mesh_bbt, global_sampling)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates,
                                                                   global_sampling)
        is_global_sample_on_proc = np.array([len(colliding_cells.links(i)) > 0 for i in range(n_global_samples)])

        # get points on proc
        n_points_on_proc = is_global_sample_on_proc.sum()
        points_on_proc = global_sampling[is_global_sample_on_proc, :]

        # find self.mesh cells colliding with global samples
        current_cells = [colliding_cells.links(i)[0]
                         for i, is_p_on_proc in enumerate(is_global_sample_on_proc) if is_p_on_proc]

        # compute af value on proc
        af_values_on_proc = self.af_old.eval(points_on_proc, current_cells)
        af_values_on_proc = af_values_on_proc.reshape(af_values_on_proc.shape[0])

        # get which global samples have af values above Tc
        af_above_Tc = (af_values_on_proc >= self.Tc)

        # compute grad af and its norm
        grad_af_values = self.grad_af_old.eval(points_on_proc, current_cells)
        norm_grad_af = np.linalg.norm(grad_af_values, axis=1)
        norm_grad_af = norm_grad_af.reshape(norm_grad_af.shape[0])

        # find where the norm is above Gm
        grad_af_above_Gm = (norm_grad_af >= self.Gm)

        # compute the negative versor pointing in the opposite direction of the gradient
        negative_grad_af_versors = np.zeros(shape=(n_points_on_proc, 3))
        for i in range(self.spatial_dimension):
            negative_grad_af_versors[:, i] = -(grad_af_values[:, i] / norm_grad_af)

        # check if there is a vessel in that direction
        point_close_to_capillary = np.zeros((n_points_on_proc,)).astype(bool)  # init
        current_cp_close_to_capillary = point_close_to_capillary.copy()
        for i in range(self.refine_rate):
            # create current checkpoint
            current_checkpoints = points_on_proc + (self.tc_mean_speed * negative_grad_af_versors)

            # find which points are on the current processor
            cell_candidates = dolfinx.geometry.compute_collisions_points(mesh_bbt, current_checkpoints)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(self.mesh, cell_candidates,
                                                                       current_checkpoints)
            is_cell_on_proc = [len(colliding_cells.links(i)) > 0 for i in range(n_points_on_proc)]
            current_cells = [colliding_cells.links(i)[0]
                             for i, is_p_on_proc in enumerate(is_cell_on_proc) if is_p_on_proc]

            # cast is_cell_on_proc to np.ndarray
            is_cell_on_proc = np.array(is_cell_on_proc, dtype=object).astype(bool)

            # find c_value at cells on proc
            c_value_at_cp_on_proc = self.c_old.eval(current_checkpoints[is_cell_on_proc, :], current_cells)

            # set test variable to False is the point is not on proc
            current_cp_close_to_capillary[~is_cell_on_proc] = False

            # for the other, set to true is c_value is over 0
            c_value_at_cp_on_proc = c_value_at_cp_on_proc.reshape((len(current_cells),))
            current_cp_close_to_capillary[is_cell_on_proc] = c_value_at_cp_on_proc > 0.

            # combine with the one computed at the previous step
            point_close_to_capillary = current_cp_close_to_capillary | point_close_to_capillary

        # get boolean index of the points on proc respecting all conditions
        boolean_index_selected_points_on_proc = af_above_Tc & grad_af_above_Gm & point_close_to_capillary

        # get boolean index of the global samples respecting all conditions
        local_boolean_index_selected_global_samples = np.zeros((n_global_samples,)).astype(bool)
        local_boolean_index_selected_global_samples[is_global_sample_on_proc] = boolean_index_selected_points_on_proc
        print(local_boolean_index_selected_global_samples.sum())

        # On Proc 0, get the global boolean index using a logical or
        list_boolean_indices = comm_world.gather(local_boolean_index_selected_global_samples, 0)
        if rank == 0:
            global_boolean_index_selected_global_samples = np.logical_or.reduce(list_boolean_indices)
        else:
            global_boolean_index_selected_global_samples = None
        global_boolean_index_selected_global_samples = comm_world.bcast(global_boolean_index_selected_global_samples, 0)

        # get selected global points
        selected_global_samples = global_sampling[global_boolean_index_selected_global_samples, :]

        # get cells colliding with the selected global samples
        _, colliding_cells = fu.get_colliding_cells_for_points(selected_global_samples,
                                                               adapted_mesh,
                                                               dolfinx.geometry.bb_tree(adapted_mesh,
                                                                                        adapted_mesh.topology.dim))

        # get edges connected to colliding cells
        adapted_mesh.topology.create_connectivity(adapted_mesh.topology.dim, 1)
        selected_edges = dolfinx.mesh.compute_incident_entities(adapted_mesh.topology,
                                                                colliding_cells,
                                                                adapted_mesh.topology.dim,
                                                                1)

        # return corresponding edges
        return selected_edges

    def _adapt_mesh(self):
        """
        Generate an adaptively refined mesh
        """
        # init
        edges = []  # init edges to refine
        adapted_mesh = dolfinx.mesh.refine(self.coarse_mesh, edges, redistribute=False)  # init adapted mesh
        global_len_high_af_res = 1  # init len of edges with high af residual
        high_res_edges = np.array([])  # init array of edges with high af residual

        # collapse old state (required from some refinement procedures)
        self.af_old_collapsed = self.af_old.collapse()
        self.c_old_collapsed = self.c_old.collapse()
        self.mu_old_collapsed = self.mu_old.collapse()

        # get current mesh bbt
        mesh_bbt = dolfinx.geometry.bb_tree(self.mesh, self.mesh.topology.dim)

        # get high c_gradient_marker
        get_high_c_gradient_edges = self._get_high_c_gradient_marker()

        # start refinement
        logger.info(f"Starting refinement")
        for refinement_cycle in range(self.max_mesh_refinement_cycles):
            logging.info(f"time {self.t} | refinement cycle {refinement_cycle}")
            # -------------------------------------------------------------------------------------------------------- #
            # Check min h value; if reached the target, stop refining
            # -------------------------------------------------------------------------------------------------------- #
            logger.debug(f"Refinement h test")
            mesh_cells_index_map = adapted_mesh.topology.index_map(adapted_mesh.topology.dim)
            local_h = adapted_mesh.h(adapted_mesh.topology.dim, range(mesh_cells_index_map.size_local))
            local_h_min = np.amin(local_h)
            global_h_min = comm_world.allreduce(local_h_min, MPI.MIN)
            # order_of_magnitude_global_h_min = 10 ** np.floor(np.log10(global_h_min))
            if global_h_min < self.mesh_refinement_epsilon:
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

            # generate adapted function space
            self._spatial_discretization(adapted_mesh)

            # locate edges to refine for af residual
            if global_len_high_af_res > 0:
                high_res_edges = self._get_high_af_residual_edges(adapted_mesh)
                global_len_high_af_res = comm_world.allreduce(len(high_res_edges), MPI.SUM)
            far_from_eq_edges = get_high_c_gradient_edges(adapted_mesh)
            tip_cells_edges = self._get_future_tip_cell_poitions_edges(adapted_mesh, mesh_bbt)

            # merge edges
            logging.debug(f"af high res edges to refine: {len(high_res_edges)}")
            logging.debug(f"far from eq edges to refine: {len(far_from_eq_edges)}")
            logging.debug(f"tip cell edges to refine: {len(tip_cells_edges)}")
            edges = np.append(high_res_edges, far_from_eq_edges)
            edges = np.append(edges, tip_cells_edges)
            edges = np.unique(edges.astype(np.int32))

            # refine mesh
            logger.debug(f"Mesh refinement")
            adapted_mesh.topology.create_entities(1)
            adapted_mesh = dolfinx.mesh.refine(adapted_mesh, edges, redistribute=False)

        # set new mesh
        self.mesh = adapted_mesh

    def _nmm_interpolate_u_old(self):
        # interpolate old state on new mesh
        logger.debug(f"NMM interpolate")
        a_af_old = nmm_interpolate(dolfinx.fem.Function(self.subV0_collapsed), self.af_old_collapsed)
        a_c_old = nmm_interpolate(dolfinx.fem.Function(self.subV1_collapsed), self.c_old_collapsed)
        a_mu_old = nmm_interpolate(dolfinx.fem.Function(self.subV2_collapsed), self.mu_old_collapsed)

        # interpolate phi on the new mesh
        self.phi = dolfinx.fem.Function(self.subV0_collapsed)
        self.phi.interpolate(self.phi_expression.eval)
        self.phi.x.scatter_forward()

        # create u_old to keep all the functions
        logger.debug(f"Assign to u_old")
        self.u_old = dolfinx.fem.Function(self.V)
        self.u_old.x.array[self.collapsedV0_to_V] = a_af_old.x.array
        self.u_old.x.array[self.collapsedV1_to_V] = a_c_old.x.array
        self.u_old.x.array[self.collapsedV2_to_V] = a_mu_old.x.array
        self.u_old.x.scatter_forward()

        # reassign
        self.af_old, self.c_old, self.mu_old = self.u_old.split()

    def _time_iteration(self, test_convergence: bool = False):
        # set dt constant
        self.min_dt = int(np.round(self.sim_parameters.get_value("dt")))
        self.dt_constant = dolfinx.fem.Constant(self.mesh, dolfinx.default_scalar_type(self.min_dt))

        # init time iteration
        self.t = int(np.round(self.t0))
        super()._set_pbar(total=self.steps)

        # init coarse mesh
        self._generate_coarse_mesh()

        # log
        logger.info(f"{self.out_folder_name}:Starting time iteration...")
        # start time iteration
        while self.t < (self.steps + 1):
            # update time
            self.t += self.dt_constant.value
            logger.info(f"Starting iteration at time {self.t}")
            logger.debug(f"Current DOFS: {self.V.dofmap.index_map.size_global}")

            # manage tip cells
            self.tip_cell_manager.activate_tip_cell(self.c_old, self.af_old, self.grad_af_old, self.t)
            self.tip_cell_manager.revert_tip_cells(self.af_old, self.grad_af_old)
            self.tip_cell_manager.move_tip_cells(self.c_old, self.af_old, self.grad_af_old)

            # store tip cells in fenics function and json file
            logger.debug(f"Saving incremental tip cells")
            self.t_c_f_function = self.tip_cell_manager.get_latest_tip_cell_function()
            self.t_c_f_function.x.scatter_forward()
            self.tip_cell_manager.save_incremental_tip_cells(f"{self.report_folder}/incremental_tipcells.json", self.t)

            # if it is time to refine
            if (self.refine_counter % self.refine_rate) == 0:
                self._adapt_mesh()  # adapt mesh

                self._spatial_discretization()  # regenerate spatial discretization

                self._nmm_interpolate_u_old()  # regenerate u_old

                # generate u on the adapted mesh
                self.u = dolfinx.fem.Function(self.V)

                # regenerate the weak form
                logger.debug(f"Regenerate weak form")
                self.dt_constant = dolfinx.fem.Constant(self.mesh,                         # regenerate dt constant
                                                        dolfinx.default_scalar_type(self.dt_constant.value))
                af, c, mu = ufl.split(self.u)
                v1, v2, v3 = ufl.TestFunctions(self.V)
                af_form = src.forms.angiogenic_factors_form_dt(af, self.af_old, self.phi, c, v1, self.sim_parameters,
                                                               dt=self.dt_constant)
                capillaries_form = src.forms.angiogenesis_form_no_proliferation(
                    c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters, dt=self.dt_constant)
                form = af_form + capillaries_form

                # define problem
                logger.info(f"Regenerating problem")
                problem = dolfinx.fem.petsc.NonlinearProblem(form, self.u)

                # define solver
                logger.info(f"Regenerating solver")
                self.solver = NewtonSolver(comm_world, problem)
                self.solver.report = True  # report iterations
                # set options for krylov solver
                opts = PETSc.Options()
                option_prefix = self.solver.krylov_solver.getOptionsPrefix()
                for o, v in self.lsp.items():
                    opts[f"{option_prefix}{o}"] = v
                self.solver.krylov_solver.setFromOptions()

            self.refine_counter += 1  # update refine counter

            self._solve_problem()  # solve problem

            # assign to old
            logger.debug(f"Updating u_old")
            self.u_old.x.array[:] = self.u.x.array
            self.u_old.x.scatter_forward()
            self.af_old, self.c_old, self.mu_old = self.u_old.split()
            # assign new value to grad_af_old
            self.grad_af_old = dolfinx.fem.Function(self.vec_V)
            project(ufl.grad(self.af_old), target_func=self.grad_af_old)
            # assign new value to phi
            self.phi_expression.update_time(self.t)
            self.phi.interpolate(self.phi_expression.eval)
            self.phi.x.scatter_forward()
            # update mesh of tip cell manager
            self.tip_cell_manager.update_mesh(self.mesh)

            # save
            if (self.t % self.save_rate == 0) or (self.t == self.steps) or self.runtime_error_occurred:
                self.t_c_f_function = dolfinx.fem.Function(self.subV0_collapsed)
                self._write_files(self.t, write_mesh=self.mesh)

            if self.runtime_error_occurred:
                break

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
                 slurm_job_id: int = None):
        # init super class
        super().__init__(spatial_dimension=spatial_dimension,
                         sim_parameters=standard_params,
                         patient_parameters=patient_parameters,
                         out_folder_name=out_folder_name,
                         out_folder_mode=out_folder_mode,
                         sim_rationale=sim_rationale,
                         slurm_job_id=slurm_job_id)

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
                logger.debug(f"Setting  {param_name} to {param_value}")
                current_sim_parameters.set_value(param_name, param_value)

            # generate sim parameters dependent initial conditions
            self._generate_sim_parameters_dependent_initial_conditions(sim_parameters=current_sim_parameters)

            # call activate tip cell
            tc_activated = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)

            # store result in dataframe
            tip_cell_activation_df = tip_cell_activation_df.append(
                {col: val for col, val in zip(columns_name, [tc_activated, *param_values])},
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
        super()._sim_mkdir_list(self.data_folder, self.report_folder, self.reproduce_folder)

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
