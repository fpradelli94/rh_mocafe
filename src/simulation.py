"""
Contains methods and classes to run simulations in 2D and 3D
"""
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
        Initialize a Simulation Object. This serves as basis for all the other Simulation Objects.

        :param spatial_dimension: 2 == 2D simulation; 3 == 3D simulation.
        :param sim_parameters: mathematical model parameters.
        :param patient_parameters: patient-specific parameters (e.g. tumor dimension).
        :param out_folder_name: folder where the simulation will be saved.
        :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in
        out_folder_name; if ``datetime``, the output folder name will be also followed by a string containing the date
        and the time of the simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``).
        The latter is recommended to run multiple simulations of the same kind. Default is ```None``.
        :param sim_rationale: provide a rationale for running this simulation (e.g. checking the effect of this
        parameter). It will be added to the simulation report in ``saved_sim/sim_name/sim_info/sim_info.html``. Default
        is "No comment".
        :param slurm_job_id: slurm job ID assigned to the simulation, if performed with slurm. It is used to generate a
        progress bar stored in ``slurm/<slurm job ID>.pbar``. Default is None.
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
            self.pbar_file = open(f"temp_pbar.o", "w")

    def _check_simulation_properties(self):
        """
        Check if the simulation was initialized correctly
        """
        # check spatial dimension
        assert (self.spatial_dimension == 2 or self.spatial_dimension == 3), \
            f"Cannot run simulation for dimension {self.spatial_dimension}"

    def _sim_mkdir_list(self, *folders_list: Path):
        """
        Generate all the relevant folders to be used in the simulation.
        """
        if rank == 0:
            # generate simulation folder
            for folder in folders_list:
                folder.mkdir(exist_ok=True, parents=True)

            # if slurm id is provided, generate slurm folder
            if self.slurm_job_id is not None:
                self.slurm_folder.mkdir(exist_ok=True)

    def _fill_reproduce_folder(self):
        """
        Store the code used to generate the simulation in the "reproduce" folder.
        """
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
        """
        Generate the mesh for the simulation
        """
        self.mesh, self.mesh_parameters = compute_mesh(self.spatial_dimension,
                                                       self.sim_parameters,
                                                       self.patient_parameters,
                                                       self.ureg)

    def _spatial_discretization(self, mesh: dolfinx.mesh.Mesh = None):
        """
        Generate the spatial discretization (by default, Continuous Lagrange elements)
        """
        logger.info(f"Generating spatial discretization")
        if mesh is None:
            self.V = fu.get_mixed_function_space(self.mesh, 3)
            self.vec_V = dolfinx.fem.FunctionSpace(self.mesh, ufl.VectorElement("P", self.mesh.ufl_cell(), 1))
        else:
            self.V = fu.get_mixed_function_space(mesh, 3)
            self.vec_V = dolfinx.fem.FunctionSpace(mesh, ufl.VectorElement("P", mesh.ufl_cell(), 1))

        self.subV0_collapsed, self.collapsedV0_to_V = self.V.sub(0).collapse()
        logger.debug(f"DOFS: {self.V.dofmap.index_map.size_global}")

    def _generate_u_old(self):
        """
        Generate the function u_old, used to store the previous staatus of the simulation
        """
        logger.info(f"Initializing u old... ")
        self.u_old = dolfinx.fem.Function(self.V)
        self.af_old, self.c_old, self.mu_old = self.u_old.split()

    def _generate_sim_parameters_independent_initial_conditions(self):
        """
        Generate initial conditions not depending on from the mathematical model parameters
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
        self._compute_af0(sim_parameters)

        # af gradient
        logger.info(f"{self.out_folder_name}:Computing grad_af0...")
        self.grad_af_old = dolfinx.fem.Function(self.vec_V)
        project(ufl.grad(self.af_old), target_func=self.grad_af_old)

        # define tip cell manager
        self.tip_cell_manager = TipCellManager(self.mesh, sim_parameters, n_checkpoints=30)

    def _compute_af0(self, sim_parameters: Parameters, options: Dict = None):
        """
        Solve equilibrium system for af considering the initial values of phi and c. It is used to generate the
        initial condition for af.
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
        """
        Generate progress bar
        """
        self.pbar = tqdm(total=total,
                         ncols=100,
                         desc=self.out_folder_name,
                         file=self.pbar_file,
                         disable=True if rank != 0 else False)

    def _end_simulation(self):
        """
        Perform all the operations that need to be carried out at the end of the simulation
        """
        logger.info("Ending simulation... ")

        # close pbar file
        if (rank == 0):
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
                 save_distributed_files_to: str or None = None):
        """
        Initialize a Simulation Object to simulate tumor-induced angiogenesis and tumor growth in time.

        :param spatial_dimension: 2 == 2D simulation; 3 == 3D simulation.
        :param sim_parameters: mathematical model's parameters.
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
        :param save_distributed_files_to: if the simulation runs in parallel different nodes, specify if you want your
        simulation results to be saved as vtk in the different nodes and the location where they will be stored.
        The location must be an existing folder present on each node. At the end of the simulation, the files will be
        moved in the actual output under the sup-folder 'pvd'.
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
        self.lsp = {  # linear solver parameters
            "ksp_type": "gmres",
            "pc_type": "gasm",
            "ksp_monitor": None
        }

        # specific flags
        self.__resumed: bool = False  # secret flag to check if simulation has been resumed
        self.save_distributed_files = (save_distributed_files_to is not None)  # Use PVD files instead of XDMF

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
        """
        Method used to set up the simulation for a convergence test. Used to check the best preconditioner or solver
        to be used in your specific system.
        """
        self._check_simulation_properties()  # Check class proprieties. Return error if something does not work.

        self._sim_mkdir()  # create all simulation folders

        self._fill_reproduce_folder()  # store current script in reproduce folder to keep track of the code

        self._generate_mesh()  # generate mesh

        self._spatial_discretization()  # initialize function space

        self._generate_initial_conditions()  # generate initial condition

        self._set_up_problem()

    def test_convergence(self, lsp: Dict = None):
        """
        Run the simulation for une step using the given linear solver parameters (lsp).
        """
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
        """
        Generate initial conditions for the simulation
        """
        # generate u_old
        super()._generate_u_old()
        # generate initial conditions independent of sim parameters
        super()._generate_sim_parameters_independent_initial_conditions()
        # generate initial conditions dependent from sim parameters
        super()._generate_sim_parameters_dependent_initial_conditions(self.sim_parameters)

    def _write_files(self, t: int, write_mesh: bool = False):
        """
        Write simulation files as XDMF or PVD (if you want your simulation to be stored in parallel on each node).
        """
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
        """
        Run the simulation in time.
        """
        # define weak form
        logger.info(f"{self.out_folder_name}:Defining weak form...")
        u = dolfinx.fem.Function(self.V)
        u.x.array[:] = self.u_old.x.array
        # assign u_old to u
        af, c, mu = ufl.split(u)
        # define test functions
        v1, v2, v3 = ufl.TestFunctions(self.V)
        # build total form
        af_form = src.forms.angiogenic_factors_form_eq(af, self.phi, c, v1, self.sim_parameters)
        capillaries_form = src.forms.angiogenesis_form_no_proliferation(
            c, self.c_old, mu, self.mu_old, v2, v3, self.sim_parameters)
        form = af_form + capillaries_form

        # define problem
        logger.info(f"{self.out_folder_name}:Defining problem...")
        problem = dolfinx.fem.petsc.NonlinearProblem(form, u)

        # define solver
        self.solver = NewtonSolver(comm_world, problem)
        self.solver.max_it = 100
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
                n_iterations, converged = self.solver.solve(u)
                logger.debug(f"Solver ended with n_interations: {n_iterations}; converged: {converged}")
                if n_iterations == 0:
                    raise RuntimeError("Newton Solver converged in 0 iteration.")
                if not converged:
                    raise RuntimeError("Newton Solver did not converge.")
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
                    self.runtime_error_occurred):
                self._write_files(step)

            # update progress bar
            self.pbar.update(1)

            # if error occurred, stop iteration
            if self.runtime_error_occurred or test_convergence:
                break

    def _end_simulation(self):
        """
        Perform all the operations that need to be carried out at the end of the simulation
        """
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
                 max_dt: int = None):
        """
        Initialize a Simulation Object to simulate tumor-induced angiogenesis and tumor growth in time using an
        adaptive time-stepping scheme.

        :param spatial_dimension: 2 == 2D simulation; 3 == 3D simulation.
        :param sim_parameters: mathematical model's parameters.
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
        :param save_distributed_files_to: if the simulation runs in parallel different nodes, specify if you want your
        simulation results to be saved as vtk in the different nodes and the location where they will be stored.
        The location must be an existing folder present on each node. At the end of the simulation, the files will be
        moved in the actual output under the sup-folder 'pvd'.
        """
        super().__init__(spatial_dimension=spatial_dimension,
                         sim_parameters=sim_parameters,
                         patient_parameters=patient_parameters,
                         steps=steps,
                         save_rate=save_rate,
                         out_folder_name=out_folder_name,
                         out_folder_mode=out_folder_mode,
                         sim_rationale=sim_rationale,
                         slurm_job_id=slurm_job_id,
                         save_distributed_files_to=save_distributed_files_to)
        self.max_dt_steps = 30 if max_dt is None else max_dt

    def _solve_problem(self):
        """
        Solve the PDE problem
        """
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
        Check if, after dt, the conditions for tc activation are met
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
        """
        Find the dt value after which the TC activation conditions are met, if any.
        """
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
        """
        Manage active tip cells and RH growth
        """
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
        """
        Run the simulation in time.
        """
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
        self.max_dt = self.max_dt_steps * self.min_dt
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
        self.solver.convergence_criterion = "incremental"
        self.solver.max_it = 100
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

            # save
            if (((self.t - self.last_writing_time) > self.save_rate)
                    or (self.t >= self.steps)
                    or self.runtime_error_occurred):
                self._write_files(self.t)
                self.last_writing_time = self.t

            # if error occurred, stop iteration
            if self.runtime_error_occurred:
                break

            # update progress bar
            self.pbar.update(int(np.round(self.dt_constant.value)))


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

    def setup(self):
        """
        Setup simulation for tip cell activation
        """
        self._check_simulation_properties()  # Check class proprieties. Return error if something does not work.

        self._sim_mkdir()  # create all simulation folders

        self._fill_reproduce_folder()  # store current script in reproduce folder to keep track of the code

        self._generate_mesh()  # generate mesh

        self._spatial_discretization()  # initialize function space

        self._generate_u_old()

        self._generate_sim_parameters_independent_initial_conditions()

    def run(self, find_min_tdf_i: bool = False, append_result_to_csv: str or None = None, **kwargs) -> bool:
        """
        Run simulation for tip cell activation. Return True if a runtime error occurred, False otherwise.
        """
        self.setup()

        # get params dictionary
        self._generate_test_parameters(**kwargs)

        # check tdf_i presence
        if find_min_tdf_i:
            assert "tdf_i" not in self.params_dictionary.keys(), "Cannot set tdf_i while looking for the min value"

        # generate column name for the output csv
        columns_name = ["tip_cell_activated",
                        *[f"{k} (range: [{np.amin(r)}, {np.amax(r)}])" for k, r in self.params_dictionary.items()]]

        # generate dataframe collecting the activation tiles (True/False for each condition)
        tip_cell_activation_df = self._setup_output_dataframe(columns_name, append_result_to_csv)

        # if required, generate dataframe collecting the minimal RH dimension
        if find_min_tdf_i:
            min_tdf_i_df = self._setup_output_dataframe(columns_name, None)
        else:
            min_tdf_i_df = None

        # init pbar
        self._set_pbar(total=len(list(product(*self.params_dictionary.values()))))

        # iterate on parameters
        for param_values in product(*self.params_dictionary.values()):
            # set parameters value
            current_sim_parameters = Parameters(self.df_standard_params)
            for param_name, param_value in zip(self.params_dictionary.keys(), param_values):
                logger.debug(f"Setting  {param_name} to {param_value}")
                current_sim_parameters.set_value(param_name, param_value)

            if find_min_tdf_i:
                # set tdf_i to 1
                current_sim_parameters.set_value("tdf_i", 1.)
                # generate initial conditions
                self._generate_sim_parameters_dependent_initial_conditions(sim_parameters=current_sim_parameters)
                # check if tc_activation occurrs
                tc_activated = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)
                # store result in dataframe
                tip_cell_activation_df = tip_cell_activation_df.append(
                    {col: val for col, val in zip(columns_name, [tc_activated, *param_values])},
                    ignore_index=True
                )
                # find min tdf_i value
                if tc_activated:
                    # find min tdf_i
                    min_tdf_i_for_condition = self._find_min_tdf_i(current_sim_parameters)
                else:
                    min_tdf_i_for_condition = None
                # store it to dataframe
                min_tdf_i_df = min_tdf_i_df.append(
                    {col: val for col, val in zip(columns_name, [min_tdf_i_for_condition, *param_values])},
                    ignore_index=True
                )

            else:
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
            if find_min_tdf_i:
                min_tdf_i_df.to_csv(f"{self.data_folder}/min_tdf_i.csv")

        self._end_simulation()  # conclude time iteration

        return self.runtime_error_occurred

    def _find_min_tdf_i(self, current_sim_parameters, min_tdf_i: float = 0., max_tdf_i: float = 1.):
        """
        Find min dimension for RH for the given parameters combination
        """
        if max_tdf_i - min_tdf_i < 0.05:
            return max_tdf_i
        else:
            # set the mean between the two tdf_i as new test tdf_i
            test_tdf_i = (max_tdf_i + min_tdf_i) / 2
            current_sim_parameters.set_value("tdf_i", test_tdf_i)
            # generate initial conditions
            self._generate_sim_parameters_dependent_initial_conditions(sim_parameters=current_sim_parameters)
            # test tip cell activation
            tc_activated = self.tip_cell_manager.test_tip_cell_activation(self.c_old, self.af_old, self.grad_af_old)
            logger.info(f"Current test tdf_i: {test_tdf_i} | tc_activated: {tc_activated}")
            if tc_activated:
                return self._find_min_tdf_i(current_sim_parameters, min_tdf_i=min_tdf_i, max_tdf_i=test_tdf_i)
            else:
                return self._find_min_tdf_i(current_sim_parameters, min_tdf_i=test_tdf_i, max_tdf_i=max_tdf_i)

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
