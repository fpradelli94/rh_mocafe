"""
Contains methods and classes to run and resume simulations in 2D and 3D
"""

import json
import fenics
import time
import pathlib
import sys
import pandas as pd
from itertools import product
from pint import Quantity, UnitRegistry
from tqdm import tqdm
import numpy as np
from PIL import Image
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
from src.simulation2d import compute_2d_mesh_from_image
from src.simulation3d import compute_3d_mesh_from_image, compute_3d_c_0


comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()


def mpi_print(msg: str, r: int = None):
    """
    Wrapper of the print function MPI-aware.
    """
    if r is None:
        print(f"p{rank}: {msg}", file=sys.stdout)
        sys.stdout.flush()
    else:
        if rank == r:
            print(f"p{rank}: {msg}", file=sys.stdout)
            sys.stdout.flush()


def load_arbitrary_units(
        local_ureg: UnitRegistry,
        parameters_df: pd.DataFrame,
        sau_name: str,
        tau_name: str,
        afau_name: str
):
    """
    Load the arbitrary units with given names in the Unit Registry.

    :param local_ureg: the Unit Registry to update
    :param parameters_df: the dataframe where the arbitrary units are saved.
    :param sau_name: the name of the Space Arbitrary Unit (sau) in the dataframe.
    :param tau_name: the name of the Time Arbitrary Unit (tau) in the dataframe.
    :param afau_name: the name of the AFs Arbitrary Unit (afau) in the dataframe.
    :return: the updated Unit Registry
    """
    # define sau according to dataframe
    local_ureg.define(f"{sau_name} = "
                      f"{parameters_df.loc[sau_name, 'real_value']} * {parameters_df.loc[sau_name, 'real_um']} = "
                      f"sau")

    # define tau according to dataframe
    local_ureg.define(f"{tau_name} = "
                      f"{parameters_df.loc[tau_name, 'real_value']} * {parameters_df.loc[tau_name, 'real_um']} = "
                      f"tau")

    # define afau according to dataframe
    local_ureg.define(f"af concentration arbitrary unit = "
                      f"{parameters_df.loc[afau_name, 'real_value']} * {parameters_df.loc[afau_name, 'real_um']} = "
                      f"afau")

    return local_ureg


def estimate_pixel_size(
        image_path: str,
        tumor_diameter: Quantity,
        local_ureg: UnitRegistry
):
    """
    Estimate the real dimension of the input image pixels given the tumor diameter. The pixels are added to the local
    unit registry to be used for Mesh Generation.

    The input image should be black and white, with white pixels representing the tumor and black pixels the
    surrounding tissues. Any grey between included in [1, 255] is converted to 255 to estimate the area, but the
    original image is not modified.

    :param image_path: path of the input image
    :param tumor_diameter: ``Quantity`` indicating the tumor diameter
    :local_ureg: the Unit Registry to update
    :return: the updated Unit Registry
    """
    # read tumour selection image as grayscale
    image = Image.open(image_path).convert('L')
    # convert to numpy array
    np_image = np.array(image)
    # transform every non-zero pixel to 1
    np_image[np_image > 0] = 1.
    # get the total number of pixels
    area_pxl = np.sum(np_image)
    # estimate the exact area
    area_um2 = np.pi * ((tumor_diameter / 2) ** 2)
    # get the area for each pixel
    um2_pxl = area_um2 / area_pxl
    # define pxl as unit of measure
    local_ureg.define(f"Pixel = {um2_pxl} = pxl")
    # get pixel size length
    pxl_side_um = np.sqrt(um2_pxl)
    # define pxl side as unit of measure
    local_ureg.define(f"Pixel Side = {pxl_side_um} = pxls")
    return local_ureg


def compute_af_0(af_old: fenics.Function,
                 V: fenics.FunctionSpace,
                 phi: fenics.Function,
                 c_old: fenics.Function,
                 sim_parameters: Parameters,
                 options: Dict = None):
    """
    Solve equilibrium system for af considering the initial values of phi and c.

    Basically, this function is used to generate the initial condition for af assuming that af is at equilibrium
    at the beginning of the simulation.

    :param af_old: function for af initial condition. Will be overwritten.
    :param V: function space for af at equilibrium
    :param phi: Initial tumor
    :param c_old: Initial vessel.
    :param sim_parameters: Simulation parameters.
    :param options: specify the options for the solver to use
    """
    # manage none dict
    if options is None:
        options = {}
    # get af variable
    af = fenics.TrialFunction(V)
    # get test function
    v = fenics.TestFunction(V)
    # built equilibrium form for af
    af_form = src.forms.angiogenic_factors_form_eq(af, phi, c_old, v, sim_parameters)
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
    ksp.solve(b, af_old.vector().vec())
    af_old.vector().update_ghost_values()


def save_resume_info(resume_folder: pathlib.Path,
                     mesh: fenics.Mesh = None,
                     fnc_dict: Dict[str, fenics.Function] = None,
                     tip_cell_manager: TipCellManager = None) -> Dict[str, str]:
    """
    Save the mesh and/or the fenics Functions in a resumable format (i.e. using the FEniCS function `write_checkpoint`).

    :param resume_folder: name of the folder where the files will be saved
    :param mesh: mesh to save
    :param fnc_dict: dict of functions to save. Each function is coupled with a name to be used for saving the file
    (e.g. the function stored in the dictionary as ``{"af": af_function}`` will be saved as ``af.xdmf``)
    :param tip_cell_manager: TipCellManager to use for saving the tip cells status. It will be saved as ``.json`` file.
    """
    # init list of saved file
    saved_files = {}
    if rank == 0:
        resume_folder.mkdir(exist_ok=True)

    # write mesh
    if mesh is not None:
        file_name = f"{resume_folder}/mesh.xdmf"
        with fenics.XDMFFile(file_name) as outfile:
            outfile.write(mesh)
        saved_files["mesh"] = str(pathlib.Path(file_name).resolve())

    # write functions
    if fnc_dict is not None:
        for name, fnc in fnc_dict.items():
            file_name = f"{resume_folder}/{name}.xdmf"
            with fenics.XDMFFile(file_name) as outfile:
                outfile.write_checkpoint(fnc, name, 0, fenics.XDMFFile.Encoding.HDF5, False)
            saved_files[name] = str(pathlib.Path(file_name).resolve())

    # store tip cells position
    if tip_cell_manager is not None:
        file_name = f"{resume_folder}/tipcells.json"
        tip_cell_manager.save_tip_cells(file_name)
        saved_files["tipcells"] = str(pathlib.Path(file_name).resolve())

    return saved_files


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
    # -- Initial setup
    # check if is possible to resume
    error_msg = "Not enough info to resume."

    sim_parameters_csv = pathlib.Path(resume_from + "/sim_info/sim_parameters.csv")
    if not sim_parameters_csv.exists():
        raise RuntimeError(error_msg + " sim_parameters.csv missing.")
    sim_parameters = Parameters(pd.read_csv(str(sim_parameters_csv)))

    mesh_parameters_csv = pathlib.Path(resume_from + "/sim_info/mesh_parameters.csv")
    if not mesh_parameters_csv.exists():
        raise RuntimeError(error_msg + " mesh_parameters.csv missing.")
    mesh_parameters = Parameters(pd.read_csv(mesh_parameters_csv))

    input_incremental_tip_cells = pathlib.Path(resume_from + "/sim_info/incremental_tipcells.json")
    if not input_incremental_tip_cells.exists():
        raise RuntimeError(error_msg + " incremental_tipcells.json missing.")
    with open(str(input_incremental_tip_cells), "r") as infile:
        input_itc = json.load(infile)

    input_resume_folder = pathlib.Path(resume_from + "/resume")
    resume_files = ["mesh.xdmf", "af.xdmf", "c.xdmf", "grad_af.xdmf", "mu.xdmf", "phi.xdmf", "tipcells.json"]
    resume_files_dict = {file_name: input_resume_folder / pathlib.Path(file_name)
                         for file_name in resume_files}
    if not input_resume_folder.exists():
        raise RuntimeError(error_msg + " resume folder missing.")
    else:
        for key, item in resume_files_dict.items():
            if not item.exists():
                raise RuntimeError(error_msg + f" {key} missing")

    # count time
    init_time = time.time()

    # set up folders
    data_folder = mansim.setup_data_folder(folder_path=f"saved_sim/{out_folder_name}",
                                           auto_enumerate=out_folder_mode)
    report_folder = data_folder / pathlib.Path("sim_info")
    if rank == 0:
        report_folder.mkdir(exist_ok=True, parents=True)
    reproduce_folder = data_folder / pathlib.Path("0_reproduce")
    if (slurm_job_id is not None) and (rank == 0):
        pathlib.Path("slurm").mkdir(exist_ok=True)

    # save script and data in reproduce folder
    if rank == 0:
        if reproduce_folder.exists():
            shutil.rmtree(str(reproduce_folder.resolve()))
        shutil.copytree(str(pathlib.Path(__file__).parent.parent.resolve()), str(reproduce_folder.resolve()),
                        ignore=shutil.ignore_patterns("README.md",
                                                      "saved_sim*",
                                                      "*.ipynb_checkpoints*",
                                                      "sif",
                                                      "visualization",
                                                      "*pycache*"))
    comm_world.Barrier()

    # setup logger
    if activate_logger:
        confgure_root_logger_with_standard_settings(data_folder)

    # set up xdmf files
    c_xdmf, af_xdmf, grad_af_xdmf, tipcells_xdmf, phi_xdmf = fu.setup_xdmf_files(
        ["c", "af", "grad_af", "tipcells", "phi"],
        data_folder,
        {"flush_output": True, "rewrite_function_mesh": False}
    )

    # --- Mesh definition
    mesh = fenics.Mesh()
    with fenics.XDMFFile(str(resume_files_dict["mesh.xdmf"])) as infile:
        infile.read(mesh)

    # --- Spatial discretization
    V = fu.get_mixed_function_space(mesh, 3)
    vec_V = fenics.VectorFunctionSpace(mesh, "CG", 1)  # for grad_af

    # --- Define linear solver parameters
    lsp = {"ksp_type": "gmres", "pc_type": "gamg"}

    # --- Initial conditions
    last_step = max([int(step.replace("step_", "")) for step in input_itc])

    # capillaries
    c_old = fenics.Function(V.sub(0).collapse())
    resume_c_xdmf = fenics.XDMFFile(str(resume_files_dict["c.xdmf"]))
    resume_c_xdmf.read_checkpoint(c_old, "c", 0)
    c_old.rename("c", "capillaries")
    c_xdmf.write(c_old, last_step)

    # mu
    mu_old = fenics.Function(V.sub(0).collapse())
    resume_mu_xdmf = fenics.XDMFFile(str(resume_files_dict["mu.xdmf"]))
    resume_mu_xdmf.read_checkpoint(mu_old, "mu", 0)

    # phi
    spatial_dimension = len(mesh.coordinates()[0])
    phi_expression = get_growing_RH_expression(sim_parameters, mesh_parameters, last_step, spatial_dimension)
    phi = fenics.interpolate(phi_expression, V.sub(0).collapse())
    phi.rename("phi", "retinal hemangioblastoma")
    phi_xdmf.write(phi, last_step)

    # tcf function
    t_c_f_function = fenics.interpolate(fenics.Constant(0.), V.sub(0).collapse())
    t_c_f_function.rename("tcf", "tip cells function")
    tipcells_xdmf.write(t_c_f_function, 0)

    # af
    af_old = fenics.Function(V.sub(0).collapse())
    resume_af_xdmf = fenics.XDMFFile(str(resume_files_dict["af.xdmf"]))
    resume_af_xdmf.read_checkpoint(af_old, "af", 0)
    af_old.rename("af", "angiogenic factor")
    af_xdmf.write(af_old, 0)

    # af gradient
    ge = GradientEvaluator()
    grad_af_old = fenics.Function(vec_V)
    resume_grad_af_xdmf = fenics.XDMFFile(str(resume_files_dict["grad_af.xdmf"]))
    resume_grad_af_xdmf.read_checkpoint(grad_af_old, "grad_af", 0)
    grad_af_old.rename("grad_af", "angiogenic factor gradient")
    grad_af_xdmf.write(grad_af_old, 0)

    # --- Weak form definition
    # define variable functions
    if spatial_dimension == 3:
        mpi_print("Defining weak form...")
    u = fenics.Function(V)
    fenics.assign(u, [af_old, c_old, mu_old])
    af, c, mu = fenics.split(u)
    # define test functions
    v1, v2, v3 = fenics.TestFunctions(V)
    # bulid total form
    af_form = src.forms.angiogenic_factors_form_eq(af, phi, c, v1, sim_parameters)
    capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(c, c_old, mu, mu_old, v2, v3,
                                                                             sim_parameters)
    form = af_form + capillaries_form

    # --- Problem solution
    if spatial_dimension == 3:
        mpi_print("Defining problem...")
    # define Jacobian
    J = fenics.derivative(form, u)
    # define problem
    problem = PETScProblem(J, form, [])
    solver = PETScNewtonSolver(lsp, mesh.mpi_comm())

    # define tip cell manager
    tip_cell_manager = TipCellManager(mesh, sim_parameters,
                                      initial_tcs=load_tip_cells_from_json(str(resume_files_dict["tipcells.json"])))

    # init time iteration
    t = last_step
    dt = sim_parameters.get_value("dt")

    if rank == 0:
        if slurm_job_id is not None:
            pbar_file = open(f"slurm/{slurm_job_id}pbar.o", 'w')
            pbar = tqdm(total=steps, ncols=100, desc=out_folder_name, file=pbar_file)
        else:
            pbar_file = None
            pbar = tqdm(total=steps, ncols=100, desc=out_folder_name)
    else:
        pbar_file = None
        pbar = None

    # log
    if spatial_dimension == 3:
        mpi_print("Starting time iteration...")

    # iterate in time
    for step in range(last_step + 1, last_step + steps + 1):
        # update time
        t += dt

        # activate tip cells
        tip_cell_manager.activate_tip_cell(c_old, af_old, grad_af_old, step)

        # revert tip cells
        tip_cell_manager.revert_tip_cells(af_old, grad_af_old)

        # move tip cells
        tip_cell_manager.move_tip_cells(c_old, af_old, grad_af_old)

        # store tip cells in fenics function and json file
        t_c_f_function.assign(tip_cell_manager.get_latest_tip_cell_function())
        tip_cell_manager.save_incremental_tip_cells(f"{report_folder}/incremental_tipcells.json", step)

        # solve
        try:
            solver.solve(problem, u.vector())
        except RuntimeError as e:
            # save sim data
            mansim.save_sim_info(data_folder=report_folder,
                                 parameters={"Sim Parameters:": sim_parameters, "Mesh parameters": mesh_parameters},
                                 execution_time=time.time() - init_time,
                                 sim_name=out_folder_name,
                                 sim_description="RUNTIME ERROR OCCURRED. \n" + sim_rationale, error_msg=str(e))
            # save parameters
            sim_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("sim_parameters.csv"))
            mesh_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("mesh_parameters.csv"))
            # write files
            af_xdmf.write(af_old, t)
            c_xdmf.write(c_old, t)
            grad_af_xdmf.write(grad_af_old, t)
            phi_xdmf.write(phi, t)
            # save tip cells current position
            tipcells_xdmf.write(t_c_f_function, t)
            save_resume_info(data_folder / pathlib.Path("resume"),
                             fnc_dict={"af": af_old, "c": c_old, "mu": mu_old, "phi": phi, "grad_af": grad_af_old},
                             tip_cell_manager=tip_cell_manager)
            # close iteration in time
            return 1

        # assign to old
        fenics.assign([af_old, c_old, mu_old], u)
        # assign new value to grad_af_old
        ge.compute_gradient(af_old, vec_V, grad_af_old, lsp)
        # assign new value to phi
        phi_expression.t = t  # update time
        phi.assign(fenics.interpolate(phi_expression, V.sub(0).collapse()))

        # save
        if (step % save_rate == 0) or (step == (last_step + steps)):
            # write files
            af_xdmf.write(af_old, t)
            c_xdmf.write(c_old, t)
            grad_af_xdmf.write(grad_af_old, t)
            phi_xdmf.write(phi, t)
            # save tip cells current position
            tipcells_xdmf.write(t_c_f_function, t)

        # update progress bar
        if rank == 0:
            pbar.update(1)

    if (rank == 0) and (slurm_job_id is not None):
        pbar_file.close()

    # save resume info
    save_resume_info(data_folder / pathlib.Path("resume"),
                     fnc_dict={"af": af_old, "c": c_old, "mu": mu_old, "phi": phi,
                               "grad_af": grad_af_old},
                     tip_cell_manager=tip_cell_manager)

    # save sim data
    mansim.save_sim_info(data_folder=report_folder,
                         parameters={"Sim parameters:": sim_parameters, "Mesh parameters": mesh_parameters},
                         execution_time=time.time() - init_time,
                         sim_name=out_folder_name,
                         sim_description=sim_rationale)
    # save parameters
    sim_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("sim_parameters.csv"))
    mesh_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("mesh_parameters.csv"))

    comm_world.Barrier()  # wait for all processes

    return 0


def run_simulation(spatial_dimension: int,
                   sim_parameters: Parameters,
                   steps: int,
                   save_rate: int = 1,
                   out_folder_name: str = mansim.default_data_folder_name,
                   out_folder_mode: str = None,
                   sim_rationale: str = "No comment",
                   slurm_job_id: int = None,
                   vessels_no_discontinuities: bool = True,
                   activate_logger: bool = False):
    """
    Resume a simulation and store the result

    :param spatial_dimension: specify if the simulation will be in 2D or 3D.
    :param sim_parameters: parameters to be used in the simulation.
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
    :param vessels_no_discontinuities: use the vessels image without discontinuities. This option was added to allow us
    to choose between two similar versions of the vessel images (see ``notebooks/vessels_image_processing.ipynb``). No
    difference in the simulations results was found using one image or another.
    :param activate_logger: activate the standard logger to save detailed information on the simulation activity.
    """
    # -- Check spatial dimension
    if not (spatial_dimension == 2 or spatial_dimension == 3):
        raise RuntimeError(f"Cannot run simulation for dimension {spatial_dimension}")

    # -- Initial setup
    init_time = time.time()  # count time

    local_ureg = UnitRegistry()

    # set up folders
    data_folder = mansim.setup_data_folder(folder_path=f"saved_sim/{out_folder_name}",
                                           auto_enumerate=out_folder_mode)

    report_folder = data_folder / pathlib.Path("sim_info")
    if rank == 0:
        report_folder.mkdir(exist_ok=True, parents=True)
    reproduce_folder = data_folder / pathlib.Path("0_reproduce")
    if (slurm_job_id is not None) and (rank == 0):
        pathlib.Path("slurm").mkdir(exist_ok=True)

    # save script and data in reproduce folder
    if rank == 0:
        if reproduce_folder.exists():
            shutil.rmtree(str(reproduce_folder.resolve()))
        shutil.copytree(str(pathlib.Path(__file__).parent.parent.resolve()), str(reproduce_folder.resolve()),
                        ignore=shutil.ignore_patterns("README.md",
                                                      "saved_sim*",
                                                      "*.ipynb_checkpoints*",
                                                      "sif",
                                                      "visualization",
                                                      "*pycache*"))
    comm_world.Barrier()

    # setup logger
    if activate_logger:
        confgure_root_logger_with_standard_settings(data_folder)

    # set up xdmf files
    c_xdmf, af_xdmf, grad_af_xdmf, tipcells_xdmf, phi_xdmf = fu.setup_xdmf_files(
        ["c", "af", "grad_af", "tipcells", "phi"],
        data_folder,
        {"flush_output": True, "rewrite_function_mesh": False}
    )

    # load arbitrary units
    local_ureg = load_arbitrary_units(local_ureg,
                                      sim_parameters.as_dataframe(),
                                      sau_name="Space Arbitrary Unit",
                                      tau_name="Time Arbitrary Unit",
                                      afau_name="AFs Arbitrary Unit")

    # --- Mesh definition
    max_tumor_diameter = 600 * local_ureg("um")
    local_ureg = estimate_pixel_size("input_images/RH_initial_selection.png",
                                     max_tumor_diameter,
                                     local_ureg)
    if spatial_dimension == 2:
        mesh, mesh_parameters = compute_2d_mesh_from_image(sim_parameters.as_dataframe(),
                                                           local_ureg,
                                                           "input_images/RH_initial_selection.png")
    else:
        mpi_print("Computing mesh...")
        mesh, mesh_parameters = compute_3d_mesh_from_image(sim_parameters.as_dataframe(),
                                                           local_ureg,
                                                           "input_images/RH_initial_selection.png",
                                                           max_tumor_diameter)
    # reload mesh to ensure consistent distribution of the dofs
    saved_files = save_resume_info(data_folder / pathlib.Path("resume"), mesh=mesh)
    del mesh
    mesh = fenics.Mesh()
    with fenics.XDMFFile(str(saved_files["mesh"])) as infile:
        infile.read(mesh)

    # --- Spatial discretization
    V = fu.get_mixed_function_space(mesh, 3)
    vec_V = fenics.VectorFunctionSpace(mesh, "CG", 1)  # for grad_af

    # --- Define linear solver parameters
    lsp = {"ksp_type": "gmres", "pc_type": "gamg"}

    # --- Initial conditions
    # capillaries
    if spatial_dimension == 2:
        if vessels_no_discontinuities:
            cb = "notebooks/out/RH_vessels_binary_ND.png"
        else:
            cb = "notebooks/out/RH_vessels_binary.png"
        c_old_expression = BWImageExpression(cb, [-1, 1], local_ureg)
        c_old = fenics.interpolate(c_old_expression, V.sub(0).collapse())
    else:
        mpi_print("Computing c0...")
        c_old = fenics.Function(V.sub(0).collapse())
        compute_3d_c_0(c_old, mesh_parameters, vessels_no_discontinuities)
    c_old.rename("c", "capillaries")
    c_xdmf.write(c_old, 0)

    # auxiliary fun for capillaries, initially set to 0
    if spatial_dimension == 3:
        mpi_print("Computing mu0...")
    mu_old = fenics.interpolate(fenics.Constant(0.), V.sub(0).collapse())

    # define initial condition for tumor
    if spatial_dimension == 3:
        mpi_print("Computing phi0...")
    phi_expression = get_growing_RH_expression(sim_parameters, mesh_parameters, 0, spatial_dimension)
    phi = fenics.interpolate(phi_expression, V.sub(0).collapse())
    phi.rename("phi", "retinal hemangioblastoma")
    phi_xdmf.write(phi, 0)

    # t_c_f_function (dynamic tip cell position)
    if spatial_dimension == 3:
        mpi_print("Computing t_c_f_function...")
    t_c_f_function = fenics.interpolate(fenics.Constant(0.), V.sub(0).collapse())
    t_c_f_function.rename("tcf", "tip cells function")
    tipcells_xdmf.write(t_c_f_function, 0)

    # af
    if spatial_dimension == 3:
        mpi_print("Computing af0...")
    af_old = fenics.Function(V.sub(0).collapse())
    compute_af_0(af_old, V.sub(0).collapse(), phi, c_old, sim_parameters, lsp)
    af_old.rename("af", "angiogenic factor")
    af_xdmf.write(af_old, 0)

    # af gradient
    if spatial_dimension == 3:
        mpi_print("Computing grad_af0...")
    ge = GradientEvaluator()
    grad_af_old = fenics.Function(vec_V)
    ge.compute_gradient(af_old, vec_V, grad_af_old, lsp)
    grad_af_old.rename("grad_af", "angiogenic factor gradient")
    grad_af_xdmf.write(grad_af_old, 0)

    # --- Weak form definition
    # define variable functions
    if spatial_dimension == 3:
        mpi_print("Defining weak form...")
    u = fenics.Function(V)
    fenics.assign(u, [af_old, c_old, mu_old])
    af, c, mu = fenics.split(u)
    # define test functions
    v1, v2, v3 = fenics.TestFunctions(V)
    # bulid total form
    af_form = src.forms.angiogenic_factors_form_eq(af, phi, c, v1, sim_parameters)
    capillaries_form = mocafe.angie.forms.angiogenesis_form_no_proliferation(c, c_old, mu, mu_old, v2, v3,
                                                                             sim_parameters)
    form = af_form + capillaries_form

    # --- Problem solution
    if spatial_dimension == 3:
        mpi_print("Defining problem...")
    # define Jacobian
    J = fenics.derivative(form, u)
    # define problem
    problem = PETScProblem(J, form, [])
    solver = PETScNewtonSolver(lsp, mesh.mpi_comm())

    # define tip cell manager
    tip_cell_manager = TipCellManager(mesh, sim_parameters)

    # init time iteration
    t = 0
    dt = sim_parameters.get_value("dt")

    if rank == 0:
        if slurm_job_id is not None:
            pbar_file = open(f"slurm/{slurm_job_id}pbar.o", 'w')
            pbar = tqdm(total=steps, ncols=100, desc=out_folder_name, file=pbar_file)
        else:
            pbar_file = None
            pbar = tqdm(total=steps, ncols=100, desc=out_folder_name)
    else:
        pbar_file = None
        pbar = None

    # log
    if spatial_dimension == 3:
        mpi_print("Starting time iteration...")

    # iterate in time
    for step in range(1, steps + 1):
        # update time
        t += dt

        # activate tip cells
        tip_cell_manager.activate_tip_cell(c_old, af_old, grad_af_old, step)

        # revert tip cells
        tip_cell_manager.revert_tip_cells(af_old, grad_af_old)

        # move tip cells
        tip_cell_manager.move_tip_cells(c_old, af_old, grad_af_old)

        # store tip cells in fenics function and json file
        t_c_f_function.assign(tip_cell_manager.get_latest_tip_cell_function())
        tip_cell_manager.save_incremental_tip_cells(f"{report_folder}/incremental_tipcells.json", step)

        # solve
        try:
            solver.solve(problem, u.vector())
        except RuntimeError as e:
            # save sim data
            mansim.save_sim_info(data_folder=report_folder,
                                 parameters={"Sim Parameters:": sim_parameters, "Mesh parameters": mesh_parameters},
                                 execution_time=time.time() - init_time,
                                 sim_name=out_folder_name,
                                 sim_description="RUNTIME ERROR OCCURRED. \n" + sim_rationale, error_msg=str(e))
            # save parameters
            sim_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("sim_parameters.csv"))
            mesh_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("mesh_parameters.csv"))
            # write files
            af_xdmf.write(af_old, t)
            c_xdmf.write(c_old, t)
            grad_af_xdmf.write(grad_af_old, t)
            phi_xdmf.write(phi, t)
            # save tip cells current position
            tipcells_xdmf.write(t_c_f_function, t)
            save_resume_info(data_folder / pathlib.Path("resume"),
                             fnc_dict={"af": af_old, "c": c_old, "mu": mu_old, "phi": phi, "grad_af": grad_af_old},
                             tip_cell_manager=tip_cell_manager)
            # close iteration in time
            return 1

        # assign to old
        fenics.assign([af_old, c_old, mu_old], u)
        # assign new value to grad_af_old
        ge.compute_gradient(af_old, vec_V, grad_af_old, lsp)
        # assign new value to phi
        phi_expression.t = t  # update time
        phi.assign(fenics.interpolate(phi_expression, V.sub(0).collapse()))

        # save
        if (step % save_rate == 0) or (step == steps):
            # write files
            af_xdmf.write(af_old, t)
            c_xdmf.write(c_old, t)
            grad_af_xdmf.write(grad_af_old, t)
            phi_xdmf.write(phi, t)
            # save tip cells current position
            tipcells_xdmf.write(t_c_f_function, t)

        # update progress bar
        if rank == 0:
            pbar.update(1)

    if (rank == 0) and (slurm_job_id is not None):
        pbar_file.close()

    # save resume info
    save_resume_info(data_folder / pathlib.Path("resume"),
                     fnc_dict={"af": af_old, "c": c_old, "mu": mu_old, "phi": phi, "grad_af": grad_af_old},
                     tip_cell_manager=tip_cell_manager)

    # save sim data
    mansim.save_sim_info(data_folder=report_folder,
                         parameters={"Sim parameters:": sim_parameters, "Mesh parameters": mesh_parameters},
                         execution_time=time.time() - init_time,
                         sim_name=out_folder_name,
                         sim_description=sim_rationale)
    # save parameters
    sim_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("sim_parameters.csv"))
    mesh_parameters.as_dataframe().to_csv(report_folder / pathlib.Path("mesh_parameters.csv"))

    comm_world.Barrier()  # wait for all processes

    return 0


def test_tip_cell_activation(spatial_dimension: int,
                             df_standard_params: pd.DataFrame,
                             out_folder_name: str = mansim.default_data_folder_name,
                             out_folder_mode: str = None,
                             slurm_job_id: int = None,
                             vessels_no_discontinuities: bool = True,
                             results_df: str = None,
                             **kwargs):
    """
    Test if tip cells activate at first simulation step for a given set of parameters. The default ranges are those
    used for Fig. 3 of the manuscript. To test a different range or another parameters, use ``kwargs``. For instance,
    you can do:

        test_tip_cell_activation(spatial_dimension=3,
                                 df_standard_params=sim_parameters,
                                 sim_name="test",
                                 min_tumor_diameter=[0.3, 0.4, 0.5])

    :param spatial_dimension: 2 if 2D, 3 if 3D.
    :param df_standard_params: dataframe containing the parameter set to be used as base for all simulations. The
    parameters will be changed according to the given ranges or to the default ranges.
    :param out_folder_name: name of the output folder.
    :param out_folder_mode: if ``None``, the output folder name will be exactly the one specified in out_folder_name; if
    ``datetime``, the output folder name will be also followed by a string containing the date and the time of the
    simulation (e.g. ``saved_sim/my_sim/2022-09-25_15-51-17-610268``). The latter is recommended to run multiple
    simulations of the same kind. Default is ```None``.
    :param slurm_job_id: slurm job id, if available. Default is None.
    :param vessels_no_discontinuities: use the vessels image without discontinuities
    :param results_df: give the dataframe where to save the tip cells activation results as input. Default is None.
    If None, a new dataframe will be created.
    :param kwargs: give a parameter name and a range as input to test whether tip cells activation occurs in the given
    parameters range. Each parameter must be associated with a list (even if a single value is given).
    """
    # -- Initial setup
    local_ureg = UnitRegistry()

    # set up folders
    data_folder = mansim.setup_data_folder(folder_path=f"{pathlib.Path('saved_sim')}/{out_folder_name}",
                                           auto_enumerate=out_folder_mode)

    # load arbitrary units
    local_ureg = load_arbitrary_units(local_ureg,
                                      df_standard_params,
                                      sau_name="Space Arbitrary Unit",
                                      tau_name="Time Arbitrary Unit",
                                      afau_name="AFs Arbitrary Unit")

    # --- Define parameters sets to test
    if kwargs:
        # if kwargs is not empty, set it as param dictionary
        if all([isinstance(v, list) for v in kwargs.values()]):
            params_dictionary = kwargs
        else:
            raise TypeError("Every parameter must be associated with a list.")

    else:
        # else create default params dictionary
        tumor_diameter_range = np.linspace(float(df_standard_params.loc["min_tumor_diameter", "sim_value"]) / 10,
                                           float(df_standard_params.loc["min_tumor_diameter", "sim_value"]),
                                           num=5,
                                           endpoint=True)
        D_af_range = [float(df_standard_params.loc["D_af", "sim_range_min"]),
                      float(df_standard_params.loc["D_af", "sim_value"] / 10),
                      float(df_standard_params.loc["D_af", "sim_value"]),
                      float(df_standard_params.loc["D_af", "sim_range_max"])]
        V_pT_af_range = np.logspace(np.log10(float(df_standard_params.loc["V_pT_af", "sim_range_min"])),
                                    np.log10(float(df_standard_params.loc["V_pT_af", "sim_range_max"])),
                                    num=5,
                                    endpoint=True)
        V_uc_af_range = np.logspace(np.log10(2 * float(df_standard_params.loc["V_d_af", "sim_range_min"])),
                                    np.log10(10 * float(df_standard_params.loc["V_uc_af", "sim_value"])),
                                    num=5,
                                    endpoint=True)
        params_dictionary = {"min_tumor_diameter": tumor_diameter_range,
                             "D_af": D_af_range,
                             "V_pT_af": V_pT_af_range,
                             "V_uc_af": V_uc_af_range}

    # generate dataframe
    columns_name = ["tip_cell_activated",
                    *[f"{k} (range: [{np.amin(r)}, {np.amax(r)}])" for k, r in params_dictionary.items()]]
    if results_df is None:
        tip_cell_activation_df = pd.DataFrame(
            columns=columns_name
        )
    else:
        tip_cell_activation_df = pd.read_csv(results_df)

    # --- Define linear solver parameters
    lsp = {"ksp_type": "gmres", "pc_type": "gamg"}

    # --- Mesh definition
    max_tumor_diameter = 600 * local_ureg("um")
    local_ureg = estimate_pixel_size("input_images/RH_initial_selection.png",
                                     max_tumor_diameter,
                                     local_ureg)
    if spatial_dimension == 2:
        mesh, mesh_parameters = compute_2d_mesh_from_image(df_standard_params,
                                                           local_ureg,
                                                           "input_images/RH_initial_selection.png")
    else:
        mpi_print("Computing mesh...")
        mesh, mesh_parameters = compute_3d_mesh_from_image(df_standard_params,
                                                           local_ureg,
                                                           "input_images/RH_initial_selection.png",
                                                           max_tumor_diameter)

    # --- Spatial discretization
    mpi_print("Starting spatial discretization")
    V = fu.get_mixed_function_space(mesh, 3)
    # define fun space for grad_af
    vec_V = fenics.VectorFunctionSpace(mesh, "CG", 1)

    # --- Define intial condition for c
    if spatial_dimension == 2:
        if vessels_no_discontinuities:
            cb = "notebooks/out/RH_vessels_binary_ND.png"
        else:
            cb = "notebooks/out/RH_vessels_binary.png"
        c_old_expression = BWImageExpression(cb, [-1, 1], local_ureg)
        c_old = fenics.interpolate(c_old_expression, V.sub(0).collapse())
    else:
        mpi_print("Computing c0...")
        c_old = fenics.Function(V.sub(0).collapse())
        compute_3d_c_0(c_old, mesh_parameters, vessels_no_discontinuities)

    # -- Define gradient evaluator to compute grad_af
    ge = GradientEvaluator()

    # define pbar
    if rank == 0:
        if slurm_job_id is not None:
            pbar_file = open(f"slurm/{slurm_job_id}pbar.o", 'w')
            pbar = tqdm(total=len(list(product(*params_dictionary.values()))),
                        ncols=100, desc=out_folder_name, file=pbar_file)
        else:
            pbar_file = None
            pbar = tqdm(total=len(list(product(*params_dictionary.values()))),
                        ncols=100, desc=out_folder_name)
    else:
        pbar_file = None
        pbar = None
    comm_world.Barrier()

    # start iteration
    for param_values in product(*params_dictionary.values()):
        # set parameters value
        sim_parameters = Parameters(df_standard_params)
        for param_name, param_value in zip(params_dictionary.keys(), param_values):
            sim_parameters.set_value(param_name, param_value)

        # --- Initial conditions
        # phi
        if spatial_dimension == 3:
            mpi_print("Computing phi0...")
        phi_expression = get_growing_RH_expression(sim_parameters, mesh_parameters, 0, spatial_dimension)
        phi = fenics.interpolate(phi_expression, V.sub(0).collapse())
        phi.rename("phi", "retinal hemangioblastoma")

        # af
        if spatial_dimension == 3:
            mpi_print("Computing af0...")
        af_old = fenics.Function(V.sub(0).collapse())
        compute_af_0(af_old, V.sub(0).collapse(), phi, c_old, sim_parameters, lsp)
        af_old.rename("af", "angiogenic factor")

        # af gradient
        if spatial_dimension == 3:
            mpi_print("Computing grad_af0...")
        grad_af_old = fenics.Function(vec_V)
        ge.compute_gradient(af_old, vec_V, grad_af_old, lsp)
        grad_af_old.rename("grad_af", "angiogenic factor gradient")

        # define tip cell manager
        tip_cell_manager = TipCellManager(mesh, sim_parameters)

        # call activate tip cell
        tip_cell_manager.activate_tip_cell(c_old, af_old, grad_af_old, 1)

        # check if tip cell has been activated
        tipcell_activated = (len(tip_cell_manager.get_global_tip_cells_list()) == 1)

        # store result in dataframe
        tip_cell_activation_df = tip_cell_activation_df.append(
            {col: val for col, val in zip(columns_name, [tipcell_activated, *param_values])},
            ignore_index=True
        )

        if rank == 0:
            # write at each step
            tip_cell_activation_df.to_csv(f"{data_folder}/tipcell_activation.csv")
            # update pbar
            pbar.update(1)

        comm_world.Barrier()

    if rank == 0 and (slurm_job_id is not None):
        pbar_file.close()
