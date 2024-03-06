import json
import logging
import time
import argparse
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
from mocafe.fenut.parameters import Parameters
from src.simulation import (RHTimeSimulation, RHTestTipCellActivation, RHAdaptiveSimulation)


# set up logger
logger = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(description="Simple CLI for RH simulations")
    # add slurm_job_id
    parser.add_argument("-slurm_job_id",
                        type=int,
                        help="Slurm job ID for the simulation")
    # add flag for test timulation
    parser.add_argument("-run_2d",
                        action="store_true",
                        help="Run the simulation in 2d to check if everything runs smoothly")
    return parser.parse_args()


def preamble():
    """
    Load general data for simulations
    """
    # load simulation parameters
    parameters_csv = "notebooks/out/g_parameters.csv"
    standard_parameters_df = pd.read_csv(parameters_csv, index_col="name")
    sim_parameters = Parameters(standard_parameters_df)

    # get cli args, if any
    args = cli()

    # load patient parameters
    with open("input_patients_data/patients_parameters.json", "r") as infile:
        patients_parameters = json.load(infile)

    if args.run_2d:
        spatial_dimension = 2
        distributed_data_folder = "temp"
    else:
        spatial_dimension = 3
        distributed_data_folder = "/local/frapra/3drh"

    return sim_parameters, patients_parameters, args.slurm_job_id, spatial_dimension, distributed_data_folder


def timer(method):
    t0 = time.perf_counter()
    method()
    tfin = time.perf_counter() - t0
    if MPI.COMM_WORLD.rank == 0:
        logger.info(f"Time (size: {MPI.COMM_WORLD.size}): {tfin} [seconds]")


def compute_initial_condition_for_each_patient():
    """
    Compute initial condition for each patient
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # For each patient, compute initial condition
    for patient_number in [0, 1, 2]:
        logger.info(f"Starting computation for patient{patient_number}")

        current_patient_parameter = patients_parameters[f"patient{patient_number}"]  # load patient params

        # run simulation
        sim = RHTimeSimulation(spatial_dimension=spatial_dimension,
                               sim_parameters=sim_parameters,
                               patient_parameters=current_patient_parameter,
                               steps=1,
                               save_rate=1,
                               out_folder_name=f"dolfinx_patient{patient_number}_initial_condition",
                               sim_rationale=f"Computed initial condition for patient{patient_number}",
                               slurm_job_id=slurm_job_id,
                               save_distributed_files_to=distributed_data_folder)
        sim.run()


def vascular_sprouting_full_volume(patient: str, n_steps: int = None):
    """
    Simulate tumor-induced angiogenesis for 300 steps under simplified assumptions
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set V_pt as max
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)
    # Set V_uc as min
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_value = V_uc_af_min
    sim_parameters.set_value("V_uc_af", V_uc_value)
    # set n steps
    n_steps = 300 if n_steps is None else n_steps

    # create sim object
    sim = RHTimeSimulation(spatial_dimension=spatial_dimension,
                           sim_parameters=sim_parameters,
                           patient_parameters=patients_parameters[patient],
                           steps=n_steps,
                           save_rate=50,
                           out_folder_name=f"dolfinx_{patient}_full_volume_V_uc{V_uc_value:.2g}",
                           sim_rationale=f"Simulation for representative case of {patient} with tdf_i = "
                                             f"{1}; V_pT_af = ({V_pT_af_max:.2e}) and V_uc_af = "
                                             f"{V_uc_value:.2e} [1 / tau]",
                           slurm_job_id=slurm_job_id,
                           save_distributed_files_to=distributed_data_folder)
    # run simulation
    sim.run()


def find_min_tdf_i():
    """
    Find parameters allowing tip cells activation and minimal diameter.
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, _ = preamble()
    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # 2. we set V_pT_af
    V_pT_af_min = float(sim_parameters_df.loc["V_pT_af", "sim_range_min"])
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = np.logspace(np.log10(V_pT_af_min), np.log10(V_pT_af_max), num=10, endpoint=True)
    # 3. We set V_uc
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_range = np.logspace(0, 4, num=10, endpoint=True) * V_uc_af_min

    # set patient
    for p in [0, 1, 2]:
        patient = f"patient{p}"
        logger.info(f"Starting tip cell activation test for {patient}")
        sim = RHTestTipCellActivation(spatial_dimension=spatial_dimension,
                                      standard_params=sim_parameters,
                                      patient_parameters=patients_parameters[patient],
                                      out_folder_name=f"dolfinx_patient{p}_find_min_tdf_i",
                                      slurm_job_id=slurm_job_id,
                                      sim_rationale=f"Finding min tdf_i")
        sim.run(find_min_tdf_i=True,
                V_pT_af=V_pT_af_range,
                V_uc_af=V_uc_af_range)


def time_adaptive_vascular_sprouting(patient: str):
    """
    Simulate tumor-induced angiogenesis for 1 month
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # parameters values leading to sprouting in patients
    tdf_i_for_patient = {
        "patient0": [0.062, 0.062, 0.062],
        "patient1": [0.77, 0.78, 0.75],
        "patient2": [0.062, 0.062, 0.062]
    }
    # set V_pT
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)
    # set V_uc
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_range = np.logspace(0, 4, num=10, endpoint=True) * V_uc_af_min
    V_uc_af_range = V_uc_af_range[:3]

    n_steps = 1662  # about one month

    for V_uc_af_value, tdf_i_value in zip(V_uc_af_range, tdf_i_for_patient[patient]):
        sim_parameters.set_value("V_uc_af", V_uc_af_value)
        sim_parameters.set_value("tdf_i", tdf_i_value)

        # run simulation
        sim = RHAdaptiveSimulation(spatial_dimension=spatial_dimension,
                                   sim_parameters=sim_parameters,
                                   patient_parameters=patients_parameters[patient],
                                   steps=n_steps,
                                   save_rate=50,
                                   out_folder_name=f"dolfinx_{patient}_vascular-sprout-adap_V_uc={V_uc_af_value:.2g}",
                                   sim_rationale=f"Simulation for representative case of {patient }with tdf_i = "
                                                 f"{tdf_i_value}; V_pT_af = ({V_pT_af_max:.2e}) and V_uc_af = "
                                                 f"{V_uc_af_value:.2e} [1 / tau], the vascular sprouting starts at "
                                                 f"tdf_i = {tdf_i_value:.2g}",
                                   slurm_job_id=slurm_job_id,
                                   save_distributed_files_to=distributed_data_folder)
        sim.run()


def time_adaptive_vascular_sprouting_full_volume_1yr(patient: str):
    """
    Simulate tumor-induced angiogenesis for 1 year
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set V_pt as max
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)
    # Set V_uc in the middle
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_range = np.logspace(0, 4, num=10, endpoint=True) * V_uc_af_min
    V_uc_value = V_uc_af_range[0]
    sim_parameters.set_value("V_uc_af", V_uc_value)
    # set M
    M_val = sim_parameters.get_value("M")
    sim_parameters.set_value("M", M_val / 2)
    # set n steps
    n_steps = 20_215
    sim = RHAdaptiveSimulation(spatial_dimension=spatial_dimension,
                               sim_parameters=sim_parameters,
                               patient_parameters=patients_parameters[patient],
                               steps=n_steps,
                               save_rate=50,
                               out_folder_name=f"dolfinx_{patient}_full_volume_1yr",
                               sim_rationale=f"Simulation for representative case of {patient} with tdf_i = "
                                             f"{1}; V_pT_af = ({V_pT_af_max:.2e}) and V_uc_af = "
                                             f"{V_uc_value:.2e} [1 / tau]",
                               slurm_job_id=slurm_job_id,
                               save_distributed_files_to=distributed_data_folder)
    sim.run()


def test_convergence_1_step(patient: str):
    """
    Test different iterative solvers and preconditioners for your system
    """
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    patient_parameters = patients_parameters[patient]

    sim = RHTimeSimulation(spatial_dimension=spatial_dimension,
                           sim_parameters=sim_parameters,
                           patient_parameters=patient_parameters,
                           steps=1,
                           save_rate=10,
                           out_folder_name=f"dolfinx_{patient}_test_convergence_for_preconditioners_gmres",
                           out_folder_mode=None,
                           sim_rationale="Testing",
                           slurm_job_id=slurm_job_id,
                           save_distributed_files_to=distributed_data_folder)

    # setup the convergence test
    sim.setup_convergence_test()

    # setup list for storing performance
    performance_dicts = []

    # setup list of linear solver parameters to test
    lsp_list = []

    # --------------------------------------------------------------------------------------------------------------- #
    # Add Iterative solvers to lsp list                                                                               #
    # --------------------------------------------------------------------------------------------------------------- #
    # create list of solver and preconditioners
    iterative_solver_list = ["cg", "gmres"]
    pc_type_list = ["jacobi", "bjacobi", "sor", "asm", "gasm", "gamg"]

    # add all combinations to lsp list
    lsp_list.extend([{"ksp_type": solver, "pc_type": pc, "ksp_monitor": None}
                     for solver, pc in product(iterative_solver_list, pc_type_list)])

    # add all combination using mumps as backend
    lsp_list.extend([{"ksp_type": solver, "pc_type": pc, "ksp_monitor": None, "pc_factor_mat_solver_type": "mumps"}
                     for solver, pc in product(iterative_solver_list, pc_type_list)])

    # add hypre preconditioners
    hypre_type_list = ["euclid", "pilut", "parasails", "boomeramg"]
    lsp_list.extend([{"ksp_type": solver, "pc_type": "hypre", "pc_hypre_type": hypre_type, "ksp_monitor": None}
                     for solver, hypre_type in product(iterative_solver_list, hypre_type_list)])

    # --------------------------------------------------------------------------------------------------------------- #
    # Add Direct solvers to lsp list                                                                                  #
    # --------------------------------------------------------------------------------------------------------------- #
    direct_solver_list = ["lu", "cholesky"]
    lsp_list.extend([{"ksp_type": "preonly", "pc_type": ds, "ksp_monitor": None}
                    for ds in direct_solver_list])
    # add also with mumps
    lsp_list.extend([{"ksp_type": "preonly", "pc_type": ds, "ksp_monitor": None, "pc_factor_mat_solver_type": "mumps"}
                    for ds in direct_solver_list])

    # --------------------------------------------------------------------------------------------------------------- #
    # Iterate
    # --------------------------------------------------------------------------------------------------------------- #
    if MPI.COMM_WORLD.rank == 0:
        pbar_file = open("convergence_pbar.o", "w")
    else:
        pbar_file = None
    pbar = tqdm(total=len(lsp_list), ncols=100, desc="convergence_test", file=pbar_file,
                disable=True if MPI.COMM_WORLD.rank != 0 else False)

    for lsp in lsp_list:
        # get characteristics of lsp
        current_solver = lsp['ksp_type']
        if lsp['pc_type'] == "hypre":
            current_pc = f"{lsp['pc_type']} ({lsp['pc_hypre_type']})"
        else:
            current_pc = lsp['pc_type']
        using_mumps = ("mumps" in lsp.values())

        # logging
        msg = f"Testing solver {current_solver} with pc {current_pc}"
        if using_mumps:
            msg += f" (MUMPS)"
        logger.info(msg)

        # set linear solver parameters
        sim.lsp = lsp

        # time solution
        time0 = time.perf_counter()
        sim.test_convergence()
        tot_time = time.perf_counter() - time0

        # check if error occurred
        error = sim.runtime_error_occurred
        error_msg = sim.error_msg

        # build performance dict
        perf_dict = {
            "solver": current_solver,
            "pc": current_pc,
            "mumps": using_mumps,
            "time": tot_time,
            "error": error,
            "error_msg": error_msg
        }

        # append dict to list
        performance_dicts.append(perf_dict)
        df = pd.DataFrame(performance_dicts)
        if MPI.COMM_WORLD.rank == 0:
            df.to_csv(sim.data_folder / Path("performance.csv"))

        # reset runtime error and error msg
        sim.runtime_error_occurred = False
        sim.error_msg = None

        # update pbar
        pbar.update(1)

    pbar_file.close()
