import json
import logging
import time
import argparse
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
from mocafe.fenut.parameters import Parameters
from mpi4py import MPI
from src.simulation import (RHTimeSimulation, RHTestTipCellActivation, RHAdaptiveSimulation, RHMeshAdaptiveSimulation)


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


def test_2d():
    # preamble
    sim_parameters, patients_parameters, _, _, _ = preamble()

    # get patient 1 parametes
    patient1_parameters = patients_parameters["patient1"]

    # run simulation for 3 steps
    sim = RHTimeSimulation(spatial_dimension=2,
                           sim_parameters=sim_parameters,
                           patient_parameters=patient1_parameters,
                           steps=3,
                           save_rate=1,
                           out_folder_name="test_2d",
                           sim_rationale="2d test",
                           save_distributed_files_to="test_2d")
    sim.run()


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
                               steps=0,
                               save_rate=10,
                               out_folder_name=f"dolfinx_patient{patient_number}_initial_condition",
                               sim_rationale=f"Computed initial condition for patient{patient_number}",
                               slurm_job_id=slurm_job_id,
                               save_distributed_files_to=distributed_data_folder)
        sim.run()


def check_tip_cell_activation_for_each_patient():
    """
    Unspecifc parameters sampling to check Tip Cell Activation
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, _ = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # Conditions to test
    # 1. We test a tdf_i range from 0.2 to 1 (i.e. initial tumor volume ranges from 20% to 100% of the size reported by
    #    clinicians).
    tdf_i_range = np.linspace(start=0.2, stop=1.0, num=5, endpoint=True)
    tdf_i_range = [tdf_i_range[3]]
    # 2. We test different value of af tumor secretion (V_pT_af). As the experimental range spans over different orders
    #    of magnitudes, we test 5 points over the log scale between the minimum and the maximal V_pT_af.
    V_pT_af_min = float(sim_parameters_df.loc["V_pT_af", "sim_range_min"])
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = np.logspace(np.log10(V_pT_af_min), np.log10(V_pT_af_max), num=5, endpoint=True)
    V_pT_af_range = [V_pT_af_max]
    # 3. We test different value of capillary uptake (V_uc_af). The values range from twice the natural VEGF degradation
    #    rate to 10 times the value provided by Travasso et al. (2011)
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_max = 10 * float(sim_parameters_df.loc["V_uc_af", "sim_value"])
    V_uc_af_range = np.logspace(np.log10(V_uc_af_min), np.log10(V_uc_af_max), num=5, endpoint=True)
    V_uc_af_range = [V_uc_af_max]
    # 4. We test different value of the minimal concentration required for tip cell activation (T_c). We test 4
    #    log points from 0 to 10 times the estimated value
    T_c_estimated = float(sim_parameters.get_value("T_c"))
    T_c_range = np.array([0., (T_c_estimated / 10), T_c_estimated, (10 * T_c_estimated)])

    for patient_number in [0]:
        logger.info(f"Starting tip cell activation test for patient{patient_number}")

        current_patient_parameter = patients_parameters[f"patient{patient_number}"]  # load patient params

        sim = RHTestTipCellActivation(spatial_dimension=spatial_dimension,
                                      standard_params=sim_parameters,
                                      patient_parameters=current_patient_parameter,
                                      out_folder_name=f"dolfinx_patient{patient_number}_tip-cell-activation",
                                      slurm_job_id=slurm_job_id)
        sim.run(tdf_i=tdf_i_range,
                V_pT_af=V_pT_af_range,
                V_uc_af=V_uc_af_range,
                T_c=T_c_range)


# def tip_cell_activation_patient1():
#     # preamble
#     sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, _ = preamble()
#
#     # get sim parameters dataframe
#     sim_parameters_df = sim_parameters.as_dataframe()
#
#     # Conditions to test
#     tdf_i_range = np.linspace(start=0.8, stop=1.0, num=21, endpoint=True)
#     V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
#     V_pT_af_range = [V_pT_af_max]
#     V_uc_af_range = [float(sim_parameters_df.loc["V_d_af", "sim_range_min"]),
#                      2.32]  # got from older tip cell activation
#
#     # Test tip cell activation
#     sim = RHTestTipCellActivation(spatial_dimension=spatial_dimension,
#                                   standard_params=sim_parameters,
#                                   patient_parameters=patients_parameters["patient1"],
#                                   out_folder_name=f"dolfinx_patient1_tip-cell-activation",
#                                   out_folder_mode="datetime",
#                                   slurm_job_id=slurm_job_id)
#     sim.run(tdf_i=tdf_i_range,
#             V_pT_af=V_pT_af_range,
#             V_uc_af=V_uc_af_range)


def vascular_sprouting_one_step(patient: str):
    """
    Experiment to run one simulation step for timing, or debugging.
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set parameters value leading to sprouting in all patients
    tdf_i = 1.
    sim_parameters.set_value("tdf_i", tdf_i)
    V_uc_af = 0.36
    sim_parameters.set_value("V_uc_af", V_uc_af)
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)

    # get number of step to reach the volume observed in patient1
    n_steps = 2

    # run simulation
    sim = RHTimeSimulation(spatial_dimension=spatial_dimension,
                           sim_parameters=sim_parameters,
                           patient_parameters=patients_parameters[patient],
                           steps=n_steps,
                           save_rate=50,
                           out_folder_name=f"dolfinx_{patient}_vascular_sprouting_one_step",
                           sim_rationale=f"Testing for 2 simulation step",
                           slurm_job_id=slurm_job_id,
                           save_distributed_files_to=distributed_data_folder)
    sim.run()


def patient1_time_adaptive_vascular_sprouting():
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set parameters value leading to sprouting in patient1
    tdf_i_range = [0.6, 0.8]  # tdf values for tip cell activation in patient1

    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_max = 10 * float(sim_parameters_df.loc["V_uc_af", "sim_value"])
    V_uc_af_range = np.logspace(np.log10(V_uc_af_min), np.log10(V_uc_af_max), num=5, endpoint=True)[0:2]

    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)

    # get number of step to reach the volume observed in patient1
    n_steps_range = [int(np.round((np.log(1 / tdf_i) / np.log(float(sim_parameters.get_value("tgr"))))))
                     for tdf_i in tdf_i_range]

    for tdf_i, V_uc_af, n_steps in zip(tdf_i_range, V_uc_af_range, n_steps_range):
        # set parameters value
        sim_parameters.set_value("tdf_i", tdf_i)
        sim_parameters.set_value("V_uc_af", V_uc_af)

        # run simulation
        sim = RHAdaptiveSimulation(spatial_dimension=spatial_dimension,
                                   sim_parameters=sim_parameters,
                                   patient_parameters=patients_parameters["patient1"],
                                   steps=n_steps,
                                   save_rate=50,
                                   out_folder_name=f"dolfinx_patient1_vascular-sprouting-adaptive_tdf_i={tdf_i:.2g}",
                                   sim_rationale=f"Testing the condition for the activation of the Tip Cells, I found "
                                     f"that when V_pT_af equals the maximum range value ({V_pT_af_max:.2e}) and "
                                     f"V_uc_af = {V_uc_af:.2e} [1 / tau], "
                                     f"the vascular sprouting starts at tdf_i = {tdf_i:.2g} (i.e., when the tumor is "
                                     f"very close to the final dimension. In this simulation, I simulated the vascular"
                                     f"sprouting until the end",
                                   slurm_job_id=slurm_job_id,
                                   save_distributed_files_to=distributed_data_folder)
        sim.run()


def find_min_rh_volume_for_tc_activation_patient_0_and_2():
    """
    Patient 0 and patient 2 have a bigger tumor volume than patient 1, so the experiment
    `check_tip_cell_activation_for_each_patient`
    """
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, _ = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # Conditions to test
    tdf_i_range = np.linspace(start=0.2/20, stop=0.2, num=20, endpoint=True)

    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = [V_pT_af_max]
    sim_parameters.set_value("V_pT_af", V_pT_af_max)

    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_max = 10 * float(sim_parameters_df.loc["V_uc_af", "sim_value"])
    V_uc_af_range = np.logspace(np.log10(V_uc_af_min), np.log10(V_uc_af_max), num=5, endpoint=True)
    V_uc_af_range = V_uc_af_range[0:-1]

    for patient in ["patient0", "patient2"]:
        sim_rationale = \
            (f"From the simulation on patient1, I found that only a few parameters values lead to tip cell activation"
             f"for such a small tumor.  V_pT_af = {V_pT_af_max:.2e} and V_uc_af = [{V_uc_af_range[0]:.2e}, "
             f"{V_uc_af_range[1]:.2e}]. With this simulation, I want to check which value of tdf_i leads to tip cells "
             f"activation for each condition in {patient}.")

        # Test tip cell activation
        sim = RHTestTipCellActivation(spatial_dimension=spatial_dimension,
                                      standard_params=sim_parameters,
                                      patient_parameters=patients_parameters[patient],
                                      out_folder_name=f"dolfinx_{patient}_tip-cell-activation",
                                      out_folder_mode="datetime",
                                      sim_rationale=sim_rationale,
                                      slurm_job_id=slurm_job_id)
        sim.run(tdf_i=tdf_i_range,
                V_pT_af=V_pT_af_range,
                V_uc_af=V_uc_af_range)


def tip_cell_activation_V_pT_and_V_uc_ranges_for_each_patient():
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, _ = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # get min tdf_i for each patient
    min_tdf_i = {
        "patient0": 0.04,
        "patient1": 0.6,
        "patient2": 0.05
    }

    # set V_uc_af range
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_range = np.linspace(start=V_uc_af_min,
                                stop=2.32,
                                num=21, endpoint=True)
    # set V_pT_af_range
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = np.linspace(start=2.,
                                stop=V_pT_af_max,
                                num=11, endpoint=True)

    for patient in ["patient1"]:
        sim_rationale = \
            (f"From Tip Cell Activation experiments, I determined the minimum volume required to induce vascular "
             f"sprouting for each patient. For this patient, the minimum volume corrspond to tdf_i = "
             f"{min_tdf_i[patient]}. In this simulation, I am checking which values of V_pT_af and V_uc_af lead"
             f" to tip cell activation.")

        # Test tip cell activation
        sim = RHTestTipCellActivation(spatial_dimension=spatial_dimension,
                                      standard_params=sim_parameters,
                                      patient_parameters=patients_parameters[patient],
                                      out_folder_name=f"dolfinx_{patient}_tip-cell-activation_Vranges",
                                      sim_rationale=sim_rationale,
                                      slurm_job_id=slurm_job_id)
        sim.run(tdf_i=[min_tdf_i[patient]],
                V_pT_af=V_pT_af_range,
                V_uc_af=V_uc_af_range)


def patient0_time_adaptive_vascular_sprouting():
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set patient
    patient = "patient0"

    # set parameters value leading to sprouting in patient0
    tdf_i = 0.05
    sim_parameters.set_value("tdf_i", tdf_i)
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)
    # get V_uc_af
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_max = 10 * float(sim_parameters_df.loc["V_uc_af", "sim_value"])
    V_uc_af_range = np.logspace(np.log10(V_uc_af_min), np.log10(V_uc_af_max), num=5, endpoint=True)
    V_uc_af = V_uc_af_range[1]
    sim_parameters.set_value("V_uc_af", V_uc_af)

    # get number of step to reach the volume observed in patient1
    n_steps = int(np.round((np.log(1 / tdf_i) / np.log(float(sim_parameters.get_value("tgr"))))))

    # run simulation
    sim = RHAdaptiveSimulation(spatial_dimension=spatial_dimension,
                               sim_parameters=sim_parameters,
                               patient_parameters=patients_parameters[patient],
                               steps=n_steps,
                               save_rate=50,
                               out_folder_name=f"dolfinx_{patient}_vascular-sprouting-adaptive_tdf_i={tdf_i:.2g}",
                               sim_rationale=f"Testing the condition for the activation of the Tip Cells, I found "
                                             f"that when V_pT_af equals the maximum range value ({V_pT_af_max:.2e}) and "
                                             f"V_uc_af = {V_uc_af:.2e} [1 / tau], "
                                             f"the vascular sprouting starts at tdf_i = {tdf_i:.2g}",
                               slurm_job_id=slurm_job_id,
                               save_distributed_files_to=distributed_data_folder)
    sim.run()


def test_convergence_1_step(patient: str):
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


def timing_mesh_refinement_patient0():
    # preamble
    sim_parameters, patients_parameters, slurm_job_id, spatial_dimension, distributed_data_folder = preamble()

    # set parameters value leading to sprouting in patient0
    tdf_i = 1.  # tdf values for tip cell activation in patient1
    sim_parameters.set_value("tdf_i", tdf_i)
    V_uc_af = 2.32  # V_uc_af values for tip cell activation in patient1
    sim_parameters.set_value("V_uc_af", V_uc_af)
    V_pT_af_max = float(sim_parameters.as_dataframe().loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)

    # set a given number of steps
    n_steps = 10
    save_rate = 5

    # selecting patient
    patient0_parameters = patients_parameters["patient0"]

    # set sim rationale
    sim_rationale = "Testing mesh refinement"

    # set traditional simulation
    simulation = RHTimeSimulation(spatial_dimension=spatial_dimension,
                                  sim_parameters=sim_parameters,
                                  patient_parameters=patient0_parameters,
                                  steps=n_steps,
                                  save_rate=save_rate,
                                  out_folder_name="dolfinx_patient0_mesh_refinement_CTRL",
                                  sim_rationale=sim_rationale + "| CTRL",
                                  slurm_job_id=slurm_job_id,
                                  save_distributed_files_to=distributed_data_folder)
    simulation.run()

    # set adaptive_simulation
    simulation = RHMeshAdaptiveSimulation(spatial_dimension=spatial_dimension,
                                          sim_parameters=sim_parameters,
                                          patient_parameters=patient0_parameters,
                                          steps=n_steps,
                                          save_rate=save_rate,
                                          out_folder_name="dolfinx_patient0_mesh_refinement_MR",
                                          sim_rationale=sim_rationale,
                                          slurm_job_id=slurm_job_id,
                                          save_distributed_files_to=distributed_data_folder)
    simulation.run()
