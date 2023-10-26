import sys
import json
import logging
import pandas as pd
import numpy as np
from mocafe.fenut.parameters import Parameters
from src.simulation import run_simulation, test_tip_cell_activation


# set up logger
logger = logging.getLogger(__name__)


def preamble():
    """
    Load general data for simulations
    """
    # load simulation parameters
    parameters_csv = "notebooks/out/g_parameters.csv"
    standard_parameters_df = pd.read_csv(parameters_csv, index_col="name")
    sim_parameters = Parameters(standard_parameters_df)

    # get slurm job id, if any
    if "-slurm_job_id" in sys.argv:
        try:
            slurm_job_id = int(sys.argv[sys.argv.index("-slurm_job_id") + 1])
        except IndexError:
            raise IndexError("slurm_job_id not specified. Found -slurm_job_id not followed by an actual id.")
    else:
        slurm_job_id = None

    # load patient parameters
    with open("input_patients_data/patients_parameters.json", "r") as infile:
        patients_parameters = json.load(infile)

    return sim_parameters, slurm_job_id, patients_parameters


def compute_initial_condition_for_each_patient():
    """
    Compute initial condition for each patient
    """
    # preamble
    sim_parameters, slurm_job_id, patients_parameters = preamble()

    # For each patient, compute initial condition
    for patient_number in [1, 2]:
        logger.info(f"Starting computation for patient{patient_number}")

        current_patient_parameter = patients_parameters[f"patient{patient_number}"]  # load patient params

        # run simulation
        run_simulation(spatial_dimension=3,
                       sim_parameters=sim_parameters,
                       patient_parameters=current_patient_parameter,
                       steps=0,
                       save_rate=10,
                       out_folder_name=f"patient{patient_number}_initial_condition",
                       sim_rationale=f"Computed initial condition for patient{patient_number}",
                       slurm_job_id=slurm_job_id,
                       recompute_mesh=True,
                       recompute_c0=True,
                       write_checkpoints=False,
                       save_distributed_files_to="/local/frapra/3drh")


def check_tip_cell_activation_for_each_patient():
    """

    """
    # preamble
    sim_parameters, slurm_job_id, patients_parameters = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # Conditions to test
    # 1. We test a tdf_i range from 0.2 to 1 (i.e. initial tumor volume ranges from 20% to 100% of the size reported by
    #    clinicians).
    tdf_i_range = np.linspace(start=0.2, stop=1.0, num=5, endpoint=True)
    # 2. We test different value of af tumor secretion (V_pT_af). As the experimental range spans over different orders
    #    of magnitudes, we test 5 points over the log scale between the minimum and the maximal V_pT_af.
    V_pT_af_min = float(sim_parameters_df.loc["V_pT_af", "sim_range_min"])
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = np.logspace(np.log10(V_pT_af_min), np.log10(V_pT_af_max), num=5, endpoint=True)
    # 3. We test different value of capillary uptake (V_uc_af). The values range from twice the natural VEGF degradation
    #    rate to 10 times the value provided by Travasso et al. (2011)
    V_uc_af_min = float(sim_parameters_df.loc["V_d_af", "sim_range_min"])
    V_uc_af_max = 10 * float(sim_parameters_df.loc["V_uc_af", "sim_value"])
    V_uc_af_range = np.logspace(np.log10(V_uc_af_min), np.log10(V_uc_af_max), num=5, endpoint=True)
    # 4. We test different value of the minimal concentration required for tip cell activation (T_c). We test 4
    #    log points from 0 to 10 times the estimated value
    T_c_estimated = float(sim_parameters.get_value("T_c"))
    T_c_range = np.array([0., (T_c_estimated / 10), T_c_estimated, (10 * T_c_estimated)])

    for patient_number in range(3):
        logger.info(f"Starting tip cell activation test for patient{patient_number}")

        current_patient_parameter = patients_parameters[f"patient{patient_number}"]  # load patient params

        test_tip_cell_activation(spatial_dimension=3,
                                 standard_sim_parameters=sim_parameters,
                                 patient_parameters=current_patient_parameter,
                                 out_folder_name=f"patient{patient_number}_tip-cell-activation",
                                 slurm_job_id=slurm_job_id,
                                 recompute_mesh=True,
                                 recompute_c0=True,
                                 tdf_i=tdf_i_range,
                                 V_pT_af=V_pT_af_range,
                                 V_uc_af=V_uc_af_range,
                                 T_c=T_c_range)


def tip_cell_activation_patient1():
    # preamble
    sim_parameters, slurm_job_id, patients_parameters = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # Conditions to test
    tdf_i_range = np.linspace(start=0.8, stop=1.0, num=21, endpoint=True)
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = [V_pT_af_max]
    V_uc_af_range = [float(sim_parameters_df.loc["V_d_af", "sim_range_min"]),
                     2.32]  # got from older tip cell activation

    # Test tip cell activation
    test_tip_cell_activation(spatial_dimension=3,
                             standard_sim_parameters=sim_parameters,
                             patient_parameters=patients_parameters["patient1"],
                             out_folder_name=f"patient1_tip-cell-activation",
                             out_folder_mode="datetime",
                             slurm_job_id=slurm_job_id,
                             recompute_mesh=True,
                             recompute_c0=True,
                             tdf_i=tdf_i_range,
                             V_pT_af=V_pT_af_range,
                             V_uc_af=V_uc_af_range)


def patient1_vascular_sprouting():
    # preamble
    sim_parameters, slurm_job_id, patients_parameters = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # set parameters value leading to sprouting in patient1
    tdf_i_range = [0.94, 0.99]  # tdf values for tip cell activation in patient1
    V_uc_af_range = [2.32, 0.36]  # V_uc_af values for tip cell activation in patient1
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    sim_parameters.set_value("V_pT_af", V_pT_af_max)

    # get number of step to reach the volume observed in patient1
    n_steps_range = [int(np.round((np.log(1 / tdf_i) / np.log(float(sim_parameters.get_value("tgr"))))))
                     for tdf_i in tdf_i_range]

    for tdf_i, V_uc_af, n_steps in zip(tdf_i_range, V_uc_af_range, n_steps_range):
        # set parameters value
        sim_parameters.set_value("tdf_i", 0.94)
        sim_parameters.set_value("V_uc_af", 2.32)

        # run simulation
        run_simulation(spatial_dimension=3,
                       sim_parameters=sim_parameters,
                       patient_parameters=patients_parameters["patient1"],
                       steps=n_steps,
                       save_rate=50,
                       out_folder_name="patient1_vascular-sprouting",
                       out_folder_mode="datetime",
                       sim_rationale=f"Testing the condition for the activation of the Tip Cells, I found "
                                     f"that when V_pT_af equals the maximum range value ({V_pT_af_max:.2e}) and "
                                     f"V_uc_af = {V_uc_af:.2e} [1 / tau], "
                                     f"the vascular sprouting starts at tdf_i = {tdf_i:.2g} (i.e., when the tumor is "
                                     f"very close to the final dimension. In this simulation, I evaluated the vascular "
                                     f"sprouting for {n_steps} steps, which is the time required to the tumor to "
                                     f"reach the final volume observed in patients.",
                       slurm_job_id=slurm_job_id,
                       recompute_mesh=True,
                       recompute_c0=True,
                       write_checkpoints=False,
                       save_distributed_files_to="/local/frapra/3drh")


def tip_cell_activation_patient0_and_2():
    # preamble
    sim_parameters, slurm_job_id, patients_parameters = preamble()

    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()

    # Conditions to test
    tdf_i_range = np.linspace(start=0.2/20, stop=0.2, num=20, endpoint=True)
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = [V_pT_af_max]
    V_uc_af_range = [float(sim_parameters_df.loc["V_d_af", "sim_range_min"]),
                     2.32]  # got from older tip cell activation

    for patient in ["patient0", "patient2"]:
        sim_rationale = \
            (f"From the simulation on patient1, I found that only a few parameters values lead to tip cell activation"
             f"for such a small tumor.  V_pT_af = {V_pT_af_max:.2e} and V_uc_af = [{V_uc_af_range[0]:.2e}, "
             f"{V_uc_af_range[1]:.2e}]. With this simulation, I want to check which value of tdf_i leads to tip cells "
             f"activation for each condition in {patient}.")

        # Test tip cell activation
        test_tip_cell_activation(spatial_dimension=3,
                                 standard_sim_parameters=sim_parameters,
                                 patient_parameters=patients_parameters[patient],
                                 out_folder_name=f"{patient}_tip-cell-activation",
                                 out_folder_mode="datetime",
                                 sim_rationale=sim_rationale,
                                 slurm_job_id=slurm_job_id,
                                 recompute_mesh=True,
                                 recompute_c0=True,
                                 tdf_i=tdf_i_range,
                                 V_pT_af=V_pT_af_range,
                                 V_uc_af=V_uc_af_range)


def tip_cell_activation_V_pT_and_V_uc_ranges_for_each_patient():
    # preamble
    sim_parameters, slurm_job_id, patients_parameters = preamble()
    # get sim parameters dataframe
    sim_parameters_df = sim_parameters.as_dataframe()
    # get min tdf_i for each patient
    min_tdf_i = {
        "patient0": 0.05,
        "patient1": 0.94,
        "patient2": 0.05
    }

    # set V_uc_af range
    V_uc_af_range = np.linspace(start=float(sim_parameters.get_value("V_d_af")),
                                stop=15.0,
                                num=21, endpoint=True)
    # set V_pT_af_range
    V_pT_af_max = float(sim_parameters_df.loc["V_pT_af", "sim_range_max"])
    V_pT_af_range = np.linspace(start=2.,
                                stop=V_pT_af_max,
                                num=11, endpoint=True)

    for patient in ["patient0", "patient1", "patient2"]:
        sim_rationale = \
            (f"From Tip Cell Activation experiments, I determined the minimum volume required to induce vascular "
             f"sprouting for each patient. For this patient, the minimum volume corrspond to tdf_i = "
             f"{min_tdf_i[patient]}. In this simulation, I am checking which values of V_pT_af and V_uc_af lead"
             f" to tip cell activation.")

        # Test tip cell activation
        test_tip_cell_activation(spatial_dimension=3,
                                 standard_sim_parameters=sim_parameters,
                                 patient_parameters=patients_parameters[patient],
                                 out_folder_name=f"{patient}_tip-cell-activation",
                                 out_folder_mode="datetime",
                                 sim_rationale=sim_rationale,
                                 slurm_job_id=slurm_job_id,
                                 recompute_mesh=True,
                                 recompute_c0=True,
                                 tdf_i=[min_tdf_i[patient]],
                                 V_pT_af=V_pT_af_range,
                                 V_uc_af=V_uc_af_range)
