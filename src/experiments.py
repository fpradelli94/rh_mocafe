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

