import fenics
import subprocess
import pandas as pd
import sys
from src.common import run_simulation, resume_simulation, test_tip_cell_activation
from mocafe.fenut.parameters import Parameters

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()

# set fenics log level
fenics.set_log_level(fenics.LogLevel.ERROR)


def tutorial_2d():
    """
    In this function, we provide some usage examples of the code to simulate our patient-specific RH.

    All the examples here are in 2D, to make the computational effort affordable for any PC.
    """
    # -- Generate simulation parameters from notebook
    parameters_nb = "notebooks/parameters.ipynb"
    if rank == 0:
        subprocess.run(["jupyter", "nbconvert", "--to", "html", "--execute", parameters_nb])

    # load parameters
    parameters_csv = "notebooks/out/g_parameters.csv"
    standard_parameters_df = pd.read_csv(parameters_csv, index_col="name")

    # wait for all processes
    comm_world.Barrier()

    # -- Get Slurm job id, if any
    if "-slurm_job_id" in sys.argv:
        try:
            slurm_job_id = int(sys.argv[sys.argv.index("-slurm_job_id") + 1])
        except IndexError:
            raise IndexError("slurm_job_id not specified. Found -slurm_job_id not followed by an actual id.")
    else:
        slurm_job_id = None

    """--- Use examples ---"""
    spatial_dimension = 2  # warning: 3D simulation require a considerable computational effort, see README.md

    # Ex. 1: run a simulation
    run_simulation(spatial_dimension=spatial_dimension,                # 2 for 2D or 3 for 3D
                   sim_parameters=Parameters(standard_parameters_df),  # simulation parameters
                   steps=100,                                          # n steps
                   save_rate=100,                                      # save every 100 steps
                   out_folder_name="sim1",                             # store result in saved_sim/sim1
                   sim_rationale=f"I want to run this simulation "     # motivation for the simulation (if you want)
                                 f"for reason x",
                   slurm_job_id=slurm_job_id)                          # slurm job id, if any (can be None)

    # Ex. 2: change some parameters value
    sim_parameters = Parameters(standard_parameters_df)
    sim_parameters.set_value("min_tumor_diameter",                     # change initial tumor diameter (d_0)
                             sim_parameters.get_value("max_tumor_diameter"))
    run_simulation(spatial_dimension=spatial_dimension,
                   sim_parameters=sim_parameters,
                   steps=100,
                   save_rate=100,
                   out_folder_name="sim2",
                   slurm_job_id=slurm_job_id)

    # Ex. 3: resume a simulation
    additional_steps = 100
    resume_simulation(resume_from="saved_sim/sim2",                    # resume previous sim
                      steps=additional_steps,                          # simulate 100 steps more
                      save_rate=100,
                      out_folder_name="sim2_resumed",                  # store result in saved_sim/sim2_resumed
                      sim_rationale=f"I want to simulate for "         # add a comment
                                    f"{additional_steps} steps more",
                      slurm_job_id=slurm_job_id)

    # Ex 4.: Test if TC activation occurs
    test_tip_cell_activation(spatial_dimension,
                             df_standard_params=standard_parameters_df,
                             out_folder_name="test_tc_activation")


def reproduce_results():
    """
    This function contains the code to reproduce the simulations discussed in the manuscript (Fig 2, Fig 3, and Fig 4).

    All these simulation are in 3D and have been executed on a powerful cluster (192 GB total RAM, 6 nodes, 96 cores).
    """
    # ---------------------------------------------------------------------------------------------------------------- #
    # PREAMBLE
    # Below, we generate the simulation parameters form the notebook, and we get the slurm job ID, if any.
    # ---------------------------------------------------------------------------------------------------------------- #

    # execute the notebook
    parameters_nb = "notebooks/parameters.ipynb"
    if rank == 0:
        subprocess.run(["jupyter", "nbconvert", "--to", "html", "--execute", parameters_nb])

    # load parameters as generated from the notebook
    parameters_csv = "notebooks/out/g_parameters.csv"
    standard_parameters_df = pd.read_csv(parameters_csv, index_col="name")

    # get slurm job id, if any
    if "-slurm_job_id" in sys.argv:
        try:
            slurm_job_id = int(sys.argv[sys.argv.index("-slurm_job_id") + 1])
        except IndexError:
            raise IndexError("slurm_job_id not specified. Found -slurm_job_id not followed by an actual id.")
    else:
        slurm_job_id = None

    # wait for all processes
    comm_world.Barrier()

    # ---------------------------------------------------------------------------------------------------------------- #
    # CODE FOR SIMULATION IN FIG 2
    # Below, you can find the code to reproduce simulation in Fig.2. In the first part we configure the simulation,
    # setting the correct values for the parameters. Then, we run the simulation using the appropriate function.
    # ---------------------------------------------------------------------------------------------------------------- #

    # load parameters
    sim_parameters = Parameters(standard_parameters_df)

    # change initial tumor diameter
    sim_parameters.set_value("min_tumor_diameter", 0.5)  # correspond to d_0 (unit of measure is [sau])

    # run simulation
    run_simulation(spatial_dimension=3,
                   sim_parameters=sim_parameters,
                   steps=200,
                   save_rate=10,
                   out_folder_name="Fig2",
                   sim_rationale=f"Reproducing simulation in Fig 2",
                   slurm_job_id=slurm_job_id)

    # ---------------------------------------------------------------------------------------------------------------- #
    # CODE FOR SIMULATIONS IN FIG 3
    # Below, you can find the code to reproduce simulations in Fig.3, where we tested tip cell activation with 500
    # one-step simulations. To do so, there is a single function that handles everything. For additional details,
    # you can check the implementation.
    # The output is a csv file saved in saved_sim/Fig3/tipcell_activation.csv. From that, we generated Fig 3 using the
    # script in ./rh_mocafe/visualization/python/tiles_plot.py
    # ---------------------------------------------------------------------------------------------------------------- #

    # test tip cell activation
    test_tip_cell_activation(spatial_dimension=3,
                             df_standard_params=standard_parameters_df,
                             out_folder_name="Fig3")

    # ---------------------------------------------------------------------------------------------------------------- #
    # CODE FOR SIMULATIONS IN FIG 4a, 4b, 4c
    # Below, you can find the code to reproduce simulations in Fig.4a, 4b, and 4c. In the first part we configure the
    # simulation, setting the correct values for the parameters. Notice that we don't set V_pT, since it is already set
    # in the Jupyter Notebook.
    # ---------------------------------------------------------------------------------------------------------------- #

    # load sim parameters
    sim_parameters = Parameters(standard_parameters_df)

    # set steps number
    steps = 300

    # compute initial tumor diameter, knowing that the final correspond to 600 um
    td_f = sim_parameters.get_value("max_tumor_diameter")  # correspond to 600 um
    tgr = sim_parameters.get_value("tgr")
    td_i = td_f / (tgr ** (steps / 3))
    sim_parameters.set_value("min_tumor_diameter", td_i)  # correspond to d_0

    # test for different uptake values
    V_uc_af_range = [21.1608695473508, 3.89679868445612, 0.7176]  # values given in [1 / tau]

    # simulate
    for sim_folder, V_uc_af_value in zip(["Fig4a", "Fig4b", "Fig4c"], V_uc_af_range):
        # update V_uc_value
        sim_parameters.set_value("V_uc_af", V_uc_af_value)

        run_simulation(spatial_dimension=3,
                       sim_parameters=sim_parameters,
                       steps=steps,
                       save_rate=10,
                       out_folder_name=sim_folder,
                       sim_rationale=f"Reproducing simulation in {sim_folder}",
                       slurm_job_id=slurm_job_id)

    # ---------------------------------------------------------------------------------------------------------------- #
    # CODE FOR SIMULATIONS IN FIG 4d
    # Below, you can find the code to reproduce simulations in Fig.4d. In the first part we configure the
    # simulation, setting the correct values for the parameters. Notice that we don't set V_pT, since it is already set
    # in the Jupyter Notebook.
    # ---------------------------------------------------------------------------------------------------------------- #

    # load sim parameters
    sim_parameters = Parameters(standard_parameters_df)

    # set steps number
    steps = 300

    # compute initial tumor diameter, knowing that the final correspond to 600 um
    td_f = sim_parameters.get_value("max_tumor_diameter")  # correspond to 600 um
    tgr = sim_parameters.get_value("tgr")
    td_i = td_f / (tgr ** (steps / 3))
    sim_parameters.set_value("min_tumor_diameter", td_i)  # correspond to d_0

    sim_parameters.set_value("V_uc_af", 21.1608695473508)  # values given in [1 / tau]

    sim_parameters.set_value("V_pT_af", 2.00243118326342)  # values given in [afau / tau]

    run_simulation(spatial_dimension=3,
                   sim_parameters=sim_parameters,
                   steps=steps,
                   save_rate=10,
                   out_folder_name="Fig4d",
                   sim_rationale=f"Reproducing simulation in Fig4d",
                   slurm_job_id=slurm_job_id)


def main():
    tutorial_2d()


if __name__ == "__main__":
    main()
