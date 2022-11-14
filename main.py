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


def main():

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
    sim_parameters.set_value("min_tumor_diameter",                     # change initial tumor diameter
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


if __name__ == "__main__":
    main()
