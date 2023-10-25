"""
Plot number of tip cells for simulation.

Note: uses a json file called `incremental_tipcells.json` usually contained in the `sim_info` folder.

Basic usage:

    python3 tipcells_in_time.py --sim /home/user/sim_folder/sim_info

If you want to concatenate the plots of a resumed simulation:

    python3 tipcells_in_time.py --sim /home/user/sim_folder/sim_info --append /home/user/sim_folder/resumed_sim
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

SIM_FOLDER_COM = "--sim"
APPEND_FOLDER_COM = "--append"


def parse_cli_args():
    parser = argparse.ArgumentParser(
        prog="tipcells-in-time",
        description="Plot Tip Cells in time given a incremental_tipcells,json"
    )

    parser.add_argument("jsonfile",
                        help="File containing the number of Tip Cells in time for the simulation (usually called "
                             "incremental_tipcells.json).")
    parser.add_argument("output_png",
                        help="PNG to store plot")
    parser.add_argument("--append", action='extend',
                        help="Append other json files to the main one.")

    return parser.parse_args()


def main():
    args = parse_cli_args()

    # load json file
    with open(args.jsonfile, "r") as infile:
        tip_cells_in_time = json.load(infile)

    # append possible additional file
    if args.append is not None:
        steps = tip_cells_in_time.keys()
        for json_file in args.append:
            # open json file
            with open(json_file, "r") as infile:
                tip_cells_to_append = json.load(infile)
            # check if there are common elements
            keys_intersection = set(tip_cells_to_append.keys()) & set(steps)
            assert len(keys_intersection) < 2, f"Found common keys = {keys_intersection}. Files are incompatible"
            # if not incompatible, append
            tip_cells_in_time.update(tip_cells_to_append)

    # plot results
    # get n_steps
    n_steps = len(tip_cells_in_time)

    # init arrays
    steps = np.zeros(n_steps)
    n_tc = np.zeros(n_steps)

    for i, key in enumerate(tip_cells_in_time.keys()):
        steps[i] = int(key.replace("step_", ""))
        n_tc[i] = len(tip_cells_in_time[key])

    # plot
    plt.plot(steps, n_tc, linewidth=3)
    plt.xlabel("steps", fontsize=20)
    plt.ylabel("n TCs", fontsize=20)

    # adjust
    figure = plt.gcf()
    figure.set_size_inches(6, 6)
    plt.tight_layout()
    plt.savefig(args.output_png)


if __name__ == "__main__":
    main()





# def load_itc_dict(folder_name: str):
#     # get tip cells incremental
#     if folder_name.endswith("/"):
#         tci_f = folder_name + "incremental_tipcells.json"
#     else:
#         tci_f = folder_name + "/incremental_tipcells.json"
#
#     # load json
#     with open(tci_f, "r") as infile:
#         tci_dict = json.load(infile)
#
#     return tci_dict
#
#
# # load folder
# if SIM_FOLDER_COM in sys.argv:
#     sim_folder_name = sys.argv[sys.argv.index(SIM_FOLDER_COM) + 1]
# else:
#     raise RuntimeError("You must specify an input")
#
# tci_dict = load_itc_dict(sim_folder_name)
#
# # if resumed
# if APPEND_FOLDER_COM in sys.argv:
#     resume_folder_name = sys.argv[sys.argv.index(APPEND_FOLDER_COM) + 1]
#     resumed_tci_dict = load_itc_dict(resume_folder_name)
#     tci_dict.update(resumed_tci_dict)


