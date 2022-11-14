"""
Plot number of tip cells for simulation.

Note: uses a json file called `incremental_tipcells.json` usually contained in the `sim_info` folder.

Basic usage:

    python3 tipcells_in_time.py --sim /home/user/sim_folder/sim_info

If you want to concatenate the plots of a resumed simulation:

    python3 tipcells_in_time.py --sim /home/user/sim_folder/sim_info --append /home/user/sim_folder/resumed_sim
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

SIM_FOLDER_COM = "--sim"
APPEND_FOLDER_COM = "--append"


def load_itc_dict(folder_name: str):
    # get tip cells incremental
    if folder_name.endswith("/"):
        tci_f = folder_name + "incremental_tipcells.json"
    else:
        tci_f = folder_name + "/incremental_tipcells.json"

    # load json
    with open(tci_f, "r") as infile:
        tci_dict = json.load(infile)

    return tci_dict


# load folder
if SIM_FOLDER_COM in sys.argv:
    sim_folder_name = sys.argv[sys.argv.index(SIM_FOLDER_COM) + 1]
else:
    raise RuntimeError("You must specify an input")

tci_dict = load_itc_dict(sim_folder_name)

# if resumed
if APPEND_FOLDER_COM in sys.argv:
    resume_folder_name = sys.argv[sys.argv.index(APPEND_FOLDER_COM) + 1]
    resumed_tci_dict = load_itc_dict(resume_folder_name)
    tci_dict.update(resumed_tci_dict)

# get n_steps
n_steps = len(tci_dict)

# init arrays
steps = np.zeros(n_steps)
n_tc = np.zeros(n_steps)

for i, key in enumerate(tci_dict.keys()):
    steps[i] = int(key.replace("step_", ""))
    n_tc[i] = len(tci_dict[key])

# plot
plt.plot(steps, n_tc, linewidth=3)
plt.xlabel("steps", fontsize=20)
plt.ylim([0, 300])
plt.xlim([0, 300])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("n TCs", fontsize=20)

# adjust
figure = plt.gcf()
figure.set_size_inches(6, 6)
plt.tight_layout()
plt.show()
