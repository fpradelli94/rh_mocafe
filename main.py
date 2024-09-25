import sys
import logging
import socket
from mpi4py import MPI
import src.experiments


# get process rank
comm_world = MPI.COMM_WORLD
rank = comm_world.rank

# set up logger
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"%(asctime)s (%(created)f) | "
                              f"host={socket.gethostname().ljust(8, ' ')}:p{str(rank).zfill(2)} | "
                              f"%(name)s:%(funcName)s:%(levelname)s: %(message)s")
ch.setFormatter(formatter)
logging.root.handlers = []  # removing default logger
logging.root.addHandler(ch)  # adding my handler
logging.root.setLevel(logging.DEBUG)  # setting root logger level


def main():
    # init patients list
    patients = ["patient0", "patient1", "patient2"]

    # Generate 100 steps simulations in Fig. 1
    for p in patients:
        src.experiments.vascular_sprouting_full_volume(p)

    # Generate tiles plot in Fig. 2
    src.experiments.find_min_tdf_i()

    # Generate 1 month simulations in Fig. 3
    for p in patients:
        src.experiments.time_adaptive_vascular_sprouting(p)

    # Generate 1 year simulation in Fig. 5
    src.experiments.time_adaptive_vascular_sprouting_full_volume_1yr("patient1")


if __name__ == "__main__":
    main()
