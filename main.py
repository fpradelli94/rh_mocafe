import sys
import logging
import socket
import random
from mpi4py import MPI
import src.experiments


# get process rank
comm_world = MPI.COMM_WORLD
rank = comm_world.rank

# init random seed
random.seed(22)

# set fenics log level
# fenics.set_log_level(fenics.LogLevel.ERROR)

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
    src.experiments.timing_adaptive_vascular_sprouting_for_patient1()


if __name__ == "__main__":
    main()
