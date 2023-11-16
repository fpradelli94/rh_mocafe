import json
import shutil
import socket
import logging
import functools
import pandas as pd
from pathlib import Path
from mpi4py import MPI
from mocafe.fenut.parameters import Parameters

# get process rank
comm_world = MPI.COMM_WORLD
rank = comm_world.rank

# set up logger
logger = logging.getLogger(__name__)


def write_parameters(parameters: Parameters, parameters_file: Path):
    if rank == 0:
        parameters.as_dataframe().to_csv(parameters_file)


def read_parameters(parameters_file: Path or str):
    # cast
    if isinstance(parameters_file, str):
        parameters_file = Path(parameters_file)

    # read on rank 0
    if rank == 0:
        mesh_parameters_df = pd.read_csv(parameters_file)
        mesh_parameters_dict = mesh_parameters_df.to_dict('records')
    else:
        mesh_parameters_dict = None
    # bcast
    mesh_parameters_dict = comm_world.bcast(mesh_parameters_dict, 0)
    mesh_parameters_df = pd.DataFrame(mesh_parameters_dict)
    return Parameters(mesh_parameters_df)


def dump_json(obj, outfile: Path or str):
    if rank == 0:
        with open(outfile, "w") as output_file:
            json.dump(obj, output_file)


def load_json(infile: Path or str):
    if rank == 0:
        with open(infile, "r") as input_file:
            obj = json.load(input_file)
    else:
        obj = None
    obj = comm_world.bcast(obj, 0)
    return obj


def execute_once_per_node(sequentially: bool):
    """
    Execute operation only once per simulation node.

    :param sequentially: if True, the operation will be performed on each node sequentially. So only one node at the
    time will be executing the operation.
    """
    def decorator(operation):
        @functools.wraps(operation)
        def wrapper(*args, **kwargs):
            hostname = socket.gethostname()  # get hostname
            hostname_and_rank = (hostname, rank)  # get tuple with hostname
            procs_tuple = comm_world.gather(hostname_and_rank, 0)  # gather all tuples
            if rank == 0:
                selected_procs_tuple = []  # init list of selected tuples
                added_host = []  # init list of selected hosts
                for current_host, current_rank in procs_tuple:
                    if current_host not in added_host:  # if host not present in added hosts
                        added_host.append(current_host)  # mark that host has been added
                        selected_procs_tuple.append((current_host, current_rank))  # add to selected tuples
            else:
                selected_procs_tuple = None
            selected_procs_tuple = comm_world.bcast(selected_procs_tuple, 0)  # share selected tuples

            # If sequentially == True, each host will perform the operation while the others will be waiting
            # else, each host will perform the operation in parallel
            if sequentially:
                for host, h_rank in selected_procs_tuple:
                    if rank == h_rank:
                        operation(*args, **kwargs)
                    comm_world.Barrier()
            else:
                host_rank, = [h_rank for host, h_rank in selected_procs_tuple if host == hostname]
                if rank == host_rank:
                    operation(*args, **kwargs)

        return wrapper
    return decorator


@execute_once_per_node(sequentially=False)
def rmtree_if_exists_once_per_node(d: Path):
    """
    Remove the given directory. This operation is executed once per node.

    :param d: directory
    """
    if d.exists():
        logger.info(f"Removing {d}")
        shutil.rmtree(str(d))


@execute_once_per_node(sequentially=True)
def move_files_once_per_node(src_folder: Path, dst: Path):
    """
    Move files contained in src_folder to the folder dst.
    """
    logger.info(f"Starting iteration on {src_folder} (type:{type(src_folder)})")
    for f in src_folder.iterdir():
        logger.info(f"Moving {f} to {dst}")
        shutil.move(src=str(f), dst=str(dst))


@execute_once_per_node(sequentially=False)
def print_once_per_node(r: int, h: str, msg: str):
    print(f"p{r} on host {h} says: {msg}", flush=True)
