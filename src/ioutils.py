import json
import fenics
import pandas as pd
from pathlib import Path
from mocafe.fenut.parameters import Parameters

# get process rank
comm_world = fenics.MPI.comm_world
rank = comm_world.Get_rank()


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
