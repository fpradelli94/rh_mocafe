# Simulating Retinal Hemangioblastoma with DOLFINx and Mocafe

<p align="center">
  <img src=".thumbs/MovieS1.gif" />
</p>

**************************************************************************

The materials contained in this folder allow the reproduction of the simulations of RH development
and angiogenesis with DOLFINx and Mocafe.

## Quick instructions
The full code to reproduce our results and to test our implementation is contained in  `main.py`. 
To ensure full reproducibility, we recommend to use [Singularity](https://github.com/sylabs/singularity):

```shell
# pull container
singularity singularity push mocafex.sif library://fpradelli94/mocafex/mocafex:dev
# execute main in the container
singularity exec mocafex.sif python3 main.py
```

We also recommend to run the script in parallel to save time. To do so, you can exploit `mpirun` typing the following:
```shell
# run over 4 cores
singularity exec mocafex.sif mpirun -n 4 python3 main.py
# run over x cores, according to your system
singularity exec mocafex.sif mpirun -n x python3 main.py
```

If you don’t have Singularity installed, just follow the instructions provided at the official documentation page for 
[SingularityCE](https://sylabs.io/docs/).

By default, the simulation output will be stored in `saved_sim`. For a detailed explanation of the generated files, 
see section "Simulation output".

> :warning: 3D simulations require a consistent computational effort, so consider using an 
> appropriate machine to reproduce that. To test the code, you can run 2D simulations.

## Slurm integration
Our code is compatible with [slurm](https://slurm.schedmd.com/documentation.html). If you use slurm to run
our code, it is recommended to use a `sbatch` script like the following:

```shell
#!/bin/bash
#SBATCH --job-name rh_mocafe

srun singularity exec sif/mocafex.sif python3 main.py -slurm_job_id $SLURM_JOB_ID
ex=$?
```

You can change the script according to your needs.

## Simulation output
The result of a simulation is a folder with the following structure:
```shell
.
├── 0_reproduce
├── sim_info
├── af.h5
├── af.xdmf
├── c.h5
├── c.xdmf
├── grad_af.h5
├── grad_af.xdmf
├── phi.h5
├── phi.xdmf
├── resume
├── sim_info
├── tipcells.h5
└── tipcells.xdmf
```
Where the `*.xdmf` and `*.h5` files are the actual simulation outputs. More precisely:

| Files                            | Content                                       |
|----------------------------------|-----------------------------------------------|
| `af.xdmf`, `af.h5`               | Angiogenic Factors (AFs) distribution in time |
| `c.xdmf`, `c.h5`                 | Capillaries field in time                     |
| `grad_af.xdmf`, `grad_af.h5`     | Gradient of the AFs distribution in time      |
| `phi.xdmf`, `phi.h5`             | Retinal Hemangioblastoma phase field in time  |
| `tipcells.xdmf`, `tipcells.xdmf` | Tip Cells phase field in time                 |

You can load these file in ParaView to visualize them. [^1]

There are also three additional folders:
- `0_reproduce` contains a copy of the script used to generate the simulation. 
Useful to check the exact code used to generate the results.
- `sim_info` contains:
  - `incremental_tipcells.json`: data for every tip cell at each time step.
  - `mesh_parameters.csv`: info regarding the mesh used for the simulation.
  - `sim_info.html`: info regarding the simulation [^2]
  - `sim_parameters.csv`: the parameters used in the simulation.
  - `patient_parameters.json`: the patient-specific parameters for the given simulation

## GitHub and Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7330072.svg)](https://doi.org/10.5281/zenodo.7330072)

This repository is available both on GitHub or on Zenodo. The GitHub version is meant to clone and reproduce the 
results. On the Zenodo repository, you can access to the generated outputs of the code, and the simulation 
results that we used in our manuscript. See the section "Repository folders" (below) for additional info.

## Structure of this repository
This repository contains several folders:

```bash
.
├── input_patients_data
├── notebooks
├── saved_sim
├── saved_sim_archive # only on Zenodo
├── src
└── visualization
```

Let's see them one by one: 

### :file_folder: `input_patients_data`
Contains patient-specific images, derived from the Optical Coherence Tomography Angiography. They are 
used in the program to generate the initial condition for the simulation. 

It also contain the patient parameters (e.g. RH dimension) in the `patients_parameters.json` file.

### :file_folder: `notebooks`
Contains the Jupyter Notebooks `parameters.ipynb`, which generates the `.csv` table of the simulation's parameters.

The output of the notebook is in `notebooks/out`. You can generate it running the notebooks. 

### :file_folder: `saved_sim`
This folder will be generated at the first simulation and will contain the simulation files, if you run some 
simulations.

### :file_folder: `saved_sim_archive`
Contains the results of the simulations shown in the manuscript in Figure 2 and Figure 4. Available only on Zenodo. 

The recommended way to visualize the simulation result is [ParaView](https://www.paraview.org/). [^1]

Also, you can consider using the visualization script provided in `visualization` subfolder.

### :file_folder: `src`
Contains the source code used in the `main.py` script. 

### :file_folder: `visualization`
Contains scripts and data related to the results' visualization. There are two sub-folders: `paraview` and `python`

1. `paraview`

Contains a script, `section.py`, to visualize a simulation in 3D with [ParaView](https://www.paraview.org/). 
It is the same used to generate the thumbnail of this `README`. It only works with 3D simulations and with ParaView >5.9.0.

To use the script, open ParaView and, on the View tab, select "Python Shell". 
A new window will appear with a button "Run Script". Press the button and select the file `section.py`.

If the script runs correctly, it will ask for two inputs: the absolute paths of the simulation's `c.xdmf` file 
and of the `phi.xdmf` file. Copy and paste them in the box that will appear and let ParaView handle the rest.

It might take some minutes to display the result.

2. `python`

Contains the scripts used to produce some plots reported in the manuscript. More precisely, there are:

- `activation_tiles.py`, which generates a tiles plot of the parameters leading to angiognesis (Fig. 3 of the manuscript)
using the table `sim_index.csv`.
- `tipcells_in_time.py`, which generates a plot of the number of tip cells at any simulation step (like in Fig. 4 of the 
manuscript).

## Meta
Franco Pradelli (franco.pradelli94@gmail.com), Giovanni Minervini, and Silvio Tosatto  

## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. 
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Footnotes

[^1]: Don't know how? See examples in the [Mocafe Demo gallery](https://biocomputingup.github.io/mocafe/build/html/demo_doc/angiogenesis_3d.html#visualize-the-result-with-paraview)

[^2]: For an example, see [Mocafe Demo gallery](https://biocomputingup.github.io/mocafe/build/html/demo_doc/multiple_pc_simulations.html#sphx-glr-demo-doc-multiple-pc-simulations-py)