"""QuakeCoRE Scientific Software Stack.

## Overview
The QuakeCoRE scientific workflow is composed of a number of independent components

- `workflow` :: This module, which implements the Cybershake workflow stages: scripts that researchers can use to build their own automated research pipeline using scientifically validated, regularly tested, well-documented components built from the software team.
- `source_modelling` :: A collection of modules for modelling sources.
- `velocity_modelling` :: The next-generation velocity model.
- `qcore` :: Utilities and common functions.

## Using the Workflow

This repository provides the workflow's stages, not a workflow to run them in.
Each stage is a command installed with the package (and in the cybershake
container, see below), so a workflow is a sequence of these commands, run by
hand, from a script, or as tasks in a workflow manager such as cylc. For a
complete cylc workflow built from these stages, see the `flow.cylc` in
[realtime_simulation_workflow](https://github.com/ucgmsim/realtime_simulation_workflow).

To run a cylc workflow on NeSI, first run `cylc/workflow-setup` from this
repository to set up your account for cylc, then install and play your workflow

```
$ cylc install cybershake # OR the name of your workflow in ~/cylc-src
$ cylc play cybershake
```

You can monitor the progress of your workflow with `cylc log cybershake` and `squeue -u $USER` to see the slurm job queue for your user.

## The Stages

The workflow repository contains all the tools necessary to build a
workflow for ground motion model simulations. We build and maintain a
Cybershake workflow, but researchers may wish to build their own. For
this reason, the workflow is build out of composable parts we intend
for anyone to reuse to build their own workflow based on the
Cybershake workflow. Below you will find documentation for all the
workflow stages, their inputs, outputs and environments.

```mermaid
flowchart LR
    A[NSHM To Realisation] --> B[SRF Generation]
    A --> C[Domain Generation]
    B --> D[Stoch Generation]
    C --> E[Velocity Model Generation]
    C --> F[Station Selection]
    C --> G[Write Model Coordinates]
    B --> H[Create EMOD3D Parameters]
    E --> H
    F --> H
    G --> H
    H --> I[EMOD3D]
    D --> J[High Frequency Simulation]
    I --> M[Broadband Simulation]
    J --> M
    M --> N[Intensity Measure Calculation]
    I -->|Optionally| K[Merge Timeslices]
    K --> L[Create Simulation Video]
```


Many of the stages will run in a _container_. A container is a self-contained execution environment with all the system and Python libraries required to execute workflow stages. It is also isolated from the host system and may not be able to access certain directories. We maintain a [cybershake container](https://hub.docker.com/r/earthquakesuc/runner) that has a copy of the latest validated workflow, and environment. Use this container as much as possible in your own scripts to run your code.

To find the documentation for a given stage, find the module that runs this stage in the above flow diagram and then click the corresponding link

| Stage                         | Module                                                |
|:------------------------------|:------------------------------------------------------|
| NSHM to Realisation           | `workflow.scripts.nshm2022_to_realisation`            |
| SRF Generation                | `workflow.scripts.realisation_to_srf`                 |
| SRF Generation (successor)    | `workflow.scripts.rupture_input`                      |
| Domain Generation             | `workflow.scripts.generate_velocity_model_parameters` |
| Stoch Generation              | `workflow.scripts.generate_stoch`                     |
| Velocity Model Generation     | `workflow.scripts.generate_velocity_model`            |
| Station Selection             | `workflow.scripts.generate_station_coordinates`       |
| Write Model Coordinates       | `workflow.scripts.generate_model_coordinates`         |
| EMOD3D                        | See below.                                            |
| High Frequency Simulation     | `workflow.scripts.hf_sim`                             |
| Broadband Simulation          | `workflow.scripts.bb_sim`                             |
| Create Simulation Video       | `workflow.scripts.plot_ts`                            |
| Intensity Measure Calculation | `workflow.scripts.im_calc`                            |


## EMOD3D

Most stages are documented in their corresponding python module

### Description
Run a low frequency ground motion simulation using EMOD3D.

### Inputs
1. A parameter file in "key=value" format,
2. An SRF file,
3. A station file list (latitude, longitude, and x, y), see .

### Outputs
1. Ground acceleration timeslice files, one per core.
2. Seismograms, one per station.

### Environment
This stage must be run on a system with MPI installed. Typically, we run this stage in Maui on NeSI HPCs or Kisti. Due to high computational requirements, this stage usually cannot be run locally.

### Usage
On an HPC with slurm enabled `srun emod3d-mpi_v3.0.8 -args "par=$CYLC_WORKFLOW_SHARE_DIR/LF/e3d.par"` will run EMOD3D. Depending on the number of cores rerequested, this may invoke multiple proesses on different compute nodes (for Maui, this will occur when the number of cores exceeds 40). EMOD3D has support for checkpointing, so repeat invocations will continue from their previous checkpointed stage.

### For More Help
See Graves, 1996[^1] for a description of the mathematical and technical details of EMOD3D's implementation.

[^1]: Graves, Robert W. "Simulating seismic wave propagation in 3D elastic media using staggered-grid finite differences." Bulletin of the seismological society of America 86.4 (1996): 1091-1106.
"""
