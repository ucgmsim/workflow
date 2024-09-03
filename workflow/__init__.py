"""
# QuakeCoRE Scientific Software Stack

## Overview
The QuakeCoRE scientific workflow is composed of a number of independent components

- `workflow` :: This module, which implements the Cybershake workflow. It defines a cylc worflow and scripts that researchers can use to build their own automated research pipeline using scientifically validated, regularly tested, well-documented components built from the software team.
- `source_modelling` :: A collection of modules for modelling sources.
- `velocity_modelling` :: The next-generation velocity model.
- `qcore` :: Utilities and common functions.

## Using the Workflow

To deploy the workflow on NeSI, first clone the repository and then install the cylc workflow

```
$ git clone https://github.com/ucgmsim/workflow
$ # add inputs into workflow/cylc/inputs
$ cp -r workflow/cylc ~/cylc-src/cybershake # OR a suitable name for your custom workflow
$ cylc install cybershake
$ cylc play cybershake
```

And that's it! You can monitor the progress of your workflow with `cylc log cybershake` and `squeue -u $USER` to see the slurm job queue for your user.

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
    I -->|Optionally| K[Merge Timeslices]
    K --> L[Create Simulation Video]
```


Many of the stages will run in a _container_. A container is a self-contained execution environment with all the system and Python libraries required to execute workflow stages. It is also isolated from the host system and may not be able to access certain directories. We maintain a [cybershake container](https://hub.docker.com/r/earthquakesuc/runner) that has a copy of the latest validated workflow, and environment. Use this container as much as possible in your own scripts to run your code.

### NSHM To Realisation

#### Description
Construct a realisation from a rupture in the [NSHM 2022](https://nshm.gns.cri.nz/RuptureMap).
#### Inputs

1. A copy of the [NSHM 2022 database](https://www.dropbox.com/scl/fi/50kww45wpsnmtf3pn2okz/nshmdb.db?rlkey=4mjuomuevl1963fjwfximgldm&st=50ax73gl&dl=0).
2. A rupture id to simulate. You can find a rupture id from the [rupture explorer](https://nshm.gns.cri.nz/RuptureMap). Alternatively, you can use the visualisation tools to find one.
3. The version of the [scientific defaults](https://github.com/ucgmsim/workflow/blob/pegasus/workflow/default_parameters/README.md#L1) to use. If you don't know what version to use, choose the latest version. Versions are specified as `YY.M.D.R`, where `R` is the resolution of the simulation (1 = 100m). For example `24.2.2.1`. The special `develop` version is for testing workflow iterations and not to be used for accurate scientific simulation.

#### Outputs
A realisation file containing:

1. The definition of all the faults in the the rupture,
2. A rupture propagation plan (i.e. how the rupture jumps between faults, and where),
3. The estimated rupture magnitude and apportionment to the involved faults.
4. The definition of the rakes.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `nshm2022-to-realisation` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

#### Usage
`nshm2022-to-realisation [OPTIONS] NSHM_DB_FILE RUPTURE_ID REALISATION_FFP DEFAULTS_VERSION`

#### For More Help
See the output of `nshm2022-to-realisation --help` or [nshm2022\_to\_realisation.py](https://github.com/ucgmsim/workflow/blob/pegasus/workflow/scripts/nshm2022_to_realisation.py).

### SRF Generation

#### Description
Produce an SRF from a realisation.

#### Inputs
A realisation file containing:
1. A source configuration,
2. A rupture propagation configuration,
3. A metadata configuration.

Typically, this information comes from a stage like [NSHM To Realisation](#nshm-to-realisation).

#### Outputs

1. An [SRF](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-SRFFormat) file containing the source slip definition for the realisation,
2. An updated realisation file containing the parameters used for SRF generation copied from the scientific defaults.

#### Environment

Can be run in the cybershake container. Can also be run from your own computer using the `realisation-to-srf` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the work directory (with the `--work-directory` flag), a 1D velocity model (`--velocity-model-ffp`), and the path to a genslip binary (`--genslip-path`).

#### Usage
`realisation-to-srf [OPTIONS] REALISATION_FFP OUTPUT_SRF_FILEPATH`

#### For More Help
See the output of `realisation-to-srf --help` or `workflow.scripts.realisation_to_srf`

#### Visualisation
You can visualise the output of this stage using the SRF plotting tools in the [source modelling](https://github.com/ucgmsim/source_modelling/blob/plots/wiki/Plotting-Tools.md) repository. Many of the tools take realisations as optional arguments to enhance the plot output.

### Stoch Generation

#### Description
Generate Stoch file for HF simulation. This file is just a down-sampled version of the SRF.

#### Inputs
A realisation file containing a metadata configuration, and a generated SRF file.

#### Outputs
A [Stoch](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-Stochformat) file containing a down-sampled version of the SRF.

#### Usage
`generate-stoch [OPTIONS] REALISATION_FFP SRF_FFP STOCH_FFP`

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `generate-stoch` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the `srf2stoch` path (`--srf2stoch-path`).

#### For More Help
See the output of `generate-stoch --help` or `workflow.scripts.generate_stoch`.

### Domain Generation
#### Description
Find a suitable simulation domain, estimating a rupture radius that captures significant ground motion, and the time the simulation should run for to capture this ground motion.

#### Inputs
A realisation file containing a metadata configuration, source definitions and rupture propagation information.

#### Outputs
A realisation file containing velocity model and domain extent parameters.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `generate-velocity-model-parameters` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

#### Usage
`generate-velocity-model-parameters [OPTIONS] REALISATION_FFP`

#### For More Help
See the output of `generate-velocity-model-parameters --help` or `workflow.scripts.generate_velocity_model_parameters`.

### Velocity Model Generation

#### Description
Generate a velocity model for a domain.

#### Inputs
A realisation file containing:

1. Domain parameters,
2. Velocity model parameters.

#### Outputs
A directory consisting of [velocity model files](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-VelocityModelFiles).

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `generate-velocity-model` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you are executing on your own computer you also need to specify the `NZVM` path (`--velocity-model-bin-path`) and the work directory (`--work-directory`).

#### Usage
`generate-velocity-model [OPTIONS] REALISATION_FFP VELOCITY_MODEL_OUTPUT`

#### For More Help
See the output of `generate-velocity-model --help` or `workflow.scripts.generate_velocity_model`

### Station Selection

#### Description
Filter a station list for in-domain stations to simulate high frequency and broadband output for.

#### Inputs
1. A station list and,
2. A realisation file containing domain parameters.

#### Outputs
1. A station list containing only stations in-domain and with unique discretised coordinate positions in two formats:
   - Stations in the format "longitude latitude name" format in "stations.ll",
   - Stations in the format "x y name" format in "stations.statcord". The x and y are the discretised positions of each station in the domain.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `generate-station-coordinates` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you do run this on your own computer, you need a version of `ll2gp` installed.

#### Usage
`generate-station-coordinates [OPTIONS] REALISATIONS_FFP OUTPUT_PATH`

#### For More Help
See the output of `generate-station-coordinates --help` or `workflow.scripts.generate_station_coordinates` for more help.

### Write Model Coordinates

#### Description
Write out model parameters for EMOD3D.

#### Inputs
1. A realisation file containing domain parameters.

#### Outputs
1. A model parameters file describing the location of the domain in latitude, longitude,
2. A grid parameters file describing the discretisation of the domain.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `generate-model-coordinates` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

#### Usage
`generate-station-coordinates [OPTIONS] REALISATIONS_FFP OUTPUT_PATH`
#### For More Help
See the output of `generate-model-coordinates --help` or `workflow.scripts.generate_model_coordinates` for more help.

### EMOD3D

#### Description
Run a low frequency ground motion simulation using EMOD3D.

#### Inputs
1. A parameter file in "key=value" format,
2. An SRF file,
3. A station file list (latitude, longitude, and x, y), see .

#### Outputs
1. Ground acceleration timeslice files, one per core.
2. Seismograms, one per station.

#### Environment
This stage must be run on a system with MPI installed. Typically, we run this stage in Maui on NeSI HPCs or Kisti. Due to high computational requirements, this stage usually cannot be run locally.

#### Usage
On an HPC with slurm enabled `srun emod3d-mpi_v3.0.8 -args "par=$CYLC_WORKFLOW_SHARE_DIR/LF/e3d.par"` will run EMOD3D. Depending on the number of cores rerequested, this may invoke multiple proesses on different compute nodes (for Maui, this will occur when the number of cores exceeds 40). EMOD3D has support for checkpointing, so repeat invocations will continue from their previous checkpointed stage.

#### For More Help
See Graves, 1996[^1] for a description of the mathematical and technical details of EMOD3D's implementation.

### High Frequency Simulation

#### Description
Generate stochastic high frequency ground acceleration data for a number of stations.

#### Inputs
1. A station list (in the "latitude longitude name" format),
2. A 1D velocity model,
3. A stoch file,
4. A realisation with domain parameters and metadata.

#### Outputs
1. A combined HF simulation output containing ground acceleration data for each station.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `hf-sim` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If you do run this on your own computer, you need a version of `hb_high_binmod` installed.

> [!NOTE]
> The high-frequency code is very brittle. It is recommended you have both versions 6.0.3 and 5.4.5 built to run with. Sometimes it is necessary to switch between versions if one does not work.

#### Usage
`hf-sim [OPTIONS] REALISATION_FFP STOCH_FFP STATION_FILE OUT_FILE`

#### For More Help
See the output of `hf-sim --help` or `workflow.scripts.hf_sim`.

### Create EMOD3D Parameters

#### Description
Write parameters for EMOD3D simulation.
#### Inputs
1. A realisation file containing domain parameters, velocity model parameters, and realisation metadata,
2. An SRF file,
3. A generated velocity model,
4. Station coordinates.
#### Outputs
An EMOD3D parameter file containing a mixture of simulations parameters. Parameters source values from the defaults specified the realisation defaults version. The `emod3d` section of the realisation file overrides default values.
#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `create-e3d-par` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.
#### Usage
`create-e3d-par [OPTIONS] REALISATION_FFP SRF_FILE_FFP VELOCITY_MODEL_FFP STATIONS_FFP GRID_FFP OUTPUT_FFP`

#### For More Help
See the output of `create-e3d-par --help` or [create_e3d_par.py](https://github.com/ucgmsim/workflow/blob/pegasus/workflow/scripts/create_e3d_par.py).
See our description of the [EMOD3D Parameters](https://wiki.canterbury.ac.nz/pages/viewpage.action?pageId=100794983) for documentation on the EMOD3D parameter file format.

### Merge Timeslices

#### Description
Merge the output timeslice files of EMOD3D.

#### Inputs
1. A directory containing EMOD3D timeslice files.

#### Outputs
1. A merged output timeslice file.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `merge-ts` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

#### Usage
`merge_ts XYTS_DIRECTORY XYTS_DIRECTORY/output.e3d`

#### For More Help
See the output of `merge-ts --help` or [merge_ts.py](https://github.com/ucgmsim/workflow/blob/pegasus/merge_ts/merge_ts.py).

### Create Simulation Video

#### Description
Create a simulation video from the low frequency simulation output.

#### Inputs
1. A merged timeslice file.

#### Outputs
1. An animation of the low frequency simulation output. See [youtube](https://www.youtube.com/watch?v=Crdk3k0Prew) for an example of these videos.

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `plot-ts` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If running on your own computer, you need to install [gmt](https://www.generic-mapping-tools.org/) and [ffmpeg](https://www.ffmpeg.org/). This stage does not run well on Windows, and is very dependent on the gmt version installed. Hypocentre is already setup to run `plot_ts.py` without installing anything.

#### Usage
`plot-ts [OPTIONS] SRF_FFP XYTS_INPUT_DIRECTORY OUTPUT_FFP`

#### For More Help
See the output of `plot-ts --help` or `workflow.scripts.plot_ts`

### Broadband Simulation

#### Description
Combine high-frequency and low-frequency simulation waveforms for each station into a broadband simulation file.

#### Inputs
1. A realisation file containing:
   - Realisation metadata,
   - Domain parameters.
2. Station list (latitude, longitude, name),
3. Stations VS30 reference values,
4. Low frequency waveform directory,
5. High frequency output file,
6. Velocity model directory.

#### Outputs
An output [broadband file](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation#FileFormatsUsedInGroundMotionSimulation-LF/HF/BBbinaryformat).

#### Environment
Can be run in the cybershake container. Can also be run from your own computer using the `bb-sim` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`. If running on your own computer, you need to configure a work directory (`--work-directory`).

#### Usage
`bb-sim REALISATION_FFP STATION_FFP STATION_VS30_FFP LOW_FREQUENCY_WAVEFORM_DIRECTORY HIGH_FREQUENCY_WAVEFORM_FILE VELOCITY_MODEL_DIRECTORY OUTPUT_FFP`

#### For More Help
See the output of `bb-sim --help` or `workflow.scripts.bb_sim` for more help.

[^1]: Graves, Robert W. "Simulating seismic wave propagation in 3D elastic media using staggered-grid finite differences." Bulletin of the seismological society of America 86.4 (1996): 1091-1106.
"""
