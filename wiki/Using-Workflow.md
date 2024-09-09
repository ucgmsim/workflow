# Using Workflow

This document is a tutorial for the new workflow. For those familiar with the old workflow, and in particular the [Rangitata Gorge](https://wiki.canterbury.ac.nz/pages/viewpage.action?pageId=181307633) tutorial, this also highlights a contrast between how the old and new workflow works.

This tutorial will simulate a simple rupture from the National Seismic Hazard Model 2022, and produce a video of the low-frequency waveforms (like the Alpine Fault [simulation video](https://www.youtube.com/watch?v=eJx9pxP_GU8)).

## Useful Resources

Your primary resource for help with workflow is, of course, the in-house software team at QuakeCoRE. However, there may be queries that are easier answered by the additional resources we list here.

Cylc is well-supported on NeSI. See the [Cylc on NeSI](https://docs.nesi.org.nz/Scientific_Computing/Supported_Applications/Cylc/) documentation for a more detailed description of using Cylc. The [official Cylc documentation](https://cylc.github.io/cylc-doc/stable/html/index.html) is another helpful resource. You might find support on the [Cylc forums](https://cylc.discourse.group/). Finally, because Cylc is maintained in-house at NeSI, you may also find help at [NeSI's office hours](https://docs.nesi.org.nz/Getting_Started/Getting_Help/Weekly_Online_Office_Hours/) for issues relating to NeSI specifically.

## Prerequisites

We'll assume that you are running the workflow on NeSI. If you want to run this example on a different environment (such as Hypocentre), refer to the [extra steps](#extra-steps) at the end of this tutorial.

All that is required for NeSI users to setup the workflow is

1. Enable Cylc and,
2. Clone the workflow repository to obtain the Cylc file.

### Enabling Cylc

To enable Cylc you need set the `CYLC_VERSION` environment variable. You also need the `cylc-src` and `cylc-run` directories in your home folder on NeSI.

``` shell
echo 'export CYLC_VERSION=8.0.1' >> ~/.bashrc
mkdir ~/cylc-src ~/cylc-run
source ~/.bashrc
```

### Cloning Workflow

The Cylc workflow file we'll use lives in the workflow repository. You must clone this repository to obtain the file.

``` shell
git clone git@github.com:ucgmsim/workflow.git
cd workflow
cp -r cylc ~/cylc-src/tutorial
mkdir ~/cylc-src/tutorial/input
```

The `tutorial` directory is the name of our workflow. The `input` subdirectory houses files that are used as input in the workflow, like your station list or a custom 1D velocity model.

## Customising Your Workflow File

The Cylc workflow we copied in [the prior section](#cloning-workflow) contains a skeleton file for the full Cybershake workflow. However, we don't need the high frequency and broadband waveforms.We can easily modify the Cylc file to only include the stages we need. Copy the following into `~/cylc-src/tutorial/flow.cylc`:

```
[scheduler]
    allow implicit tasks = True
[scheduling]
    [[graph]]
        R1 = """
            nshm_to_realisation => realisation_to_srf & generate_velocity_model_parameters
            generate_velocity_model_parameters => generate_velocity_model & generate_station_coordinates & generate_model_coordinates
            realisation_to_srf & generate_velocity_model & generate_station_coordinates & generate_model_coordinates =>  create_e3d_par
            create_e3d_par => run_emod3d
            run_emod3d => plot_ts
            """
[runtime]
    [[root]]
        platform = mahuika-slurm
        pre-script = """
        module load Apptainer
        """
        [[[directives]]]
            --account = nesi00213
    [[nshm_to_realisation]]
        platform = localhost
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif nshm2022-to-realisation /nshmdb.db <CHANGE ME> /share/realisation.json <CHANGE ME>
    [[realisation_to_srf]]
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif realisation-to-srf /share/realisation.json /share/realisation.srf
    [[generate_stoch]]
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif generate-stoch /share/realisation.srf /share/realisation.stoch
    [[generate_velocity_model_parameters]]
        platform = localhost
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif generate-velocity-model-parameters /share/realisation.json
    [[generate_velocity_model]]
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif sh -c 'generate-velocity-model /share/realisation.json /share/Velocity_Model --num-threads $(nproc)'
        [[[directives]]]
                --cpus-per-task = 32
                --time = 01:00:00
    [[generate_station_coordinates]]
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif generate-station-coordinates /share/realisation.json /share/stations
    [[generate_model_coordinates]]
        platform = localhost
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif generate-model-coordinates /share/realisation.json /share/model
    [[create_e3d_par]]
        platform = localhost
        script = apptainer exec /nesi/nobackup/nesi00213/containers/runner_latest.sif create-e3d-par $CYLC_WORKFLOW_SHARE_DIR/realisation.json $CYLC_WORKFLOW_SHARE_DIR/realisation.srf $CYLC_WORKFLOW_SHARE_DIR/Velocity_Model $CYLC_WORKFLOW_SHARE_DIR/stations $CYLC_WORKFLOW_SHARE_DIR/model $CYLC_WORKFLOW_SHARE_DIR/LF --emod3d-path /nesi/project/nesi00213/opt/maui/hybrid_sim_tools/emod3d-mpi_v3.0.8 --scratch-ffp $CYLC_WORKFLOW_SHARE_DIR/LF
    [[run_emod3d]]
        platform = maui-xc-slurm
        pre-script = ""
        script = srun /nesi/project/nesi00213/opt/maui/hybrid_sim_tools/emod3d-mpi_v3.0.8 -args "par=$CYLC_WORKFLOW_SHARE_DIR/LF/e3d.par"
        [[[directives]]]
            --ntasks = 80
            --hint = nomultithread
            --time = 00:30:00
    [[run_hf]]
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif run-hf /share/realisation.json /share/realisation.stoch /share/stations.ll /share/realisation.hf
        [[directives]]
                --cpus-per-task = 128
                --time = 01:00:00
    [[plot_ts]]
        script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif plot-ts /share/LF/OutBin /share/simulation.m4v
```

At this point have a folder `~/cylc-src/tutorial` containing the following directory structure

```bash
$ tree ~/cylc-src/tutorial
.
├── flow.cylc
└── input

2 directories, 1 file
```

**You are not done yet!**. See the next section where we will pick a rupture to simulate.

## Picking a Rupture and Defaults
Observant readers may have noticed the `<CHANGE ME>` blocks hiding inside the `nshm_to_realisation` stage. This stage selects a rupture from the National Seismic Hazard Model database and generates a _realisation_. A realisation is JSON file specifying a single rupture scenario. It contains, initially, a specification of the source geometry, magnitude, and metadata for a rupture. It is updated overtime as stages complete.

The `nshm_to_realisation` stage expects two variables we want to change: the rupture id specifying the rupture to simulate, and a the _defaults_ versions. For the rupture id, we will pick rupture 0. This is a simple single-segment rupture of the Acton fault. The _defaults_ version points to a set of default values to use for simulation. The defaults specify things like:

- The spatial and temporal resolution of the simulation,
- The simulation length,
- Where the cutoff between low and high frequency simulation is,
- The topography type for the velocity model,
- etc...

For most of your simulation defaults, you should pick `24.2.2.X` where `X` is the simulation resolution (in 100's of metres). We will do a low resolution 400m simulation and thus pick `24.2.2.4`.

Change line 22 of `~/cylc-src/tutorial/flow.cylc` to read

```
script = apptainer exec -c --bind "$PWD:/out,$CYLC_WORKFLOW_SHARE_DIR:/share,$CYLC_WORKFLOW_RUN_DIR/input:/input:ro" /nesi/nobackup/nesi00213/containers/runner_latest.sif nshm2022-to-realisation /nshmdb.db 0 /share/realisation.json 24.2.2.4
```

## Installing the Workflow

To install the workflow to run invoke

``` shell
cylc install tutorial
```

This will copy the source files in `~/cylc-src/tutorial` to `~/cylc-run/tutorial/run1`. Every subsequent install of this workflow will create a new folder under `tutorial`, so running `cylc install tutorial` again creates `~/cylc-run/tutorial/run2`, and so on.

> [!NOTE]
> To access the latest run of a given workflow, you can visit `~/cylc-run/tutorial/runN`.

Let's look at the tutorial workflow directory.

``` shell
~/cylc-run/tutorial/runN@maui $ ls
log  share  work  flow.cylc
```

Besides the `flow.cylc` file we have three additional directories:

1. The `log` directory which contains the log files for the run. You never need to read this file directly, because Cylc provides convenience commands to read logs.
2. The `share` directory. This directory contains files that are shared between jobs, like the `realisation.json` file, but also the final outputs like our animation.
3. The `work` directory. Some jobs, like EMOD3D, produce many files that other jobs don't care about. These are saved in the work directory `work/<job name>/...` to promote job isolation.

When the workflow is done, your output will live in the `share` directory.

## Running the Workflow

This step is simple!

``` shell
cylc play tutorial
```

The above command instructs Cylc to run your workflow. You can now monitor it in two places. The first place is the Cylc logs accessed via `cylc log tutorial`. The second logging location is the NeSI slurm queue. The slurm queue is the queue all jobs running on HPC must wait in. You can access the slurm queue for your jobs with `squeue -u $USER`.

Cylc has other ways to monitor your workflow, including a GUI. See NeSI's [documentation](https://docs.nesi.org.nz/Scientific_Computing/Supported_Applications/Cylc/#different-ways-to-interact-with-cylc) on the different ways they support interacting with cylc including the terminal user interface, GUI and Jupyter notebooks.

## Extra Steps
