# Compiling Tools in EMOD3D and NZVM for Workflow

The workflow uses a few (and hopefully fewer in the future) C binaries built in EMOD3D. The Cybershake container has these prebuilt, so if you run your Python code in this container you do not need to build any binaries. The [EMOD3D Readme](https://github.com/ucgmsim/EMOD3D?tab=readme-ov-file#emod3d) has generic build instructions for tools in EMOD3D, but this page has more detailed instructions for each tool required by the workflow.

If you are using a system with `gcc` version 14 or later, then you need to see the [extra steps](#extra-steps-for-gcc14-users). You can tell if you have a system with version 14 or later by executing `gcc --version` in the command line. You are using version 14 if you see something like the following.
```
gcc (GCC) 14.2.1 20240910
Copyright (C) 2024 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
**Hypocentre and NeSI are not using version 14 at this stage.**


> [!NOTE]
> Before you start! Clone the EMOD3D repository somewhere with the following command: `git clone https://github.com/ucgmsim/EMOD3D`

## SRF Generation

If you are generating a SRF with `realisation-to-srf`, you need to build genslip. **After cloning EMOD3D**, change directory (using `cd`) into the `EMOD3D` folder and execute the following in the command line (one line per command):

``` shell
mkdir build
cd build
cmake ..
cmake --build . --target genslip_5.4.2
```

You then need to pass the `--genslip-path` flag with the path to this binary when you use `realisation-to-srf`

``` shell
realisation-to-srf realisation.json realisation.srf realisation.stoch --genslip-path <PATH TO EMOD3D DIRECTORY>/tools/srf2stoch --work-directory ... --velocity-model-ffp ...
```

Replace `<PATH TO EMOD3D DIRECTORY>` with the path to your cloned copy of EMOD3D where you built the tools.


## Stoch Generation
If you are generating a Stoch file with `generate-stoch`, you need to build srf2stoch. **After cloning EMOD3D**, change directory (using `cd`) into the `EMOD3D` folder and execute the following in the command line (one line per command):

``` shell
mkdir build
cd build
cmake ..
cmake --build . --target srf2stoch
```

You then need to pass the `--srf2stoch-path` flag with the path to this binary when you use `generate-stoch`

``` shell
generate-stoch realisation.json realisation.srf realisation.stoch --srf2stoch-path <PATH TO EMOD3D DIRECTORY>/tools/srf2stoch
```

Replace `<PATH TO EMOD3D DIRECTORY>` with the path to your cloned copy of EMOD3D where you built the tools.


## High Frequency Simulation

If you are locally running high-frequency simulations using `hf-sim` then you need to build the high-frequency simulation binaries.
**After cloning EMOD3D**, change directory (using `cd`) into the `EMOD3D` folder and execute the following in the command line (one line per command):

``` shell
mkdir build
cd build
cmake ..
cmake --build . --target hb_high_binmod_v6.0.3
cmake --build . --target hb_high_binmod_v5.4.5.3
```

You then need to pass the `--hf-sim-path` flag with the path to this binary when you use `hf-sim`

``` shell
hf-sim realisation.json realisation.stoch stations.ll HF.bin --hf-sim-path <PATH TO EMOD3D DIRECTORY>/tools/hb_high_binmod_v6.0.3 --velocity-model ... --work-directory ...
```

Replace `<PATH TO EMOD3D DIRECTORY>` with the path to your cloned copy of EMOD3D where you built the tools.

### My High-Frequency Simulation Fails!

The high-frequency Fortran code is very brittle and old. Sometimes you need to use a different version to make it work. Try running the same `hf-sim` command with version `5.4.5.3`.

``` shell
hf-sim realisation.json realisation.stoch stations.ll HF.bin --hf-sim-path <PATH TO EMOD3D DIRECTORY>/tools/hb_high_binmod_v5.4.5.3 --velocity-model ... --work-directory ...
```

## Velocity Modelling

To create a velocity model you have to build `NZVM`. To build `NZVM`, run the following

``` shell
git clone https://github.com/ucgmsim/Velocity-Model
cd Velocity-Model
make
```

Then when running the velocity model pass the path to the `NZVM` binary using the `--velocity-model-bin-path` flag.

``` shell
generate-velocity-model realisation.json velocity_model_folder --velocity-model-bin-path <PATH TO VELOCITY MODEL DIRECTORY>/NZVM --work-directory ...
```

## Extra Steps for GCC14 Users

After cloning the EMOD3D repository, running the following

``` shell
cd EMOD3D
git checkout gcc14
```
