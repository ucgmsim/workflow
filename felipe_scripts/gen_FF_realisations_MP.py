'''
Last modified on 07-07-2026
Felipe Kuncar (felipe.kuncar@canterbury.ac.nz)
'''

# Workflow branch: pegasus

# =====================================================================================================================
# IMPORT LIBRARIES, FUNCTIONS, CLASSES
# =====================================================================================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import requests
import json

# Avoid issue with RankWarning
import warnings
if not hasattr(np, "RankWarning"):
    np.RankWarning = RuntimeWarning

from source_modelling.community_fault_model import NodalPlane
from workflow.scripts.gcmt_to_realisation import gcmt_to_realisation, NodalPlaneChoice
from workflow.scripts.generate_domain import generate_domain_from_realisation
from workflow.realisations import VelocityModelParameters, IntensityMeasureCalculationParameters
from workflow.defaults import DefaultsVersion

# =====================================================================================================================
# PARAMETERS
# =====================================================================================================================

defaults_version = '24.2.2.1'
vel_model_version = '2.09'

# Source type
source_type='finite-fault'
# source_type = 'point-source'

# Rrup Interpolants (Lee et al., 2024)
mag_vec, rrup_vec = np.loadtxt(Path('Mw_rrup_mod.txt'), unpack=True)
rrup_interpolants = np.array([mag_vec, rrup_vec], dtype=np.float32)

# Read vibration periods and FAS frequencies
IMs_DIR = Path("IMs")
periods_csv = IMs_DIR / "periods.csv"
frequencies_csv = IMs_DIR / "frequencies.csv"
# Vibration periods
period_array: np.ndarray = pd.read_csv(periods_csv)["valid_periods"].to_numpy(dtype=np.float64)
# FAS frequencies
frequency_array: np.ndarray = pd.read_csv(frequencies_csv)["fas_frequencies"].to_numpy(dtype=np.float64)

# =====================================================================================================================
# PATHS
# =====================================================================================================================

rootDir = Path(__file__).parents[1]

input_path = Path(rootDir / 'step1' / 'output')

if source_type=='finite-fault':
    output_path = Path('output') / 'realisations_finite-fault'
elif source_type=='point-source':
    output_path = Path('output') / 'realisations_point-source'
os.makedirs(output_path, exist_ok=True)

# =====================================================================================================================
# FUNCTION TO GET NODAL PLANES
# =====================================================================================================================

def get_gcmt_solution(gcmt_event_id):

    MOMENT_TENSOR_SOLUTION_URL = "https://raw.githubusercontent.com/GeoNet/data/main/moment-tensor/GeoNet_CMT_solutions.csv"
    AUTOMATED_TENSOR_URL = "https://gcmt-realtime-database-default-rtdb.asia-southeast1.firebasedatabase.app/c059adc9c34b9b1c77a3bfc04e4059ea/earthquakes.json"
    NAN_PUBLIC_ID = "9999999"

    gcmt_solutions = pd.read_csv(MOMENT_TENSOR_SOLUTION_URL)
    automated_gcmt_solutions = requests.get(AUTOMATED_TENSOR_URL).json()

    gcmt_solutions = gcmt_solutions[
        gcmt_solutions["PublicID"] != NAN_PUBLIC_ID
        ].set_index("PublicID")

    if gcmt_event_id in gcmt_solutions.index:
        solution = gcmt_solutions.loc[gcmt_event_id]
        nodal_plane_1 = NodalPlane(solution["strike1"], solution["dip1"], solution["rake1"])
        nodal_plane_2 = NodalPlane(solution["strike2"], solution["dip2"], solution["rake2"])
    elif gcmt_event_id in automated_gcmt_solutions:
        solution = automated_gcmt_solutions[gcmt_event_id]
        nodal_plane_1 = NodalPlane(**solution["nodalPlanes"][0])
        nodal_plane_2 = NodalPlane(**solution["nodalPlanes"][1])
    else:
        print(f"GCMT event ID {gcmt_event_id} not found in either the published GCMT solutions or automated solutions.")

    return solution, nodal_plane_1, nodal_plane_2

# =====================================================================================================================
# FUNCTION TO PROCESS ONE EVENT
# =====================================================================================================================

def process_event(args):
    evid, selected_strike, rrup_interpolants, realisationFiles_dir, source_type = args
    try:
        print(f"Processing event {evid}...")

        solution, nodal_plane1, nodal_plane2 = get_gcmt_solution(evid)

        print('Selected strike:', selected_strike)

        if selected_strike == nodal_plane1.strike:
            selected_plane = nodal_plane1
            nodal_plane_choice = NodalPlaneChoice.PLANE_1
        elif selected_strike == nodal_plane2.strike:
            selected_plane = nodal_plane2
            nodal_plane_choice = NodalPlaneChoice.PLANE_2
        else:
            raise ValueError(
                f"Selected strike {selected_strike}° does not match either nodal plane "
                f"(strike1={nodal_plane1.strike}°, strike2={nodal_plane2.strike}°)"
            )
        print(f"Selected plane: {selected_plane}")
        print(f"Nodal plane choice: {nodal_plane_choice}")

        # json file that contains all the information regarding the parameters considered
        realisationFile_name = 'realisation.json'
        realisationFiles_dir.mkdir(parents=True, exist_ok=True)
        temp_realisation_file = Path(realisationFiles_dir / realisationFile_name)

        # Generate realisation (.json file)
        gcmt_to_realisation(
            gcmt_event_id=evid,
            defaults_version=defaults_version,
            realisation_ffp=temp_realisation_file,
            source_type=source_type,
            nodal_plane =nodal_plane_choice
        )

        # Modify velocity model version and rrup_interpolants
        vm_params = VelocityModelParameters.read_from_defaults(DefaultsVersion.v24_2_2_1)
        vm_params.version = vel_model_version
        vm_params.rrup_interpolants = rrup_interpolants
        vm_params.write_to_realisation(temp_realisation_file)

        # Generate domain parameters
        generate_domain_from_realisation(temp_realisation_file)

        # Modify valid_periods and fas_frequencies
        im_params = IntensityMeasureCalculationParameters.read_from_defaults(DefaultsVersion.v24_2_2_1)
        im_params.valid_periods = period_array
        im_params.fas_frequencies = frequency_array
        im_params.write_to_realisation(temp_realisation_file)

        print(f"Event {evid} done")
    except Exception as e:
        print(f"Error processing event {evid}: {e}")

# =====================================================================================================================
# MAIN
# =====================================================================================================================

if __name__ == "__main__":

    # Read events
    #selected_events_df = pd.read_csv(Path(input_path / 'selected_events.csv'), dtype={"evid": str})
    selected_events_df = pd.read_csv('selected_events_crustal_moderate.csv', dtype={"evid": str})
    selected_events = selected_events_df['evid'].to_list()
    # For testing:
    #selected_events = ['2398629', '2012p001887', '2016p118944']
    print('Number of events considered:', len(selected_events))

    # Read selected nodal planes
    selected_nodal_plane_df = pd.read_csv(Path('preferred_nodal_planes') / 'CMT_solutions.csv', dtype={"PublicID": str})

    # Set number of parallel processes
    nproc = min(8, cpu_count())
    print(f"Using {nproc} parallel processes")

    if source_type == 'finite-fault':

        args_list = [(evid,
                      selected_nodal_plane_df.loc[selected_nodal_plane_df['PublicID'] == evid, 'strike1'].iloc[0],
                      rrup_interpolants,
                      Path(output_path / evid),
                      source_type) for evid in selected_events]

    # Process events in parallel
    with Pool(processes=nproc) as pool:
        pool.map(process_event, args_list)
