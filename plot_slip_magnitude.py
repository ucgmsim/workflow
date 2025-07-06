#!/usr/bin/env python
"""Plot slips rate as a function of magnitude for point sources at constant depth.

This script creates a plot showing how slips varies with magnitude for point sources
at a constant depth of 5 km. It uses the calc_point_source_slip function from the
realisation_to_srf module to compute slips values for different magnitudes.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the workflow module to the Python path
sys.path.insert(0, str(Path(__file__).parent / "source_modelling"))
sys.path.insert(0, str(Path(__file__).parent / "workflow"))

from qcore.uncertainties import mag_scaling
from source_modelling import magnitude_scaling, sources
from workflow import defaults
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RupturePropagationConfig,
    SourceConfig,
    SRFConfig,
    VelocityModel1D,
)
from workflow.scripts.realisation_to_srf import (
    SRFRealisationContext,
    calc_point_source_slip,
)


def create_default_velocity_model() -> VelocityModel1D:
    """Create a default velocity model using workflow defaults.

    Returns
    -------
    VelocityModel1D
        A velocity model with the develop defaults values.
    """
    return VelocityModel1D.read_from_defaults(defaults.DefaultsVersion.develop)


def create_point_source_at_depth(
    depth_km: float, length_km: float = 1.0
) -> sources.Point:
    """Create a simple point source at a given depth.

    Parameters
    ----------
    depth_km : float
        The depth of the point source in kilometers.
    length_km : float, optional
        The length scale of the point source in kilometers.

    Returns
    -------
    sources.Point
        A point source at the specified depth.
    """
    # Create a point source at a simple location (Wellington area coordinates)
    point_coordinates = np.array(
        [-41.2865, 174.7762, depth_km * 1000]
    )  # depth in meters

    return sources.Point.from_lat_lon_depth(
        point_coordinates=point_coordinates,
        length_m=length_km * 1000,  # convert km to meters
        strike=45.0,  # arbitrary strike
        dip=90.0,  # vertical dip
        dip_dir=135.0,  # dip direction (strike + 90)
    )


def create_srf_context_for_magnitude(
    magnitude: float, depth_km: float
) -> SRFRealisationContext:
    """Create an SRFRealisationContext for a given magnitude and depth.

    Parameters
    ----------
    magnitude : float
        The magnitude of the earthquake.
    depth_km : float
        The depth in kilometers.

    Returns
    -------
    SRFRealisationContext
        A context object for SRF calculation.
    """
    # Create a point source at the specified depth
    point_source = create_point_source_at_depth(depth_km)

    # Create the required configuration objects
    source_config = SourceConfig(source_geometries={"point_source": point_source})
    magnitudes = Magnitudes(magnitudes={"point_source": magnitude})
    rakes = Rakes(rakes={"point_source": 0.0})  # arbitrary rake
    velocity_model_1d = create_default_velocity_model()
    srf_config = SRFConfig(resolution=1.0, genslip_dt=0.02, genslip_version="5.4.2")

    # Create SRF context
    rupture_propagation_config = RupturePropagationConfig(
        rupture_causality_tree={
            "point_source": None
        },  # Single point source with no parent
        jump_points={},  # No jumps for single point source
        hypocentre=np.array([0.5, 0.5]),  # Center of the point source
    )

    return SRFRealisationContext(
        source_config=source_config,
        rupture_propagation_config=rupture_propagation_config,
        magnitudes=magnitudes,
        rakes=rakes,
        velocity_model_1d=velocity_model_1d,
        srf_config=srf_config,
    )


def main() -> None:
    """Main function to create the slips vs magnitude plot."""

    velocity_model_1d = VelocityModel1D.read_from_defaults(
        defaults.DefaultsVersion.v24_2_2_4
    )
    velocity_model_df = velocity_model_1d.model
    velocity_model_df["depth_km"] = velocity_model_df["thickness"].cumsum()

    source_depth_km = 5.0

    # Find the index of the closest depth in the velocity model
    idx = np.argmin(np.abs(velocity_model_df["depth_km"] - source_depth_km))
    vs_km_per_s = velocity_model_df.iloc[idx]["Vs"]
    rho_g_per_cm3 = velocity_model_df.iloc[idx]["rho"]

    magnitudes = [4.0, 4.25, 4.5, 4.75, 5.0]
    moments_dyne_cm = []
    areas_km2 = []
    slips = []

    for magnitude in magnitudes:
        moment_dyne_cm = mag_scaling.mag2mom(magnitude)
        area_km2 = magnitude_scaling.leonard_magnitude_to_area(magnitude, rake=110)

        moments_dyne_cm.append(moment_dyne_cm)
        areas_km2.append(area_km2)

        slips.append(
            calc_point_source_slip(moment_dyne_cm, area_km2, vs_km_per_s, rho_g_per_cm3)
        )

    srf_slips = [0.0288, 0.0344]
    gsf_slips = [8.37838582945137, 3.648887967004306]
    mags = [4.5, 4]
    event_ids = ["2025p013176", "2025p186425"]  # Example event name

    plt.plot(magnitudes, slips)

    plt.plot(magnitudes, slips, "bo-", linewidth=2, markersize=8, label="theoretical")
    plt.plot(mags, gsf_slips, "ro-", linewidth=2, markersize=8, label="GSF")

    plt.xlabel("Magnitude")
    plt.ylabel("Slip (cm)")

    plt.grid(True, alpha=0.3)

    plt.show()

    print()

    # # Fixed depth
    # depth_km = 5.0

    # # Calculate slips for each magnitude
    # slips = []
    # for magnitude in magnitudes:
    #     context = create_srf_context_for_magnitude(magnitude, depth_km)
    #     slips = calc_point_source_slip_wrapper(context, "point_source")
    #     slips.append(slips)
    #     print(f"Magnitude {magnitude}: Slip = {slips:.3f} cm")

    # # Create the plot

    # # Add some styling
    # plt.tight_layout()

    # # Add text annotations for each point
    # for i, (mag, slips) in enumerate(zip(magnitudes, slips)):
    #     plt.annotate(
    #         f"{slips:.2f} cm",
    #         (mag, slips),
    #         textcoords="offset points",
    #         xytext=(0, 10),
    #         ha="center",
    #         fontsize=9,
    #     )

    # # Show the plot
    # plt.show()

    # # Also save the plot
    # output_path = Path(__file__).parent / "slip_magnitude_plot.png"
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # print(f"\nPlot saved to: {output_path}")

    # # Print summary statistics
    # print("\nSummary:")
    # print(f"Depth: {depth_km} km")
    # print(f"Magnitude range: {min(magnitudes)} - {max(magnitudes)}")
    # print(f"Slip range: {min(slips):.3f} - {max(slips):.3f} cm")
    # print(f"Slip ratio (max/min): {max(slips) / min(slips):.2f}")


if __name__ == "__main__":
    main()
