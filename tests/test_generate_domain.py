from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from hypothesis import given
from hypothesis import strategies as st

from source_modelling import magnitude_scaling, sources
from workflow.realisations import (
    Magnitudes,
    Rakes,
    SourceConfig,
    VelocityModelParameters,
)
from workflow.scripts import generate_domain


# Slow because of openquake import
@pytest.mark.slow
def test_significant_duration_calculation() -> None:
    """Basic integration test for OQW, tests that the interface works as we expect still."""
    ds595 = generate_domain.get_significant_duration(
        magnitude=6.5, distance=100.0, vs30=500.0, rake=180.0, z1pt0=5.0
    )
    assert isinstance(ds595, float)  # Should not be lying about the type
    assert np.isfinite(ds595), "Ds595 is invalid"
    # Check that ds595 is not in log-space or similar
    assert ds595 > 0, "Ds595 should be positive"
    assert ds595 < 1000, "Ds595 unrealistically high"
    # Combined with type checking this should be enough to catch most of these problems.


def test_simulation_max_depth_increases_with_mw() -> None:
    depth_mw5p0 = generate_domain.simulation_max_depth(magnitude=5.0, bottom_depth=10.0)
    depth_mw7p0 = generate_domain.simulation_max_depth(magnitude=6.0, bottom_depth=10.0)
    depth_mw9p0 = generate_domain.simulation_max_depth(magnitude=9.0, bottom_depth=10.0)
    assert depth_mw5p0 < depth_mw7p0 < depth_mw9p0, (
        "Depths do not increase with magnitude"
    )


@given(
    magnitude=st.floats(min_value=3.5, max_value=9.0),
    depth=st.floats(min_value=1.0, max_value=250),
)
def test_simulation_max_depth_more_than_bottom_depth(
    magnitude: float, depth: float
) -> None:
    simulation_depth = generate_domain.simulation_max_depth(magnitude, depth)
    assert np.isfinite(simulation_depth), "Invalid simulation depth"
    assert depth <= simulation_depth <= 350, "Sensible simulation depths are applied"


def test_estimate_domain_contains_fault_geometry() -> None:
    fault_coords = [(100000, 100000), (110000, 100000)]
    fault_geom = shapely.LineString(fault_coords)

    mock_fault = SimpleNamespace(geometry=fault_geom)
    source_config = SimpleNamespace(source_geometries={"fault_a": mock_fault})

    rrups = {"fault_a": 5000.0}

    nz_outline = shapely.box(0, 0, 500000, 500000)

    result_domain = generate_domain.estimate_domain(
        source_config=source_config,  # ty: ignore[invalid-argument-type]
        rrups=rrups,
        nz_outline=nz_outline,
        fault_buffer=2000.0,
    )

    assert result_domain.polygon.contains(fault_geom), (
        f"Domain polygon should contain the original fault geometry.\n"
        f"Fault bounds: {fault_geom.bounds}\n"
        f"Domain bounds: {result_domain.polygon.bounds}"
    )


# Slow because of openquake import
@pytest.mark.slow
def test_generate_domain() -> None:
    """Basic E2E test to check that domain generation works without crashing or producing a silly domain."""
    source = sources.Point(
        np.array([-43.0, -172.0, 10000.0]),
        length_m=1000,
        width_m=1000,
        strike=90.0,
        dip=45.0,
        dip_dir=180.0,
    )
    source_config = SourceConfig({"source": source})
    magnitudes = Magnitudes({"source": magnitude_scaling.BoldM(6.0)})
    rakes = Rakes({"source": 180.0})

    velocity_model_parameters = VelocityModelParameters(
        min_vs=500.0,
        version="2.09",
        topo_type="BULLDOZED",
        ds_multiplier=1.2,
        vs30=500.0,
        fault_buffer=2000,
        s_wave_velocity=3500,
        rrup_interpolants=np.array([[5.0, 8.0], [50.0, 50.0]]),
    )
    domain_parameters = generate_domain.generate_domain(
        source_config,
        magnitudes,
        rakes,
        velocity_model_parameters,
    )
    assert shapely.contains(domain_parameters.domain.polygon, source.geometry)
