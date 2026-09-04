import numpy as np
import pytest

from rupture_generator.config.geometry import FaultConfig, PointConfig
from rupture_generator.config.rupture import (
    ComputedPropagation,
    PerFaultSourceConfig,
    PointSourceConfig,
    PredeterminedPropagation,
    RampConfig,
)
from source_modelling import sources
from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RealisationMetadata,
    RupturePropagationConfig,
    RuptureVelocity,
    Seeds,
    SourceConfig,
    SRFConfig,
    VelocityModel1D,
)
from workflow.scripts import rupture_input_template


@pytest.fixture
def srf_config() -> SRFConfig:
    return SRFConfig.read_from_defaults(DefaultsVersion.v24_2_2_1)


@pytest.fixture
def rupture_velocity() -> RuptureVelocity:
    return RuptureVelocity(
        rvfrac=0.8,
        rvfrac_shal=0.55,
        rvfrac_deep=0.65,
        rvfrac_slip_sig=None,
        shallow_depth=6.5,
        shallow_transition_range=1.5,
        deep_depth=17.5,
        deep_transition_range=2.5,
    )


@pytest.fixture
def velocity_model_1d() -> VelocityModel1D:
    return VelocityModel1D.read_from_defaults(DefaultsVersion.v24_2_2_1)


@pytest.fixture
def metadata() -> RealisationMetadata:
    return RealisationMetadata(
        name="test",
        version="1",
        defaults_version=DefaultsVersion.v24_2_2_1,
        tag=None,
    )


def a_fault(strike: float = 45.0, dip: float = 70.0) -> sources.Fault:
    """One planar fault, hung from a trace point by strike and dip."""
    return sources.Fault(
        planes=[
            sources.Plane.from_centroid_strike_dip(
                centroid=np.array([-43.5, 172.6, 4.5]),
                strike=strike,
                dip_dir=strike + 90,
                dip=dip,
                dtop=1.0,
                dbottom=8.0,
                length=10.0,
                width=(8.0 - 1.0) / np.sin(np.radians(dip)),
            )
        ]
    )


def test_geometry_mirrors_the_source(
    srf_config: SRFConfig, metadata: RealisationMetadata
) -> None:
    """A realisation's fault reaches the generator as the same fault.

    The dip, the depths and the discretisation are what the mesh is built from, so each
    is pinned against the `source_modelling` geometry it was read off.
    """
    fault = a_fault()
    source_config = SourceConfig(source_geometries={"a": fault})

    geometry = rupture_input_template.build_geometry(
        source_config, srf_config, metadata
    )

    (surface,) = geometry.surfaces
    assert isinstance(surface, FaultConfig)
    assert surface.name == "a"
    assert surface.top_depth_km == pytest.approx(1.0)
    assert geometry.crs.to_epsg() == rupture_input_template.NZTM_EPSG
    # A projected CRS, which the generator requires and refuses a geographic one for.
    assert geometry.crs.is_projected

    (plane,) = surface.planes
    assert plane.dip_deg == pytest.approx(70.0)
    assert plane.bottom_depth_km == pytest.approx(8.0)
    assert plane.discretisation.subfault_size_km == srf_config.resolution


def test_point_source_becomes_a_point_surface(
    srf_config: SRFConfig, metadata: RealisationMetadata
) -> None:
    """A point source is a point surface, not a one-plane fault."""
    point = sources.Point.from_lat_lon_depth(
        np.array([-43.5, 172.6, 8000.0]),
        length_m=3000.0,
        width_m=3000.0,
        strike=64.0,
        dip=58.0,
        dip_dir=154.0,
    )
    geometry = rupture_input_template.build_geometry(
        SourceConfig(source_geometries={"a": point}), srf_config, metadata
    )

    (surface,) = geometry.surfaces
    assert isinstance(surface, PointConfig)
    assert surface.depth_km == pytest.approx(8.0)
    assert surface.dip_deg == pytest.approx(58.0)
    assert surface.size_km == pytest.approx(3.0)


def test_rupture_injects_what_the_srf_section_omits(
    srf_config: SRFConfig,
    rupture_velocity: RuptureVelocity,
    velocity_model_1d: VelocityModel1D,
    metadata: RealisationMetadata,
) -> None:
    """The `srf` section is splatted through; everything else is injected.

    The rupture-speed profile is the half worth pinning: it lives in
    `rupture_velocity` rather than in `srf` because `hf-sim` reads the same physical
    values, so this checks it arrives in the generator's `timing` and `field`.
    """
    fault = a_fault()
    source_config = SourceConfig(source_geometries={"a": fault, "b": a_fault()})
    rupture_propagation = RupturePropagationConfig(
        rupture_causality_tree={"a": None, "b": "a"},
        jump_points={},
        hypocentre=np.array([0.5, 0.25]),
    )

    config = rupture_input_template.build_rupture(
        source_config,
        srf_config,
        rupture_propagation,
        Magnitudes(magnitudes={"a": 6.0, "b": 5.5}),
        Rakes(rakes={"a": 110.0, "b": 90.0}),
        velocity_model_1d,
        rupture_velocity,
        Seeds(
            nshm_to_realisation_seed=1,
            rupture_propagation_seed=2,
            rupture_seed=1234,
            hf_seed=4,
        ),
        metadata,
    )

    # Splatted through unchanged.
    assert (
        config.slip.coefficient_of_variation
        == (srf_config.slip["coefficient_of_variation"])
    )
    assert config.timing.sample_interval_s == srf_config.timing["sample_interval_s"]

    # Injected, because the `srf` section does not carry them.
    assert config.field.velocity_fraction == rupture_velocity.rvfrac
    assert config.timing.shallow_speed_factor == rupture_velocity.rvfrac_shal
    assert config.timing.deep_speed_factor == rupture_velocity.rvfrac_deep
    assert config.timing.shallow_ramp == RampConfig(
        centre_km=rupture_velocity.shallow_depth,
        half_width_km=rupture_velocity.shallow_transition_range,
    )
    assert config.timing.deep_ramp == RampConfig(
        centre_km=rupture_velocity.deep_depth,
        half_width_km=rupture_velocity.deep_transition_range,
    )
    assert config.random.seed == 1234

    # The nested ramps are the generator's own type, not the raw dicts the realisation
    # holds them as -- the stages read attributes off them.
    assert isinstance(config.timing.rise_time_blend, RampConfig)

    # The hazard model's division of the moment, kept rather than re-derived.
    assert isinstance(config.source, PerFaultSourceConfig)
    assert config.source.magnitudes == {"a": 6.0, "b": 5.5}
    assert config.source.rakes == {"a": 110.0, "b": 90.0}

    # Fractions along strike and down dip become in-fault arc lengths.
    assert config.hypocentre.fault == "a"
    assert config.hypocentre.strike_km == pytest.approx(0.5 * fault.length)
    assert config.hypocentre.dip_km == pytest.approx(0.25 * fault.width)

    # Layer thicknesses become depths to each layer's bottom.
    assert config.velocity_model.bottom_depth_km == pytest.approx(
        np.cumsum(velocity_model_1d.model["thickness"])
    )

    assert isinstance(config.propagation, PredeterminedPropagation)
    assert config.propagation.parents == {"b": "a"}


def test_single_segment_leaves_propagation_alone(
    srf_config: SRFConfig,
    rupture_velocity: RuptureVelocity,
    velocity_model_1d: VelocityModel1D,
    metadata: RealisationMetadata,
) -> None:
    """One fault has no edges to state, and an empty tree is refused rather than written.

    The generator will not accept a `predetermined` propagation with nothing in it, so
    the section is left at its default instead of being written out saying nothing.
    """
    config = rupture_input_template.build_rupture(
        SourceConfig(source_geometries={"a": a_fault()}),
        srf_config,
        RupturePropagationConfig(
            rupture_causality_tree={"a": None},
            jump_points={},
            hypocentre=np.array([0.5, 0.5]),
        ),
        Magnitudes(magnitudes={"a": 6.0}),
        Rakes(rakes={"a": 110.0}),
        velocity_model_1d,
        rupture_velocity,
        Seeds(
            nshm_to_realisation_seed=1,
            rupture_propagation_seed=2,
            rupture_seed=3,
            hf_seed=4,
        ),
        metadata,
    )

    assert isinstance(config.propagation, ComputedPropagation)


def test_lone_point_source_takes_a_point_rupture(
    srf_config: SRFConfig,
    rupture_velocity: RuptureVelocity,
    velocity_model_1d: VelocityModel1D,
    metadata: RealisationMetadata,
) -> None:
    """A point source draws no fields, so it carries a rise time and no slip section.

    The rise time is derived from the moment through the generator's own relation, since
    the realisation carries the Graves & Pitarka coefficient rather than a rise time.
    """
    point = sources.Point.from_lat_lon_depth(
        np.array([-43.5, 172.6, 8000.0]),
        length_m=3000.0,
        width_m=3000.0,
        strike=64.0,
        dip=58.0,
        dip_dir=154.0,
    )
    config = rupture_input_template.build_rupture(
        SourceConfig(source_geometries={"a": point}),
        srf_config,
        RupturePropagationConfig(
            rupture_causality_tree={"a": None},
            jump_points={},
            hypocentre=np.array([0.5, 0.5]),
        ),
        Magnitudes(magnitudes={"a": 5.1}),
        Rakes(rakes={"a": 131.0}),
        velocity_model_1d,
        rupture_velocity,
        Seeds(
            nshm_to_realisation_seed=1,
            rupture_propagation_seed=2,
            rupture_seed=3,
            hf_seed=4,
        ),
        metadata,
    )

    assert isinstance(config.source, PointSourceConfig)
    assert config.source.magnitude == 5.1
    assert config.source.average_dip_deg == pytest.approx(58.0)
    assert config.source.average_rake_deg == pytest.approx(131.0)
    # Seconds, not the coefficient it was derived from.
    assert 0.0 < config.source.rise_time_s < 1.0
