from types import SimpleNamespace

import numpy as np
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

import hf_simulation
from hf_simulation import (
    FaultSegment,
    HfConfig,
    PathDurationModel,
    Ray,
    RecordParameters,
    Simulator,
    SlipModel,
    VelocityModel1D,
)
from workflow.realisations import (
    HFConfig,
    RuptureVelocity,
)
from workflow.scripts import hf_sim


def test_build_config_mirrors_the_realisation() -> None:
    """The realisation's `hf` section reaches `hf_simulation.HfConfig` unchanged.

    The two structures mirror each other group for group, so `build_config` is a splat plus
    the two values the realisation deliberately does not carry: the record duration, which
    the domain computes, and the rupture-velocity multipliers, which live in their own
    section because SRF generation reads them too. The three per-subfault and per-station
    values are left out, to be broadcast elsewhere. This pins all three parts of that.
    """
    hf_config = HFConfig(
        source={
            "stress_drop_bars": 50.0,
            "corner_frequency_constant": 2.5,
            "corner_frequency_alpha": 0.1,
            "rupture_velocity": {"sigma": 0.1},
        },
        path={"rayset": [1, 2], "q_frequency_exponent": 0.6, "path_duration_model": 11},
        site={"kappa_s": 0.045, "fmax_hz": 20.0},
        record={"dt": 0.005},
    )
    rupture_velocity = RuptureVelocity(
        rvfrac=0.8,
        rvfrac_shal=0.7,
        rvfrac_deep=0.9,
        shallow_depth=1.0,
        shallow_transition_range=1,
        deep_depth=2.0,
        deep_transition_range=1,
        rvfrac_slip_sig=None,
    )
    # A bounding box is not needed to read one field off it.
    domain = SimpleNamespace(duration=100.0)

    config = hf_sim._build_config(hf_config, rupture_velocity, domain)  # ty: ignore[invalid-argument-type]

    # Splatted through unchanged.
    assert config.source.corner_frequency_constant == 2.5
    assert config.site.fmax_hz == 20.0
    assert config.record.dt == 0.005
    # Ints become the enums the simulation takes.
    assert config.path.rayset == (Ray.DIRECT, Ray.MOHO_REFLECTION)
    assert config.path.path_duration_model is PathDurationModel.BOORE_THOMPSON_2014
    # Injected, because the `hf` section does not carry them.
    assert config.record.duration_s == 100.0
    assert config.source.rupture_velocity.fraction == 0.8
    assert config.source.rupture_velocity.shallow == 0.7
    assert config.source.rupture_velocity.deep == 0.9
    # ... but the sigma does come from the `hf` section.
    assert config.source.rupture_velocity.sigma == 0.1
    # Left out: hf-sim broadcasts them per subfault and per station instead.
    assert not hasattr(config.source, "stress_drop_bars")
    assert not hasattr(config.path, "q_frequency_exponent")
    assert not hasattr(config.site, "kappa_s")


def test_simulate_chunk_broadcasts_kappa_and_q_to_every_station() -> None:
    """A block simulated with run-wide kappa and Q equals passing them per station."""
    segment = FaultSegment(
        longitude_deg=173.0,
        latitude_deg=-43.0,
        strike_deg=220.0,
        dip_deg=70.0,
        rake_deg=160.0,
        top_depth_km=0.0,
        subfault_length_km=1.5,
        subfault_width_km=1.5,
        hypocentre_along_strike_km=0.0,
        hypocentre_down_dip_km=1.5,
        slip=np.full((3, 4), 50.0, np.float32),
        rise_time_s=np.full((3, 4), 0.5, np.float32),
        rupture_time_s=np.zeros((3, 4), np.float32),
        stress_drop_bars=np.full((3, 4), 50.0, np.float32),
    )
    simulator = Simulator(
        SlipModel([segment]),
        VelocityModel1D(
            thickness_km=np.array([1.0, 2.0, 0.0], np.float32),
            vp_km_s=np.array([3.0, 5.0, 6.0]),
            vsh_km_s=np.array([1.5, 2.8, 3.5]),
            density_g_cm3=np.array([2.1, 2.5, 2.7]),
            quality_factor_p=np.array([100.0, 200.0, 400.0], np.float32),
            quality_factor_s=np.array([50.0, 100.0, 200.0], np.float32),
            vs_moho_km_s=999.9,
        ),
        HfConfig(record=RecordParameters(duration_s=40.0)),
    )
    names = ["CHCH", "LINC"]
    latitude = np.array([-43.4, -43.6], np.float32)
    longitude = np.array([172.6, 172.4], np.float32)
    seeds = hf_simulation.station_seeds(1234, names)
    chunk = xr.Dataset(
        {
            "latitude": ("station", latitude),
            "longitude": ("station", longitude),
            "seed": ("station", seeds),
        },
        coords={"station": names},
    )
    # The default 0.005 s sample interval over the 40 s record.
    time = np.arange(8000) * 0.005

    broadcast = hf_sim._simulate_chunk(
        chunk, time, simulator, kappa_s=0.045, q_frequency_exponent=0.6
    )
    direct = simulator.run_stations(
        latitude_deg=latitude,
        longitude_deg=longitude,
        station_seed=seeds,
        kappa_s=np.full(2, 0.045, np.float32),
        q_frequency_exponent=np.full(2, 0.6, np.float32),
    )
    assert np.abs(direct).max() > 0
    np.testing.assert_array_equal(
        broadcast.sel(component=list(hf_simulation.COMPONENTS)).values, direct
    )


STATION_STRATEGY = st.text(
    min_size=0, max_size=8, alphabet=st.characters(codec="ascii")
)


def test_station_seeds() -> None:
    seed = hf_simulation.station_seeds(0, ["station"])
    assert seed.dtype == np.uint64
    assert seed.shape == (1,)
    # Seeds should be referentially transparent: i.e. depend only on the seed and station name
    seed_1 = hf_simulation.station_seeds(0, ["station"])
    assert seed.item() == seed_1.item()


@given(
    # Non-negative: SeedSequence rejects negative entropy, which `station_seeds` says.
    seed=st.integers(min_value=0, max_value=(1 << 31) - 1),
    stations=st.lists(STATION_STRATEGY, min_size=1, unique=True),
)
def test_station_seeds_on_name_only(seed: int, stations: list[str]) -> None:
    station_seeds = hf_simulation.station_seeds(seed, stations)

    # check that station hashes depend on name only and not the order that the stations are supplied in
    reordered_station_seeds = hf_simulation.station_seeds(seed, stations[::-1])
    assert (reordered_station_seeds[::-1] == station_seeds).all()

    # Check the subset property: If we hash the station seed on its own, the station seed remains the same.
    # Note a station seed that was derived from stations order in a
    # sorted list of stations would pass the first test, but not this
    # one.
    for station, expected_seed in zip(stations, station_seeds):
        assert hf_simulation.station_seeds(seed, [station]).item() == expected_seed
