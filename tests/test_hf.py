from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from workflow.realisations import (
    HFConfig,
    Resolution,
    RuptureVelocity,
)
from workflow.scripts import hf_sim


def test_build_hf_input_serialisation() -> None:
    stoch_ffp = Path("/path/to/stoch")
    velocity_model = Path("/path/to/vmodel")

    hf_config = HFConfig(
        sdrop=50.0,
        rayset=[1, 2],
        no_siteamp=False,
        nbu=1,
        ift=0,
        flo=0.1,
        fhi=10.0,
        fmax=20.0,
        kappa=0.045,
        qfexp=0.6,
        czero=2.5,
        calpha=0.0,
        mom=None,
        rupv=1.2,
        vs_moho=3.5,
        nl_skip=0,
        vp_sig=0.1,
        vsh_sig=0.1,
        rho_sig=0.1,
        qs_sig=0.1,
        ic_flag=True,
        velocity_name="test_vel",
        fa_sig1=0.2,
        fa_sig2=0.2,
        rv_sig1=0.1,
        path_dur=11,
        t_sec=0.0,
        site_specific=False,
        dpath_pert=0,
        stoch_dx=2.0,
        stoch_dy=2.0,
        stress_parameter_adjustment_fault_area=None,
        stress_parameter_adjustment_target_magnitude=None,
        stress_parameter_adjustment_tect_type=0,
    )

    res = Resolution(resolution=0.1)
    rv = RuptureVelocity(
        rvfrac=0.8,
        rvfrac_shal=0.7,
        rvfrac_deep=0.9,
        shallow_depth=1.0,
        shallow_transition_range=1,
        deep_depth=2.0,
        deep_transition_range=1,
        rvfrac_slip_sig=None
    )
    # Rather than create DomainParameters with a bounding box, we simplify with a mock object
    domain = SimpleNamespace(duration=100.0)

    result = hf_sim.build_hf_input(
        stoch_ffp,
        velocity_model,
        res,
        hf_config,
        rv,
        domain,  # type: ignore[invalid-argument-type]
    )

    lines = result.split("\n")

    assert lines[1] == "50.0"  # sdrop
    assert lines[2] == "{station_input_file}"  # placeholder for station file
    assert lines[3] == "{output_file}"  # placeholder for output file
    assert lines[4] == "2 1 2"  # rayset count + rays
    assert lines[5] == "1"  # int(not no_siteamp) -> int(not False) -> 1
    assert lines[7] == "{seed}"  # seed placeholder
    assert lines[9] == "100.0 0.005 20.0 0.045 0.6"  # Domain and resolution parameters
    assert lines[10] == "0.8 0.7 0.9 2.5 0.0"  # rupture velocity + czero,alpha
    assert lines[11] == "0.0 2.0 1.0 3.0"  # shallow depth, deep depth
    assert lines[12] == "-1 1.2"  # mom (None -> -1) and rupv
    assert lines[13] == str(stoch_ffp)  # Stoch file path
    assert lines[16] == "0 0.1 0.1 0.1 0.1 1"  # Sigs and ic_flag (True -> 1)
    assert lines[21] == "-1 -1 -1"  # Optional stress parameters


STATION_STRATEGY = st.text(
    min_size=0, max_size=8, alphabet=st.characters(codec="ascii")
)


@given(station=STATION_STRATEGY)
def test_stable_hash(station: str) -> None:
    # Check that stable_hash output is always a valid 32-bit integer
    assert -(1 << 31) <= hf_sim.stable_hash(station) <= (1 << 31) - 1


def test_station_seeds() -> None:
    seed = hf_sim.station_seeds(0, ["station"])
    assert seed.dtype == np.int32
    assert seed.shape == (1,)
    # Seeds should be referentially transparent: i.e. depend only on the seed and station name
    seed_1 = hf_sim.station_seeds(0, ["station"])
    assert seed.item() == seed_1.item()


@given(
    seed=st.integers(min_value=-(1 << 31), max_value=(1 << 31) - 1),
    stations=st.lists(STATION_STRATEGY, min_size=1, unique=True),
)
def test_station_seeds_on_name_only(seed: int, stations: list[str]) -> None:
    station_seeds = hf_sim.station_seeds(seed, stations)

    # check that station hashes depend on name only and not the order that the stations are supplied in
    reordered_station_seeds = hf_sim.station_seeds(seed, stations[::-1])
    assert (reordered_station_seeds[::-1] == station_seeds).all()

    # Check the subset property: If we hash the station seed on its own, the station seed remains the same.
    # Note a station seed that was derived from stations order in a
    # sorted list of stations would pass the first test, but not this
    # one.
    for station, expected_seed in zip(stations, station_seeds):
        assert hf_sim.station_seeds(seed, [station]).item() == expected_seed


def test_create_hf_dataset_structure() -> None:
    # 1. Setup Mock Data
    n_stations = 2
    n_components = 3  # Fixed by function logic
    n_time = 100

    names = ["station_a", "station_b"]
    waveform = np.random.rand(n_components, n_stations, n_time).astype(np.float32)
    lat = np.array([-43.5, -43.6])
    lon = np.array([172.6, 172.7])
    dist = np.array([10.5, 20.1])
    seeds = np.array([123, 456])
    vrefs = np.array([300.0, 350.0])
    dt = 0.02
    start_sec = 0.0

    ds = hf_sim.create_hf_dataset(
        waveform=waveform,
        latitude=lat,
        longitude=lon,
        names=names,
        epicentre_distance=dist,
        seed=seeds,
        vref=vrefs,
        dt=dt,
        start_sec=start_sec,
    )

    assert ds.sizes == {"component": 3, "station": 2, "time": 100}

    np.testing.assert_array_equal(ds.station.values, names)
    np.testing.assert_array_equal(ds.component.values, ["x", "y", "z"])
    assert ds.time.values[1] == pytest.approx(0.02)
    assert ds.lat.dims == ("station",)
    assert ds.lon.dims == ("station",)

    assert "waveform" in ds.data_vars
    assert ds.waveform.dims == ("component", "station", "time")
    np.testing.assert_allclose(ds.waveform.values, waveform)

    assert ds.attrs["dt"] == dt
    assert ds.attrs["nt"] == n_time
    assert ds.attrs["units"] == "cm/s^2"
