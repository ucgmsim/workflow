import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

import hf_simulation
from hf_simulation import PathDurationModel, Ray
from workflow.realisations import (
    HFConfig,
    HFVelocityModel1D,
    RuptureVelocity,
)
from workflow.scripts import hf_sim
from workflow.waveforms import Component


def _hf_config() -> HFConfig:
    return HFConfig(
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


def _rupture_velocity() -> RuptureVelocity:
    return RuptureVelocity(
        rvfrac=0.8,
        rvfrac_shal=0.7,
        rvfrac_deep=0.9,
        shallow_depth=1.0,
        shallow_transition_range=1,
        deep_depth=2.0,
        deep_transition_range=1,
        rvfrac_slip_sig=None,
    )


def test_build_config_mirrors_the_realisation() -> None:
    """The realisation's `hf` section reaches `hf_simulation.HfConfig` unchanged.

    The two structures mirror each other group for group, so `build_config` is a splat plus
    the two values the realisation deliberately does not carry: the record duration, which
    the domain computes, and the rupture-velocity multipliers, which live in their own
    section because SRF generation reads them too. This pins both halves of that.
    """
    hf_config = _hf_config()
    rupture_velocity = _rupture_velocity()
    # A bounding box is not needed to read one field off it.
    domain = SimpleNamespace(duration=100.0)

    config = hf_sim._build_config(hf_config, rupture_velocity, domain)  # ty: ignore[invalid-argument-type]

    # Splatted through unchanged.
    assert config.source.stress_drop_bars == 50.0
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


def test_build_hf_input_serialisation() -> None:
    """The `hf` section lands on the lines `hb_high` reads it from."""
    stoch_ffp = Path("/path/to/stoch")
    velocity_model_ffp = Path("/path/to/vmodel")
    velocity_model = HFVelocityModel1D(
        model=pd.DataFrame(
            {
                "thickness": [1.0],
                "Vp": [3.0],
                "Vs": [1.5],
                "rho": [2.2],
                "Qp": [100.0],
                "Qs": [50.0],
            }
        ),
        vs_moho=3.5,
    )

    lines = hf_sim.build_hf_input(
        _hf_config(),
        _rupture_velocity(),
        velocity_model,
        100.0,
        stoch_ffp,
        velocity_model_ffp,
    ).split("\n")

    assert lines[1] == "50.0"  # stress drop
    assert lines[2] == "{station_input_file}"
    assert lines[3] == "{output_file}"
    assert lines[4] == "2 1 2"  # ray count, then rays
    assert lines[5] == "1"  # site amplification on
    assert lines[7] == "{seed}"
    assert lines[9] == "100.0 0.005 20.0 0.045 0.6"  # duration, dt, fmax, kappa, qfexp
    assert lines[10] == "0.8 0.7 0.9 2.5 0.1"  # rupture velocity, czero, calpha
    assert lines[12] == str(stoch_ffp)
    assert lines[13] == str(velocity_model_ffp)
    assert lines[14] == "3.5"  # vs_moho
    assert lines[17] == "0.0 0.0 0.1"  # rv_sig1 is the rupture velocity sigma
    assert lines[18] == "11"  # path duration model
    assert lines[20] == "-1 -1 -1"  # stress parameter adjustment off
    assert len(lines) == 23


def test_fortran_station_seeds() -> None:
    seed = hf_sim.fortran_station_seeds(0, ["station"])
    assert seed.dtype == np.int32
    assert seed.shape == (1,)
    assert seed.item() == hf_sim.fortran_station_seeds(0, ["station"]).item()


@given(
    seed=st.integers(min_value=0, max_value=(1 << 31) - 1),
    stations=st.lists(STATION_STRATEGY, min_size=1, unique=True),
)
def test_fortran_station_seeds_on_name_only(seed: int, stations: list[str]) -> None:
    station_seeds = hf_sim.fortran_station_seeds(seed, stations)

    reordered_station_seeds = hf_sim.fortran_station_seeds(seed, stations[::-1])
    assert (reordered_station_seeds[::-1] == station_seeds).all()

    for station, expected_seed in zip(stations, station_seeds):
        assert hf_sim.fortran_station_seeds(seed, [station]).item() == expected_seed


FAKE_HB_HIGH = """\
import sys

import numpy as np

lines = sys.stdin.read().split("\\n")
nt = int(float(lines[9].split()[0]) / float(lines[9].split()[1]))
# hb_high writes (nt, 3) float32 columns: 090, 000, ver.
waveform = np.empty((nt, 3), dtype=np.float32)
waveform[:] = [90.0, 0.0, 1.0]
waveform[:, 2] = int(lines[7])
waveform.tofile(lines[3])
"""


@pytest.fixture
def fake_hb_high(tmp_path: Path) -> Path:
    """An executable that answers an `hb_high` deck the way `hb_high` does.

    Its 090 column is 90, its 000 column is 0, and its ver column is the station seed.
    """
    script = tmp_path / "hb_high"
    script.write_text(f"#!{sys.executable}\n{FAKE_HB_HIGH}")
    script.chmod(0o755)
    return script


def test_fortran_chunk_stores_components_in_workflow_order(
    fake_hb_high: Path,
) -> None:
    """`hb_high`'s 090/000/ver columns are stored under their labels, in `Component` order."""
    nt = 10
    time = np.arange(nt) * 0.005
    hf_input = hf_sim.build_hf_input(
        _hf_config(),
        _rupture_velocity(),
        HFVelocityModel1D(model=pd.DataFrame(), vs_moho=999.9),
        nt * 0.005,
        Path("stoch"),
        Path("vmodel"),
    )
    stations = xr.Dataset(
        {
            "latitude": ("station", [-43.5, -43.6]),
            "longitude": ("station", [172.6, 172.7]),
            "seed": ("station", np.array([7, -3], dtype=np.int32)),
        },
        coords={"station": ["AAAA", "BBBB"]},
    )

    waveform = hf_sim._simulate_chunk_fortran(
        stations, time, hf_sim_path=fake_hb_high, hf_input=hf_input
    )

    assert list(waveform["component"].values) == list(Component)
    assert waveform.dims == ("component", "station", "time")
    assert (waveform.sel(component=Component.EAST) == 90.0).all()
    assert (waveform.sel(component=Component.NORTH) == 0.0).all()
    assert (waveform.sel(component=Component.UP, station="AAAA") == 7).all()
    assert (waveform.sel(component=Component.UP, station="BBBB") == -3).all()


def test_fortran_chunk_rejects_short_output(fake_hb_high: Path) -> None:
    """A record of the wrong length is an error, not a silently misaligned waveform."""
    hf_input = hf_sim.build_hf_input(
        _hf_config(),
        _rupture_velocity(),
        HFVelocityModel1D(model=pd.DataFrame(), vs_moho=999.9),
        10 * 0.005,
        Path("stoch"),
        Path("vmodel"),
    )
    stations = xr.Dataset(
        {
            "latitude": ("station", [-43.5]),
            "longitude": ("station", [172.6]),
            "seed": ("station", np.array([1], dtype=np.int32)),
        },
        coords={"station": ["AAAA"]},
    )

    with pytest.raises(RuntimeError, match="expected 11"):
        hf_sim._simulate_chunk_fortran(
            stations,
            np.arange(11) * 0.005,
            hf_sim_path=fake_hb_high,
            hf_input=hf_input,
        )
