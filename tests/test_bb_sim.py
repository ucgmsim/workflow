import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from site_calculation import amplification
from workflow import sites
from workflow.defaults import DefaultsVersion
from workflow.realisations import BroadbandParameters, RealisationMetadata
from workflow.scripts import bb_sim, im_calc
from workflow.waveforms import Component, time_axis

STATIONS = ["CHCH", "RICC"]


def site_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [-43.5, -43.52],
            "longitude": [172.5, 172.53],
            "vs30": [400.0, 250.0],
            "z1pt0": [0.2, 0.4],
            "z2pt5": [1.2, 2.0],
            "kappa_s": [0.045, 0.03],
            "q_frequency_exponent": [0.6, 0.5],
        },
        index=pd.Index(STATIONS, name="station"),
    )


def waveforms(dt: float, nt: int, seed: int) -> xr.Dataset:
    """Random waveforms for the stations, in the layout hf-sim and lf-to-xarray write."""
    rng = np.random.default_rng(seed)
    table = site_table()
    return xr.Dataset(
        {
            "waveform": (
                ("component", "station", "time"),
                rng.normal(size=(len(Component), len(STATIONS), nt)).astype(np.float32),
            ),
            "vref": ("station", [500.0] * len(STATIONS)),
        },
        coords={
            "component": list(Component),
            "station": STATIONS,
            "time": np.arange(nt) * dt,
            "latitude": ("station", table["latitude"].to_numpy()),
            "longitude": ("station", table["longitude"].to_numpy()),
        },
        attrs={"dt": dt, "start_sec": 0.0, "nt": nt},
    )


def test_site_parameters_reach_im_calc(tmp_path: Path) -> None:
    """im-calc reads back the site parameters bb-sim copies from the site table."""
    realisation = tmp_path / "realisation.json"
    RealisationMetadata(
        name="Test_REL01", version="1", defaults_version=DefaultsVersion.v24_2_2_1
    ).write_to_realisation(realisation)
    sites_ffp = tmp_path / "stations.parquet"
    sites.write_sites(site_table(), sites_ffp)
    lf_ffp = tmp_path / "realisation.lf"
    waveforms(dt=0.02, nt=500, seed=1).drop_vars("vref").to_netcdf(
        lf_ffp, engine="h5netcdf"
    )
    hf_ffp = tmp_path / "realisation.hf"
    waveforms(dt=0.01, nt=1000, seed=2).to_netcdf(hf_ffp, engine="h5netcdf")
    bb_ffp = tmp_path / "realisation.bb"

    bb_sim.combine_hf_and_lf(realisation, sites_ffp, lf_ffp, hf_ffp, bb_ffp)

    with xr.open_dataset(bb_ffp) as broadband:
        site_parameters = im_calc.read_site_parameters(broadband)
    for name, values in site_parameters.as_dict().items():
        np.testing.assert_allclose(
            values.sel(station=STATIONS).values,
            site_table()[name].to_numpy(),
            rtol=1e-6,
            err_msg=name,
        )


def test_an_event_time_stamps_the_broadband_time_axis(tmp_path: Path) -> None:
    """Timestamped LF and HF inputs give the same broadband, on a timestamped axis."""
    event_time = datetime.datetime(2010, 9, 3, 16, 35, 46, tzinfo=datetime.UTC)
    sites_ffp = tmp_path / "stations.parquet"
    sites.write_sites(site_table(), sites_ffp)

    def broadband(
        name: str, metadata_event_time: datetime.datetime | None
    ) -> xr.Dataset:
        directory = tmp_path / name
        directory.mkdir()
        realisation = directory / "realisation.json"
        RealisationMetadata(
            name="Test_REL01",
            version="1",
            defaults_version=DefaultsVersion.v24_2_2_1,
            event_time=metadata_event_time,
        ).write_to_realisation(realisation)
        lf = waveforms(dt=0.02, nt=500, seed=1).drop_vars("vref")
        hf = waveforms(dt=0.01, nt=1000, seed=2)
        for dataset, ffp in ((lf, directory / "lf.nc"), (hf, directory / "hf.nc")):
            dataset.assign_coords(
                time=time_axis(dataset["time"].values, metadata_event_time)
            ).to_netcdf(ffp, engine="h5netcdf")
        bb_ffp = directory / "bb.nc"
        bb_sim.combine_hf_and_lf(
            realisation, sites_ffp, directory / "lf.nc", directory / "hf.nc", bb_ffp
        )
        with xr.open_dataset(bb_ffp) as dataset:
            return dataset.load()

    seconds = broadband("seconds", None)
    stamped = broadband("stamped", event_time)

    np.testing.assert_array_equal(
        stamped["time"].values, time_axis(seconds["time"].values, event_time)
    )
    np.testing.assert_array_equal(
        stamped["waveform"].values, seconds["waveform"].values
    )


def onset_sine(frequency: float, time: np.ndarray, start_sec: float) -> np.ndarray:
    """A sine that rises smoothly from rest at `start_sec`, as a simulated record does."""
    onset = 1 - np.exp(-(((time - start_sec) / 2.0) ** 2))
    return onset * np.sin(2 * np.pi * frequency * time)


def sine_dataset(frequency: float, dt: float, nt: int, start_sec: float) -> xr.Dataset:
    """A single-station, single-component `onset_sine` starting at `start_sec`."""
    time = start_sec + np.arange(nt) * dt
    return xr.Dataset(
        {
            "waveform": (
                ("component", "station", "time"),
                onset_sine(frequency, time, start_sec)[np.newaxis, np.newaxis, :],
            )
        },
        coords={"component": ["000"], "station": ["CHCH"], "time": time},
        attrs={"dt": dt, "start_sec": start_sec, "nt": nt},
    ).chunk(time=-1)


def test_resampling_keeps_the_signal_and_its_start_time() -> None:
    """Resampling a sine spans the input's time range and matches it before the taper."""
    frequency, old_dt, dt, nt, start_sec = 0.5, 0.02, 0.005, 2000, -3.0

    resampled = bb_sim._resample_signal(
        sine_dataset(frequency, old_dt, nt, start_sec), dt
    ).compute()

    time = resampled["time"].values
    expected_nt = round((nt - 1) * old_dt / dt) + 1
    assert resampled.sizes["time"] == expected_nt
    assert resampled.attrs["dt"] == dt
    np.testing.assert_allclose(time, start_sec + np.arange(expected_nt) * dt)
    np.testing.assert_allclose(time[-1], start_sec + (nt - 1) * old_dt)
    # Outside the tapered tail the resampled signal is the analytic one from
    # the first sample on. The record starts from rest and the taper brings the
    # tail to rest, so the periodic FFT wraps smoothly from end to start. A
    # record that starts abruptly (non-zero value or slope) would ring at its
    # start under any band-limited resampler, wrap or no wrap.
    core = slice(0, int(expected_nt * (1 - bb_sim.TAIL_TAPER_FRACTION)))
    np.testing.assert_allclose(
        resampled["waveform"].values[0, 0, core],
        onset_sine(frequency, time[core], start_sec),
        atol=1e-5,
    )


def test_alignment_pads_each_leg_at_its_original_times() -> None:
    """LF and HF with different starts and lengths share an axis spanning both."""
    dt = 0.01
    lf = sine_dataset(1.0, dt, nt=100, start_sec=-3.0)
    hf = sine_dataset(2.0, dt, nt=500, start_sec=0.0)

    lf_aligned, hf_aligned = (
        waveform.compute() for waveform in bb_sim._align_datasets(lf, hf, dt)
    )

    # From the LF start at -3 s to the HF end at 4.99 s.
    common_nt = 800
    common_time = -3.0 + np.arange(common_nt) * dt
    for aligned in (lf_aligned, hf_aligned):
        assert aligned.sizes["time"] == common_nt
        np.testing.assert_allclose(aligned["time"].values, common_time)
    # The LF occupies the first 100 samples, the HF the last 500.
    np.testing.assert_array_equal(lf_aligned.values[..., :100], lf["waveform"].values)
    np.testing.assert_array_equal(lf_aligned.values[..., 100:], 0.0)
    np.testing.assert_array_equal(hf_aligned.values[..., :300], 0.0)
    np.testing.assert_array_equal(hf_aligned.values[..., 300:], hf["waveform"].values)


BB_CONFIG = BroadbandParameters.read_from_defaults(DefaultsVersion.v24_2_2_1)
BB_DT = 0.01
BB_NT = 6000


def broadband_chunk(
    lf: np.ndarray,
    hf: np.ndarray,
    filter_legs: bb_sim.FilterLeg,
    pga: float = 100.0,
) -> np.ndarray:
    """The broadband of one station and component from LF and HF in cm/s^2."""
    coords = {
        "component": ["000"],
        "station": ["CHCH"],
        "time": np.arange(BB_NT) * BB_DT,
    }
    dims = ("component", "station", "time")
    station = {"station": ["CHCH"]}
    bb = bb_sim._process_bb_chunk(
        xr.DataArray(lf[np.newaxis, np.newaxis, :], coords=coords, dims=dims),
        xr.DataArray(hf[np.newaxis, np.newaxis, :], coords=coords, dims=dims),
        xr.DataArray([400.0], coords=station, dims="station"),
        xr.DataArray([500.0], coords=station, dims="station"),
        xr.DataArray(
            [[pga]],
            coords={"component": ["000"]} | station,
            dims=("component", "station"),
        ),
        dt=BB_DT,
        config=BB_CONFIG,
        filter_legs=filter_legs,
    )
    return bb.values[0, 0]


@pytest.mark.parametrize("filter_legs", list(bb_sim.FilterLeg))
def test_filter_legs_filter_only_the_chosen_legs(
    filter_legs: bb_sim.FilterLeg,
) -> None:
    """Each `--filter` choice low-passes the LF and high-passes the HF only as asked."""
    rng = np.random.default_rng(3)
    lf = rng.normal(size=BB_NT)
    hf = rng.normal(size=BB_NT)
    zero = np.zeros(BB_NT)

    lf_only = broadband_chunk(lf, zero, filter_legs)
    hf_only = broadband_chunk(zero, hf, filter_legs)
    amplified_hf = broadband_chunk(zero, hf, bb_sim.FilterLeg.NONE).astype(np.float64)

    # The output is in g, from LF and HF in cm/s^2.
    expected_lf = lf * bb_sim.G
    if filter_legs in (bb_sim.FilterLeg.LF, bb_sim.FilterLeg.BOTH):
        expected_lf = amplification.bwfilter(
            expected_lf, BB_DT, BB_CONFIG.flo, amplification.Band.LOWPASS
        )
    expected_hf = amplified_hf
    if filter_legs in (bb_sim.FilterLeg.HF, bb_sim.FilterLeg.BOTH):
        expected_hf = amplification.bwfilter(
            expected_hf, BB_DT, BB_CONFIG.flo, amplification.Band.HIGHPASS
        )
    np.testing.assert_allclose(lf_only, expected_lf, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(hf_only, expected_hf, rtol=1e-4, atol=1e-7)
    # The broadband is the sum of the two legs.
    np.testing.assert_allclose(
        broadband_chunk(lf, hf, filter_legs), lf_only + hf_only, rtol=1e-4, atol=1e-7
    )


def test_both_filters_pass_a_low_frequency_lf() -> None:
    """An LF well below the merge frequency passes the low-pass mostly unchanged."""
    time = np.arange(BB_NT) * BB_DT
    lf = np.sin(2 * np.pi * 0.1 * time)

    bb = broadband_chunk(lf, np.zeros(BB_NT), bb_sim.FilterLeg.BOTH)

    # Away from the edges of the record.
    core = slice(BB_NT // 4, 3 * BB_NT // 4)
    np.testing.assert_allclose(bb[core], lf[core] * bb_sim.G, atol=0.02 * bb_sim.G)


def test_the_amplification_model_gets_the_hf_pga_in_g(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The site amplification model gets the HF PGA converted from cm/s^2 to g."""
    amp_model_fn, amp_model_freqs = bb_sim.SITE_AMP_MODELS[BB_CONFIG.site_amp_version]
    pgas = []

    def recording_amp_model(
        vs30: np.ndarray, vs30_sim: np.ndarray, pga: np.ndarray
    ) -> np.ndarray:
        pgas.append(pga)
        return amp_model_fn(vs30, vs30_sim, pga)

    monkeypatch.setitem(
        bb_sim.SITE_AMP_MODELS,
        BB_CONFIG.site_amp_version,
        (recording_amp_model, amp_model_freqs),
    )

    broadband_chunk(np.zeros(BB_NT), np.zeros(BB_NT), bb_sim.FilterLeg.BOTH, pga=490.5)

    assert len(pgas) == 1
    np.testing.assert_allclose(pgas[0], [0.5])
