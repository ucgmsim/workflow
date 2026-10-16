from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from workflow import sites
from workflow.defaults import DefaultsVersion
from workflow.realisations import RealisationMetadata
from workflow.scripts import bb_sim, im_calc
from workflow.waveforms import Component

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
