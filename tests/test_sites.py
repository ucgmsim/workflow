from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow import sites


def site_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [-43.5, -43.6],
            "longitude": [172.6, 172.4],
            "vs30": [400.0, 250.0],
            "z1pt0": [0.2, 0.4],
            "z2pt5": [1.2, 2.0],
            "kappa_s": [0.045, 0.03],
            "q_frequency_exponent": [0.6, 0.5],
            "network": ["NZ", "NZ"],
        },
        index=pd.Index(["CHCH", "LINC"], name="station"),
    )


def test_round_trip_keeps_only_site_parameters(tmp_path: Path) -> None:
    sites_ffp = tmp_path / "sites.parquet"
    sites.write_sites(site_table(), sites_ffp)
    pd.testing.assert_frame_equal(
        sites.read_sites(sites_ffp), site_table()[list(sites.COLUMNS)]
    )


def test_index_must_be_the_station() -> None:
    with pytest.raises(ValueError, match="indexed by 'station'"):
        sites.validate(site_table().reset_index())


def test_stations_must_not_repeat() -> None:
    table = site_table().rename(index={"LINC": "CHCH"})
    with pytest.raises(ValueError, match=r"repeats stations: \['CHCH'\]"):
        sites.validate(table)


@pytest.mark.parametrize("column", sites.COLUMNS)
def test_every_column_is_required(column: str) -> None:
    with pytest.raises(ValueError, match=f"missing columns: \\['{column}'\\]"):
        sites.validate(site_table().drop(columns=column))


def test_every_station_needs_every_value() -> None:
    table = site_table()
    table.loc["LINC", "z1pt0"] = np.nan
    with pytest.raises(ValueError, match=r"\{'z1pt0': \['LINC'\]\}"):
        sites.validate(table)


def test_structure_ignores_missing_values() -> None:
    table = site_table()
    table.loc["LINC", "z1pt0"] = np.nan
    pd.testing.assert_frame_equal(
        sites.validate_structure(table), table[list(sites.COLUMNS)]
    )
    assert sites.missing_values(table) == {"z1pt0": ["LINC"]}


def test_missing_values_are_not_written(tmp_path: Path) -> None:
    table = site_table()
    table.loc["CHCH", "kappa_s"] = np.nan
    sites_ffp = tmp_path / "sites.parquet"
    with pytest.raises(ValueError):
        sites.write_sites(table, sites_ffp)
    assert not sites_ffp.exists()
