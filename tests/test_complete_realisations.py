"""Tests for the complete_realisations campaign tool."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import complete_realisations as cr

DATA = Path(__file__).parent / "data"
FELIPE = DATA / "felipe_reference_realisation.json"
SAMPLE = DATA / "minimal_realisation_sample.json"
BROKEN = DATA / "broken_minimal_stub.json"
FELIPE_SCRIPTS = Path(__file__).parents[1] / "felipe_scripts"


def test_load_overrides_shapes_and_dtypes() -> None:
    overrides = cr.load_overrides(FELIPE_SCRIPTS, vm_version="2.09")
    assert overrides.vm_version == "2.09"
    assert overrides.rrup_interpolants.shape == (2, 29)
    assert overrides.rrup_interpolants.dtype == np.float32
    assert overrides.valid_periods.shape == (111,)
    assert overrides.fas_frequencies.shape == (389,)


def test_load_overrides_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cr.load_overrides(tmp_path)


def test_is_valid_minimal_true_for_full_stub() -> None:
    assert cr.is_valid_minimal(json.loads(SAMPLE.read_text())) is True


def test_is_valid_minimal_false_for_broken_stub() -> None:
    assert cr.is_valid_minimal(json.loads(BROKEN.read_text())) is False


def test_normalize_key_order_matches_canonical() -> None:
    scrambled = {"bb": 1, "sources": 2, "metadata": 3, "surprise": 4, "domain": 5}
    assert list(cr.normalize_key_order(scrambled)) == [
        "metadata",
        "sources",
        "domain",
        "bb",
        "surprise",
    ]


@pytest.mark.slow
def test_complete_one_produces_full_realisation(tmp_path: Path) -> None:
    overrides = cr.load_overrides(FELIPE_SCRIPTS)
    dst = tmp_path / "completed.json"

    cr.complete_one(SAMPLE, dst, DefaultsVersion.v24_2_2_1, overrides)
    completed = json.loads(dst.read_text())

    # Exactly the 18 canonical sections, in canonical order.
    assert list(completed) == cr.FELIPE_SECTION_ORDER
    # Domain computed and sane.
    assert completed["domain"]["depth"] > 0
    assert completed["domain"]["duration"] > 0
    # Overrides applied.
    assert completed["velocity_model"]["version"] == "2.09"
    assert len(completed["im"]["valid_periods"]) == 111
    assert len(completed["im"]["fas_frequencies"]) == 389
    assert completed["metadata"]["defaults_version"] == "24.2.2.1"
    # Rupture-independent sections identical to Felipe's reference.
    felipe = json.loads(FELIPE.read_text())
    for section in [
        "velocity_model",
        "im",
        "emod3d",
        "hf",
        "bb",
        "resolution",
        "srf",
        "velocity_model_1d",
        "hf_velocity_model_1d",
    ]:
        assert completed[section] == felipe[section], f"{section} differs from Felipe"


def test_complete_one_does_not_touch_source(tmp_path: Path) -> None:
    # Guard the read-only-inputs constraint: complete_one must copy, never edit src.
    src = tmp_path / "realisation_114741.json"
    shutil.copy(SAMPLE, src)
    before = src.read_bytes()
    dst = tmp_path / "completed.json"
    try:
        cr.complete_one(
            src, dst, DefaultsVersion.v24_2_2_1, cr.load_overrides(FELIPE_SCRIPTS)
        )
    except Exception:  # noqa: BLE001 -- even on failure src must be untouched
        pass
    assert src.read_bytes() == before


def test_summary_row_fields() -> None:
    felipe = json.loads(FELIPE.read_text())
    row = cr.summary_row(felipe, "3528839")
    assert row["rupture_id"] == "3528839"
    assert row["n_faults"] == 1
    assert row["fault_names"] == "3528839"
    assert row["vm_version"] == "2.09"
    assert row["n_valid_periods"] == 111
    assert row["n_fas_frequencies"] == 389
    assert isinstance(row["total_magnitude_mw"], float)
    assert row["domain_depth_km"] == felipe["domain"]["depth"]


@pytest.mark.slow
def test_complete_realisations_end_to_end(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    shutil.copy(SAMPLE, input_dir / "realisation_114741.json")
    shutil.copy(BROKEN, input_dir / "realisation_59421.json")

    cr.complete_realisations(
        input_dir,
        output_dir,
        defaults_version=DefaultsVersion.v24_2_2_1,
        felipe_scripts_dir=FELIPE_SCRIPTS,
        vm_version="2.09",
        workers=1,
    )

    completed = output_dir / "realisation_114741.json"
    assert completed.exists()
    assert list(json.loads(completed.read_text())) == cr.FELIPE_SECTION_ORDER
    # Broken stub skipped, not written.
    assert not (output_dir / "realisation_59421.json").exists()
    # Reports written.
    assert (output_dir / "completion_summary.csv").exists()
    assert "114741" in (output_dir / "completion_summary.csv").read_text()
    assert "59421" in (output_dir / "error_log.txt").read_text()
