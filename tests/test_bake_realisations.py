"""Tests for the bake_realisations campaign tool."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import bake_realisations as br

DATA = Path(__file__).parent / "data"
FELIPE = DATA / "felipe_reference_realisation.json"
SAMPLE = DATA / "minimal_realisation_sample.json"
BROKEN = DATA / "broken_minimal_stub.json"
FELIPE_SCRIPTS = Path(__file__).parents[1] / "felipe_scripts"


def test_load_overrides_shapes_and_dtypes() -> None:
    overrides = br.load_overrides(FELIPE_SCRIPTS, vm_version="2.09")
    assert overrides.vm_version == "2.09"
    assert overrides.rrup_interpolants.shape == (2, 29)
    assert overrides.rrup_interpolants.dtype == np.float32
    assert overrides.valid_periods.shape == (111,)
    assert overrides.fas_frequencies.shape == (389,)


def test_load_overrides_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        br.load_overrides(tmp_path)


def test_is_valid_minimal_true_for_full_stub() -> None:
    assert br.is_valid_minimal(json.loads(SAMPLE.read_text())) is True


def test_is_valid_minimal_false_for_broken_stub() -> None:
    assert br.is_valid_minimal(json.loads(BROKEN.read_text())) is False


def test_normalize_key_order_matches_canonical() -> None:
    scrambled = {"bb": 1, "sources": 2, "metadata": 3, "surprise": 4, "domain": 5}
    assert list(br.normalize_key_order(scrambled)) == [
        "metadata",
        "sources",
        "domain",
        "bb",
        "surprise",
    ]


@pytest.mark.slow
def test_bake_one_produces_complete_realisation(tmp_path: Path) -> None:
    overrides = br.load_overrides(FELIPE_SCRIPTS)
    dst = tmp_path / "baked.json"

    br.bake_one(SAMPLE, dst, DefaultsVersion.v24_2_2_1, overrides)
    baked = json.loads(dst.read_text())

    # Exactly the 18 canonical sections, in canonical order.
    assert list(baked) == br.FELIPE_SECTION_ORDER
    # Domain computed and sane.
    assert baked["domain"]["depth"] > 0
    assert baked["domain"]["duration"] > 0
    # Overrides applied.
    assert baked["velocity_model"]["version"] == "2.09"
    assert len(baked["im"]["valid_periods"]) == 111
    assert len(baked["im"]["fas_frequencies"]) == 389
    assert baked["metadata"]["defaults_version"] == "24.2.2.1"
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
        assert baked[section] == felipe[section], f"{section} differs from Felipe"


def test_bake_one_does_not_touch_source(tmp_path: Path) -> None:
    # Guard the read-only-inputs constraint: bake_one must copy, never edit src.
    src = tmp_path / "realisation_114741.json"
    shutil.copy(SAMPLE, src)
    before = src.read_bytes()
    dst = tmp_path / "baked.json"
    try:
        br.bake_one(
            src, dst, DefaultsVersion.v24_2_2_1, br.load_overrides(FELIPE_SCRIPTS)
        )
    except Exception:  # noqa: BLE001 -- even on failure src must be untouched
        pass
    assert src.read_bytes() == before
