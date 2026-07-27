"""Tests for the generate_realisations_from_csv campaign driver."""

import json
import subprocess
from pathlib import Path

import pytest

from workflow.defaults import DefaultsVersion
from workflow.scripts import generate_realisations_from_csv as gr


def test_generate_one_deletes_the_partial_stub_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nshm2022-to-realisation writes metadata and seeds before the point at which
    # a disconnected rupture graph fails, so a crash leaves this behind.
    target = tmp_path / "realisation_59421.json"
    target.write_text('{"metadata": {}, "seeds": {}}', encoding="utf-8")

    def fail(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1, cmd, output="some stdout", stderr="graph must be connected"
        )

    monkeypatch.setattr(subprocess, "run", fail)

    message = gr.generate_one(
        Path("nshmdb.db"), 59421, target, DefaultsVersion.v24_2_2_1
    )

    assert message is not None
    assert "59421" in message
    assert "graph must be connected" in message
    assert not target.exists()


def test_generate_one_returns_none_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"

    def succeed(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", succeed)

    assert (
        gr.generate_one(Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1)
        is None
    )
    assert target.exists()


def test_generate_one_passes_the_scientific_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}
    target = tmp_path / "realisation_42.json"

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        Path(cmd[3]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture)

    gr.generate_one(Path("nshmdb.db"), 42, target, DefaultsVersion.v24_2_2_1)

    assert captured["cmd"] == [
        "nshm2022-to-realisation",
        "nshmdb.db",
        "42",
        str(target),
        "24.2.2.1",
        "--dip-delta",
        "20",
    ]


def test_generate_one_writes_seeds_into_the_stub_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    seeds = {
        "nshm_to_realisation_seed": 531798913,
        "rupture_propagation_seed": 31268976,
        "genslip_seed": 513004717,
        "srfgen_seed": 1837842819,
        "hf_seed": 1524796118,
    }
    seen: dict[str, dict[str, object]] = {}

    def capture_stub(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen["stub"] = json.loads(Path(cmd[3]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture_stub)

    result = gr.generate_one(
        Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1, seeds=seeds
    )

    assert result is None
    assert seen["stub"]["seeds"] == seeds


def test_generate_one_writes_no_stub_when_seeds_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    seen: dict[str, bool] = {}

    def note_absence(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen["existed_before"] = Path(cmd[3]).exists()
        Path(cmd[3]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", note_absence)

    gr.generate_one(Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1)

    assert seen["existed_before"] is False


def test_read_inherited_seeds_reads_the_deployed_seed_block(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "100932").mkdir(parents=True)
    seeds = {
        "nshm_to_realisation_seed": 531798913,
        "rupture_propagation_seed": 31268976,
        "genslip_seed": 513004717,
        "srfgen_seed": 1837842819,
        "hf_seed": 1524796118,
    }
    (events_dir / "100932" / "realisation.json").write_text(
        json.dumps(
            {
                "metadata": {"name": "Rupture 100932"},
                "magnitudes": {"AlpineK2T": 7.1},
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )

    # Only the seed block is taken; every other field is ignored.
    assert gr.read_inherited_seeds(events_dir, 100932) == seeds


def test_read_inherited_seeds_returns_none_when_the_event_is_absent(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()

    assert gr.read_inherited_seeds(events_dir, 999999) is None


def test_read_inherited_seeds_returns_none_when_the_seed_block_is_missing(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    (events_dir / "100932").mkdir(parents=True)
    (events_dir / "100932" / "realisation.json").write_text(
        json.dumps({"metadata": {"name": "Rupture 100932"}}), encoding="utf-8"
    )

    assert gr.read_inherited_seeds(events_dir, 100932) is None
