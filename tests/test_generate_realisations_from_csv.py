"""Tests for the generate_realisations_from_csv campaign driver."""

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
