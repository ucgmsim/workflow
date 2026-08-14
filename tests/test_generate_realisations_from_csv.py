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


def test_read_inherited_seeds_rejects_a_partial_block(tmp_path: Path) -> None:
    # A partial block would otherwise be counted as inherited, then fail schema
    # validation inside the subprocess, where the failure is easy to miss.
    events_dir = tmp_path / "events"
    (events_dir / "100932").mkdir(parents=True)
    (events_dir / "100932" / "realisation.json").write_text(
        json.dumps({"seeds": {"nshm_to_realisation_seed": 531798913}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seeds"):
        gr.read_inherited_seeds(events_dir, 100932)


RUPTURE_PROPAGATION = {
    "rupture_causality_tree": {"Beacon Hill": "Little Hillfoot", "Little Hillfoot": None},
    "jump_points": {
        "Beacon Hill": {
            "from_point": {"s": 1.4087856255707695e-34, "d": 0.00011441641893337497},
            "to_point": {"s": 1.9679106304718342e-10, "d": 6.646602576300802e-05},
        }
    },
    "hypocentre": {"s": 0.42683003552928284, "d": 0.874844163014458},
}


def write_deployed_realisation(
    events_dir: Path, rupture_id: int, **sections: object
) -> Path:
    event_dir = events_dir / str(rupture_id)
    event_dir.mkdir(parents=True)
    realisation_ffp = event_dir / "realisation.json"
    realisation_ffp.write_text(json.dumps(sections), encoding="utf-8")
    return realisation_ffp


def test_read_inherited_rupture_propagation_reads_the_deployed_block(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        metadata={"name": "Rupture 100932"},
        magnitudes={"AlpineK2T": 7.1},
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    assert (
        gr.read_inherited_rupture_propagation(events_dir, 100932) == RUPTURE_PROPAGATION
    )


def test_read_inherited_rupture_propagation_returns_none_when_the_event_is_absent(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()

    assert gr.read_inherited_rupture_propagation(events_dir, 999999) is None


def test_read_inherited_rupture_propagation_returns_none_when_the_block_is_missing(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(events_dir, 100932, metadata={"name": "Rupture 100932"})

    assert gr.read_inherited_rupture_propagation(events_dir, 100932) is None


def test_read_inherited_rupture_propagation_rejects_a_partial_block(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        rupture_propagation={"rupture_causality_tree": {"Beacon Hill": None}},
    )

    with pytest.raises(ValueError, match="rupture_propagation"):
        gr.read_inherited_rupture_propagation(events_dir, 100932)


def test_generate_one_overwrites_the_generated_rupture_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"

    def generate_a_different_tree(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text(
            json.dumps(
                {
                    "metadata": {"name": "Rupture 100932"},
                    "rupture_propagation": {
                        "rupture_causality_tree": {"Beacon Hill": None},
                        "jump_points": {},
                        "hypocentre": {"s": 0.1, "d": 0.2},
                    },
                    "magnitudes": {"AlpineK2T": 7.1},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", generate_a_different_tree)

    result = gr.generate_one(
        Path("nshmdb.db"),
        100932,
        target,
        DefaultsVersion.v24_2_2_1,
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    assert result is None
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["rupture_propagation"] == RUPTURE_PROPAGATION
    # Every other section is still whatever the generator derived.
    assert written["magnitudes"] == {"AlpineK2T": 7.1}
    # The section keeps its position, so complete-realisations still sees the
    # order nshm2022-to-realisation wrote.
    assert list(written) == ["metadata", "rupture_propagation", "magnitudes"]


def test_generate_one_leaves_rupture_propagation_alone_when_not_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    derived = {
        "rupture_causality_tree": {"Beacon Hill": None},
        "jump_points": {},
        "hypocentre": {"s": 0.1, "d": 0.2},
    }

    def generate(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text(
            json.dumps({"rupture_propagation": derived}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", generate)

    gr.generate_one(Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1)

    assert json.loads(target.read_text(encoding="utf-8"))["rupture_propagation"] == derived


def test_generate_one_reports_a_generated_file_with_no_rupture_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Overwriting a section the generator did not write would invent one, so this
    # has to fail loudly rather than produce a file that looks complete.
    target = tmp_path / "realisation_100932.json"

    def generate_without_propagation(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text(json.dumps({"metadata": {}}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", generate_without_propagation)

    message = gr.generate_one(
        Path("nshmdb.db"),
        100932,
        target,
        DefaultsVersion.v24_2_2_1,
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    assert message is not None
    assert "rupture_propagation" in message
    assert not target.exists()


def run_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inherit_seeds_from: Path | None = None,
    inherit_rupture_propagation_from: Path | None = None,
) -> None:
    csv_file = tmp_path / "ruptures.csv"
    csv_file.write_text("chosen_nshm_id\n100932\n", encoding="utf-8")

    def generate(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text(
            json.dumps(
                {
                    "rupture_propagation": {
                        "rupture_causality_tree": {"Beacon Hill": None},
                        "jump_points": {},
                        "hypocentre": {"s": 0.1, "d": 0.2},
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", generate)

    gr.generate_realisations_from_csv(
        tmp_path / "nshmdb.db",
        csv_file,
        tmp_path / "out",
        DefaultsVersion.v24_2_2_1,
        inherit_seeds_from=inherit_seeds_from,
        inherit_rupture_propagation_from=inherit_rupture_propagation_from,
    )


def test_main_warns_when_seeds_are_inherited_without_rupture_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        seeds={
            "nshm_to_realisation_seed": 531798913,
            "rupture_propagation_seed": 31268976,
            "genslip_seed": 513004717,
            "srfgen_seed": 1837842819,
            "hf_seed": 1524796118,
        },
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    run_campaign(
        tmp_path,
        monkeypatch,
        inherit_seeds_from=events_dir,
        inherit_rupture_propagation_from=None,
    )

    warning = capsys.readouterr().out
    assert "WARNING" in warning
    assert "--inherit-rupture-propagation-from" in warning


def test_main_does_not_warn_when_both_are_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        seeds={
            "nshm_to_realisation_seed": 531798913,
            "rupture_propagation_seed": 31268976,
            "genslip_seed": 513004717,
            "srfgen_seed": 1837842819,
            "hf_seed": 1524796118,
        },
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    run_campaign(
        tmp_path,
        monkeypatch,
        inherit_seeds_from=events_dir,
        inherit_rupture_propagation_from=events_dir,
    )

    output = capsys.readouterr().out
    assert "WARNING" not in output
    assert "Inherited rupture propagation for 1 of 1" in output


def test_main_logs_and_skips_a_rupture_whose_inherited_block_is_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        rupture_propagation={"rupture_causality_tree": {"Beacon Hill": None}},
    )

    run_campaign(
        tmp_path, monkeypatch, inherit_rupture_propagation_from=events_dir
    )

    assert not (tmp_path / "out" / "realisation_100932.json").exists()
    error_log = (tmp_path / "out" / "error_log.txt").read_text(encoding="utf-8")
    assert "100932" in error_log
    assert "rupture_propagation" in error_log
