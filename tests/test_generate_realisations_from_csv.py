"""Tests for the generate_realisations_from_csv campaign driver."""

import json
import subprocess
from pathlib import Path

import pytest
import typer

from workflow import inheritance as ih
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


SEEDS = {
    "nshm_to_realisation_seed": 531798913,
    "rupture_propagation_seed": 31268976,
    "genslip_seed": 513004717,
    "srfgen_seed": 1837842819,
    "hf_seed": 1524796118,
}

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


def test_read_inherited_sections_reads_only_what_was_asked_for(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        metadata={"name": "Rupture 100932"},
        magnitudes={"magnitudes": {"AlpineK2T": 7.1}},
        seeds=SEEDS,
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    assert gr.read_inherited_sections(events_dir, 100932, ["seeds"]) == {"seeds": SEEDS}


def test_read_inherited_sections_reads_several_at_once(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        seeds=SEEDS,
        magnitudes={"magnitudes": {"AlpineK2T": 7.1}},
        rupture_propagation=RUPTURE_PROPAGATION,
    )

    inherited = gr.read_inherited_sections(
        events_dir, 100932, ["seeds", "magnitudes", "rupture_propagation"]
    )

    assert inherited == {
        "seeds": SEEDS,
        "magnitudes": {"magnitudes": {"AlpineK2T": 7.1}},
        "rupture_propagation": RUPTURE_PROPAGATION,
    }


def test_read_inherited_sections_is_empty_when_the_event_is_absent(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()

    assert gr.read_inherited_sections(events_dir, 999999, ["seeds"]) == {}


def test_read_inherited_sections_omits_a_section_the_file_lacks(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(events_dir, 100932, seeds=SEEDS)

    inherited = gr.read_inherited_sections(
        events_dir, 100932, ["seeds", "rupture_propagation"]
    )

    assert inherited == {"seeds": SEEDS}


def test_read_inherited_sections_rejects_a_partial_seed_block(tmp_path: Path) -> None:
    # A partial block would otherwise be counted as inherited, then fail schema
    # validation inside the subprocess, where the failure is easy to miss.
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir, 100932, seeds={"nshm_to_realisation_seed": 531798913}
    )

    with pytest.raises(ValueError, match="seeds"):
        gr.read_inherited_sections(events_dir, 100932, ["seeds"])


def test_read_inherited_sections_rejects_a_partial_rupture_propagation_block(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        rupture_propagation={"rupture_causality_tree": {"Beacon Hill": None}},
    )

    with pytest.raises(ValueError, match="rupture_propagation"):
        gr.read_inherited_sections(events_dir, 100932, ["rupture_propagation"])


def test_section_keys_covers_every_inheritable_section() -> None:
    # A new enum member with no key set would KeyError at read time, on a real
    # campaign run rather than here.
    assert {section.value for section in gr.InheritableSection} == set(gr.SECTION_KEYS)


# A tree the same as RUPTURE_PROPAGATION to within the section tolerance: the
# jump-point s coordinates are both effectively zero, delivered as different
# denormal residue, which a relative-only comparison would call a 100% mismatch.
NOISY_RUPTURE_PROPAGATION = {
    "rupture_causality_tree": {"Beacon Hill": "Little Hillfoot", "Little Hillfoot": None},
    "jump_points": {
        "Beacon Hill": {
            "from_point": {"s": 3.2e-07, "d": 0.00011441641893337499},
            "to_point": {"s": 4.5e-12, "d": 6.646602576300804e-05},
        }
    },
    "hypocentre": {"s": 0.42683003552928289, "d": 0.874844163014458},
}

A_DIFFERENT_TREE = {
    "rupture_causality_tree": {"Beacon Hill": None},
    "jump_points": {},
    "hypocentre": {"s": 0.1, "d": 0.2},
}


def generator_writing(section_value: object) -> object:
    def generate(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text(
            json.dumps(
                {
                    "metadata": {"name": "Rupture 100932"},
                    "rupture_propagation": section_value,
                    "magnitudes": {"magnitudes": {"AlpineK2T": 7.1}},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return generate


def test_generate_one_adopts_the_inherited_value_when_the_difference_is_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    monkeypatch.setattr(subprocess, "run", generator_writing(NOISY_RUPTURE_PROPAGATION))

    result = gr.generate_one(
        Path("nshmdb.db"),
        100932,
        target,
        DefaultsVersion.v24_2_2_1,
        inherited={"rupture_propagation": RUPTURE_PROPAGATION},
    )

    assert result is None
    written = json.loads(target.read_text(encoding="utf-8"))
    # Byte-exact adoption: the derived value differed, so equality proves the
    # deployed one was written rather than the derived one kept.
    assert written["rupture_propagation"] == RUPTURE_PROPAGATION
    assert written["magnitudes"] == {"magnitudes": {"AlpineK2T": 7.1}}
    # The section keeps its position, so complete-realisations still sees the
    # order nshm2022-to-realisation wrote.
    assert list(written) == ["metadata", "rupture_propagation", "magnitudes"]


def test_generate_one_refuses_a_real_divergence_without_a_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    monkeypatch.setattr(subprocess, "run", generator_writing(A_DIFFERENT_TREE))

    message = gr.generate_one(
        Path("nshmdb.db"),
        100932,
        target,
        DefaultsVersion.v24_2_2_1,
        inherited={"rupture_propagation": RUPTURE_PROPAGATION},
    )

    assert message is not None
    assert "rupture_propagation" in message
    assert "Record a choice for" in message
    # No half-inherited file is left behind.
    assert not target.exists()


def test_generate_one_keeps_the_inherited_value_when_a_decision_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    monkeypatch.setattr(subprocess, "run", generator_writing(A_DIFFERENT_TREE))

    result = gr.generate_one(
        Path("nshmdb.db"),
        100932,
        target,
        DefaultsVersion.v24_2_2_1,
        inherited={"rupture_propagation": RUPTURE_PROPAGATION},
        decisions={
            "rupture_propagation": ih.InheritanceDecision(
                choice=ih.InheritanceChoice.inherited,
                reason="the derived tree is an unseeded draw",
                decided="2026-08-14",
            )
        },
    )

    assert result is None
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["rupture_propagation"] == RUPTURE_PROPAGATION


def test_generate_one_takes_the_derived_value_when_a_decision_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    monkeypatch.setattr(subprocess, "run", generator_writing(A_DIFFERENT_TREE))

    result = gr.generate_one(
        Path("nshmdb.db"),
        100932,
        target,
        DefaultsVersion.v24_2_2_1,
        inherited={"rupture_propagation": RUPTURE_PROPAGATION},
        decisions={
            "100932.rupture_propagation": ih.InheritanceDecision(
                choice=ih.InheritanceChoice.derived,
                reason="adopting the new causality tree for this event",
                decided="2026-08-14",
            )
        },
    )

    assert result is None
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["rupture_propagation"] == A_DIFFERENT_TREE


def test_generate_one_leaves_rupture_propagation_alone_when_not_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "realisation_100932.json"
    monkeypatch.setattr(subprocess, "run", generator_writing(A_DIFFERENT_TREE))

    gr.generate_one(Path("nshmdb.db"), 100932, target, DefaultsVersion.v24_2_2_1)

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["rupture_propagation"] == A_DIFFERENT_TREE


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
        inherited={"rupture_propagation": RUPTURE_PROPAGATION},
    )

    assert message is not None
    assert "rupture_propagation" in message
    assert not target.exists()


def run_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inherit_from: Path | None = None,
    inherit: list[gr.InheritableSection] | None = None,
) -> None:
    csv_file = tmp_path / "ruptures.csv"
    csv_file.write_text("chosen_nshm_id\n100932\n", encoding="utf-8")

    def generate(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(cmd[3]).write_text(
            json.dumps(
                {
                    "sources": {
                        "source_geometries": {"Beacon Hill": [[1.0000000000000002, 1.0]]}
                    },
                    "rupture_propagation": NOISY_RUPTURE_PROPAGATION,
                    "magnitudes": {
                        "magnitudes": {"Beacon Hill": 7.900000000000001}
                    },
                    "rakes": {"rakes": {"Beacon Hill": 90.0}},
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
        inherit_from=inherit_from,
        inherit=inherit,
    )


def deployed_events(tmp_path: Path) -> Path:
    events_dir = tmp_path / "events"
    write_deployed_realisation(
        events_dir,
        100932,
        seeds=SEEDS,
        sources={"source_geometries": {"Beacon Hill": [[1.0, 1.0]]}},
        rupture_propagation=RUPTURE_PROPAGATION,
        magnitudes={"magnitudes": {"Beacon Hill": 7.9}},
        rakes={"rakes": {"Beacon Hill": 90.0}},
    )
    return events_dir


def test_main_warns_when_inheriting_without_rupture_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_campaign(
        tmp_path,
        monkeypatch,
        inherit_from=deployed_events(tmp_path),
        inherit=[gr.InheritableSection.seeds],
    )

    warning = capsys.readouterr().out
    assert "WARNING" in warning
    assert "rupture_propagation" in warning


def test_main_does_not_warn_when_rupture_propagation_is_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_campaign(
        tmp_path,
        monkeypatch,
        inherit_from=deployed_events(tmp_path),
        inherit=[gr.InheritableSection.seeds, gr.InheritableSection.rupture_propagation],
    )

    output = capsys.readouterr().out
    assert "WARNING" not in output
    assert "Inherited rupture_propagation for 1 of 1" in output


def test_main_carries_over_every_requested_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The float-noise sections -- sources and magnitudes -- reproduce to within a
    # couple of ULP rather than exactly, so they are carried over too.
    events_dir = deployed_events(tmp_path)

    run_campaign(
        tmp_path,
        monkeypatch,
        inherit_from=events_dir,
        inherit=list(gr.InheritableSection),
    )

    written = json.loads(
        (tmp_path / "out" / "realisation_100932.json").read_text(encoding="utf-8")
    )
    deployed = json.loads(
        (events_dir / "100932" / "realisation.json").read_text(encoding="utf-8")
    )
    for section in ("sources", "rupture_propagation", "magnitudes"):
        assert written[section] == deployed[section], section


def test_main_exits_1_when_inherit_is_given_without_inherit_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(typer.Exit) as exit_info:
        run_campaign(tmp_path, monkeypatch, inherit=[gr.InheritableSection.seeds])

    assert exit_info.value.exit_code == 1


def test_main_exits_1_when_inherit_from_is_given_without_inherit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(typer.Exit) as exit_info:
        run_campaign(tmp_path, monkeypatch, inherit_from=deployed_events(tmp_path))

    assert exit_info.value.exit_code == 1


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
        tmp_path,
        monkeypatch,
        inherit_from=events_dir,
        inherit=[gr.InheritableSection.rupture_propagation],
    )

    assert not (tmp_path / "out" / "realisation_100932.json").exists()
    error_log = (tmp_path / "out" / "error_log.txt").read_text(encoding="utf-8")
    assert "100932" in error_log
    assert "rupture_propagation" in error_log
