"""Tests for the complete_realisations campaign tool."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import typer

from workflow.defaults import DefaultsVersion
from workflow.scripts import complete_realisations as cr
from workflow.scripts.reconcile_parameters import Decision, value_fingerprint

DATA = Path(__file__).parent / "data"
FELIPE = DATA / "felipe_reference_realisation.json"
SAMPLE = DATA / "minimal_realisation_sample.json"
BROKEN = DATA / "broken_minimal_stub.json"
FELIPE_SCRIPTS = Path(__file__).parents[1] / "felipe_scripts"


@pytest.fixture
def minimal_stub() -> dict[str, object]:
    return json.loads(SAMPLE.read_text())


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
        "emod3d",
        "hf",
        "bb",
        "resolution",
        "srf",
        "velocity_model_1d",
        "hf_velocity_model_1d",
    ]:
        assert completed[section] == felipe[section], f"{section} differs from Felipe"

    # `im` matches Felipe everywhere except where the campaign decided otherwise.
    for key in ["valid_periods", "fas_frequencies"]:
        assert completed["im"][key] == felipe["im"][key], f"im.{key} differs from Felipe"

    # im.ims deliberately diverges: the scientific defaults gained PGD in
    # ec2fb25 and the campaign adopted it. Asserting the exact expected list --
    # rather than dropping the check -- keeps any *other* drift in ims a failure.
    assert completed["im"]["ims"] == [
        *felipe["im"]["ims"][:2],
        "PGD",
        *felipe["im"]["ims"][2:],
    ]


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


def test_apply_parameters_overwrites_only_the_decided_keys() -> None:
    realisation = {
        "im": {"ims": ["PGA"], "valid_periods": [0.1]},
        "magnitudes": {"A": 7.1},
    }

    cr.apply_parameters(realisation, {"im.ims": ["PGA", "PGD"]})

    assert realisation["im"]["ims"] == ["PGA", "PGD"]
    assert realisation["im"]["valid_periods"] == [0.1]
    assert realisation["magnitudes"] == {"A": 7.1}


def test_apply_parameters_creates_an_absent_section() -> None:
    realisation: dict[str, object] = {}

    cr.apply_parameters(realisation, {"im.ims": ["PGA"]})

    assert realisation == {"im": {"ims": ["PGA"]}}


def test_resolve_parameters_rejects_a_decision_whose_source_moved(
    tmp_path: Path,
) -> None:
    # The decision recorded a fingerprint for a value the defaults no longer hold.
    decisions = {
        "im.ims": Decision(
            source="defaults",
            reason="adopt PGD",
            decided="2026-07-27",
            sha256=value_fingerprint(["THIS", "IS", "STALE"]),
        )
    }

    with pytest.raises(ValueError, match="moved since"):
        cr.resolve_parameters(
            decisions, DefaultsVersion.v24_2_2_1, Path("felipe_scripts"), None
        )


def test_resolve_parameters_requires_an_events_dir_for_deployed_decisions() -> None:
    decisions = {
        "im.ims": Decision(
            source="deployed", reason="pin", decided="2026-07-27", sha256="x"
        )
    }

    with pytest.raises(ValueError, match="--deployed-from"):
        cr.resolve_parameters(
            decisions, DefaultsVersion.v24_2_2_1, Path("felipe_scripts"), None
        )


def test_complete_realisations_refuses_to_deploy_over_existing(
    tmp_path: Path, minimal_stub: dict[str, object]
) -> None:
    source = tmp_path / "minimal"
    source.mkdir()
    (source / "realisation_100932.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    with pytest.raises(typer.Exit) as exit_info:
        cr.complete_realisations(
            source,
            tmp_path / "complete",
            felipe_scripts_dir=Path("felipe_scripts"),
            workers=1,
            deploy_dir=events,
        )

    assert exit_info.value.exit_code == 1
    # The deployed file is untouched: the gate held.
    assert (events / "100932" / "realisation.json").read_text() == '{"old": 1}'


def test_complete_realisations_deploys_when_overwrite_is_given(
    tmp_path: Path, minimal_stub: dict[str, object]
) -> None:
    source = tmp_path / "minimal"
    source.mkdir()
    (source / "realisation_100932.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    events = tmp_path / "events"
    (events / "100932").mkdir(parents=True)
    (events / "100932" / "realisation.json").write_text('{"old": 1}', encoding="utf-8")

    cr.complete_realisations(
        source,
        tmp_path / "complete",
        felipe_scripts_dir=Path("felipe_scripts"),
        workers=1,
        deploy_dir=events,
        overwrite_existing=True,
    )

    deployed = json.loads((events / "100932" / "realisation.json").read_text())
    assert "domain" in deployed


def test_complete_realisations_does_not_deploy_without_the_flag(
    tmp_path: Path, minimal_stub: dict[str, object]
) -> None:
    source = tmp_path / "minimal"
    source.mkdir()
    (source / "realisation_100932.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    events = tmp_path / "events"
    events.mkdir()

    cr.complete_realisations(
        source, tmp_path / "complete", felipe_scripts_dir=Path("felipe_scripts"),
        workers=1,
    )

    # No --deploy-dir: nothing outside the output directory is written.
    assert list(events.iterdir()) == []


def test_complete_realisations_refuses_to_deploy_when_a_stub_is_broken(
    tmp_path: Path, minimal_stub: dict[str, object]
) -> None:
    source = tmp_path / "minimal"
    source.mkdir()
    (source / "realisation_100932.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    shutil.copy(BROKEN, source / "realisation_59421.json")
    events = tmp_path / "events"

    with pytest.raises(typer.Exit) as exit_info:
        cr.complete_realisations(
            source,
            tmp_path / "complete",
            felipe_scripts_dir=Path("felipe_scripts"),
            workers=1,
            deploy_dir=events,
        )

    assert exit_info.value.exit_code == 1
    # The guard refuses the whole run, before copy_realisations ever runs:
    # even the realisation that completed successfully (100932) is not
    # deployed.
    assert not events.exists()


def test_complete_realisations_refuses_to_deploy_when_a_completion_fails(
    tmp_path: Path, minimal_stub: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "minimal"
    source.mkdir()
    (source / "realisation_100932.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    (source / "realisation_100933.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    events = tmp_path / "events"
    real_complete_one = cr.complete_one

    def flaky_complete_one(
        src: Path,
        dst: Path,
        defaults_version: DefaultsVersion,
        overrides: cr.Overrides,
        resolved: dict[str, object] | None = None,
    ) -> None:
        if src.name == "realisation_100933.json":
            raise RuntimeError("simulated completion failure")
        real_complete_one(src, dst, defaults_version, overrides, resolved)

    monkeypatch.setattr(cr, "complete_one", flaky_complete_one)

    with pytest.raises(typer.Exit) as exit_info:
        cr.complete_realisations(
            source,
            tmp_path / "complete",
            felipe_scripts_dir=Path("felipe_scripts"),
            workers=1,
            deploy_dir=events,
        )

    assert exit_info.value.exit_code == 1
    # The guard refuses the whole run: even the realisation that completed
    # successfully (100932) is not deployed.
    assert not events.exists()


def test_complete_realisations_ignores_overwrite_existing_without_deploy_dir(
    tmp_path: Path, minimal_stub: dict[str, object]
) -> None:
    source = tmp_path / "minimal"
    source.mkdir()
    (source / "realisation_100932.json").write_text(
        json.dumps(minimal_stub), encoding="utf-8"
    )
    output_dir = tmp_path / "complete"
    events = tmp_path / "events"

    cr.complete_realisations(
        source,
        output_dir,
        felipe_scripts_dir=Path("felipe_scripts"),
        workers=1,
        overwrite_existing=True,
    )

    # overwrite_existing is inert without --deploy-dir: no exception, no
    # events path created, and the output directory is written normally.
    assert not events.exists()
    assert (output_dir / "realisation_100932.json").exists()
    assert (output_dir / "completion_summary.csv").exists()


# -- Domain carry-over ------------------------------------------------------
#
# The computed domain differs from the deployed one by a handful of ULP, which
# is noise rather than a change, so it can be carried over verbatim. Doing so is
# only sound while the inputs it was derived from still agree, and "velocity
# model changed, domain unchanged" is exactly what a clean pass looks like --
# so the guard against that is tested harder than the happy path.


DEPLOYED_DOMAIN = {
    "domain": [{"latitude": -45.5, "longitude": 174.9}],
    "depth": 60.0,
    "duration": 234.8667508646568,
}


def write_deployed(tmp_path: Path, **sections: object) -> Path:
    events_dir = tmp_path / "events"
    (events_dir / "100932").mkdir(parents=True)
    (events_dir / "100932" / "realisation.json").write_text(
        json.dumps({"domain": DEPLOYED_DOMAIN, **sections}), encoding="utf-8"
    )
    return events_dir


def test_inherit_domain_replaces_the_computed_domain(tmp_path: Path) -> None:
    events_dir = write_deployed(tmp_path, velocity_model={"version": "2.09"}, sources={})
    realisation = {
        "domain": {"domain": [], "depth": 1.0, "duration": 2.0},
        "velocity_model": {"version": "2.09"},
        "sources": {},
    }

    cr._inherit_domain(realisation, events_dir, "100932")

    assert realisation["domain"] == DEPLOYED_DOMAIN


def test_inherit_domain_tolerates_float_noise_in_the_inputs(tmp_path: Path) -> None:
    # sources reproduces to within a couple of ULP rather than exactly. Requiring
    # bit-equality here would refuse every event the carry-over exists to serve.
    events_dir = write_deployed(
        tmp_path,
        velocity_model={"version": "2.09"},
        sources={"corner": -44.593361302171196},
    )
    realisation = {
        "domain": {"domain": [], "depth": 1.0, "duration": 2.0},
        "velocity_model": {"version": "2.09"},
        "sources": {"corner": -44.59336130217119},
    }

    cr._inherit_domain(realisation, events_dir, "100932")

    assert realisation["domain"] == DEPLOYED_DOMAIN


def test_inherit_domain_refuses_when_the_velocity_model_moved(tmp_path: Path) -> None:
    events_dir = write_deployed(tmp_path, velocity_model={"version": "2.09"}, sources={})
    realisation = {
        "domain": {"domain": [], "depth": 1.0, "duration": 2.0},
        "velocity_model": {"version": "2.07"},
        "sources": {},
    }

    with pytest.raises(ValueError, match="velocity_model"):
        cr._inherit_domain(realisation, events_dir, "100932")


def test_inherit_domain_refuses_when_the_sources_moved(tmp_path: Path) -> None:
    events_dir = write_deployed(
        tmp_path, velocity_model={"version": "2.09"}, sources={"corner": 1.0}
    )
    realisation = {
        "domain": {"domain": [], "depth": 1.0, "duration": 2.0},
        "velocity_model": {"version": "2.09"},
        "sources": {"corner": 1.5},
    }

    with pytest.raises(ValueError, match="sources"):
        cr._inherit_domain(realisation, events_dir, "100932")


def test_inherit_domain_keeps_the_computed_domain_for_a_new_rupture(
    tmp_path: Path,
) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    computed = {"domain": [], "depth": 1.0, "duration": 2.0}
    realisation = {"domain": computed, "velocity_model": {}, "sources": {}}

    cr._inherit_domain(realisation, events_dir, "999999")

    assert realisation["domain"] == computed


@pytest.mark.slow
def test_complete_one_carries_the_deployed_domain_through(tmp_path: Path) -> None:
    overrides = cr.load_overrides(FELIPE_SCRIPTS)
    reference = tmp_path / "reference.json"
    cr.complete_one(SAMPLE, reference, DefaultsVersion.v24_2_2_1, overrides)
    completed_reference = json.loads(reference.read_text())

    # A deployed file whose inputs match, but whose domain differs -- so a pass
    # cannot be explained by the computed domain already being equal.
    rupture_id = cr._rupture_id_from_path(SAMPLE)
    events_dir = tmp_path / "events"
    (events_dir / rupture_id).mkdir(parents=True)
    deployed_domain = dict(completed_reference["domain"], duration=999.5)
    (events_dir / rupture_id / "realisation.json").write_text(
        json.dumps(
            {
                "domain": deployed_domain,
                "velocity_model": completed_reference["velocity_model"],
                "sources": completed_reference["sources"],
            }
        ),
        encoding="utf-8",
    )

    dst = tmp_path / "completed.json"
    src = tmp_path / f"realisation_{rupture_id}.json"
    shutil.copy(SAMPLE, src)
    cr.complete_one(
        src, dst, DefaultsVersion.v24_2_2_1, overrides, None, events_dir
    )

    completed = json.loads(dst.read_text())
    assert completed["domain"] == deployed_domain
    assert list(completed) == cr.FELIPE_SECTION_ORDER
