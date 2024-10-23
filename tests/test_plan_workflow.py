import difflib
from pathlib import Path

from typer.testing import CliRunner

from workflow.scripts.plan_workflow import (
    app,
)

runner = CliRunner()
EXPECTED_FLOW_CYLC_FILES = Path("tests") / "flows"


def compare_files(expected_file: Path, actual_file: Path) -> list[str]:
    return list(
        difflib.unified_diff(
            expected_file.read_text().splitlines(), actual_file.read_text().splitlines()
        )
    )


def test_single_realisation(tmp_path: Path):
    result = runner.invoke(
        app, ["Darfield", str(tmp_path / "flow.cylc"), "--goal", "create_e3d_par"]
    )
    assert result.exit_code == 0
    expected_output_file = EXPECTED_FLOW_CYLC_FILES / "expected_single_realisation.cylc"
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_multiple_realisations(tmp_path: Path):
    result = runner.invoke(
        app,
        ["Darfield:2", "Ahuriri:2", str(tmp_path / "flow.cylc"), "--goal", "hf_sim"],
    )
    assert result.exit_code == 0
    expected_output_file = (
        EXPECTED_FLOW_CYLC_FILES / "expected_multiple_realisations.cylc"
    )
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_excluding_stages(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "Darfield",
            str(tmp_path / "flow.cylc"),
            "--goal",
            "im_calc",
            "--excluding",
            "hf_sim",
        ],
    )
    assert result.exit_code == 0
    expected_output_file = EXPECTED_FLOW_CYLC_FILES / "expected_excluding_stages.cylc"
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_excluding_group(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "Darfield",
            str(tmp_path / "flow.cylc"),
            "--goal",
            "plot_ts",
            "--excluding-group",
            "preprocessing",
        ],
    )
    assert result.exit_code == 0
    expected_output_file = EXPECTED_FLOW_CYLC_FILES / "expected_excluding_group.cylc"
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_group_goal(tmp_path: Path):
    result = runner.invoke(
        app,
        ["Darfield", str(tmp_path / "flow.cylc"), "--group-goal", "preprocessing"],
    )
    assert result.exit_code == 0
    expected_output_file = EXPECTED_FLOW_CYLC_FILES / "expected_group_goal.cylc"
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_different_target_host(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "Darfield",
            str(tmp_path / "flow.cylc"),
            "--goal",
            "plot_ts",
            "--goal",
            "im_calc",
            "--target-host",
            "hypocentre",
        ],
    )
    assert result.exit_code == 0
    expected_output_file = (
        EXPECTED_FLOW_CYLC_FILES / "expected_different_target_host.cylc"
    )
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


def test_different_source(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "Darfield",
            str(tmp_path / "flow.cylc"),
            "--goal",
            "create_e3d_par",
            "--source",
            "nshm",
            "--defaults-version",
            "24.2.2.2",
        ],
    )
    assert result.exit_code == 0
    expected_output_file = EXPECTED_FLOW_CYLC_FILES / "expected_different_source.cylc"
    actual_output_file = tmp_path / "flow.cylc"
    diff = compare_files(expected_output_file, actual_output_file)
    assert diff == [], "Unexpected file contents:\n" + "\n".join(diff)


EXPECTED_OUTPUT = """You require the following files for your simulation:

┐
└── cylc-src
    └── WORKFLOW_NAME
        ├── flow.cylc: Your workflow file (the file {output_path}).
        └── input
            └── Darfield
                ├── realisation.json: Realisation file for event containing: metadata, rupture_propagation, sources.
                └── realisation.srf: Slip model of source (Section: SRF Format).

Refer to the indicated sections in https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation
Refer to the realisation glossary at URL HERE for details on filling in the realisation files.
"""


def test_show_required_files(tmp_path: Path):
    output_path = str(tmp_path / "flow.cylc")
    result = runner.invoke(
        app,
        [
            "Darfield",
            output_path,
            "--goal",
            "create_e3d_par",
            "--show-required-files",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == EXPECTED_OUTPUT.format(output_path=output_path)
