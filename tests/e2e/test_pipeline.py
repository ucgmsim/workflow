"""End-to-end smoke test of the simulation pipeline.

Runs every stage of the SW4 workflow, from a GCMT solution to intensity
measures, as the installed command-line tools, on a deliberately tiny problem:
a Mw 4.3 event near Christchurch, a 25 km by 10 km deep domain on a 1 km grid,
five stations, and NZCVM's synthetic data root in place of the real velocity
model data. Nothing here checks the science. It checks that each stage accepts
what the stage before it wrote, so that a dependency update which changes a
realisation field, a CLI flag or a file format fails here rather than halfway
through a real simulation.

The scientific defaults are only shrunk where they set the problem size (domain
reach and depth, fault buffer, supergrid width, mesh refinements) and
redirected where they name data (the NZCVM data root); every other default
runs as shipped.

Only the solver is stubbed. `fake_sw4.py` checks SW4's input file and writes
a recording in SW4's output format. Set `SW4` to an SW4 binary (and
`SW4_LAUNCHER` to, e.g., `mpirun -n 4`) to run the real solver instead.

The test needs what the runner container provides, so it is skipped unless
`WORKFLOW_E2E=1`. Once enabled, missing tools fail the test rather than skip
it. Binary locations default to the container's and can be overridden with
`GENSLIP` and `GENERIC_SLIP2SRF`.

    apptainer exec runner.sif env WORKFLOW_E2E=1 pytest tests/e2e

Each stage is its own test, in pipeline order, so a failure names the
interface that broke. A stage whose inputs were never written is skipped.
"""

import dataclasses
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from workflow.defaults import DefaultsVersion
from workflow.realisations import (
    DomainParameters,
    NZCVMSettings,
    Refinement,
    Refinements,
    SW4Parameters,
    VelocityModelParameters,
    find_command,
)

if os.environ.get("WORKFLOW_E2E") != "1":
    pytest.skip(
        "end-to-end smoke test; set WORKFLOW_E2E=1 to run", allow_module_level=True
    )

DATA = Path(__file__).parent / "data"
FAKE_SW4 = Path(__file__).parent / "fake_sw4.py"

EVENT = "e2esmoke"
DEFAULTS_VERSION = DefaultsVersion.v26_7_1Hz
STATIONS = ["E2E01", "E2E02", "E2E03", "E2E04", "E2E05"]

# The shrunk problem. Reach and fault buffer are in kilometres; supergrid
# width and grid spacing are in metres.
REACH_KM = 7.0
FAULT_BUFFER_KM = 8.0
SUPERGRID_WIDTH = 2000.0
GRID_SPACING = 1000.0
# SW4 stretches each refinement layer to at least this many cells. The
# default (12) assumes 100 m cells; at 1 km it would push the domain far below
# the velocity model `create-nzvm-input` asks NZCVM for.
NZ_MIN = 4
NZCVM_DATA_PREFIX = "/nzcvm/"
# NZCVM's synthetic world: a patch of Canterbury coastline.
SYNTHETIC_LON = (172.0, 172.6)
SYNTHETIC_LAT = (-43.8, -43.4)
# Kilometres. The synthetic tomography stops at 30 km, and `create-sw4-input`
# deepens the domain to fit SW4's minimum layer thickness and sponge.
MAX_DEPTH_KM = 10.0

GENSLIP = os.environ.get("GENSLIP", "/EMOD3D/tools/genslip_v5.6.2")
GENERIC_SLIP2SRF = os.environ.get("GENERIC_SLIP2SRF", "/EMOD3D/tools/generic_slip2srf")


def run(*command: str | Path, cwd: Path | None = None) -> None:
    """Run a pipeline command, failing the test with its output if it fails."""
    command_line = [str(part) for part in command]
    result = subprocess.run(
        command_line, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(
            f"`{shlex.join(command_line)}` exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout[-5000:]}\n"
            f"--- stderr ---\n{result.stderr[-5000:]}",
            pytrace=False,
        )


def require(*paths: Path) -> None:
    """Skip a stage whose inputs an earlier, failed stage never wrote."""
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"upstream stage did not produce {', '.join(missing)}")


def require_domain(realisation: Path) -> None:
    """Skip a stage that needs the domain if `generate-domain` never wrote it."""
    require(realisation)
    if "domain" not in json.loads(realisation.read_text()):
        pytest.skip("upstream stage did not write the domain")


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The run directory every stage reads from and writes to."""
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="module")
def realisation(work: Path) -> Path:
    """The realisation every stage reads."""
    return work / "realisation.json"


@pytest.fixture(scope="module")
def nzcvm_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """NZCVM's synthetic data root, laid out like the real one.

    Built with the installed nzcvm, the way nzcvm's own `just synthetic`
    recipe builds it, minus the basins (which need nzcvm's meshing extra).
    """
    root = tmp_path_factory.mktemp("nzcvm")
    scratch = tmp_path_factory.mktemp("nzcvm-inputs")
    (root / "resources").mkdir()
    (root / "models").mkdir()
    run("nzcvm", "synthetic", "dem", scratch / "dem.h5")
    run("nzcvm", "surface", "convert", scratch / "dem.h5", root / "resources/dem.zarr")
    run("nzcvm", "synthetic", "vs30", scratch / "vs30.h5")
    run(
        "nzcvm", "surface", "convert", scratch / "vs30.h5",
        root / "resources/vs30.zarr", "--scalar-key", "vs30", "--no-flip",
    )  # fmt: skip
    run("nzcvm", "synthetic", "coastline", root / "resources/coastline.wkb.gz")
    run("nzcvm", "synthetic", "tomography", scratch / "tomography.csv")
    run(
        "nzcvm", "tomography", "convert", scratch / "tomography.csv",
        root / "models/tomography.zarr",
    )  # fmt: skip
    return root


def test_tools_installed() -> None:
    """The container provides every tool the pipeline calls."""
    commands = [
        "gcmt-to-realisation", "generate-domain", "realisation-to-srf", "check-srf",
        "srf-to-hdf5", "generate-stoch", "generate-station-coordinates",
        "create-nzvm-input", "nzcvm", "validate-sfile", "create-sw4-input",
        "lf-to-xarray", "hf-sim", "bb-sim", "im-calc",
    ]  # fmt: skip
    missing = [command for command in commands if shutil.which(command) is None]
    missing += [
        binary
        for binary in (GENSLIP, GENERIC_SLIP2SRF)
        if not os.access(binary, os.X_OK)
    ]
    assert not missing, f"missing tools: {missing}"


def test_gcmt_to_realisation(realisation: Path) -> None:
    """A realisation from a (local) GeoNet CMT solution."""
    run(
        "gcmt-to-realisation", EVENT, DEFAULTS_VERSION, realisation, "finite-fault",
        "--solution-origin", DATA / "cmt_solutions.csv",
    )  # fmt: skip
    assert json.loads(realisation.read_text())["metadata"]["name"] == EVENT


def test_generate_domain(realisation: Path, nzcvm_data_root: Path) -> None:
    """Shrink the problem through the realisation API, then size the domain.

    This uses the API `size_domain.py` in realtime_simulation_workflow uses:
    configs read from the defaults, `dataclasses.replace`d and written back.
    """
    require(realisation)
    velocity_model = VelocityModelParameters.read_from_defaults(DEFAULTS_VERSION)
    interpolants = np.asarray(velocity_model.rrup_interpolants, dtype=float)
    interpolants[1] = np.minimum(interpolants[1], REACH_KM)
    dataclasses.replace(
        velocity_model, rrup_interpolants=interpolants, fault_buffer=FAULT_BUFFER_KM
    ).write_to_realisation(realisation)

    sw4_params = SW4Parameters.read_from_defaults(DEFAULTS_VERSION)
    supergrid = find_command(sw4_params.commands, "supergrid")
    assert supergrid is not None, "26.7.1Hz defaults have no supergrid command"
    supergrid.parameters = {"width": SUPERGRID_WIDTH}
    dataclasses.replace(sw4_params, nz_min=NZ_MIN).write_to_realisation(realisation)

    Refinements(
        refinements=[Refinement(resolution=GRID_SPACING, bottom=2000.0)],
        unbounded_refinement_resolution=GRID_SPACING,
    ).write_to_realisation(realisation)

    nzcvm_settings = NZCVMSettings.read_from_defaults(DEFAULTS_VERSION)
    nzcvm_settings.write_to_realisation(realisation)
    redirected = realisation.read_text().replace(
        f'"{NZCVM_DATA_PREFIX}', f'"{nzcvm_data_root}/'
    )
    realisation.write_text(redirected)
    assert NZCVM_DATA_PREFIX not in json.dumps(json.loads(redirected)["nzcvm"])

    run("generate-domain", realisation, "--solver", "sw4")

    domain = DomainParameters.read_from_realisation(realisation)
    dataclasses.replace(
        domain, depth=min(domain.depth, MAX_DEPTH_KM)
    ).write_to_realisation(realisation)
    longitudes = domain.domain.corners[:, 1]
    latitudes = domain.domain.corners[:, 0]
    assert SYNTHETIC_LON[0] < longitudes.min() and longitudes.max() < SYNTHETIC_LON[1]
    assert SYNTHETIC_LAT[0] < latitudes.min() and latitudes.max() < SYNTHETIC_LAT[1]


def test_realisation_to_srf(realisation: Path, work: Path) -> None:
    """genslip and generic_slip2srf turn the realisation into an SRF."""
    require(realisation)
    srf_work = work / "srf_work"
    srf_work.mkdir()
    run(
        "realisation-to-srf", realisation, work / "realisation.srf",
        "--work-directory", srf_work,
        "--genslip-path", GENSLIP, "--generic-slip2srf-path", GENERIC_SLIP2SRF,
        cwd=srf_work,  # genslip drops dump_last_seed.txt in its working directory
    )  # fmt: skip


def test_check_srf(realisation: Path, work: Path) -> None:
    """The SRF agrees with the realisation it came from."""
    require(work / "realisation.srf")
    run("check-srf", realisation, work / "realisation.srf")


def test_srf_to_hdf5(work: Path) -> None:
    """The SRF in the HDF5 form SW4's `rupturehdf5` reads."""
    require(work / "realisation.srf")
    run("srf-to-hdf5", work / "realisation.srf", work / "srf.h5")


def test_generate_stoch(realisation: Path, work: Path) -> None:
    """The stoch file the HF simulation reads."""
    require(work / "realisation.srf")
    run(
        "generate-stoch",
        realisation,
        work / "realisation.srf",
        work / "realisation.stoch",
    )


def test_generate_station_coordinates(realisation: Path, work: Path) -> None:
    """Every fixture station lands in the domain and in SW4's station file."""
    require_domain(realisation)
    stations = work / "stations"
    stations.mkdir()
    run(
        "generate-station-coordinates", realisation, DATA / "stations.ll", stations,
        "--format", "sw4",
    )  # fmt: skip
    with h5py.File(stations / "stations.h5") as handle:
        assert sorted(handle) == STATIONS


def test_velocity_model(realisation: Path, work: Path) -> None:
    """NZCVM samples its (synthetic) data onto SW4's grid as a valid sfile."""
    require_domain(realisation)
    velocity_model = work / "velocity_model"
    velocity_model.mkdir()
    run(
        "create-nzvm-input", realisation, velocity_model / "sw4.json", "--format", "sw4"
    )
    run(
        "nzcvm", "generate", velocity_model / "sw4.json", velocity_model / "sw4.sfile",
        "--format", "sfile", "--n-threads", "2",
    )  # fmt: skip
    run("validate-sfile", velocity_model / "sw4.sfile")


def test_sw4(realisation: Path, work: Path) -> None:
    """SW4 (or its stand-in) runs the input file `create-sw4-input` writes."""
    sfile = work / "velocity_model/sw4.sfile"
    stations = work / "stations/stations.h5"
    require(work / "srf.h5", stations, sfile)
    sw4_work = work / "sw4"
    sw4_work.mkdir()
    run(
        "create-sw4-input", realisation, stations, work / "srf.h5", sfile, sw4_work,
        sw4_work / "input.in",
    )  # fmt: skip
    if sw4 := os.environ.get("SW4"):
        launcher = shlex.split(os.environ.get("SW4_LAUNCHER", ""))
        run(*launcher, sw4, "input.in", cwd=sw4_work)
    else:
        run(sys.executable, FAKE_SW4, "input.in", cwd=sw4_work)
    assert (sw4_work / "out.h5").exists()


def test_lf_to_xarray(work: Path) -> None:
    """SW4's station recording becomes the workflow's LF waveform dataset."""
    require(work / "sw4/out.h5")
    run("lf-to-xarray", work / "sw4/out.h5", work / "realisation.lf", "--format", "sw4")


def test_hf_sim(realisation: Path, work: Path) -> None:
    """The stochastic HF simulation at every station."""
    require(work / "realisation.stoch", work / "stations/stations.ll")
    run(
        "hf-sim", realisation, work / "realisation.stoch", work / "stations/stations.ll",
        work / "realisation.hf",
    )  # fmt: skip


def test_bb_sim(realisation: Path, work: Path) -> None:
    """LF and HF merge into broadband, as SW4 runs do (LF prefiltered)."""
    require(work / "realisation.lf", work / "realisation.hf")
    run(
        "bb-sim", realisation, DATA / "stations.vs30", work / "realisation.lf",
        work / "realisation.hf", work / "realisation.bb", "--filter", "hf",
    )  # fmt: skip


def test_im_calc(realisation: Path, work: Path) -> None:
    """Intensity measures for every station, all finite."""
    require(work / "realisation.bb")
    ko_matrices = work / "ko_matrices"
    ko_matrices.mkdir()
    output = work / "intensity_measures.h5"
    run(
        "im-calc", realisation, work / "realisation.bb", output,
        "--ko-directory", ko_matrices,
    )  # fmt: skip

    intensity_measures = xr.open_datatree(output)
    found_stations: set[str] = set()
    for node in intensity_measures.subtree:
        for name, variable in node.dataset.data_vars.items():
            assert np.isfinite(variable.values).all(), f"{node.path}/{name} has NaNs"
        if "station" in node.dataset.coords:
            found_stations.update(node.dataset.coords["station"].values.tolist())
    assert found_stations >= set(STATIONS)
