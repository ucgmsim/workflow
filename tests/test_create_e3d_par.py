"""Tests for `workflow.scripts.create_e3d_par`."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from velocity_modelling.bounding_box import BoundingBox
from workflow import defaults
from workflow.realisations import (
    DomainParameters,
    EMOD3DParameters,
    LogTrail,
    RealisationMetadata,
    Resolution,
    VelocityModelParameters,
)
from workflow.scripts import create_e3d_par

DEFAULTS_VERSION = defaults.DefaultsVersion.v24_2_2_4

# A 20 x 20 x 10 km domain at 0.5 km resolution: 40 x 40 x (20 + 1) points.
NX, NY, NZ = 40, 40, 21
VELOCITY_MODEL_FILE_SIZE = NX * NY * NZ * np.dtype(np.float32).itemsize


@dataclasses.dataclass
class Inputs:
    """The input files `create_e3d_par` needs."""

    realisation: Path
    srf: Path
    velocity_model: Path
    stations: Path
    output: Path


def write_velocity_model(directory: Path, size: int) -> None:
    """Write p, s and density files of `size` bytes, named as the defaults expect."""
    parameters = EMOD3DParameters.read_from_defaults(DEFAULTS_VERSION)
    for filename in [parameters.pmodfile, parameters.smodfile, parameters.dmodfile]:
        (directory / filename).write_bytes(b"\0" * size)


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    """A realisation with a velocity model of the right size for its domain.

    Resolution 0.5 km, min_vs 0.5 km/s and a 35 s duration are chosen so the
    duration parameters come out exact: flo = 0.2 Hz, the simulation is
    extended by 3 / flo = 15 s to 50 s, and dt = 0.025 s.
    """
    realisation = tmp_path / "realisation.json"
    RealisationMetadata(
        name="Test_REL01", version="1", defaults_version=DEFAULTS_VERSION
    ).write_to_realisation(realisation)
    DomainParameters(
        domain=BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.5, 172.5]),
            bearing=30.0,
            extent_x=20.0,
            extent_y=20.0,
        ),
        depth=10.0,
        duration=35.0,
    ).write_to_realisation(realisation)
    VelocityModelParameters(
        min_vs=0.5,
        version="2.09",
        topo_type="BULLDOZED",
        ds_multiplier=1.2,
        vs30=500.0,
        s_wave_velocity=3500.0,
        rrup_interpolants=np.array([[5.0, 8.0], [50.0, 50.0]]),
        fault_buffer=2.0,
    ).write_to_realisation(realisation)
    Resolution(resolution=0.5).write_to_realisation(realisation)

    srf = tmp_path / "realisation.srf"
    srf.write_text("")
    velocity_model = tmp_path / "Velocity_Model"
    velocity_model.mkdir()
    write_velocity_model(velocity_model, VELOCITY_MODEL_FILE_SIZE)
    stations = tmp_path / "stations"
    stations.mkdir()
    (stations / "stations.statcords").write_text("")

    return Inputs(realisation, srf, velocity_model, stations, tmp_path / "LF")


def run(inputs: Inputs, **kwargs: str) -> dict[str, str]:
    """Run `create_e3d_par` and parse the written e3d.par into a dictionary."""
    create_e3d_par.create_e3d_par(
        inputs.realisation,
        inputs.srf,
        inputs.velocity_model,
        inputs.stations,
        inputs.output,
        **kwargs,
    )
    lines = (inputs.output / "e3d.par").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines)


def test_create_e3d_par(inputs: Inputs) -> None:
    par = run(inputs)

    # Domain: the grid matches the velocity model, centred on the domain origin.
    domain = DomainParameters.read_from_realisation(inputs.realisation).domain
    assert (int(par["nx"]), int(par["ny"]), int(par["nz"])) == (NX, NY, NZ)
    assert float(par["h"]) == 0.5
    assert float(par["modellat"]) == pytest.approx(domain.origin[0])
    assert float(par["modellon"]) == pytest.approx(domain.origin[1])
    assert float(par["modelrot"]) == pytest.approx(domain.great_circle_bearing)

    # Duration: 50 s at dt = 0.025 s, with timeslices every dtts = 20 steps.
    assert float(par["flo"]) == pytest.approx(0.2)
    assert float(par["dt"]) == pytest.approx(0.025)
    assert int(par["nt"]) == 2000
    # The computed nt overrides the default dump_itinc of 4000.
    assert int(par["dump_itinc"]) == 2000
    assert int(par["ts_total"]) == 100
    assert int(par["restart_itinc"]) == 667

    # Strings and paths are quoted, numbers are not.
    assert par["faultfile"] == f'"{inputs.srf}"'
    assert par["vmoddir"] == f'"{inputs.velocity_model}"'
    assert par["seiscords"] == f'"{inputs.stations / "stations.statcords"}"'
    assert par["name"] == par["restartname"] == '"Test_REL01"'
    assert par["version"] == '"3.0.13-mpi"'
    assert par["pmodfile"] == '"vp3dfile.p"'
    assert par["stype"] == '"2tri-p10-h20"'
    assert par["dtts"] == "20"
    assert par["tzero"] == "0.6"

    # Output directories are created, and the timeslice file lives in OutBin.
    for key, name in [
        ("main_dump_dir", "OutBin"),
        ("seisdir", "SeismoBin"),
        ("restartdir", "Restart"),
        ("logdir", "Log"),
        ("ts_out_dir", "TSFiles"),
        ("slipout", "SlipOut"),
    ]:
        assert par[key] == f'"{inputs.output / name}"'
        assert (inputs.output / name).is_dir()
    assert par["ts_file"] == f'"{inputs.output / "OutBin" / "Test_REL01_xyts.e3d"}"'

    # Running the stage is recorded in the realisation, and the defaults it
    # used are written back.
    assert len(LogTrail.read_from_realisation(inputs.realisation).log) == 1
    assert EMOD3DParameters.read_from_realisation(
        inputs.realisation
    ) == EMOD3DParameters.read_from_defaults(DEFAULTS_VERSION)


def test_create_e3d_par_is_rerunnable(inputs: Inputs) -> None:
    """A second run over existing output directories overwrites e3d.par."""
    run(inputs)
    par = run(inputs, emod3d_version="3.0.8")
    assert par["version"] == '"3.0.8-mpi"'
    assert len(LogTrail.read_from_realisation(inputs.realisation).log) == 2


def test_realisation_emod3d_section_overrides_defaults(inputs: Inputs) -> None:
    parameters = dataclasses.replace(
        EMOD3DParameters.read_from_defaults(DEFAULTS_VERSION),
        pmodfile="custom.p",
        dtts=10,
        fmax=10.0,
    )
    parameters.write_to_realisation(inputs.realisation)
    (inputs.velocity_model / "custom.p").write_bytes(b"\0" * VELOCITY_MODEL_FILE_SIZE)

    par = run(inputs)

    assert par["pmodfile"] == '"custom.p"'
    assert par["fmax"] == "10.0"
    assert par["dtts"] == "10"
    # Halving dtts doubles the number of timeslices.
    assert int(par["ts_total"]) == 200


@pytest.mark.parametrize("size_change", [-4, 4])
def test_velocity_model_size_mismatch(inputs: Inputs, size_change: int) -> None:
    """A velocity model generated for a different domain is rejected."""
    smodfile = EMOD3DParameters.read_from_defaults(DEFAULTS_VERSION).smodfile
    (inputs.velocity_model / smodfile).write_bytes(
        b"\0" * (VELOCITY_MODEL_FILE_SIZE + size_change)
    )

    with pytest.raises(
        RuntimeError,
        match=rf"vs3dfile\.s.*expected: {VELOCITY_MODEL_FILE_SIZE}, "
        rf"found: {VELOCITY_MODEL_FILE_SIZE + size_change}",
    ):
        run(inputs)
    assert not (inputs.output / "e3d.par").exists()


def test_missing_velocity_model_files_only_warn(inputs: Inputs) -> None:
    """Velocity model files not on disk yet cannot be checked, which is not an error."""
    for path in inputs.velocity_model.iterdir():
        path.unlink()

    with capture_logs() as logs:
        par = run(inputs)

    warnings = [log for log in logs if log["log_level"] == "warning"]
    assert len(warnings) == 3
    assert all(isinstance(log["error"], FileNotFoundError) for log in warnings)
    assert int(par["nx"]) == NX


@pytest.mark.parametrize("key", ["faultfile", "seiscords", "vmoddir"])
def test_missing_inputs(inputs: Inputs, key: str) -> None:
    missing = {
        "faultfile": inputs.srf,
        "seiscords": inputs.stations / "stations.statcords",
        "vmoddir": inputs.velocity_model,
    }[key]
    if missing.is_dir():
        for path in missing.iterdir():
            path.unlink()
        missing.rmdir()
    else:
        missing.unlink()

    with pytest.raises(ValueError, match=f"The {key} path does not exist"):
        run(inputs)
    assert not (inputs.output / "e3d.par").exists()
