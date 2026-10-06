"""Stand-in for the SW4 solver in the end-to-end smoke test.

SW4 is far too expensive to run in CI, so this replaces it with a script that
checks the input file `create-sw4-input` wrote is one SW4 could run (every
command it needs is present and every file it references exists), then writes
a station recording in SW4's `rechdf5` output format (SW4 User Guide, Section
12.9) for the stations SW4 would have recorded.

The recordings are a decaying sinusoid, not physics. The point is that the
files on either side of the solver line up, so `lf-to-xarray` and everything
after it get a recording of the shape a real run produces.

Usage: python fake_sw4.py input.in
"""

import shlex
import sys
from pathlib import Path

import h5py
import numpy as np

REQUIRED_COMMANDS = {"fileio", "grid", "time", "rupturehdf5", "sfile", "rechdf5"}
DT = 0.02
"""Time step of the fake recording, in seconds."""


def parse_input(input_path: Path) -> dict[str, list[dict[str, str]]]:
    """Read an SW4 input file into a map of command name to its occurrences."""
    commands: dict[str, list[dict[str, str]]] = {}
    for line in input_path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, *parameters = shlex.split(line)
        commands.setdefault(name, []).append(
            dict(parameter.split("=", 1) for parameter in parameters)
        )
    return commands


def check_input(commands: dict[str, list[dict[str, str]]]) -> None:
    """Fail the way SW4 would if the input file cannot be run."""
    missing = REQUIRED_COMMANDS - commands.keys()
    if missing:
        sys.exit(f"fake-sw4: input file is missing commands {sorted(missing)}")

    fileio_path = Path(commands["fileio"][0]["path"])
    references = [
        fileio_path,
        Path(commands["rupturehdf5"][0]["file"]),
        Path(commands["sfile"][0]["directory"]) / commands["sfile"][0]["filename"],
        Path(commands["rechdf5"][0]["infile"]),
    ]
    for reference in references:
        if not reference.exists():
            sys.exit(
                f"fake-sw4: input file references {reference}, which does not exist"
            )

    grid = commands["grid"][0]
    for key in ("h", "lat", "lon", "az"):
        if key not in grid:
            sys.exit(f"fake-sw4: grid command has no {key}=")
    float(commands["time"][0]["t"])


def write_recording(commands: dict[str, list[dict[str, str]]]) -> Path:
    """Write SW4's `rechdf5` output for every station in the station file."""
    rechdf5 = commands["rechdf5"][0]
    output_path = Path(commands["fileio"][0]["path"]) / rechdf5["outfile"]
    duration = float(commands["time"][0]["t"])
    npts = int(duration / DT) + 1
    time = np.arange(npts) * DT

    supergrid = commands.get("supergrid", [{}])[0]
    supergrid_width = float(supergrid.get("width", 0.0))

    with (
        h5py.File(rechdf5["infile"], "r") as stations,
        h5py.File(output_path, "w") as output,
    ):
        output.create_dataset("DELTA", data=np.array([DT], dtype=np.float32))
        output.create_dataset("DOWNSAMPLE", data=np.array([1], dtype=np.int32))
        output.create_dataset("UNIT", data=np.bytes_("m/s"))
        output.create_dataset("SGWIDTH", data=np.array([supergrid_width]))
        for i, (name, station) in enumerate(stations.items()):
            group = output.create_group(name)
            group.create_dataset("STLA,STLO,STDP", data=station["STLA,STLO,STDP"][:])
            group.create_dataset("NPTS", data=np.array([npts], dtype=np.int32))
            group.create_dataset("ISNSEW", data=np.array([1], dtype=np.int32))
            group.create_dataset("SGDEPTH", data=np.array([supergrid_width]))
            group.create_dataset("SGDEPTHGP", data=np.array([1.0]))
            for j, component in enumerate(("EW", "NS", "UP")):
                phase = 0.3 * i + 0.7 * j
                waveform = 0.01 * np.exp(-time / 3) * np.sin(2 * np.pi * time + phase)
                group.create_dataset(component, data=waveform.astype(np.float32))
    return output_path


def main() -> None:
    """Check the input file and write a fake station recording."""
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    commands = parse_input(Path(sys.argv[1]))
    check_input(commands)
    print(f"fake-sw4: wrote {write_recording(commands)}")


if __name__ == "__main__":
    main()
