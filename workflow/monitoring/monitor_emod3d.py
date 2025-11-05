"""Monitor EMOD3D job progress from the command line."""

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from qcore import cli

app = typer.Typer()


NT_REGEX = re.compile(r"^\s+nt=\s*(?P<nt>\d+)\s*$")
STEP_REGEX = re.compile(
    r"""
    ^\s*
    (?P<time_step>\d+)          # Time step number
    \s+
    (?P<cpu_transfer>[\d.]+)    # CPU (Data Transfer)
    \s+
    (?P<mpi_only>[\d.]+)        # MPI_only
    \s+
    (?P<mbyte>[\d.]+)           # Mbyte
    \s+
    (?P<real_transfer>[\d.]+)   # %Real (Data Transfer)
    \s+
    (?P<cpu_comp>[\d.]+)        # CPU (Transfer & Computation)
    \s+
    (?P<real_comp>[\d.]+)       # %Real (Transfer & Computation)
    \s+
    (?P<cpu_cumulative>\d+)\.          # CPU (Cumulative) - integer followed by a dot
    \s+
    (?P<real_cumulative>[\d.]+)        # %Real (Cumulative)
    \s*$
    """,
    re.VERBOSE,
)


def find_fresh_files(
    root: Path, now: float, file_glob: str, stale_seconds: int
) -> list[Path]:
    """Recursively find all files matching a given file glob in a directory modified after a given time.

    Parameters
    ----------
    root : Path
        The path to begin searching for.
    file_glob : str
        The file glob for files to consider.
    stale_seconds : int
        The number of seconds to consider a file as recently modified.

    Returns
    -------
    list[Path]
        The paths matching `file_glob` modified within the last `stale_seconds` seconds.
    """
    return [
        f.resolve()
        for f in root.rglob(file_glob)
        if now - f.stat().st_mtime < stale_seconds
    ]


class EventType(Enum):
    """Event type enum."""

    CREATED = auto()
    """File was created."""
    MODIFIED = auto()
    """File was modified."""
    STALE = auto()
    """File has been deleted or was not recently modified."""


@dataclass
class ProgressUpdate:
    """Dataclass describing file updates."""

    event_type: EventType
    """The type of event."""
    path: Path
    """The path of the file updated."""
    nt: int | None = None
    """The discovered total number of timesteps."""
    current: int | None = None
    """The current timestep."""
    cumulative_time: int | None = None
    """The cumulative time for the simulation since the first timestep."""
    # Exponential weighted moving average is used here instead of simple
    # average to allow the process to adapt to transient changes in the cluster
    # speed, and to converge quickly to a good estimate of completion time
    # (the first 100 timesteps are typically the slowest).
    tps: float | None = None
    """Timesteps per second (exponential weighted average)."""
    seconds_remaining: int | None = None
    """Seconds remaining estimated from tps"""


def log_progress_update(event_path: Path) -> ProgressUpdate:
    """Parse an EMOD3D log file and return a progress update.

    Parameters
    ----------
    event_path : Path
        The path of the log file to read.

    Returns
    -------
    ProgressUpdate
        The simulation's current progress.
    """
    nt = None
    current = None
    cumulative_time = None
    tps = None
    # An alpha = 0.3 means that 30% of the average tps value is represented
    # by the time to compute the last 100 timesteps.
    alpha = 0.3
    with open(event_path, "r") as fp:
        for line in fp:
            if m := NT_REGEX.match(line):
                nt = int(m.group("nt"))
            elif m := STEP_REGEX.match(line):
                tps = tps or 0.0
                cumulative_time = cumulative_time or 0
                current = current or 0
                next_ts = int(m.group("time_step"))
                now = int(m.group("cpu_cumulative"))
                dt = now - cumulative_time
                instant_tps = (next_ts - current) / dt
                current = next_ts
                cumulative_time = now
                tps = alpha * instant_tps + (1 - alpha) * tps
    if tps and nt and current:
        seconds_remaining = round((nt - current) / tps)
    return ProgressUpdate(
        event_type=EventType.MODIFIED,
        path=event_path,
        nt=nt,
        current=current,
        cumulative_time=cumulative_time,
        tps=tps,
        seconds_remaining=seconds_remaining,
    )


async def monitor_files(
    queue: asyncio.Queue[ProgressUpdate | None],
    root: Path,
    file_glob: str,
    stale_seconds: int,
) -> None:
    """Monitor LF log files in a given directory and publish updates to an event queue.

    Parameters
    ----------
    event_queue : Queue
        The event queue to publish updates to.
    root : Path
        The path to watch for new LF log files in.
    file_glob : str
        The glob pattern used to match LF log files.
    stale_seconds : int
        The number of seconds that must pass for a file to be considered stale to stop watching.
    """
    # Contains a dictionary of path -> modified time. This is the most reliable
    # way to track file changes without hashing. Three other methods were
    # also considered:
    #
    # 1. INotify. This is canonical on local filesystems, but doesn't work
    # on network mounted filesystems (i.e. most filesystems on HPC) because
    # they don't typically implement inotify.
    #
    # 2. Using a global modified time and comparing file modification times to
    # the global modified time - poll interval. Network mounted filesystems
    # record the modification time on the filesystem they've mounted. They
    # can take some time to sync changes, which means that you often miss
    # updates that occur out-of-order.
    #
    # 3. Using a dictionary of path -> file size. This works just fine but
    # is less robust that checking if the file is modified. It also means
    # you can't prompt an update using touch.
    tracked: dict[Path, float] = {}
    for tracked_file in find_fresh_files(root, time.time(), file_glob, stale_seconds):
        await queue.put(ProgressUpdate(EventType.CREATED, tracked_file))
        tracked[tracked_file] = tracked_file.stat().st_mtime
        await queue.put(log_progress_update(tracked_file))

    poll_interval = 5
    while True:
        await asyncio.sleep(poll_interval)
        now = time.time()
        fresh_files = set(find_fresh_files(root, now, file_glob, stale_seconds))
        tracked_files = set(tracked)
        for stale_file in tracked_files - fresh_files:
            await queue.put(ProgressUpdate(EventType.STALE, stale_file))
            del tracked[stale_file]
        for updated_file in tracked_files & fresh_files:
            mtime = updated_file.stat().st_mtime
            if tracked[updated_file] != mtime:
                await queue.put(log_progress_update(updated_file))
                tracked[updated_file] = mtime
        for tracked_file in fresh_files - tracked_files:
            await queue.put(ProgressUpdate(EventType.CREATED, tracked_file))
            tracked[tracked_file] = tracked_file.stat().st_size


async def track_rlog_progress(
    event_queue: asyncio.Queue[ProgressUpdate | None],
    root: Path,
    file_glob: str,
    suffix: str,
    stale_seconds: int,
) -> None:
    """Track LF progress using Rich progress bars.

    Parameters
    ----------
    event_queue : Queue
        The event queue to publish updates to.
    root : Path
        The path to watch for new LF log files in.
    file_glob : str
        The glob pattern used to match LF log files.
    suffix : str
        The common suffix of rlog files to strip for presenting the name with the progress bars.
    stale_seconds : int
        The number of seconds that must pass for a file to be considered stale to stop watching.
    """
    asyncio.create_task(monitor_files(event_queue, root, file_glob, stale_seconds))
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.fields[tps]:.2f} steps/s"),
        TextColumn("[blue]{task.fields[elapsed_time]}"),
        TextColumn("[cyan]{task.fields[time_remaining]}"),
    )
    progress_tasks = {}
    started = set()
    with progress:
        while True:
            event = await event_queue.get()
            match event:
                case ProgressUpdate(event_type=EventType.CREATED):
                    task_id = progress.add_task(
                        "",
                        filename=event.path.name[: -len(suffix)],
                        total=None,
                        tps=0.0,
                        elapsed_time=timedelta(seconds=0),
                        time_remaining="-:--:--",
                        start=False,
                    )
                    progress_tasks[event.path] = task_id
                case ProgressUpdate(event_type=EventType.MODIFIED):
                    task_id = progress_tasks[event.path]
                    if event.nt and event.current and task_id not in started:
                        progress.update(task_id, total=event.nt)
                        progress.start_task(task_id)
                        started.add(task_id)

                    progress.update(
                        task_id,
                        completed=event.current,
                        tps=event.tps or 0.0,
                        elapsed_time=timedelta(seconds=event.cumulative_time or 0),
                        time_remaining=timedelta(seconds=event.seconds_remaining)
                        if event.seconds_remaining
                        else "-:--:---",
                    )
                case ProgressUpdate(event_type=EventType.STALE):
                    task_id = progress_tasks.pop(event.path)
                    progress.remove_task(task_id)
                case None:
                    # This will occur if any early termination logic is implemented.
                    return


@cli.from_docstring(app)
def monitor_lf(
    log_dir: Annotated[Path, typer.Option(file_okay=False, readable=True)] = Path("."),
    file_glob: str = "*.rlog",
    stale_seconds: Annotated[int, typer.Option(min=1)] = 5 * 60,
    suffix: str = "-00000.rlog",
) -> None:
    """Monitor a directory for LF progress and publish the results as progress bars.

    Paramaters
    ----------
    log_dir : Path
        The directory to search for log files.
    file_glob : str
        A glob for files to consider for processing.
    stale_seconds : int
        The timeout for a file to be considered stale (usually this means EMOD3D terminated). These jobs are removed.
    suffix : str
        The suffix to strip from the run log files when displaying them alongside progress.
    """
    event_queue = asyncio.Queue()

    asyncio.run(
        track_rlog_progress(event_queue, log_dir, file_glob, suffix, stale_seconds),
    )


if __name__ == "__main__":
    app()
