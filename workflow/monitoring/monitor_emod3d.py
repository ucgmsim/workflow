import asyncio
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
from pathlib import Path

import typer
from asyncinotify import Event, Inotify, Mask, Watch
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

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
    return [
        f.resolve()
        for f in root.rglob(file_glob)
        if now - f.stat().st_mtime < stale_seconds
    ]


class EventType(Enum):
    CREATED = auto()
    MODIFIED = auto()
    STALE = auto()


@dataclass
class ProgressUpdate:
    event_type: EventType
    path: Path
    nt: int | None = None
    current: int | None = None
    cumulative_time: int | None = None
    tps: float | None = None


def log_progress_update(event_path: Path) -> ProgressUpdate:
    nt = None
    current = None
    cumulative_time = None
    tps = None
    alpha = 0.3
    with open(event_path, "r") as fp:
        for line in fp:
            if m := NT_REGEX.match(line):
                print(line)
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

    return ProgressUpdate(
        event_type=EventType.MODIFIED,
        path=event_path,
        nt=nt,
        current=current,
        cumulative_time=cumulative_time,
        tps=tps,
    )


async def monitor_files(
    queue: asyncio.Queue[ProgressUpdate | None],
    root: Path,
    file_glob: str,
    stale_seconds: int,
) -> None:
    tracked: dict[Path, Watch] = {}
    with Inotify() as inotify:
        for tracked_file in find_fresh_files(
            root, time.time(), file_glob, stale_seconds
        ):
            await queue.put(ProgressUpdate(EventType.CREATED, tracked_file))
            tracked[tracked_file] = inotify.add_watch(tracked_file.parent, Mask.MODIFY)
            await queue.put(log_progress_update(tracked_file))

        last_scan = time.time()
        while True:
            try:
                event: Event = await asyncio.wait_for(
                    inotify.get(), timeout=stale_seconds
                )
            except asyncio.TimeoutError:
                await queue.put(None)
                print('Bailing from file monitoring!')
                return
            print(event)

            if event_path := event.path:
                await queue.put(log_progress_update(event_path))

            now = time.time()
            if now - last_scan > stale_seconds:
                fresh_files = set(find_fresh_files(root, now, file_glob, stale_seconds))

                for stale_file in set(tracked) - fresh_files:
                    await queue.put(ProgressUpdate(EventType.STALE, stale_file))
                    watch = tracked.pop(stale_file)
                    inotify.rm_watch(watch)
                for tracked_file in fresh_files:
                    await queue.put(ProgressUpdate(EventType.CREATED, tracked_file))
                    tracked[tracked_file] = inotify.add_watch(
                        tracked_file.parent, Mask.MODIFY
                    )
                last_scan = now


async def track_rlog_progress(
    event_queue: asyncio.Queue[ProgressUpdate | None],
    root: Path,
    file_glob: str,
    suffix: str,
    stale_seconds: int,
):
    asyncio.create_task(monitor_files(event_queue, root, file_glob, stale_seconds))
    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.fields[tps]:.2f} steps/s"),
        TextColumn("[blue]{task.fields[elapsed_time]}"),
        TimeRemainingColumn(),
        speed_estimate_period=stale_seconds,
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
                        start=False,
                    )
                    progress_tasks[event.path] = task_id
                case ProgressUpdate(event_type=EventType.MODIFIED):
                    task_id = progress_tasks[event.path]
                    print(event)
                    if event.nt and event.current and task_id not in started:
                        progress.update(task_id, total=event.nt)
                        progress.start_task(task_id)
                        started.add(task_id)

                    progress.update(
                        task_id,
                        completed=event.current,
                        tps=event.tps or 0.0,
                        elapsed_time=timedelta(seconds=event.cumulative_time or 0),
                    )
                case ProgressUpdate(event_type=EventType.STALE):
                    task_id = progress_tasks.pop(event.path)
                    progress.remove_task(task_id)
                case None:
                    return


@app.command()
def monitor_lf(
    log_dir: Path = Path("."),
    file_glob: str = "*.rlog",
    stale_seconds: int = 5 * 60,
    suffix: str = "-00000.rlog",
) -> None:
    event_queue = asyncio.Queue()

    asyncio.run(
        track_rlog_progress(event_queue, log_dir, file_glob, suffix, stale_seconds),
    )


if __name__ == "__main__":
    app()
