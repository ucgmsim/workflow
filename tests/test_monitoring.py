import asyncio
import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from workflow.monitoring import (
    monitor_emod3d,
)


@pytest.fixture
def sample_log_content() -> str:
    """Provides sample content for an EMOD3D log file."""
    return """
    Some Header Info
    nt= 2000
    
    Time Step Header Line (Ignored)
      100   1.5   0.5   100.0  90.0  2.0  95.0  10.  92.5
      200   1.5   0.5   100.0  90.0  2.0  95.0  20.  92.5
    """


def test_regex_nt_match() -> None:
    line = "   nt= 2000 "
    match = monitor_emod3d.NT_REGEX.match(line)
    assert match is not None
    assert match.group("nt") == "2000"


def test_regex_step_match() -> None:
    line = "  100   0.5   0.1   10.0   99.0   1.0   98.0  123.  97.0"
    match = monitor_emod3d.STEP_REGEX.match(line)
    assert match is not None
    assert match.group("time_step") == "100"
    assert match.group("cpu_cumulative") == "123"


def test_log_progress_update_parsing(tmp_path: Path, sample_log_content: str) -> None:
    log_file = tmp_path / "test.rlog"
    log_file.write_text(sample_log_content)

    update = monitor_emod3d.log_progress_update(log_file)

    assert update.event_type == monitor_emod3d.EventType.MODIFIED
    assert update.path == log_file
    assert update.nt == 2000
    assert update.current == 200
    assert update.cumulative_time == 20

    assert update.tps == pytest.approx(5.1)

    assert update.seconds_remaining == 353


def test_find_fresh_files(tmp_path: Path) -> None:
    now = time.time()

    old_file = tmp_path / "old.rlog"
    old_file.touch()
    timestamp_old = now - 600

    os.utime(old_file, (timestamp_old, timestamp_old))

    new_file = tmp_path / "new.rlog"
    new_file.touch()

    fresh = monitor_emod3d.find_fresh_files(tmp_path, now, "*.rlog", stale_seconds=300)

    assert new_file in fresh
    assert old_file not in fresh


@pytest.mark.asyncio
async def test_monitor_files_producer(tmp_path: Path) -> None:
    """
    Tests the 'Producer' logic with mocked time and sleep.
    1. Finds a new file -> Emits CREATED and MODIFIED.
    2. Waits.
    3. Sees file hasn't changed -> Emits nothing.
    4. File gets deleted (or becomes stale) -> Emits STALE.
    """
    queue = asyncio.Queue()
    f = tmp_path / "sim.rlog"
    f2 = tmp_path / "sim2.log"
    f2.touch()
    f.touch()

    with (
        patch("workflow.monitoring.monitor_emod3d.find_fresh_files") as mock_find,
        patch("workflow.monitoring.monitor_emod3d.time.time") as mock_time,
        patch("workflow.monitoring.monitor_emod3d.asyncio.sleep"),
    ):
        mock_find.side_effect = [[f], [f2, f], [f2], []]

        mock_time.side_effect = [1000.0, 1005.0, 1010.0, 1015.0, 1020.0]

        task = asyncio.create_task(
            monitor_emod3d.monitor_files(queue, tmp_path, "*.rlog", 60)
        )
        timeout = 1.0
        event_created = await asyncio.wait_for(queue.get(), timeout)
        assert event_created.event_type == monitor_emod3d.EventType.CREATED
        assert event_created.path.stem == "sim"

        event_mod = await asyncio.wait_for(queue.get(), timeout)
        assert event_mod.event_type == monitor_emod3d.EventType.MODIFIED
        assert event_mod.path.stem == "sim"

        event_created = await asyncio.wait_for(queue.get(), timeout)
        assert event_created.event_type == monitor_emod3d.EventType.CREATED
        assert event_created.path.stem == "sim2"

        event_mod = await asyncio.wait_for(queue.get(), timeout)
        assert event_mod.event_type == monitor_emod3d.EventType.MODIFIED
        assert event_mod.path.stem == "sim2"

        event_stale = await asyncio.wait_for(queue.get(), timeout)
        assert event_stale.event_type == monitor_emod3d.EventType.STALE
        assert event_stale.path.stem == "sim"

        event_stale = await asyncio.wait_for(queue.get(), timeout)
        assert event_stale.event_type == monitor_emod3d.EventType.STALE
        assert event_stale.path.stem == "sim2"

        task.cancel()


@pytest.mark.asyncio
async def test_track_rlog_progress_state_machine(tmp_path: Path) -> None:
    """
    Tests the 'Consumer' logic (State Machine) by mocking the Progress class.
    """
    queue = asyncio.Queue()

    p1 = tmp_path / "run_1-00000.rlog"
    p2 = tmp_path / "run_2-00000.rlog"
    suffix = "-00000.rlog"

    await queue.put(monitor_emod3d.ProgressUpdate(monitor_emod3d.EventType.CREATED, p1))

    await queue.put(
        monitor_emod3d.ProgressUpdate(
            monitor_emod3d.EventType.MODIFIED,
            p1,
            nt=1000,
            current=10,
            tps=5.0,
            cumulative_time=2,
            seconds_remaining=200,
        )
    )

    await queue.put(monitor_emod3d.ProgressUpdate(monitor_emod3d.EventType.CREATED, p2))

    await queue.put(monitor_emod3d.ProgressUpdate(monitor_emod3d.EventType.STALE, p1))

    await queue.put(None)

    with (
        patch("workflow.monitoring.monitor_emod3d.monitor_files"),
        patch("workflow.monitoring.monitor_emod3d.Progress") as mock_progress_call,
    ):
        mock_progress_instance = mock_progress_call.return_value
        mock_progress_instance.__enter__.return_value = mock_progress_instance

        await monitor_emod3d.track_rlog_progress(queue, tmp_path, "*.rlog", suffix, 60)

    mock_progress_instance.add_task.assert_any_call(
        "",
        filename="run_1",
        total=None,
        tps=0.0,
        elapsed_time=timedelta(seconds=0),
        time_remaining="-:--:--",
        start=False,
    )

    mock_progress_instance.add_task.assert_any_call(
        "",
        filename="run_2",
        total=None,
        tps=0.0,
        elapsed_time=timedelta(seconds=0),
        time_remaining="-:--:--",
        start=False,
    )

    assert mock_progress_instance.add_task.call_count == 2

    assert mock_progress_instance.start_task.called
    assert mock_progress_instance.update.called

    assert mock_progress_instance.remove_task.called
