import concurrent.futures
import subprocess
import time
import types
import uuid
from pathlib import Path
from typing import Any

import pytest
import structlog.testing

from workflow import log_utils


@log_utils.log_call()
def foo(a: int, b: int) -> int:
    return a + b


def test_basic_log() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    _ = foo(1, 2)

    assert len(log_capture.entries) == 2
    call_log = log_capture.entries[0]
    assert call_log["event"] == "called"
    assert call_log["function"] == "foo"
    assert call_log["a"] == 1
    assert call_log["b"] == 2

    return_log = log_capture.entries[1]
    assert return_log["event"] == "completed"
    assert return_log["function"] == "foo"
    assert return_log["result"] == 3


@log_utils.log_call(exclude_args={"b"})
def foo_less_b(a: int, b: int) -> int:
    return a + b


def test_excluded_log() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    _ = foo_less_b(1, 2)

    assert len(log_capture.entries) == 2
    call_log = log_capture.entries[0]
    assert call_log["event"] == "called"
    assert call_log["function"] == "foo_less_b"
    assert call_log["a"] == 1
    assert "b" not in call_log

    return_log = log_capture.entries[1]
    assert return_log["event"] == "completed"
    assert return_log["function"] == "foo_less_b"
    assert return_log["result"] == 3


@log_utils.log_call(action_name="FOOBAR")
def bar(a: Any) -> None:
    pass


def test_renamed_bar() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    bar(1)

    assert len(log_capture.entries) == 2
    call_log = log_capture.entries[0]
    assert call_log["event"] == "called"
    assert call_log["function"] == "FOOBAR"
    assert call_log["a"] == 1

    return_log = log_capture.entries[1]
    assert return_log["event"] == "completed"
    assert return_log["function"] == "FOOBAR"


@log_utils.log_call(include_result=False)
def baz(a: Any) -> int:
    return 1


def test_no_result() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    baz(1)

    assert len(log_capture.entries) == 2
    call_log = log_capture.entries[0]
    assert call_log["event"] == "called"
    assert call_log["function"] == "baz"
    assert call_log["a"] == 1

    return_log = log_capture.entries[1]
    assert return_log["event"] == "completed"
    assert return_log["function"] == "baz"
    assert "result" not in return_log


@log_utils.log_call()
def failing_function() -> None:
    raise ValueError("This function should fail!")


def test_failing_function() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    with pytest.raises(ValueError):
        failing_function()

    assert len(log_capture.entries) == 2
    return_log = log_capture.entries[1]
    assert return_log["event"] == "failed"
    assert return_log["function"] == "failing_function"
    assert "error" in return_log


def test_successful_check_call_log(tmp_path: Path) -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    test_file = tmp_path / "test.txt"
    test_file.touch()
    log_utils.log_check_call(["ls", str(tmp_path)])

    assert len(log_capture.entries) == 2
    execution_message = log_capture.entries[0]
    assert execution_message["event"] == "executing"
    assert execution_message["command"] == "ls"

    completion_message = log_capture.entries[1]
    assert completion_message["event"] == "completed"
    assert "stdout" in completion_message and "test.txt" in completion_message["stdout"]


def test_failing_check_call_log() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    with pytest.raises(subprocess.CalledProcessError):
        log_utils.log_check_call(["ls", "/bad-path"])

    assert len(log_capture.entries) == 2
    completion_message = log_capture.entries[1]
    assert completion_message["event"] == "failed"
    assert completion_message["command"] == "ls"
    assert (
        "stderr" in completion_message and "/bad-path" in completion_message["stderr"]
    )


def test_repeated_logs() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    logger_name = "test_repeated"
    for _ in range(3):
        log_utils.get_logger(logger_name)

    logger = log_utils.get_logger(logger_name)
    logger.info("Test message for repeated logs")

    assert (
        sum(
            1
            for log in log_capture.entries
            if log["event"] == "Test message for repeated logs"
        )
        == 1
    )


def _thread_worker(logger_name: str) -> None:
    logger = log_utils.get_logger(logger_name)
    logger.info("Threaded log message")


def test_thread_safety() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    logger_name = "test_thread"
    num_threads = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(_thread_worker, [logger_name] * num_threads)

    time.sleep(0.1)

    assert (
        sum(1 for log in log_capture.entries if log["event"] == "Threaded log message")
        == num_threads
    )


def test_log_call_includes_result_by_default() -> None:
    # Decorate inside the test so the decorator default is evaluated at call time.
    @log_utils.log_call()
    def add(a: int, b: int) -> int:
        return a + b

    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    assert add(2, 5) == 7

    assert len(log_capture.entries) == 2
    assert log_capture.entries[1]["event"] == "completed"
    assert log_capture.entries[1]["result"] == 7


def test_check_call_logs_id_and_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    logger_names: list[Any] = []
    original_get_logger = log_utils.get_logger

    def recording_get_logger(name: str) -> Any:
        logger_names.append(name)
        return original_get_logger(name)

    monkeypatch.setattr(log_utils, "get_logger", recording_get_logger)

    (tmp_path / "test.txt").touch()
    log_utils.log_check_call(["ls", "-1", str(tmp_path)])
    log_utils.log_check_call(["ls", "-1", str(tmp_path)])

    assert len(logger_names) == 2
    assert all(isinstance(name, str) for name in logger_names)

    assert len(log_capture.entries) == 4
    first_execute, first_complete, second_execute, _ = log_capture.entries
    assert first_execute["event"] == "executing"
    assert first_execute["args"] == ["-1", str(tmp_path)]
    assert first_complete["stdout"] == "test.txt\n"

    first_id = first_execute["id"]
    assert isinstance(first_id, str)
    assert str(uuid.UUID(first_id)) == first_id
    assert first_complete["id"] == first_id
    assert second_execute["id"] != first_id


def test_failing_check_call_logs_code_and_output() -> None:
    log_capture = structlog.testing.LogCapture()
    structlog.configure(processors=[log_capture])

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        log_utils.log_check_call(["sh", "-c", "echo out; echo err >&2; exit 3"])

    assert exc_info.value.returncode == 3
    assert len(log_capture.entries) == 2
    failure = log_capture.entries[1]
    assert failure["event"] == "failed"
    assert failure["command"] == "sh"
    assert failure["code"] == 3
    assert failure["stdout"] == "out\n"
    assert failure["stderr"] == "err\n"


def test_check_call_without_current_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log_utils.inspect, "currentframe", lambda: None)

    with pytest.raises(RuntimeError, match=r"^Could not get the current frame\.$"):
        log_utils.log_check_call(["true"])


def test_check_call_without_parent_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        log_utils.inspect, "currentframe", lambda: types.SimpleNamespace(f_back=None)
    )

    with pytest.raises(
        RuntimeError, match=r"^Could not traverse to parent of current frame\.$"
    ):
        log_utils.log_check_call(["true"])
