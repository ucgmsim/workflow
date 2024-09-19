import json
import logging
import os
import re
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from workflow import log_utils


def test_raw_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    log_utils.log("message", log_utils.LoggingFormat.JSON)
    log_utils.log("text formatted message", log_utils.LoggingFormat.TEXT)
    assert len(caplog.messages) == 2
    basic_message = caplog.messages[0]
    assert re.match(
        '{"message": "message", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        basic_message,
    )
    text_message = caplog.messages[1]
    assert re.match(".*?\tINFO\ttext formatted message", text_message)


@log_utils.log_call()
def foo(a, b):
    return a + b


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_basic_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    _ = foo(1, 2)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        '{"function": "foo", "id": ".*?", "args": {"a": 1, "b": 2}, "message": "called", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        '{"function": "foo", "id": ".*?", "result": 3, "message": "completed", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        return_log,
    )


@log_utils.log_call(exclude_args={"b"})
def foo_less_b(a, b):
    return a + b


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_excluded_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)

    _ = foo_less_b(1, 2)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        '{"function": "foo_less_b", "id": ".*?", "args": {"a": 1}, "message": "called", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        '{"function": "foo_less_b", "id": ".*?", "result": 3, "message": "completed", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        return_log,
    )


@log_utils.log_call(action_name="FOOBAR")
def bar(a):
    pass


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_renamed_bar(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    bar(1)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        '{"function": "FOOBAR", "id": ".*?", "args": {"a": 1}, "message": "called", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        '{"function": "FOOBAR", "id": ".*?", "message": "completed", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        return_log,
    )


@log_utils.log_call(include_result=False)
def baz(a):
    return 1


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_no_result(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    baz(1)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        '{"function": "baz", "id": ".*?", "args": {"a": 1}, "message": "called", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        '{"function": "baz", "id": ".*?", "message": "completed", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        return_log,
    )


@log_utils.log_call()
def failing_function():
    raise ValueError("This function should fail!")


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_failing_function(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    with pytest.raises(ValueError):
        failing_function()

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        '{"function": "failing_function", "id": ".*?", "args": {}, "message": "called", "level": "INFO", "time": ".*?", "thread": "MainThread"}',
        call_log,
    )
    return_log = json.loads(caplog.messages[1])
    assert return_log["level"] == "ERROR"
    assert return_log["function"] == "failing_function"
    assert return_log["message"] == "failed"
    assert return_log["error"].startswith("Traceback")


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_successful_check_call_log(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    caplog.set_level(logging.INFO)
    test_file = tmp_path / "test.txt"
    test_file.touch()
    log_utils.log_check_call(["ls", str(tmp_path)])

    assert len(caplog.messages) == 2
    execution_message = json.loads(caplog.messages[0])
    assert execution_message["args"] == [str(tmp_path)]
    assert execution_message["command"] == "ls"
    completion_message = json.loads(caplog.messages[1])
    assert completion_message["stdout"] == "test.txt\n"


@mock.patch.dict(os.environ, {"LOG_FORMAT": "JSON"})
def test_failing_check_call_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    with pytest.raises(subprocess.CalledProcessError):
        log_utils.log_check_call(["ls", "/bad-path"])

    assert len(caplog.messages) == 2
    execution_message = json.loads(caplog.messages[0])
    assert execution_message["args"] == ["/bad-path"]
    assert execution_message["command"] == "ls"
    completion_message = json.loads(caplog.messages[1])
    assert completion_message["level"] == "ERROR"
    assert (
        completion_message["stderr"]
        == "ls: cannot access '/bad-path': No such file or directory\n"
    )
    assert completion_message["stdout"] == ""
    assert completion_message["code"] == 2
