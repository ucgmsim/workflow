import logging
import re
import subprocess
from pathlib import Path

import pytest
import logging

from workflow import log_utils


def test_structured_log():
    assert log_utils.structured_log('test', a=1, b=2, c=3) == 'test\ta=1\tb=2\tc=3'

@log_utils.log_call()
def foo(a, b):
    return a + b


def test_basic_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    _ = foo(1, 2)

    assert len(caplog.messages) == 2

    call_log = caplog.messages[0]
    assert re.match(
        'called\tfunction=foo\tid=.*\ta=1\tb=2',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        'completed\tfunction=foo\tid=.*\tresult=3',
        return_log,
    )


@log_utils.log_call(exclude_args={"b"})
def foo_less_b(a, b):
    return a + b


def test_excluded_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)

    _ = foo_less_b(1, 2)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        'called\tfunction=foo_less_b\tid=.*\ta=1',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        'completed\tfunction=foo_less_b\tid=.*\tresult=3',
        return_log,
    )


@log_utils.log_call(action_name="FOOBAR")
def bar(a):
    pass


def test_renamed_bar(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    bar(1)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        'called\tfunction=FOOBAR\tid=.*\ta=1',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        'completed\tfunction=FOOBAR\tid=.*',
        return_log,
    )


@log_utils.log_call(include_result=False)
def baz(a):
    return 1


def test_no_result(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    baz(1)

    assert len(caplog.messages) == 2
    call_log = caplog.messages[0]
    assert re.match(
        'called\tfunction=baz\tid=.*\ta=1',
        call_log,
    )
    return_log = caplog.messages[1]
    assert re.match(
        'completed\tfunction=baz\tid=.*',
        return_log,
    )


@log_utils.log_call()
def failing_function():
    raise ValueError("This function should fail!")


def test_failing_function(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    with pytest.raises(ValueError):
        failing_function()

    assert len(caplog.messages) == 2
    return_log = caplog.messages[1]
    assert re.match('failed\tfunction=failing_function\tid=.*\terror=Traceback.*', return_log)


def test_successful_check_call_log(caplog: pytest.LogCaptureFixture, tmp_path: Path):
    caplog.set_level(logging.INFO)
    test_file = tmp_path / "test.txt"
    test_file.touch()
    log_utils.log_check_call(["ls", str(tmp_path)])

    assert len(caplog.messages) == 2
    execution_message =caplog.messages[0]
    assert re.match("executing\tcommand=ls\targs=\\['.*'\\]\tid=.*", execution_message)
    completion_message = caplog.messages[1]
    assert re.match("completed\tcommand=ls\tstdout=test.txt\n\tid=.*", completion_message)



def test_failing_check_call_log(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    with pytest.raises(subprocess.CalledProcessError):
        log_utils.log_check_call(["ls", "/bad-path"])

    assert len(caplog.messages) == 2
    completion_message = caplog.messages[1]
    assert re.match("failed\tcommand=ls\tid=.*\tcode=2\tstdout=\tstderr=ls: cannot access '/bad-path': No such file or directory", completion_message)
