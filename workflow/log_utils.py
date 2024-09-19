"""Structured Logging Utilities for `logging`.

This module support for different structured-logging formats (JSON and
text) and decorators for logging function calls and command
executions. The logging functions can capture and structure log
messages along with additional contextual information such as function
arguments, execution status, and timestamps.

Examples
--------

>>> @log_call()
>>> def foo(a, b):
>>>     return a + b
>>> foo(1, 2)
{"function": "foo", "id": "...", "args": {"a": 1, "b": 2}, "message": "called", "level": "INFO", "time": "...", "thread": "MainThread"}
{"function": "foo", "id": "...", "result": 3, "message": "completed", "level": "INFO", "time": "...", "thread": "MainThread"}
3
>>> log('hello world', LoggingFormat.TEXT, counter=1)
2024-09-18 22:00:51.498268+00:00 :: INFO :: hello world counter=1

"""

import enum
import functools
import inspect
import json
import logging
import os
import subprocess
import threading
import traceback
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


class LoggingFormat(Enum):
    """Enumeration of possible logging format outputs."""

    JSON = enum.auto()
    """Output log in JSON structured-logging format."""
    TEXT = enum.auto()
    """Output log in Heroku-style key=value structured-logging format."""


logging.basicConfig(level=logging.INFO, format="%(message)s")


class LogEncoder(json.JSONEncoder):
    """Custom JSON encoder for logging arbitrary values."""

    def default(self, obj: Any) -> Any:
        """Encode the JSON representation of obj.

        The encoder will default to repr(obj) if the base encoder
        fails.

        Parameters
        ----------
        obj : Any
            Object to encode.

        Returns
        -------
        Any
            The encoded JSON object.
        """
        try:
            return super().default(obj)
        except TypeError:
            return repr(obj)


def log(
    message: str,
    format: Optional[LoggingFormat] = None,
    level: int = logging.INFO,
    /,
    **kwargs: Any,
) -> None:
    """Log a message in a structured logging format.

    Parameters
    ----------
    message : str
        The message to log.
    format : Optional[LoggingFormat]
        The logging format to use, defaulting to the value of the
        environment variable LOG_FORMAT or `LoggingFormat.JSON` if the
        environment variable is not present.
    level : int
        The level of the log. For example, the default is `logging.INFO`.
    kwargs : Any
        Keyword arguments to log in a structured logging format.
    """
    log_kwargs = {key: repr(value) for key, value in kwargs.items()}
    now = str(datetime.now(timezone.utc))
    level_name = logging.getLevelName(level)
    format = format or LoggingFormat[os.environ.get("LOG_FORMAT", "TEXT").upper()]

    match format:
        case LoggingFormat.JSON:
            logging.log(
                level,
                json.dumps(
                    kwargs
                    | {
                        "message": message,
                        "level": level_name,
                        "time": now,
                        "thread": threading.current_thread().name,
                    },
                    cls=LogEncoder,
                ),
            )
        case LoggingFormat.TEXT:
            structured_log_data = "\t".join(
                f"{key}={value}" for key, value in log_kwargs.items()
            )
            logging.log(
                level, f"{now}\t{level_name}\t{message}\t{structured_log_data}"
            )


def log_call(
    action_name: Optional[str] = None,
    exclude_args: Optional[Iterable[str]] = None,
    include_result: bool = True,
) -> Callable:
    """Wrap a function with logging calls of the arguments and success status.

    Parameters
    ----------
    action_name : Optional[str]
        An alternative identifier for the function in the log output.
        If None, will use `f.__name__` as the identifier.
    exclude_args : Optional[Iterable[str]]
        Arguments to exclude from log reports.
    include_result : bool
        If True, log the result of function call.

    Returns
    -------
    Callable
        A decorator that logs it's wrapped function's arguments every
        time the function is called, and logs once it has completed
        (with it's return value if `include_result` is True).
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            nonlocal exclude_args
            signature = inspect.signature(f)
            function_id = str(uuid.uuid4())
            exclude_args = exclude_args or set()
            unified_arguments = {
                parameter: arg
                for parameter, arg in zip(signature.parameters, args)
                if parameter not in exclude_args
            } | {key: value for key, value in kwargs.items() if key not in exclude_args}
            name = action_name or f.__name__
            log("called", function=name, id=function_id, args=unified_arguments)
            try:
                result = f(*args, **kwargs)
            except:
                log(
                    "failed",
                    None,
                    logging.ERROR,
                    function=name,
                    id=function_id,
                    error=traceback.format_exc(),
                )
                raise
            if result and include_result:
                log("completed", function=name, id=function_id, result=result)
            else:
                log("completed", function=name, id=function_id)
            return result

        return wrapper

    return decorator


def log_check_call(args: list[str], **kwargs: Any) -> str:
    """Execute a command, and log its output.

    Parameters
    ----------
    args : list[str]
        The command line arguments to execute.
    kwargs : Any
        Additional keyword arguments for `subprocess.check_output`.

    Returns
    -------
    str
        The stdout of the command, in UTF-8.

    Raises
    ------
    CalledProcessError
        If the process fails. The contents of this error is as in `subprocess.check_output`.
    """
    cmd_uuid = str(uuid.uuid4())
    log("executing", command=args[0], args=args[1:], id=cmd_uuid)
    try:
        kwargs["stderr"] = subprocess.PIPE
        output = subprocess.check_output(args, **kwargs).decode("utf-8")
        log("completed", command=args[0], stdout=output, id=cmd_uuid)
    except subprocess.CalledProcessError as e:
        log(
            "failed",
            None,
            logging.ERROR,
            command=args[0],
            id=cmd_uuid,
            code=e.returncode,
            stdout=e.output.decode("utf-8"),
            stderr=e.stderr.decode("utf-8"),
        )
        raise
    return output
