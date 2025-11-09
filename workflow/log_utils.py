r"""Structured Logging Utilities for `logging`.

This module supports different structured-logging formats (JSON and
text) and decorators for logging function calls and command
executions. The logging functions capture and structure log
messages along with additional contextual information such as function
arguments, execution status, and timestamps.

Examples
--------

>>> @log_call()
>>> def foo(a, b):
>>>     return a + b
>>> foo(1, 2)
{"timestamp": "2025-02-25T22:00:51.498268Z", "level": "info", "logger": "example", "thread": "MainThread", "event": "called", "function": "foo", "id": "...", "a": 1, "b": 2}
{"timestamp": "2025-02-25T22:00:51.498268Z", "level": "info", "logger": "example", "thread": "MainThread", "event": "completed", "function": "foo", "id": "...", "result": 3}
3
>>> logger = get_logger('example')
>>> logger.info('hello world', counter=1)
{"timestamp": "2025-02-25T22:00:51.498268Z", "level": "info", "logger": "example", "thread": "MainThread", "event": "hello world", "counter": 1}
"""

import functools
import inspect
import subprocess
import traceback
import uuid
from collections.abc import Callable, Iterable
from types import FunctionType
from typing import Any

import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a formatted logger with name.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    structlog.BoundLogger
        The logger with name `name`.
    """
    return structlog.get_logger(name)


def log_call(
    action_name: str | None = None,
    exclude_args: Iterable[str] | None = None,
    include_result: bool = True,
) -> Callable:
    """Wrap a function with logging calls of the arguments and success status.

    Parameters
    ----------
    action_name : str | None
        An alternative identifier for the function in the log output.
        If None, will use `f.__name__` as the identifier.
    exclude_args : Iterable[str] | None
        Arguments to exclude from log reports.
    include_result : bool
        If True, log the result of function call.

    Returns
    -------
    Callable
        A decorator that logs it's wrapped function's arguments every
        time the function is called, and logs once it has completed
        (with it's return value if include_result is True).
    """

    def decorator(f: FunctionType) -> Callable:  # numpydoc ignore=GL08
        @functools.wraps(f)
        def wrapper(
            *args: list[Any], **kwargs: dict[str, Any]
        ) -> Any:  # numpydoc ignore=GL08
            nonlocal exclude_args
            signature = inspect.signature(f)
            function_id = str(uuid.uuid4())
            exclude_args = set(exclude_args or [])
            name = f.__globals__["__name__"]
            logger = get_logger(name).bind(
                function=action_name or f.__name__, id=function_id
            )

            unified_arguments = {
                parameter: arg
                for parameter, arg in zip(signature.parameters, args)
                if parameter not in exclude_args
            } | {key: value for key, value in kwargs.items() if key not in exclude_args}

            logger.info("called", **unified_arguments)
            try:
                result = f(*args, **kwargs)
            except Exception:
                logger.error("failed", error=traceback.format_exc())
                raise

            if include_result:
                logger.info("completed", result=result)
            else:
                logger.info("completed")

            return result

        return wrapper

    return decorator


def log_check_call(args: list[str], **kwargs: Any) -> None:
    """Execute a command, and log its output.

    Parameters
    ----------
    args : list[str]
        The command line arguments to execute.
    **kwargs : Any
        Additional keyword arguments for `subprocess.check_output`.

    Returns
    -------
    str
        The stdout of the command, in UTF-8.

    Raises
    ------
    CalledProcessError
        If the process fails. The contents of this error is as in `subprocess.check_output`.
    RuntimeError
        If the caller or caller's parent stackframe cannot be traversed to annotate logging output.
    """
    cmd_uuid = str(uuid.uuid4())
    current_frame = inspect.currentframe()
    if not current_frame:
        raise RuntimeError("Could not get the current frame.")
    parent_frame = current_frame.f_back
    if not parent_frame:
        raise RuntimeError("Could not traverse to parent of current frame.")
    name = parent_frame.f_globals["__name__"]
    logger = get_logger(name).bind(command=args[0], id=cmd_uuid)

    logger.info("executing", args=args[1:])
    try:
        kwargs["stderr"] = subprocess.PIPE
        output = subprocess.check_output(args, **kwargs).decode("utf-8")
        logger.info("completed", stdout=output)
        return output
    except subprocess.CalledProcessError as e:
        logger.error(
            "failed",
            code=e.returncode,
            stdout=e.output.decode("utf-8"),
            stderr=e.stderr.decode("utf-8"),
        )
        raise
