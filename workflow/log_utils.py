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
3
>>> log('hello world', counter=1)
2024-09-18 22:00:51.498268+00:00 :: INFO :: hello world counter=1

"""

import functools
import inspect
import logging
import subprocess
import traceback
import uuid
from collections.abc import Iterable
from typing import Any, Callable, Optional

def get_logger(name: str) -> logging.Logger:
    """Get a formatted logger with `name`.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The logger with name `name`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(name)s\t%(threadName)s\t%(message)s"))
    logger.addHandler(handler)
    return logger


def structured_log(
    message: str,
    **kwargs: Any,
) -> str:
    """Format a log message in a structured logging format.

    Parameters
    ----------
    message : str
        The message to log.
    kwargs : Any
        Keyword arguments to log in a structured logging format.

    Returns
    -------
    str
        A message in a structured log format
    """
    log_kwargs = {key: str(value) for key, value in kwargs.items()}
    structured_log_data = "\t".join(
        f"{key}={value}" for key, value in log_kwargs.items()
    )
    return f'{message}\t{structured_log_data}'



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
            name = f.__globals__['__name__']
            logger = get_logger(name)
            unified_arguments = {
                parameter: arg
                for parameter, arg in zip(signature.parameters, args)
                if parameter not in exclude_args
            } | {key: value for key, value in kwargs.items() if key not in exclude_args}
            name = action_name or f.__name__
            logger.info(structured_log("called", function=name, id=function_id, **unified_arguments))
            try:
                result = f(*args, **kwargs)
            except:
                logger.error(structured_log(
                    "failed",
                    function=name,
                    id=function_id,
                    error=traceback.format_exc(),
                ))
                raise
            if result and include_result:
                logger.info(structured_log("completed", function=name, id=function_id, result=result))
            else:
                logger.info(structured_log("completed", function=name, id=function_id))
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
    name = inspect.currentframe().f_back.f_globals['__name__']
    logger = logging.getLogger(name)
    logger.info(structured_log("executing", command=args[0], args=args[1:], id=cmd_uuid))
    try:
        kwargs["stderr"] = subprocess.PIPE
        output = subprocess.check_output(args, **kwargs).decode("utf-8")
        logger.info(structured_log("completed", command=args[0], stdout=output, id=cmd_uuid))
    except subprocess.CalledProcessError as e:
        logger.error(structured_log(
            "failed",
            command=args[0],
            id=cmd_uuid,
            code=e.returncode,
            stdout=e.output.decode("utf-8"),
            stderr=e.stderr.decode("utf-8"),
        ))
        raise
    return output
