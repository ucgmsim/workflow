import enum
import functools
import inspect
import json
import logging
import os
import subprocess
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional


class LoggingFormat(Enum):
    JSON = enum.auto()
    TEXT = enum.auto()


logging.basicConfig(level=logging.INFO, format="%(message)s")


class LogEncoder(json.JSONEncoder):
    def default(self, obj):
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
    log_kwargs = {key: repr(value) for key, value in kwargs.items()}
    now = str(datetime.now(timezone.utc))
    level_name = logging.getLevelName(level)
    format = format or LoggingFormat[os.environ.get("LOG_FORMAT", "JSON").upper()]

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
            structured_log_data = " ".join(
                f"{key}={value}" for key, value in log_kwargs.items()
            )
            logging.log(
                level, f"{now} :: {level_name} :: {message} {structured_log_data}"
            )


def log_call(
    f: Callable,
    action_name: Optional[str] = None,
    exclude_args: Optional[Iterable[str]] = None,
    include_result: bool = True,
) -> Callable:
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
        log("called", function=f.__name__, id=function_id, args=unified_arguments)
        try:
            result = f(*args, **kwargs)
        except Exception as e:
            log("failed", function=f.__name__, id=function_id, error=e)
            raise
        if result and include_result:
            log("completed", function=f.__name__, id=function_id, result=result)
        else:
            log("completed", function=f.__name__, id=function_id)
        return result

    return wrapper


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
            command=args[0],
            id=cmd_uuid,
            code=e.returncode,
            stdout=e.output.decode("utf-8"),
            stderr=e.stderr.decode("utf-8"),
        )
        raise
    return output
