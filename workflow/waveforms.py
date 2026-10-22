"""Component and time-axis conventions shared by the workflow's waveform files."""

import datetime
from enum import StrEnum

import numpy as np
import numpy.typing as npt


class Component(StrEnum):
    """Component labels of the LF, HF and BB waveform files.

    Every file stores its components in this order: 000, 090, then ver. That is
    also the order the IM package reads components in, by position. Each
    producer translates its solver's own convention into these labels as it
    writes the file, so `list(Component)` is always the stored order.
    """

    NORTH = "000"
    EAST = "090"
    UP = "ver"
    """Vertical, positive up."""


def time_axis(
    seconds: npt.ArrayLike, event_time: datetime.datetime | None
) -> np.ndarray:
    """The time coordinate of a waveform file.

    Waveforms are computed against seconds from the origin time. A realisation
    with an event time stamps them instead, so records line up with observations.

    Parameters
    ----------
    seconds : array-like
        Sample times, in seconds from the origin time.
    event_time : datetime.datetime or None
        The (timezone-aware) origin time, or None to keep seconds.

    Returns
    -------
    np.ndarray
        `seconds` unchanged if `event_time` is None, and otherwise UTC
        `datetime64[ns]` timestamps (numpy datetimes carry no timezone).
    """
    if event_time is None:
        return np.asarray(seconds)
    origin = np.datetime64(
        event_time.astimezone(datetime.UTC).replace(tzinfo=None), "ns"
    )
    offsets = np.round(np.asarray(seconds, dtype=np.float64) * 1e9).astype(np.int64)
    return origin + offsets.astype("timedelta64[ns]")
