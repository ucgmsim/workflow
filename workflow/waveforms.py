"""Component convention shared by the workflow's waveform files."""

from enum import StrEnum


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
