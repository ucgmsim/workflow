#!/usr/bin/env python3
"""Distribute completed realisations into per-event CyberShake directories.

For every ``*.json`` in ``SOURCE_DIR`` this script extracts the integer id from
the file name, creates a directory named after that id under ``EVENTS_DIR``, and
copies the realisation into it as ``realisation.json``.

e.g. ``complete_realisations/realisation_100932.json``
  -> ``.../flow/events/100932/realisation.json``
"""

import re
import shutil
from pathlib import Path

SOURCE_DIR = Path("/home/arr65/src/workflow/complete_realisations")
EVENTS_DIR = Path("/home/arr65/src/cybershake_nshm_2022/flow/events")


def main() -> None:
    copied = 0
    skipped: list[str] = []

    for src in sorted(SOURCE_DIR.glob("*.json")):
        match = re.search(r"\d+", src.name)
        if match is None:
            skipped.append(src.name)
            continue
        event_id = match.group()

        event_dir = EVENTS_DIR / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        # Copy straight to the final name (copy + rename in one step).
        shutil.copy2(src, event_dir / "realisation.json")
        copied += 1

    print(f"Copied {copied} realisation(s) into {EVENTS_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s) with no integer id:")
        for name in skipped:
            print(f"  {name}")


if __name__ == "__main__":
    main()
