#!/usr/bin/env python3
"""Verify the provenance recorded in realisation files.

Description
-----------
``log_trail`` records the ``workflow`` version reported by
``importlib.metadata.version``, which reads a **cached** ``.dist-info`` stamped
by setuptools-scm at *install* time. An editable install does not refresh it
when source files change or when ``HEAD`` moves, and ``uv run`` re-syncs only
when ``pyproject.toml`` or ``uv.lock`` change. The recorded version can
therefore name a commit that has nothing to do with the code that ran -- which is
exactly what happened to the first, untraceable batch of realisations.

This tool exists to make that failure impossible to miss.

Usage
-----
``verify-realisation-provenance --preflight`` checks the environment *before* a
campaign, and refuses to proceed unless the tree is clean and the installed
metadata matches ``HEAD``.

``verify-realisation-provenance REALISATION_DIR`` audits finished realisations.
"""

import dataclasses
import re
import subprocess
from pathlib import Path

import typer

app = typer.Typer()


SCM_VERSION_RE = re.compile(
    r"^(?P<public>[^+]+)\+g(?P<sha>[0-9a-f]+)(?:\.d(?P<dirty>\d{8}))?$"
)

EXPECTED_UTILITIES: tuple[str, ...] = (
    "nshm2022-to-realisation",
    "complete-realisations",
)

REINSTALL_COMMAND = "uv sync --reinstall-package workflow --all-extras --dev"


@dataclasses.dataclass(frozen=True)
class ScmVersion:
    """A setuptools-scm version string, decomposed.

    Attributes
    ----------
    raw : str
        The version string as recorded.
    sha : str
        The abbreviated commit SHA from the local version segment.
    dirty : bool
        Whether the version carries a ``.d<YYYYMMDD>`` dirty suffix.
    """

    raw: str
    sha: str
    dirty: bool


def parse_scm_version(version: str) -> ScmVersion:
    """Decompose a setuptools-scm version string.

    Parameters
    ----------
    version : str
        A version of the form ``0.1.dev1286+gb541da03a``, optionally carrying a
        ``.d<YYYYMMDD>`` dirty suffix.

    Returns
    -------
    ScmVersion
        The decomposed version.

    Raises
    ------
    ValueError
        If the version has no ``+g<sha>`` local segment, and so identifies no
        commit at all.
    """
    match = SCM_VERSION_RE.match(version)
    if match is None:
        raise ValueError(
            f"Version {version!r} has no +g<sha> local segment: it identifies no commit."
        )
    return ScmVersion(raw=version, sha=match["sha"], dirty=match["dirty"] is not None)


def git_head_sha(repo_root: Path) -> str:
    """Return the full 40-character SHA of ``HEAD``.

    Parameters
    ----------
    repo_root : Path
        Root of the git repository.

    Returns
    -------
    str
        The full SHA of ``HEAD``.
    """
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_is_clean(repo_root: Path) -> bool:
    """Return whether the working tree has no modified tracked files.

    Untracked files are ignored, matching setuptools-scm's own dirty check,
    which considers only tracked modifications. This matters: the campaign writes
    into gitignored directories and so cannot dirty its own tree.

    Parameters
    ----------
    repo_root : Path
        Root of the git repository.

    Returns
    -------
    bool
        True if no tracked file is modified.
    """
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return status == ""


def preflight_problems(repo_root: Path, installed_version: str) -> list[str]:
    """Return the reasons this environment is unfit to run a traceable campaign.

    Parameters
    ----------
    repo_root : Path
        Root of the workflow git repository.
    installed_version : str
        The version reported by ``importlib.metadata.version("workflow")``.

    Returns
    -------
    list of str
        One message per problem. Empty means the environment is fit to run.
    """
    problems: list[str] = []

    if not git_is_clean(repo_root):
        problems.append(
            "Working tree has modified tracked files. setuptools-scm will stamp a "
            "'.d<date>' dirty suffix and the recorded SHA will not identify the code "
            "that ran."
        )

    try:
        version = parse_scm_version(installed_version)
    except ValueError as exc:
        problems.append(str(exc))
        return problems

    if version.dirty:
        problems.append(
            f"Installed version {version.raw!r} carries a dirty suffix. Commit or stash, "
            f"then run: {REINSTALL_COMMAND}"
        )

    head = git_head_sha(repo_root)
    if not head.startswith(version.sha):
        problems.append(
            f"STALE METADATA: the installed version names commit {version.sha}, but HEAD "
            f"is {head[: len(version.sha)]}. importlib.metadata reads a cached .dist-info "
            f"that is not refreshed when HEAD moves, so every realisation would record "
            f"the wrong commit. Run: {REINSTALL_COMMAND}"
        )

    return problems
