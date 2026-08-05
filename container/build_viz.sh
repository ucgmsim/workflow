#!/usr/bin/env bash
# Build viz.sif, exporting the current workflow branch into the image.
#
# WHY A WRAPPER RATHER THAN A PLAIN `apptainer build`
# ---------------------------------------------------
# viz.def installs `workflow` from a source tarball rather than a git URL, so
# the image can be built from a branch that has not been pushed. That costs two
# things a def file cannot do for itself:
#
#   1. The tarball has to exist before the build starts, at the exact path
#      %files names, and %files paths resolve against $PWD -- so the build has
#      to run from the repo root or it fails with a confusing "no such file".
#   2. `git archive` strips .git, and this project's version comes from
#      setuptools_scm, which without a repo cannot compute one and aborts the
#      install. The version therefore has to be computed out here, where the
#      repo still exists, and handed in as a file.
#
# Both are easy to get wrong by hand and neither fails in an obvious way, hence
# this script. It refuses to build from a dirty tree: the image records a commit
# SHA in its version string, and that string is a lie if uncommitted edits went
# into the tarball.
#
# Usage:
#     container/build_viz.sh [output.sif]        # default: viz.sif in $PWD
#     ALLOW_DIRTY=1 container/build_viz.sh       # escape hatch, stamps +dirty

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

OUT="${1:-viz.sif}"
SRC=container/workflow-src.tar.gz
VERFILE=container/workflow-version.txt

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    if [[ "${ALLOW_DIRTY:-}" != "1" ]]; then
        echo "FATAL: working tree has uncommitted changes." >&2
        echo "       The image stamps a commit SHA into its version; building" >&2
        echo "       from a dirty tree makes that stamp wrong. Commit first, or" >&2
        echo "       re-run with ALLOW_DIRTY=1 to stamp +dirty instead." >&2
        git status --short --untracked-files=no >&2
        exit 1
    fi
    DIRTY=".dirty"
else
    DIRTY=""
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
SHA=$(git rev-parse HEAD)
# Mirror setuptools_scm's own node-and-date-ish format so the version in the
# image looks like one built normally: 0.1.dev<commits>+g<short sha>.
VERSION="0.1.dev$(git rev-list --count HEAD)+g$(git rev-parse --short=9 HEAD)${DIRTY}"

echo "repo    : $REPO_ROOT"
echo "branch  : $BRANCH"
echo "commit  : $SHA"
echo "version : $VERSION"
echo "output  : $OUT"

# HEAD, not the branch name: with ALLOW_DIRTY the tree may differ from HEAD, and
# archiving HEAD makes the discrepancy explicit rather than pretending it away.
# apptainer unpacks the entire container into $APPTAINER_TMPDIR and only then
# runs mksquashfs, so it needs room for the sandbox AND the finished .sif at the
# same time -- about 16 GB for this image (~12 GB unpacked plus a ~4 GB SIF).
# The default is /tmp, which on a machine where /tmp is a tmpfs is RAM, and the
# build then dies at the very last step, after ten-plus minutes of work, with
#     FATAL: while creating squashfs: mksquashfs command failed:
#     Write failed because Disk quota exceeded
# which reads like a permissions problem rather than a full disk. Stage next to
# the output instead, on whatever real filesystem that lives on.
if [[ -z "${APPTAINER_TMPDIR:-}" ]]; then
    APPTAINER_TMPDIR="$(dirname "$(readlink -f "$OUT")")/.apptainer-build-tmp"
    OURS_TMPDIR=1
fi
export APPTAINER_TMPDIR
mkdir -p "$APPTAINER_TMPDIR"

# Check up front rather than at minute twelve.
NEED_GB=20
AVAIL_GB=$(df -BG --output=avail "$APPTAINER_TMPDIR" | tail -1 | tr -dc '0-9')
echo "tmpdir  : $APPTAINER_TMPDIR (${AVAIL_GB} GB free, need ~${NEED_GB})"
if (( AVAIL_GB < NEED_GB )); then
    echo "FATAL: not enough space at $APPTAINER_TMPDIR." >&2
    echo "       Set APPTAINER_TMPDIR to a filesystem with ~${NEED_GB} GB free." >&2
    exit 1
fi

git archive --format=tar.gz --prefix=workflow/ -o "$SRC" HEAD
printf '%s\n' "$VERSION" > "$VERFILE"
# Only remove the tmpdir if we chose it; a caller-supplied one is theirs.
trap 'rm -f "$SRC" "$VERFILE"; [[ -n "${OURS_TMPDIR:-}" ]] && rm -rf "$APPTAINER_TMPDIR"' EXIT

echo "exported $(du -h "$SRC" | cut -f1) of source"
echo

# --fakeroot: the def's %post runs apt-get, which needs root inside the build.
apptainer build --fakeroot --force "$OUT" container/viz.def

echo
echo "built $OUT ($(du -h "$OUT" | cut -f1)) from $BRANCH @ ${SHA:0:9}"
echo "Verify before shipping it to BSC — %test is deliberately session-free, so"
echo "the figure render is NOT covered by the build. See viz_container_build.md."
