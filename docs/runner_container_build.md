# Building the simulation (`runner.sif`) container

Instructions for building the Apptainer container that runs the simulation
workflow (`realisation-to-srf`, `hf-sim`, `bb-sim`, `im-calc`, …) plus the EMOD3D
binaries on an HPC cluster.

> The container definition is in this repo at `container/runner.def`. It is short
> — the heavy lifting happens in the Docker base image, built by CI from
> `container/Dockerfile`.

## What this container is

`runner.sif` is a thin Apptainer layer over the Docker base image
`earthquakesuc/workflow-bootstrap:latest`, which already carries the whole Python
stack (workflow, `source_modelling`, `velocity_modelling`, `qcore`, `IM-calculation`,
openquake, pygmt) on a self-compiled Python in `/sw/venv`. All `runner.def` adds is
**four compiled EMOD3D tools** in `/EMOD3D/tools`, which it builds from source:

| binary | used by |
|---|---|
| `genslip_v5.6.2` | `realisation-to-srf` (finite-fault SRF generation) |
| `generic_slip2srf` | `realisation-to-srf` (point-source SRF generation) |
| `srf2stoch` | `generate-stoch` |
| `hb_high_binmod_v6.0.3` | `hf-sim` |

`%post` compiles those four targets, then deletes everything under `/EMOD3D`
except `tools/`, so the source tree (and its `.git`) is **not** in the shipped
image. `viz.sif` is a sibling built on the same base — it plots, this one
simulates. Result is a ~1.5 GB `.sif`.

## Prerequisites

- `apptainer` (the 1.4.5 deb on the dev box; rootless build works with no extra
  setup — the deb ships the AppArmor userns profile).
- A checkout of `github.com/ucgmsim/EMOD3D` (dev box: `~/src/EMOD3D`).
- Network access to pull the **public** `earthquakesuc/workflow-bootstrap:latest`
  from Docker Hub, or the base cached locally.
- ~5 GB free disk for the image plus build scratch.

## Build

```bash
# 1. Separate worktree on pegasus, so your main checkout can stay on a feature branch
git worktree add ~/src/workflow-pegasus pegasus

# 2. Clean EMOD3D export into the build directory
mkdir -p ~/build/runner-pegasus \
  && rm -rf ~/build/runner-pegasus/EMOD3D \
  && git -C ~/src/EMOD3D archive master --prefix=EMOD3D/ \
     | tar -x -C ~/build/runner-pegasus

# 3. Build — the cd is load-bearing, see gotcha 1
cd ~/build/runner-pegasus \
  && apptainer build --force runner.sif ~/src/workflow-pegasus/container/runner.def \
       2>&1 | tee build.log
```

Takes ~2 minutes; the base-image pull and SIF packing dominate, the EMOD3D compile
is short. There is no `%test` section, so a clean exit is **not** by itself a
verification — run the checks under "Verify" below.

## Why the build looks the way it does (gotchas)

1. **The `cd` into the build directory is load-bearing.** `%files` reads
   `EMOD3D /EMOD3D` — a bare relative source, which Apptainer resolves against the
   directory you launch `apptainer build` from, **not** against the def file's
   directory. Build from anywhere else and it fails to find `EMOD3D`. This is why
   the def lives in `container/` but the build runs from `~/build/runner-pegasus/`.
   Symlinking `EMOD3D` into place is not a fix — `%files` copies with `cp -a`
   semantics and preserves the symlink rather than following it.

2. **Use `git archive`, never a copy of your working checkout.** `%post` starts
   with `mkdir build`, which **fails on an existing `build/` directory** — and a
   working EMOD3D checkout normally has one. The archive also guarantees no stray
   local edits get baked into an image whose provenance is otherwise
   unrecoverable (see gotcha 5).

3. **Build EMOD3D from `master`.** The `hf-profiling` and `gcc14` branches look
   "ahead" but are ~109 commits *behind* master; `hf-profiling` is an unrelated
   benchmarking harness that must not ship in the image.

4. **`-std=gnu89` in `CMAKE_C_FLAGS` is load-bearing.** The base image's GCC is
   15.x, which defaults to C23. Genslip v5.6.2 is K&R-era C: it declares
   `FILE *fpr, *fopfile();`, where empty parens now mean `(void)`, so every
   `fopfile(f, "r")` becomes "too many arguments"; it also relies on implicit
   declarations (`setpar`/`mstpar`/`getpar`/`endpar`/`isblank`) and implicit-int
   `main()`. C89 permits all of it, so the one flag covers every case. Without it
   the build **hard-errors**. Added in PR #123. Note CI builds only the Docker
   base, never `runner.sif`, so nothing upstream catches a break here.

5. **The image records no EMOD3D provenance.** `%post` strips the source tree
   including `.git`, so once built there is no way to ask an image which EMOD3D
   commit it came from. Record the commit yourself at build time —
   `git -C ~/src/EMOD3D rev-parse HEAD` — if you need to attribute a result later.

6. **Which workflow code you get is decided by the *base* image, not by anything
   here.** `container/Dockerfile` takes `ARG WORKFLOW_BRANCH=pegasus` and
   `container.yml` passes no `build-args`, so the published base always contains
   **pegasus**, whichever branch triggered the build; `workflow_dispatch` from a
   feature branch does *not* build that branch. `runner.def` is identical across
   branches, so building from your feature-branch checkout changes nothing. To get
   unmerged work into a container you must build the Docker base yourself with
   `--build-arg WORKFLOW_BRANCH=…`, from a **pushed** branch.

7. **The base image floats.** The Dockerfile installs with plain `pip` from
   GitHub and ignores `uv.lock`, resolving fresh from PyPI on every CI build. Two
   images built from the same commit on different days can ship different
   dependency versions, and an upstream release can break the build with no repo
   change. (This is why `openquake.engine<3.26` is capped in the Dockerfile —
   3.26 pins a `gdal` that will not build against Ubuntu's libgdal.) Record the
   versions you shipped; don't assume they follow from the commit.

8. **If the build fails on Docker Hub auth** with
   `FATAL: ... unable to retrieve auth token: invalid username/password`, this is
   not a problem with the def — Apptainer sends whatever stale credential sits in
   `~/.docker/config.json` even though the image is public. Either `docker login`
   to refresh, or build from a local docker-daemon copy of the base. See the
   equivalent section in `viz_container_build.md` for the full workaround. On a
   box with no `~/.docker/config.json` at all this cannot occur.

## Verify

There is no `%test`, so verify explicitly. Use `--containall`: without it Apptainer
binds `$HOME` and the CWD, and a local editable install will shadow the container's
packages, silently reporting **your working tree's** version instead of the image's.

```bash
cd /tmp   # stay out of a workflow checkout

# The four binaries exist and run (each should reach its own argument validation,
# not a linker error):
apptainer exec --containall runner.sif ls /EMOD3D/tools/
apptainer exec --containall runner.sif ldd /EMOD3D/tools/genslip_v5.6.2
apptainer exec --containall runner.sif bash -lc \
    'timeout 5 /EMOD3D/tools/genslip_v5.6.2 </dev/null 2>&1 | head -3'
    # expect: "***** input error / GridSpacing = 0.0, exiting..."

# Python side:
apptainer exec --containall runner.sif realisation-to-srf --help
apptainer exec --containall runner.sif python -c \
    "import importlib.metadata as m; print(m.version('workflow'), m.version('source-modelling'))"

# Record what you actually shipped (see gotcha 7):
apptainer exec --containall runner.sif sha256sum /EMOD3D/tools/*
```

### Confirming the base against Docker Hub

You do **not** need a local Docker build for this, and a local build could not
answer it anyway: the Dockerfile re-resolves from PyPI (gotcha 7), so a rebuild
today differs from the published image by construction. It would tell you "does
this still build", not "does this match".

Instead, the finished `.sif` carries the base image's OCI labels:

```bash
apptainer inspect runner.sif | grep opencontainers
#   org.opencontainers.image.revision: <workflow commit baked into the base>
#   org.opencontainers.image.version:  <branch>
#   org.opencontainers.image.created:  <when CI built it>
```

That `revision` is in-image proof of which workflow commit you are running. Cross-check
it against the registry — CI tags each build `sha-<commit>`, so `latest`, the branch
tag and `sha-<commit>` should all resolve to one digest:

```bash
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:earthquakesuc/workflow-bootstrap:pull" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
curl -sI -H "Authorization: Bearer $TOKEN" \
     -H "Accept: application/vnd.oci.image.index.v1+json" \
     https://registry-1.docker.io/v2/earthquakesuc/workflow-bootstrap/manifests/latest \
     | grep -i docker-content-digest
```

## Reference build (2026-07-27)

For comparison when something looks off:

- Base `earthquakesuc/workflow-bootstrap:latest` = `sha256:b4a635750ae6a5b5…`,
  built 2026-07-21, `image.revision` `7e465c5244c51a60882280a833dd1676decdb281`
  (pegasus HEAD).
- EMOD3D `master` at `d8fed8a`. Image 1.55 GB, 133 Python distributions.
- Shipped: `source_modelling` 2026.7.2, `velocity_modelling` 2026.4.1,
  `oq-wrapper` 2026.5.2, `openquake.engine` 3.25.1, `nshmdb` 2025.12.1,
  `IM-calculation` 2026.3.1, `pygmt` 0.19.0, numpy 2.4.6, pandas 2.3.3.
- Binary hashes:

  | binary | sha256 (first 16) |
  |---|---|
  | `genslip_v5.6.2` | `b5f300e5a2770007` |
  | `srf2stoch` | `d68d75daad5edae4` |
  | `generic_slip2srf` | `c893524c59d9f982` |
  | `hb_high_binmod_v6.0.3` | `b3700c2d65e42dc0` |

The EMOD3D compile is largely reproducible: against the previous image (2026-07-14,
same EMOD3D `master`) three of the four binaries are **bit-identical**. Only
`generic_slip2srf` differs, which is consistent with that older image predating
PR #123 and having been built with a local `-std=gnu17` patch instead of `gnu89`.
If a rebuild changes `genslip_v5.6.2` or `srf2stoch` without an EMOD3D source
change, something is wrong.

## Deploy to BSC

The dev box has `bsc_transfer{1..4}` configured as rclone SFTP remotes
(`transfer1.bsc.es`). Paths are home-relative, so this lands it at `~/runner.sif`:

```bash
rclone copy ~/build/runner-pegasus/runner.sif bsc_transfer1: --progress
# verify end-to-end (hash computed cluster-side):
rclone hashsum md5 bsc_transfer1:runner.sif
md5sum ~/build/runner-pegasus/runner.sif
```

Use `rclone copy`, never `sync` (sync deletes remote files absent locally).
Roughly 6 MiB/s, so ~4 minutes for 1.5 GB. Existing sifs on the cluster use
descriptive prefixes (e.g. `multifault_version_bug_runner.sif`) — worth doing when
the image is tied to a specific investigation.
