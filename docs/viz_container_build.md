# Building the visualisation (`viz.sif`) container

Instructions for building the Apptainer container that runs the `visualisation`
plotting/animation tools (`plot-srf`, `plot-ts`, …) on an HPC cluster.

> The container definition is in this repo at `container/viz.def`. The
> `visualisation` package itself lives in a separate repo
> (`github.com/ucgmsim/visualisation`); test SRFs referenced below are its
> fixtures (`visualisation/tests/srfs/`).

## What this container is

`viz.sif` is built **on top of the simulation base image**
`earthquakesuc/workflow-bootstrap:latest` — the same base `runner.sif` uses. It is
a **sibling** of `runner.sif`, not a child: it reuses the base's verified
scientific stack (pygmt, pygmt_helper, cartopy, matplotlib, `source_modelling`
2026.7.2, workflow, qcore) and adds only what's needed for figures/animations. It
deliberately does **not** contain EMOD3D — it plots, it does not simulate.

On top of the base it adds:

- **apt:** `gmt`, `gmt-gshhg`, `gmt-dcw` (GMT library + coastline/border data),
  `ffmpeg` (animation encoding), `ghostscript` (`gs`, used by pygmt's `psconvert`
  to rasterise figures — the `gmt` package does not pull it in).
- **pip:** `ffmpeg-python`, and `visualisation` itself (installed `--no-deps`).
- **baked data:** pygmt_helper's NZ topo/coastline grids (~1.8 GB) **and** cartopy's
  Natural Earth vectors (43 MB), so plots work with no internet on compute nodes.
  These are two independent data systems serving two different tools — see
  gotcha 6; provisioning one does nothing for the other.

Result is a ~9 GB `.sif` (the baked grids dominate; see "Grid data" below).

## Prerequisites

- `apptainer` (rootless build works on the dev box; see the workflow repo's
  apptainer notes for the AppArmor userns setup).
- Either the base image cached locally, or network access to pull the **public**
  `earthquakesuc/workflow-bootstrap:latest` from Docker Hub.
- ~10 GB free disk for the image, plus build scratch.
- Internet during build, from three separate sources: Docker Hub (base image),
  Dropbox (pygmt_helper grids) and `naturalearth.s3.amazonaws.com` (cartopy
  vectors). A firewall that allows only some of these fails partway through.
- The `uidmap` package. If the building user has `/etc/subuid` entries, apptainer
  selects fakeroot mode, which needs the `newuidmap`/`newgidmap` helpers — without
  them the build dies immediately with `newuidmap was not found in PATH`. Having
  the subuid mappings is what *triggers* the requirement, so mappings alone are
  not enough.

## Build

```bash
cd workflow/container        # where viz.def lives
apptainer build ~/build/runner/viz.sif viz.def
```

Build takes ~10 min, most of it the grid download. The `%test` section runs at the
end and **fails the build** if GMT won't load, the entry points don't import, or
the baked topo grids are missing — so a clean exit is a real verification.

### If the build fails on Docker Hub auth

```
FATAL: ... unable to retrieve auth token: invalid username/password:
       unauthorized: incorrect username or password
```

This is **not** a problem with the def. Apptainer re-resolves the moving `:latest`
tag against Docker Hub on each build and sends whatever credential is in
`~/.docker/config.json`. Web-login tokens there expire; once expired, Apptainer
sends a stale credential and the registry rejects it — even though the image is
public and pulls fine anonymously.

**Reliable fix — build from the local docker daemon image** (no registry, no auth):
if you have `earthquakesuc/workflow-bootstrap:latest` in your local docker (check
`docker image ls`), copy the def and point its bootstrap at the daemon:

```bash
sed 's/^Bootstrap: docker$/Bootstrap: docker-daemon/' viz.def > /tmp/viz.daemon.def
apptainer build --force ~/build/runner/viz.sif /tmp/viz.daemon.def
```

An empty `DOCKER_CONFIG` (`DOCKER_CONFIG=<dir with just '{}'> apptainer build …`)
makes `apptainer exec` pull anonymously, but was **not** reliable for
`apptainer build` here — its conveyor took a different credential path and still
401'd. Use docker-daemon, or `docker login` to refresh the token. The canonical
`viz.def` stays on `Bootstrap: docker` for reproducibility; the daemon swap is a
local build-time workaround only.

## Why the def looks the way it does (gotchas)

Six non-obvious things, each of which causes a silent or confusing failure if
dropped:

1. **`libgmt.so` symlink.** Ubuntu's `gmt` package ships only the *versioned*
   `libgmt.so.6`; the unversioned `libgmt.so` normally comes from `libgmt-dev`.
   pip's pygmt `dlopen`s the **unversioned** name and fails with
   `GMTCLibNotFoundError` without it. Setting `GMT_LIBRARY_PATH` does **not** fix
   this (verified). The def creates the symlink in `%post`, before anything imports
   pygmt. Installing `libgmt-dev` would also work but pulls needless headers.

2. **`visualisation` must be installed `--no-deps`.** Its `requirements.txt`
   cannot resolve cleanly against this image:
   - it lists `qcore @ git+…`, but that repo's package name is `qcore-utils`
     (already present) — pip discards the URL over the name mismatch and then
     can't find a `qcore` on PyPI, so a normal install **errors out entirely**;
   - it lists `source_modelling @ git+…` with no version floor, which would
     **clobber the pinned 2026.7.2** release with a git-HEAD build.
   `--no-deps` sidesteps both and reuses the base image's already-correct stack.

3. **`ffmpeg-python` is a required, missing dependency.** `visualisation/plot_ts.py`
   does `import ffmpeg` (the `ffmpeg-python` package — distinct from the `ffmpeg`
   *binary*). It is not in the base image and, under `--no-deps`, won't come in
   automatically, so it's installed explicitly. (`diffimg`, also in
   `requirements.txt`, is imported nowhere in the package — skipped.)

4. **Grid data baked at a fixed cache path.** pygmt_helper fetches NZ grids at
   runtime via `pooch.os_cache("pygmt_helper")`, which honours `XDG_CACHE_HOME`.
   The def sets `XDG_CACHE_HOME=/opt/cache` during the fetch **and** exports the
   same value in `%environment`, so the baked data is found at runtime regardless
   of `$HOME` — including under `apptainer exec -c` (contained). Without this,
   plotting on an offline compute node would fail trying to download grids.

5. **Writable cache dirs for a read-only image.** The `.sif` filesystem is
   read-only at runtime, but two libraries need to *write* a cache on import/use
   and hard-fail (not warn) if they can't:
   - `source_modelling` uses numba `@njit(cache=True)`, which writes a JIT cache;
     with nowhere writable, importing it raises and **`plot-srf`/`plot-ts` fail
     to even start** (this is not caught by an import test on a *writable* dev
     box — only in the read-only container).
   - matplotlib writes a font cache.
   `%environment` redirects both to per-user `/tmp` dirs
   (`NUMBA_CACHE_DIR`, `MPLCONFIGDIR`), evaluated at container start. `/tmp` is
   writable under Apptainer at runtime; the cost is a few seconds of numba
   recompilation on a fresh `/tmp`. (`XDG_CACHE_HOME` stays pointed at the
   read-only baked grids — pooch only reads those, so that's fine.)

6. **cartopy's Natural Earth data is a second, independent bake.** Gotcha 4 covers
   pygmt_helper/pooch, which serves `plot-srf` (pygmt → GSHHG + SRTM). It does
   **nothing** for `plot-ts`, which draws its `--simple-map` basemap through
   cartopy (`cfeature.LAND/OCEAN/COASTLINE/BORDERS/LAKES.with_scale()`) and fetches
   from `naturalearth.s3` on first use. Two separate systems, two separate bakes —
   a container can be perfectly offline-capable for one tool and not the other,
   which is exactly the state this image was in before.

   Three things make it fiddly:
   - cartopy **writes** downloads to `config['data_dir']` (`$XDG_DATA_HOME/cartopy`)
     but **reads** from `config['pre_existing_data_dir']` (`CARTOPY_DATA_DIR`).
     Different settings — so `%post` points the writer at `/opt` and `%environment`
     exports the reader as `/opt/cartopy`.
   - It must be `CARTOPY_DATA_DIR`, not `XDG_DATA_HOME`. Callers legitimately set
     `XDG_DATA_HOME` (the BSC plotting jobs do), and that would only move the
     download target; pinning the read path keeps the baked data winning whatever
     the caller's environment does.
   - Geometries resolve **lazily, at `savefig()`**, not at `add_feature()`. Offline
     that means the render dies after all the compute is done rather than failing
     fast — and it has already been mis-handled once by a `try` wrapped around
     `add_feature` that the actual fetch happened outside of.

   `%test` asserts the **files exist on disk** rather than asking the cartopy API
   to resolve them, because `%test` runs on the build host, which has internet: an
   API check would silently re-download anything missing and pass, hiding the very
   failure the bake prevents. Note `BORDERS` lands in `natural_earth/cultural/` as
   `ne_<scale>_admin_0_boundary_lines_land` while the other four are `physical/`.

## Grid data (size decision)

The baked NZ grids are ~1.8 GB, dominated by the high-resolution 1-arc-second
topography:

| file | size | note |
|---|---|---|
| `srtm_NZ.grd` | 50 MB | low-res topo |
| `srtm_NZ_i5.grd` | 133 MB | low-res shading |
| `srtm_NZ_1s.grd` | 476 MB | **high-res** topo |
| `srtm_NZ_1s_i5.grd` | 1135 MB | **high-res** shading |
| vector paths + CPTs | ~tens of MB | small |

Korea grids (`srtm_KR*`, ~100 MB) are skipped. If you never plot high-resolution
topo, dropping the two `*_1s*` files from the bake saves ~1.6 GB — edit the `skip`
set in the `%post` Python block.

## Verify (inside the container)

The build's `%test` section only covers session-free checks (imports, `--help`,
`gs`/`ffmpeg` presence, grid files) — anything that starts a GMT session fails in
the read-only build sandbox even though it works at runtime. So the **render** is
the check that actually matters, and it must be run with `apptainer exec` (writable
`/tmp`/`$HOME`), not in `%test`.

```bash
# imports + entry points
apptainer exec viz.sif python -c "from visualisation.sources import plot_srf; from visualisation import plot_ts"
apptainer exec viz.sif plot-srf --help
apptainer exec viz.sif plot-ts --help

# source_modelling must still be the pinned release, not a git-HEAD clobber:
apptainer exec viz.sif python -c "import importlib.metadata as m; print(m.version('source_modelling'))"   # 2026.7.2
apptainer exec viz.sif python -c "import source_modelling.srf_parser as p, inspect; print(inspect.signature(p.parse_srf))"  # (buffer)

# THE key check — an actual figure render (exercises libgmt + ghostscript):
apptainer exec viz.sif python -c "
import pygmt
f = pygmt.Figure()
f.basemap(region=[166,179,-47,-34], projection='M8c', frame=True)
f.coast(shorelines=True, land='gray')
f.savefig('/tmp/nz_test.png')
print('render OK')
"

# and an animation render (exercises ffmpeg):
apptainer exec viz.sif python -c "
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
fig, ax = plt.subplots()
a = FuncAnimation(fig, lambda i: ax.plot(np.arange(i))[0], frames=5)
a.save('/tmp/anim_test.mp4', writer=FFMpegWriter(fps=5))
print('anim OK')
"

# cartopy basemap data resolves from the BAKED copy with no network call.
# Promoting DownloadWarning to an error is what makes this a real offline test
# on a machine that HAS internet: without it, a missing shapefile is silently
# re-downloaded and the check passes here but fails on an offline compute node.
apptainer exec viz.sif python -c "
import warnings, cartopy
import cartopy.feature as cf
from cartopy.io import DownloadWarning
warnings.simplefilter('error', DownloadWarning)
print('read path:', cartopy.config['pre_existing_data_dir'])
for s in ('10m', '50m'):
    for n in ('LAND', 'OCEAN', 'COASTLINE', 'BORDERS', 'LAKES'):
        len(list(getattr(cf, n).with_scale(s).geometries()))
print('cartopy basemap data OK (no downloads)')
"
```

If `pygmt` can't load `libgmt.so`, or `savefig` errors with
`psconvert ... Cannot execute Ghostscript`, the corresponding apt package/symlink
is missing (see gotcha 1 and the `ghostscript` note).

## Running it

The entry points need **no** manual environment — `NUMBA_CACHE_DIR`,
`MPLCONFIGDIR`, and `XDG_CACHE_HOME` are baked into `%environment`. Bind the input
SRF (or its dir) and an output dir, then call the tool. End-to-end example that
renders a real SRF to PNG (this is the check that actually proves the whole chain
— numba JIT → SRF parse → pygmt → ghostscript — works in the read-only image):

```bash
apptainer exec --bind /path/to/srfs:/in --bind /path/to/out:/out \
    viz.sif plot-srf /in/rupture.srf /out/rupture.png
```

`plot-ts` (animations) works the same way; run `apptainer exec viz.sif plot-ts --help`
for its subcommands.

## Deploy to BSC

The dev box has `bsc_transfer{1..4}` configured as rclone SFTP remotes
(`transfer1.bsc.es`). Paths are home-relative, so this lands it at `~/viz.sif`:

```bash
rclone copy ~/build/runner/viz.sif bsc_transfer1: --progress
# verify end-to-end (hash computed cluster-side):
rclone hashsum md5 bsc_transfer1:viz.sif
md5sum ~/build/runner/viz.sif
```

Use `rclone copy`, never `sync` (sync deletes remote files absent locally).

Because the grids are baked in, no cache pre-warming or compute-node internet is
required — the container is self-contained for NZ plotting.
