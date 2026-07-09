#!/usr/bin/env python3
"""Check an SW4 sfile (HDF5 velocity model) for issues.

Checks performed
----------------
  Structure    : required groups/attributes, dataset counts match ngrids
  Attributes   : types, value ranges, NaN/Inf
  Z_interfaces : NaN/Inf, strict monotonicity z_values_0 < z_values_1 < ... at
                 every (i,j) (interfaces stored at different resolutions are
                 nearest-resampled to a common grid, not skipped),
                 zero/negative-thickness layers (triggers hv=0 division in
                 SW4), large topographic gradients between cells, consistency
                 between Min/max depth attribute and actual data
  Material     : NaN/Inf, Rho > 0 (zero density is the most common SW4 crash),
                 Cp > 0, Cs >= 0, Qp/Qs > 0 (when attenuation=1), Vp/Vs >= 1,
                 nk >= 2 (nk=1 causes 0*inf=NaN in SW4 interpolation)
  Boundaries   : grid corners in geographic coordinates

The geographic projection assumed is:
    proj=tmerc  ellps=GRS80  lon_0=173.0  lat_0=0.0  k=0.9996
    (NZTM2000 central meridian and scale but without false origin offsets)
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pyproj
import typer

app = typer.Typer()
PROJ_STRING = (
    "+proj=tmerc +ellps=GRS80 +lon_0=173.0 +lat_0=0.0 +k=0.9996 +units=m +no_defs"
)

REQUIRED_ATTRS = [
    "Attenuation",
    "ngrids",
    "Min, max depth",
    "Origin longitude, latitude, azimuth",
]
SPACING_ATTRS = ["Finest horizontal grid spacing", "Coarsest horizontal grid spacing"]
REQUIRED_VARS = {"Cp", "Cs", "Rho"}
ATTN_VARS = {"Qp", "Qs"}
SQRT2 = np.sqrt(2.0)


def _sorted_keys(group):
    return sorted(group.keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))


def _scalar(v):
    return float(np.asarray(v).flat[0])


class Validator:
    def __init__(self, path, chunk_rows=20, verbose=False):
        self.path = path
        self.chunk_rows = chunk_rows
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        # State shared across phases
        self.ngrids = None
        self.attn = 0
        self.zmin_attr = self.zmax_attr = None
        self.origin = None  # (lon, lat, az)
        self.zi_keys = []
        self.grid_shapes = {}  # gname -> (ni, nj, nk)
        self.grid_h = {}  # gname -> h

    def error(self, msg):
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg):
        self.warnings.append(msg)
        print(f"  [WARN]  {msg}")

    def info(self, msg):
        print(f"         {msg}")

    # ── resampling helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _resample_to(src, shape):
        """Nearest-neighbour resample 2-D `src` onto `shape`, treating both
        arrays as covering the same physical extent corner-to-corner (as SW4's
        nested sfile interface grids do). Nearest (not bilinear) is deliberate:
        it preserves every stored node value — mirroring how SW4 samples
        interfaces via integer strides — so a crossing sitting on a single
        coarse node is never smoothed away, and well-separated layers never
        produce a spurious crossing."""
        sr, sc = src.shape
        tr, tc = shape
        if (sr, sc) == (tr, tc):
            return src
        ri = np.rint(np.linspace(0.0, sr - 1, tr)).astype(int)
        ci = np.rint(np.linspace(0.0, sc - 1, tc)).astype(int)
        return src[np.ix_(ri, ci)]

    @classmethod
    def _align(cls, a, b):
        """Resample the coarser of `a`, `b` onto the finer one's grid so they
        can be compared elementwise. Returns (a, b, resampled?)."""
        if a.shape == b.shape:
            return a, b, False
        target = a.shape if a.size >= b.size else b.shape
        return cls._resample_to(a, target), cls._resample_to(b, target), True

    @staticmethod
    def _worst_loc(field, want_max=True):
        """(i, j, fx%, fy%) of the extremum of `field` — for locating the worst
        crossing/pinch as a fraction of the domain (axis0=i/x, axis1=j/y)."""
        idx = np.unravel_index(
            int(np.argmax(field) if want_max else np.argmin(field)), field.shape
        )
        ni, nj = field.shape
        fx = 100.0 * idx[0] / (ni - 1) if ni > 1 else 0.0
        fy = 100.0 * idx[1] / (nj - 1) if nj > 1 else 0.0
        return idx[0], idx[1], fx, fy

    # ── entry ────────────────────────────────────────────────────────────────────

    def run(self):
        print(f"\n{'=' * 72}\n  SW4 sfile validator\n  {self.path}\n{'=' * 72}\n")
        try:
            f = h5py.File(self.path, "r")
        except Exception as exc:
            self.error(f"Cannot open file: {exc}")
            self._summary()
            return

        with f:
            self._check_attrs(f)
            self._check_structure(f)
            if "Z_interfaces" in f:
                self._check_z_interfaces(f)
            if "Material_model" in f:
                self._check_material(f)
            self._check_cross(f)
            self._boundaries(f)
        self._summary()

    # ── attributes ───────────────────────────────────────────────────────────────

    def _check_attrs(self, f):
        print("--- Root Attributes ---")
        attrs = f.attrs

        for name in REQUIRED_ATTRS:
            if name not in attrs:
                self.error(f"Missing required attribute '{name}'")

        if not any(n in attrs for n in SPACING_ATTRS):
            self.error(f"Missing spacing attribute; need one of: {SPACING_ATTRS}")
        for name in SPACING_ATTRS:
            if name not in attrs:
                continue
            h = _scalar(attrs[name])
            self.info(f"{name}: {h} m")
            if h <= 0 or not np.isfinite(h):
                self.error(f"'{name}' must be finite and > 0, got {h}")

        if "Attenuation" in attrs:
            att = int(_scalar(attrs["Attenuation"]))
            self.info(f"Attenuation: {att}")
            if att not in (0, 1):
                self.error(f"Attenuation must be 0 or 1, got {att}")
            self.attn = att

        if "ngrids" in attrs:
            ng = int(_scalar(attrs["ngrids"]))
            self.info(f"ngrids: {ng}")
            if ng <= 0:
                self.error(f"ngrids must be > 0, got {ng}")
            self.ngrids = ng

        if "Min, max depth" in attrs:
            mmd = np.asarray(attrs["Min, max depth"]).ravel()
            if len(mmd) < 2:
                self.error("'Min, max depth' must have 2 values")
            else:
                zmin, zmax = float(mmd[0]), float(mmd[1])
                self.info(
                    f"Min, max depth: [{zmin:.2f}, {zmax:.2f}] m  (depth-positive; negative = above sea level)"
                )
                if not (np.isfinite(zmin) and np.isfinite(zmax)):
                    self.error("'Min, max depth' contains NaN or Inf")
                elif zmin >= zmax:
                    self.error(
                        f"Min depth ({zmin:.2f}) must be < max depth ({zmax:.2f})"
                    )
                self.zmin_attr, self.zmax_attr = zmin, zmax

        if "Origin longitude, latitude, azimuth" in attrs:
            ola = np.asarray(attrs["Origin longitude, latitude, azimuth"]).ravel()
            if len(ola) < 3:
                self.error("'Origin longitude, latitude, azimuth' must have 3 values")
            else:
                lon0, lat0, az = float(ola[0]), float(ola[1]), float(ola[2])
                self.info(f"Origin: lon={lon0:.6f}°  lat={lat0:.6f}°  az={az:.4f}°")
                if not (-180 <= lon0 <= 180):
                    self.error(f"Origin longitude {lon0} out of range [-180, 180]")
                if not (-90 <= lat0 <= 90):
                    self.error(f"Origin latitude {lat0} out of range [-90, 90]")
                if not np.isfinite(az):
                    self.error(f"Origin azimuth {az} is not finite")
                self.origin = (lon0, lat0, az)
        print()

    # ── structure ─────────────────────────────────────────────────────────────────

    def _check_structure(self, f):
        print("--- File Structure ---")
        expect = f"{self.ngrids + 1}" if self.ngrids else "?"

        if "Z_interfaces" not in f:
            self.error("Missing 'Z_interfaces' group")
        else:
            self.info(
                f"Z_interfaces:  {len(f['Z_interfaces'])} dataset(s)  (expect ngrids+1 = {expect})"
            )

        if "Material_model" not in f:
            self.error("Missing 'Material_model' group")
        else:
            n = len(f["Material_model"])
            self.info(f"Material_model: {n} grid(s)")
            if self.ngrids is not None and n != self.ngrids:
                self.error(f"ngrids={self.ngrids} but Material_model has {n} grid(s)")
        print()

    # ── z_interfaces ─────────────────────────────────────────────────────────────

    def _check_z_interfaces(self, f):
        print("--- Z_interfaces ---")
        zi = f["Z_interfaces"]
        keys = _sorted_keys(zi)
        self.zi_keys = keys

        if self.ngrids is not None and len(keys) != self.ngrids + 1:
            self.error(
                f"Expected {self.ngrids + 1} interface datasets, found {len(keys)}"
            )

        arrays = {}
        for name in keys:
            ds = zi[name]
            if ds.ndim != 2:
                self.error(f"{name}: expected 2-D, got shape {ds.shape}")
                continue
            arr = ds[()].astype(np.float64)
            arrays[name] = arr

            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan:
                self.error(f"{name}: {n_nan} NaN value(s)")
            if n_inf:
                self.error(f"{name}: {n_inf} Inf value(s)")

            valid = arr[np.isfinite(arr)]
            z_lo = float(valid.min()) if valid.size else float("nan")
            z_hi = float(valid.max()) if valid.size else float("nan")
            self.info(
                f"{name}: shape={ds.shape}  z=[{z_lo:.2f}, {z_hi:.2f}] m  elev=[{-z_hi:.1f}, {-z_lo:.1f}] m ASL"
            )

        # Monotonicity — compare each consecutive pair. Interface grids may be
        # stored at different resolutions; resample the coarser onto the finer
        # rather than skip (crossed/pinched layers are a leading cause of
        # localized SW4 instabilities right at the grid-refinement interfaces).
        for a_name, b_name in zip(keys, keys[1:]):
            a, b = arrays.get(a_name), arrays.get(b_name)
            if a is None or b is None:
                continue
            a, b, resampled = self._align(a, b)
            note = " (after nearest resample to common grid)" if resampled else ""
            diff = a - b  # >= 0 where b <= a: deeper interface not below shallower
            n_bad = int((diff >= 0.0).sum())
            if n_bad:
                i, j, fx, fy = self._worst_loc(diff, want_max=True)
                self.error(
                    f"{b_name} <= {a_name} at {n_bad} point(s){note} — "
                    f"worst gap {float(diff.max()):.2f} m at (i,j)=({i},{j}) "
                    f"≈ ({fx:.1f}%, {fy:.1f}%) of domain — "
                    f"SW4 patch-selection requires strict depth ordering"
                )

        # Terrain gradient on z_values_0
        topo = arrays.get(keys[0]) if keys else None
        if topo is not None and topo.shape[0] > 1 and topo.shape[1] > 1:
            max_grad = max(
                float(np.abs(np.diff(topo, axis=0)).max()),
                float(np.abs(np.diff(topo, axis=1)).max()),
            )
            if max_grad > 500:
                self.warn(
                    f"{keys[0]}: max cell-to-cell z jump {max_grad:.1f} m — check for fill-value boundaries"
                )
            else:
                self.info(f"  max terrain gradient: {max_grad:.1f} m/cell")

        # Min/max depth attribute vs actual data
        if len(keys) >= 2 and self.zmin_attr is not None:
            a0, aN = arrays.get(keys[0]), arrays.get(keys[-1])
            if a0 is not None and aN is not None:
                checks = [
                    ("min", self.zmin_attr, float(a0.min())),
                    ("max", self.zmax_attr, float(aN.max())),
                ]
                for label, attr_v, actual_v in checks:
                    if abs(attr_v - actual_v) > 1.0:
                        self.warn(
                            f"'Min, max depth' {label}={attr_v:.2f} m but data {label} is {actual_v:.2f} m ({actual_v - attr_v:+.2f} m diff)"
                        )
        print()

    # ── material model ────────────────────────────────────────────────────────────

    def _check_material(self, f):
        print("--- Material Model ---")
        mm = f["Material_model"]
        for gname in _sorted_keys(mm):
            print(f"\n  [{gname}]")
            self._check_grid(mm[gname], gname)
        print()

    def _check_grid(self, g, gname):
        if "Horizontal grid size" not in g.attrs:
            self.error(f"{gname}: missing 'Horizontal grid size'")
        else:
            h = _scalar(g.attrs["Horizontal grid size"])
            self.info(f"  h = {h} m")
            if h <= 0 or not np.isfinite(h):
                self.error(f"{gname}: h must be > 0, got {h}")
            else:
                self.grid_h[gname] = h

        if "Number of components" in g.attrs:
            nc = int(_scalar(g.attrs["Number of components"]))
            if nc not in (3, 5):
                self.warn(
                    f"{gname}: unexpected Number of components={nc} (expected 3 or 5)"
                )

        for name in sorted(REQUIRED_VARS):
            if name not in g:
                self.error(f"{gname}: missing required dataset '{name}'")
        if self.attn:
            for name in sorted(ATTN_VARS):
                if name not in g:
                    self.error(f"{gname}: attenuation=1 but '{name}' is absent")

        if "Cp" not in g:
            return
        shape = tuple(g["Cp"].shape)
        if len(shape) != 3:
            self.error(f"{gname}/Cp: expected 3-D, got {shape}")
            return

        ni, nj, nk = shape
        self.grid_shapes[gname] = shape
        self.info(f"  shape (ni, nj, nk) = {shape}")
        if nk < 2:
            self.error(
                f"{gname}: nk={nk} — SW4 computes hv=thickness/(nk-1); nk<2 produces NaN"
            )

        vars_to_check = sorted(REQUIRED_VARS | (ATTN_VARS if self.attn else set()))
        for name in vars_to_check:
            if name in g:
                self._check_dataset(g[name], f"{gname}/{name}", name)

        if "Cp" in g and "Cs" in g:
            self._check_vp_vs(g["Cp"], g["Cs"], gname)

    def _dataset_stats(self, ds):
        n_nan = n_inf = n_zero = n_neg = n_total = 0
        val_min, val_max = np.inf, -np.inf
        for start in range(0, ds.shape[0], self.chunk_rows):
            chunk = ds[start : start + self.chunk_rows].astype(np.float64)
            n_total += chunk.size
            n_nan += int(np.isnan(chunk).sum())
            n_inf += int(np.isinf(chunk).sum())
            n_zero += int((chunk == 0.0).sum())
            n_neg += int((chunk < 0.0).sum())
            finite = chunk[np.isfinite(chunk)]
            if finite.size:
                val_min = min(val_min, float(finite.min()))
                val_max = max(val_max, float(finite.max()))
        return dict(
            min=val_min,
            max=val_max,
            n_nan=n_nan,
            n_inf=n_inf,
            n_zero=n_zero,
            n_neg=n_neg,
            n_total=n_total,
        )

    def _check_dataset(self, ds, label, kind):
        s = self._dataset_stats(ds)

        if self.verbose:
            self.info(
                f"  {kind}: min={s['min']:.4g}  max={s['max']:.4g}  "
                f"zeros={s['n_zero']}  nan={s['n_nan']}  inf={s['n_inf']}"
            )
        if s["n_nan"]:
            self.error(f"{label}: {s['n_nan']} NaN value(s)")
        if s["n_inf"]:
            self.error(f"{label}: {s['n_inf']} Inf value(s)")

        if kind == "Rho":
            bad = s["n_zero"] + s["n_neg"]
            if bad:
                self.error(
                    f"{label}: {bad} point(s) with density <= 0 "
                    f"(zeros={s['n_zero']}, neg={s['n_neg']}) — SW4 aborts with Density=0"
                )
            if not self.verbose:
                self.info(f"  Rho: [{s['min']:.2f}, {s['max']:.2f}] kg/m³")
            if 0 < s["min"] < 100:
                self.warn(f"{label}: suspiciously low density min={s['min']:.2f} kg/m³")

        elif kind == "Cp":
            if s["min"] <= 0:
                self.error(f"{label}: {s['n_zero'] + s['n_neg']} point(s) with Vp <= 0")
            if not self.verbose:
                self.info(f"  Vp:  [{s['min']:.1f}, {s['max']:.1f}] m/s")
            if 0 < s["min"] < 200:
                self.warn(f"{label}: suspiciously low Vp min={s['min']:.1f} m/s")

        elif kind == "Cs":
            if s["n_neg"]:
                self.error(f"{label}: {s['n_neg']} negative Vs value(s)")
            if not self.verbose:
                self.info(f"  Vs:  [{s['min']:.1f}, {s['max']:.1f}] m/s")
            if s["n_zero"]:
                self.warn(
                    f"{label}: {s['n_zero']} zero Vs value(s) (fluid cells?) — verify intentional"
                )

        elif kind in ("Qp", "Qs"):
            bad = s["n_zero"] + s["n_neg"]
            if bad:
                self.error(
                    f"{label}: {bad} point(s) with Q <= 0 (required > 0 when attenuation=1)"
                )
            if not self.verbose:
                self.info(f"  {kind}: [{s['min']:.3g}, {s['max']:.3g}]")

    def _check_vp_vs(self, cp_ds, cs_ds, gname):
        n_below_1 = n_below_sqrt2 = 0
        ratio_min, ratio_max = np.inf, -np.inf
        for start in range(0, cp_ds.shape[0], self.chunk_rows):
            cp = cp_ds[start : start + self.chunk_rows].astype(np.float64)
            cs = cs_ds[start : start + self.chunk_rows].astype(np.float64)
            solid = cs > 0.0
            if not solid.any():
                continue
            ratio = np.where(solid, cp / np.where(solid, cs, 1.0), np.nan)
            r = ratio[solid]
            ratio_min = min(ratio_min, float(r.min()))
            ratio_max = max(ratio_max, float(r.max()))
            n_below_1 += int((solid & (ratio < 1.0)).sum())
            n_below_sqrt2 += int((solid & (ratio < SQRT2)).sum())

        if ratio_min < np.inf:
            self.info(f"  Vp/Vs (solid cells): [{ratio_min:.3f}, {ratio_max:.3f}]")
        if n_below_1:
            self.error(
                f"{gname}: {n_below_1} solid point(s) with Vp/Vs < 1 (physically impossible)"
            )
        elif n_below_sqrt2:
            self.warn(
                f"{gname}: {n_below_sqrt2} solid point(s) with Vp/Vs < √2 ≈ 1.414"
            )

    # ── cross-consistency ─────────────────────────────────────────────────────────

    def _check_cross(self, f):
        print("--- Cross-Consistency ---")
        ok = True

        if "Z_interfaces" in f and "Material_model" in f:
            n_zi, n_mm = len(f["Z_interfaces"]), len(f["Material_model"])
            if n_zi != n_mm + 1:
                self.error(
                    f"Z_interfaces has {n_zi} datasets, Material_model has {n_mm} grids — need n_zi = n_mm+1"
                )
                ok = False

        # All material grids must span the same horizontal domain
        extents_x = [
            (self.grid_shapes[gn][0] - 1) * h
            for gn, h in self.grid_h.items()
            if gn in self.grid_shapes
        ]
        extents_y = [
            (self.grid_shapes[gn][1] - 1) * h
            for gn, h in self.grid_h.items()
            if gn in self.grid_shapes
        ]
        if extents_x:
            tol = max(1.0, max(extents_x) * 0.001)
            if max(extents_x) - min(extents_x) > tol:
                self.warn(
                    f"Inconsistent x-extents across grids: {[f'{x:.0f}m' for x in extents_x]}"
                )
                ok = False
            if max(extents_y) - min(extents_y) > tol:
                self.warn(
                    f"Inconsistent y-extents across grids: {[f'{y:.0f}m' for y in extents_y]}"
                )
                ok = False

        # Per-patch layer thickness and nk check
        if "Z_interfaces" not in f or len(self.zi_keys) < 2:
            if ok:
                self.info("Cross-consistency OK")
            print()
            return

        zi = f["Z_interfaces"]
        for gi, gname in enumerate(_sorted_keys(f["Material_model"])):
            shape = self.grid_shapes.get(gname)
            top_key = self.zi_keys[gi] if gi < len(self.zi_keys) else None
            bot_key = self.zi_keys[gi + 1] if gi + 1 < len(self.zi_keys) else None
            if not shape or top_key is None or bot_key is None:
                continue

            nk = shape[2]
            top = zi[top_key][()].astype(np.float64)
            bot = zi[bot_key][()].astype(np.float64)
            # Bracketing interfaces may be at different resolutions — resample
            # to a common grid rather than silently skip the thickness check.
            top, bot, resampled = self._align(top, bot)
            note = " (interfaces resampled to common grid)" if resampled else ""

            thickness = bot - top
            t_min, t_max = float(thickness.min()), float(thickness.max())
            self.info(
                f"{gname} ({top_key}→{bot_key}): thickness=[{t_min:.1f}, {t_max:.1f}] m  nk={nk}{note}"
            )

            n_bad = int((thickness <= 0).sum())
            if n_bad:
                i, j, fx, fy = self._worst_loc(thickness, want_max=False)
                self.error(
                    f"{gname}: {n_bad} point(s) with zero/negative thickness{note} — "
                    f"min {t_min:.2f} m at (i,j)=({i},{j}) ≈ ({fx:.1f}%, {fy:.1f}%) of domain — "
                    f"SW4 hv=thickness/(nk-1) → NaN"
                )
                ok = False
            elif nk >= 2 and 0 < t_min / (nk - 1) < 0.01:
                self.warn(
                    f"{gname}: min vertical cell size {t_min / (nk - 1):.4f} m — extremely thin"
                )

        if ok:
            self.info("Cross-consistency OK")
        print()

    # ── boundaries ────────────────────────────────────────────────────────────────

    def _boundaries(self, f):
        print("--- Model Boundaries ---")

        if self.origin is None:
            self.warn("Cannot compute boundaries: origin attribute missing")
            print()
            return

        lon0, lat0, az = self.origin

        by_h = sorted(
            (h, gn) for gn, h in self.grid_h.items() if gn in self.grid_shapes
        )
        if not by_h:
            self.warn("Cannot compute extent: no valid material grid found")
            print()
            return

        best_h, best_gname = by_h[0]
        ni, nj, nk = self.grid_shapes[best_gname]
        x_m, y_m = (ni - 1) * best_h, (nj - 1) * best_h

        self.info(f"Using {best_gname} (h={best_h:.1f} m, shape {ni}×{nj}×{nk})")
        self.info(
            f"Extent:     {x_m / 1e3:.3f} km × {y_m / 1e3:.3f} km  (x at azimuth {az:.2f}° from north)"
        )
        if self.zmin_attr is not None:
            self.info(
                f"Depth:      {self.zmin_attr:.0f} m to {self.zmax_attr:.0f} m  "
                f"(~{(self.zmax_attr - max(0.0, self.zmin_attr)) / 1e3:.1f} km rock column)"
            )

        if self.zi_keys and "Z_interfaces" in f:
            topo = f["Z_interfaces"][self.zi_keys[0]][()].astype(np.float64)
            t_lo, t_hi = float(np.nanmin(topo)), float(np.nanmax(topo))
            self.info(
                f"Topography: z=[{t_lo:.2f}, {t_hi:.2f}] m  elev=[{-t_hi:.1f}, {-t_lo:.1f}] m ASL"
            )

        try:
            proj = pyproj.Proj(PROJ_STRING)
        except Exception as exc:
            self.warn(f"pyproj failed: {exc}")
            print()
            return

        e0, n0 = proj(lon0, lat0)
        # SW4 azimuth: angle from north to x-axis, clockwise
        # x-unit in (east, north) = (sin_az, cos_az); y-unit = (cos_az, -sin_az)
        sin_az = np.sin(np.radians(az))
        cos_az = np.cos(np.radians(az))

        corners = [
            (0, 0, "origin (SW)"),
            (x_m, 0, "far-x"),
            (0, y_m, "far-y"),
            (x_m, y_m, "far corner"),
        ]
        lons, lats = [], []
        print()
        self.info(
            f"  {'Corner':<14} {'Longitude':>12}  {'Latitude':>11}  {'Easting':>13}  {'Northing':>13}"
        )
        self.info("  " + "-" * 68)
        for sx, sy, label in corners:
            e = e0 + sx * sin_az + sy * cos_az
            n = n0 + sx * cos_az - sy * sin_az
            lon_c, lat_c = proj(e, n, inverse=True)
            lons.append(lon_c)
            lats.append(lat_c)
            self.info(
                f"  {label:<14} {lon_c:12.6f}°  {lat_c:11.6f}°  {e:13.1f}  {n:13.1f}"
            )

        print()
        self.info(
            f"Bounding box:  lon [{min(lons):.6f}°, {max(lons):.6f}°]   lat [{min(lats):.6f}°, {max(lats):.6f}°]"
        )
        print()

    # ── summary ───────────────────────────────────────────────────────────────────

    def _summary(self):
        n_e, n_w = len(self.errors), len(self.warnings)
        print("=" * 72)
        if n_e == 0 and n_w == 0:
            print("  RESULT: PASS — no issues found")
        elif n_e == 0:
            print(f"  RESULT: PASS with {n_w} warning(s)")
        else:
            print(f"  RESULT: FAIL — {n_e} error(s), {n_w} warning(s)")
        for e in self.errors:
            print(f"    • {e}")
        if self.errors and self.warnings:
            print()
        for w in self.warnings:
            print(f"    ~ {w}")
        print("=" * 72)


@app.command()
def validate_sfile(sfile: Path, chunk_rows: int = 20, verbose: bool = True) -> None:
    v = Validator(sfile, chunk_rows=chunk_rows, verbose=verbose)
    v.run()
    sys.exit(1 if v.errors else 0)
