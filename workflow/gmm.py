"""Ground motion models available to the workflow, backed by `gm_models`.

Realisations name tectonic types and models with plain strings (the
`empirical` section of a realisation). This module resolves those names to
`gm_models` tectonic types and model runners.
"""

from collections.abc import Callable, Sequence

import pandas as pd
import xarray as xr

import gm_models
from gm_models.constants import TectType

TECT_TYPES: dict[str, TectType] = {
    "active_shallow": TectType.ACTIVE_SHALLOW,
    "volcanic": TectType.VOLCANIC,
    "subduction_interface": TectType.SUBDUCTION_INTERFACE,
    "subduction_slab": TectType.SUBDUCTION_SLAB,
}
"""Realisation tectonic type name to `gm_models` tectonic type."""

ModelRunner = Callable[
    [TectType, pd.DataFrame, str, Sequence[float] | None], xr.Dataset
]
"""A model evaluated as ``runner(tect_type, rupture_df, im, periods)``."""


def _run_as16(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None,  # Unused: durations have no period.
) -> xr.Dataset:
    """Run AS_16 with the `ModelRunner` signature."""
    return gm_models.run_as16(tect_type, rupture_df, im)


def _run_nshm2022(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None,
) -> xr.Dataset:
    """Run the NSHM 2022 logic tree with the `ModelRunner` signature.

    The logic tree rejects every IM but pSA, which needs periods.
    """
    return gm_models.run_nshm2022(tect_type, rupture_df, im, periods or ())


MODELS: dict[str, ModelRunner] = {
    "A_18": gm_models.run_a18,
    "A_22": gm_models.run_a22,
    "AG_20": gm_models.run_ag20,
    "AG_20_NZ": gm_models.run_ag20_nz,
    "AS_16": _run_as16,
    "ASK_14": gm_models.run_ask14,
    "BA_18": gm_models.run_ba18,
    "BCH_16": gm_models.run_bch16,
    "Br_13": gm_models.run_br13,
    "BSSA_14": gm_models.run_bssa14,
    "CB_10": gm_models.run_cb10,
    "CB_14": gm_models.run_cb14,
    "CY_14": gm_models.run_cy14,
    "K_20": gm_models.run_k20,
    "K_20_NZ": gm_models.run_k20_nz,
    "P_20": gm_models.run_p20,
    "P_21": gm_models.run_p21,
    "S_22": gm_models.run_s22,
    "ZA_06": gm_models.run_za06,
    "NSHM2022": _run_nshm2022,
}
"""Ground motion model (or logic tree) name to its `gm_models` runner."""


def run_model(
    model: str,
    tect_type: str,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Evaluate a ground motion model (or logic tree) by name.

    Parameters
    ----------
    model : str
        The model name, a key of `MODELS`.
    tect_type : str
        The tectonic type name, a key of `TECT_TYPES`.
    rupture_df : pd.DataFrame
        One row per site, with the rupture and site columns the model needs.
    im : str
        The intensity measure, e.g. ``"PGA"`` or ``"pSA"``.
    periods : Sequence[float], optional
        The periods to evaluate pSA at. Required, and only used, for pSA.

    Returns
    -------
    xr.Dataset
        The model's ``ln_mean`` and standard deviations (``std_total`` and,
        for single models, ``std_inter`` and ``std_intra``) over the rows of
        `rupture_df` (and ``period`` for pSA).

    Raises
    ------
    ValueError
        If the model does not support the tectonic type or intensity
        measure, or `rupture_df` lacks a column the model needs.
    """
    return MODELS[model](
        TECT_TYPES[tect_type], rupture_df, im, periods if im == "pSA" else None
    )
