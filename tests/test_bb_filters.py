"""Tests for the broadband source-filter correction in `bb_sim`.

The correction exists because both solvers low-pass their *source time
functions* and `bb_sim` then low-passes the LF leg again, which leaves a hole in
the transition band. These tests pin the three things that can go wrong
silently: the analytic model of `qcore`'s filter drifting from the real one, the
correction not actually flattening the recombination, and the solver filters
being read wrongly out of a realisation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from qcore import timeseries
from workflow.realisations import RealisationMetadata
from workflow.scripts.bb_sim import (
    _HIGHPASS_SHIFT,
    _LOWPASS_SHIFT,
    BB_FILTER_ORDER,
    BB_FILTER_PASSES,
    CorrectionLeg,
    Solver,
    SourceLowpass,
    butterworth_gain,
    recombination_gains,
    solver_source_lowpass,
)

DT = 0.005
FLO = 1.0
# The two configurations actually in use: SW4's v26.7.1Hz `prefilter` command,
# and EMOD3D's `bfilt=4` with `flo = min_vs / (5 * resolution) = 1.0 Hz`.
SOURCES = [
    SourceLowpass(order=2, passes=2, corner=1.0),
    SourceLowpass(order=4, passes=2, corner=1.0),
]
# The band the correction is aimed at. Outside it the LF leg carries nothing
# (above) or the filters are flat (below), so neither says anything useful.
BAND = (0.2, 3.0)


def measured_response(band: timeseries.Band, frequencies: np.ndarray) -> np.ndarray:
    """The magnitude response of `bwfilter` itself, from an impulse.

    The impulse sits in the middle of the record, not at the start: `bwfilter`
    passes ``padtype=None`` to `sosfiltfilt`, so an impulse at index 0 has its
    acausal half truncated and the measured response is meaningless.
    """
    nt = 8192
    impulse = np.zeros(nt)
    impulse[nt // 2] = 1.0
    response = timeseries.bwfilter(impulse, DT, FLO, band)
    spectrum = np.abs(np.fft.rfft(response))
    return np.interp(frequencies, np.fft.rfftfreq(nt, DT), spectrum)


@pytest.mark.parametrize(
    ("band", "shift", "btype"),
    [
        (timeseries.Band.LOWPASS, _LOWPASS_SHIFT, "lowpass"),
        (timeseries.Band.HIGHPASS, _HIGHPASS_SHIFT, "highpass"),
    ],
)
def test_analytic_gain_matches_qcore(
    band: timeseries.Band, shift: float, btype: str
) -> None:
    """`butterworth_gain` must reproduce what `bwfilter` actually does.

    The shifts are mirrored from qcore rather than imported, so this is the
    guard against the two definitions drifting apart.
    """
    frequencies = np.linspace(*BAND, 200)
    analytic = butterworth_gain(
        frequencies, BB_FILTER_ORDER, FLO * shift, BB_FILTER_PASSES, DT, band=btype
    )
    np.testing.assert_allclose(
        analytic, measured_response(band, frequencies), atol=2e-3
    )


def test_matched_pair_is_power_complementary() -> None:
    """Without a source filter the legs already sum to one in power."""
    frequencies = np.linspace(*BAND, 400)
    low = butterworth_gain(
        frequencies, BB_FILTER_ORDER, FLO * _LOWPASS_SHIFT, BB_FILTER_PASSES, DT
    )
    high = butterworth_gain(
        frequencies,
        BB_FILTER_ORDER,
        FLO * _HIGHPASS_SHIFT,
        BB_FILTER_PASSES,
        DT,
        band="highpass",
    )
    # The pair is only approximately complementary by design; 0.05 in ln is the
    # deviation the shifts leave behind, and it is what the correction restores.
    assert np.abs(np.log(np.sqrt(low**2 + high**2))).max() < 0.05


def test_no_source_filter_is_a_no_op() -> None:
    frequencies = np.linspace(*BAND, 50)
    lf_gain, hf_gain = recombination_gains(frequencies, DT, FLO, None, CorrectionLeg.LF)
    np.testing.assert_array_equal(lf_gain, 1.0)
    np.testing.assert_array_equal(hf_gain, 1.0)


@pytest.mark.parametrize("source", SOURCES, ids=["sw4", "emod3d"])
def test_uncorrected_recombination_has_a_hole(source: SourceLowpass) -> None:
    """The defect the correction exists for: without it there is a real deficit."""
    frequencies = np.linspace(*BAND, 400)
    low = butterworth_gain(
        frequencies, BB_FILTER_ORDER, FLO * _LOWPASS_SHIFT, BB_FILTER_PASSES, DT
    )
    high = butterworth_gain(
        frequencies,
        BB_FILTER_ORDER,
        FLO * _HIGHPASS_SHIFT,
        BB_FILTER_PASSES,
        DT,
        band="highpass",
    )
    source_gain = butterworth_gain(
        frequencies, source.order, source.corner, source.passes, DT
    )
    deficit = np.log(np.sqrt((source_gain * low) ** 2 + high**2))
    assert deficit.min() < -0.1, "expected a transition-band hole to correct"


def power_sum_error(
    source: SourceLowpass, leg: CorrectionLeg, frequencies: np.ndarray
) -> np.ndarray:
    """|ln| of the corrected power sum against the matched pair's own."""
    low = butterworth_gain(
        frequencies, BB_FILTER_ORDER, FLO * _LOWPASS_SHIFT, BB_FILTER_PASSES, DT
    )
    high = butterworth_gain(
        frequencies,
        BB_FILTER_ORDER,
        FLO * _HIGHPASS_SHIFT,
        BB_FILTER_PASSES,
        DT,
        band="highpass",
    )
    source_gain = butterworth_gain(
        frequencies, source.order, source.corner, source.passes, DT
    )
    lf_gain, hf_gain = recombination_gains(frequencies, DT, FLO, source, leg)
    corrected = (lf_gain * source_gain * low) ** 2 + (hf_gain * high) ** 2
    return np.abs(np.log(np.sqrt(corrected / (low**2 + high**2))))


@pytest.mark.parametrize("source", SOURCES, ids=["sw4", "emod3d"])
def test_both_restores_the_power_sum_exactly(source: SourceLowpass) -> None:
    """Scaling the two legs together is exact everywhere, by construction."""
    frequencies = np.linspace(*BAND, 400)
    assert power_sum_error(source, CorrectionLeg.BOTH, frequencies).max() < 1e-9


@pytest.mark.parametrize("source", SOURCES, ids=["sw4", "emod3d"])
def test_lf_restores_the_power_sum_where_the_lf_leg_matters(
    source: SourceLowpass,
) -> None:
    """Exact below the matching frequency; only clipped where the LF is spent."""
    below = np.linspace(BAND[0], FLO, 200)
    assert power_sum_error(source, CorrectionLeg.LF, below).max() < 1e-9
    # Above it the boost is clipped, but the LF leg carries so little there that
    # the total barely moves. This bound is the promise the docstring makes.
    assert (
        power_sum_error(source, CorrectionLeg.LF, np.linspace(*BAND, 400)).max() < 0.01
    )


@pytest.mark.parametrize("source", SOURCES, ids=["sw4", "emod3d"])
def test_hf_cannot_restore_below_the_matching_frequency(
    source: SourceLowpass,
) -> None:
    """A known limitation, pinned so it cannot become a silent surprise.

    Filling an LF deficit from the HF leg below `flo` needs a boost of ten or
    more on stochastic content with no valid long-period part, so the correction
    clips and the power sum stays short.
    """
    below = np.linspace(BAND[0], FLO, 200)
    _, hf_gain = recombination_gains(below, DT, FLO, source, CorrectionLeg.HF)
    assert hf_gain.max() > 5.0, "expected the HF correction to be ill-conditioned"


@pytest.mark.parametrize("source", SOURCES, ids=["sw4", "emod3d"])
def test_both_is_the_best_conditioned_leg(source: SourceLowpass) -> None:
    """`both` is the option that never needs a large boost."""
    frequencies = np.linspace(*BAND, 400)
    lf_gain, hf_gain = recombination_gains(
        frequencies, DT, FLO, source, CorrectionLeg.BOTH
    )
    assert max(lf_gain.max(), hf_gain.max()) < 1.5


def test_solver_filters_are_read_from_the_realisation(tmp_path: Path) -> None:
    """The declared parameters must reach `SourceLowpass` unmangled.

    A realisation carrying only `metadata` and `resolution` falls back to the
    defaults for everything else, which is what pins the shipped values: SW4's
    `prefilter order=2 passes=2 fc2=1.0`, and EMOD3D's `bfilt=4` with a corner
    derived the way `create_e3d_par` derives it. The EMOD3D corner coming out at
    1.0 Hz is the same number `e3d.par` carries as `flo`.
    """
    realisation_ffp = tmp_path / "realisation.json"
    realisation_ffp.write_text(
        json.dumps(
            {
                "metadata": {
                    "name": "test",
                    "version": "1",
                    "defaults_version": "26.7.1Hz",
                    "tag": "gcmt",
                },
                "resolution": {"resolution": 0.1},
            }
        )
    )
    version = RealisationMetadata.read_from_realisation(
        realisation_ffp
    ).defaults_version

    assert solver_source_lowpass(Solver.SW4, realisation_ffp, version) == SourceLowpass(
        order=2, passes=2, corner=1.0
    )
    assert solver_source_lowpass(
        Solver.EMOD3D, realisation_ffp, version
    ) == SourceLowpass(order=4, passes=2, corner=1.0)
