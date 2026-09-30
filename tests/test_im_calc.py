"""Tests for workflow.scripts.im_calc."""

import functools

import numpy as np
from IM.ims import IM

from workflow.scripts import im_calc


def test_im_function_map_covers_every_im_except_fas() -> None:
    # FAS is added separately by the caller because it additionally requires
    # a KO matrix directory, so it is expected to be absent here.
    function_map = im_calc._im_function_map(dt=0.01, psa_periods=np.array([1.0]))

    assert set(function_map) == set(IM) - {IM.FAS}


def test_cav5_uses_a_5_cm_per_s_squared_threshold() -> None:
    function_map = im_calc._im_function_map(dt=0.01, psa_periods=np.array([1.0]))

    cav5_function = function_map[IM.CAV5]
    assert isinstance(cav5_function, functools.partial)
    assert cav5_function.keywords["threshold"] == 5.0
