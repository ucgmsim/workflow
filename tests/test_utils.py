import os
from unittest.mock import patch

from workflow import utils


def test_get_available_cores_slurm_cpus_on_node():
    with patch.dict(os.environ, {"SLURM_CPUS_ON_NODE": "4"}):
        assert utils.get_available_cores() == 4


def get_available_cores_slurm_nprocs():
    with patch.dict(os.environ, {"SLURM_NPROCS": "8"}):
        assert utils.get_available_cores() == 8


def get_available_cores_no_slurm():
    with patch("multiprocessing.cpu_count", return_value=16):
        assert utils.get_available_cores() == 16
