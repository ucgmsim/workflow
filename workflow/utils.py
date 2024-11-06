"""Miscellaneous workflow utilities that couldn't go anywhere else."""

import multiprocessing
import os


def get_available_cores() -> int:
    """Get the avaiable number of cores for a job.

    Returns
    -------
    int
        Either the reported number of cores from the multiprocessing
        module, or the number of allocated cores if running in a slurm
        environment.
    """
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])

    if "SLURM_NPROCS" in os.environ:
        return int(os.environ["SLURM_NPROCS"])

    return multiprocessing.cpu_count()
