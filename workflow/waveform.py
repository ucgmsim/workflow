"""API for loading compressed waveform datasets."""

from pathlib import Path

import dask.array as da
import flacarray
import h5py
import numpy as np
import xarray as xr
from numpy.lib import stride_tricks


class FlacH5Wrapper:
    """An array-like interface for FLAC compressed waveforms that only allocates memory for the compressed raw bytes."""

    def __init__(
        self,
        group: h5py.Group,
        shape: tuple,
        dtype: np.dtype,
    ) -> None:
        """Initialise the FLAC hdf5 wrapper.


        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from
        shape : tuple
            Shape of output array.
        dtype : np.dtype
            Data type of output array.
        """
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.stream_gains = np.array(group["stream_gains"])
        self.stream_offsets = np.array(group["stream_offsets"])
        self.stream_starts = np.array(group["stream_starts"])
        self.stream_nbytes = np.array(group["stream_bytes"])
        self.raw_bytes = np.array(group["compressed"])

    def __getitem__(self, key: tuple[int | slice, ...]) -> np.ndarray:
        """Getitem implementation that decompresses a slice of the compressed array data.



        Parameters
        ----------
        key : tuple[int | slice, ...]
            indexing key

        Returns
        -------
        np.ndarray
            Decompressed waveform array.
        """
        keep = np.zeros(self.stream_starts.shape, dtype=bool)
        stream_indexer = key[:-1]
        sample_indexer = key[-1]
        keep[stream_indexer] = True

        if isinstance(sample_indexer, int):
            first_sample = sample_indexer
            last_sample = sample_indexer + 1
        else:
            first_sample = (
                sample_indexer.start if sample_indexer.start is not None else 0
            )
            last_sample = (
                sample_indexer.stop
                if sample_indexer.stop is not None
                else self.shape[-1]
            )

        decompressed_data, _ = flacarray.decompress.array_decompress_slice(
            compressed=self.raw_bytes,
            stream_size=self.shape[-1],  # assuming last dim is time/samples
            stream_starts=self.stream_starts,
            stream_nbytes=self.stream_nbytes,
            stream_offsets=self.stream_offsets,
            stream_gains=self.stream_gains,
            keep=keep,
            first_stream_sample=first_sample,
            last_stream_sample=last_sample,
            is_int64=(self.dtype == np.float64),
        )

        dummy = np.zeros(1, dtype=np.int8)
        # The following is a "virtual array" with the same shape as
        # the theoretical array but the strides (how much memory each
        # element consumes) is zero so the effective memory
        # consumption of this array is zero. We are using this to get
        # an array-like that can calculate the expected shape from the
        # indexing operation without requiring potentially tens of gb
        # of memory.
        virtual_array = stride_tricks.as_strided(
            dummy, shape=self.shape, strides=(0,) * len(self.shape)
        )
        target_shape = virtual_array[key].shape

        return decompressed_data.reshape(target_shape)


def load_waveform_dataset(dataset_ffp: Path) -> xr.Dataset:
    """Read a compressed waveform array as an xarray dataset.

    Parameters
    ----------
    dataset_ffp : Path
        Path to xarray dataset.

    Returns
    -------
    xr.Dataset
        The xarray dataset read from disk.
    """
    ds = xr.open_dataset(
        dataset_ffp, engine="h5netcdf", drop_variables=["_flac_compressed_waveform"]
    )

    with h5py.File(dataset_ffp, "r") as h5:
        group = h5["_flac_compressed_waveform"]
        shape = tuple(group.attrs["shape"])
        dtype = np.dtype(group.attrs["dtype"])
        name = group.attrs["name"]
        dims = group.attrs["dims"]

        wrapper = FlacH5Wrapper(group, shape, dtype)

    waveform_da = da.from_array(
        wrapper,
        chunks="auto",
        name=name,
    )

    ds[name] = (dims, waveform_da)
    return ds
