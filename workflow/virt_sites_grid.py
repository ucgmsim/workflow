from pathlib import Path

import pooch
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from qcore import coordinates
from velocity_modelling.constants import get_data_root
from velocity_modelling.registry import CVMRegistry

# # GENERAL_GRID_FFP = Path("/home/claudy/dev/tmp_share/virt_sites/general_land_mask_grid.nc")
# GENERAL_GRID_FFP = Path(
#     "/Users/claudy/dev/tmp_share/virt_sites/general_grid.nc"
# )

GRID_DATA = pooch.create(
    path=pooch.os_cache("virt_sites"),
    base_url="",
    registry={
        "general_grid.nc": "sha256:06d675c60d90545f29573ab505db17e7662f7958270093d4a8808d6fc379a984",
        "nz_coastline.parquet": "sha256:9a965326690c560d372720b4b64b57480ede10e479b2fe62ba75760707bc521c",
    },
    urls={
        "general_grid.nc": "https://www.dropbox.com/scl/fi/4x987t01kvbz8bbyeid81/general_grid.nc?rlkey=rpztofwihh839500jn2bkw255&st=rpazzciz&dl=1",
        "nz_coastline.parquet": "https://www.dropbox.com/scl/fi/nx7hm82ern7s2b4v7u6zq/nz_coastline.parquet?rlkey=xzvh7gumefaigzhbgyjtuxu2p&st=e4gwa9a9&dl=1",
    },
)


class GeneralGrid:

    def __init__(self, land_mask_grid: xr.DataArray, spacing: int):
        self.land_mask_grid = land_mask_grid
        self.site_ids = np.arange(land_mask_grid.size).reshape(land_mask_grid.shape)
        self.spacing = spacing

        self.grid_lat, self.grid_lon = xr.broadcast(
            self.land_mask_grid.lat, self.land_mask_grid.lon
        )
        # TODO: Optimize memory usage, high atm
        self.grid_lat, self.grid_lon = (
            self.grid_lat.values,
            self.grid_lon.values,
        )

        grid_nztm = coordinates.wgs_depth_to_nztm(
            np.vstack((self.grid_lat.ravel(), self.grid_lon.ravel())).T
        )
        self.grid_nztm_x = grid_nztm[:, 1].reshape(self.land_mask_grid.shape)
        self.grid_nztm_y = grid_nztm[:, 0].reshape(self.land_mask_grid.shape)

    @property
    def shape(self):
        return self.land_mask_grid.shape

    @classmethod
    def load(cls, land_mask_grid_ffp: Path):
        return cls(xr.load_dataarray(land_mask_grid_ffp), 50)


class CustomGrid:

    def __init__(self):
        self.general_grid = GeneralGrid.load(GRID_DATA.fetch("general_grid.nc"))

        self._and_mask = np.ones(self.general_grid.shape, dtype=bool)
        self._or_mask = np.ones(self.general_grid.shape, dtype=bool)

        self._in_basin_mask = None

    @property
    def mask(self):
        return self._and_mask & self._or_mask

    def add_region_filter(self, region: shapely.Polygon):
        # Convert to NZTM
        region = shapely.transform(region, lambda x: coordinates.wgs_depth_to_nztm(x))

        region_mask = shapely.contains_xy(
            region,
            self.general_grid.grid_nztm_y.ravel(),
            self.general_grid.grid_nztm_x.ravel(),
        ).reshape(self.general_grid.shape)
        self._and_mask &= region_mask

        return self

    def add_uniform_spacing_filter(self, spacing: int):
        if spacing < self.general_grid.spacing:
            raise ValueError(
                "Uniform spacing must be greater than or equal to the general grid spacing."
            )
        if spacing % self.general_grid.spacing != 0:
            raise ValueError(
                f"Uniform spacing must be a multiple of the general grid spacing {self.general_grid.spacing}."
            )

        idx_interval = spacing // self.general_grid.spacing
        spacing_mask = np.zeros(self.general_grid.shape, dtype=bool)
        spacing_mask[::idx_interval, ::idx_interval] = True
        self._or_mask &= spacing_mask

        return self

    def add_land_only_filter(self):
        self._and_mask &= self.general_grid.land_mask_grid.values.astype(bool)

        return self

    def add_basin_spacing_filter(self, vel_model_version: str, spacing: int):
        if spacing < self.general_grid.spacing:
            raise ValueError(
                "Basin spacing must be greater than or equal to the general grid spacing."
            )
        if spacing % self.general_grid.spacing != 0:
            raise ValueError(
                f"Basin spacing must be a multiple of the general grid spacing {self.general_grid.spacing}."
            )

        basin_boundaries = get_basin_boundaries(vel_model_version)
        comb_basin_boundaries = shapely.union_all(list(basin_boundaries.values()))
        comb_basin_boundaries = shapely.transform(
            comb_basin_boundaries, lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1])
        )
        self._in_basin_mask = shapely.contains_xy(
            comb_basin_boundaries,
            self.general_grid.grid_nztm_y.ravel(),
            self.general_grid.grid_nztm_x.ravel(),
        ).reshape(self.general_grid.shape)

        idx_interval = spacing // self.general_grid.spacing
        spacing_mask = np.zeros(self.general_grid.shape, dtype=bool)
        spacing_mask[::idx_interval, ::idx_interval] = True

        self._or_mask |= self._in_basin_mask & spacing_mask
        return self

    def get_site_df(self):
        site_ids = self.general_grid.site_ids[self.mask]
        lons = self.general_grid.grid_lon[self.mask]
        lats = self.general_grid.grid_lat[self.mask]

        site_df = pd.DataFrame(
            {"site_id": site_ids, "lon": lons, "lat": lats, "in_basin": False}
        ).set_index("site_id")

        if self._in_basin_mask is not None:
            site_df["in_basin"] = self._in_basin_mask[self.mask]

        return site_df


def get_basin_boundaries(vel_model_version: str):
    cvm_registry = CVMRegistry(vel_model_version, get_data_root())
    basin_data = cvm_registry.load_basin_data(cvm_registry.global_params["basins"])

    basin_boundaries = {}
    for cur_basin_data in basin_data:
        if len(cur_basin_data.boundaries) == 1:
            basin_boundaries[cur_basin_data.name] = shapely.Polygon(
                cur_basin_data.boundaries[0]
            )
        else:
            basin_boundaries[cur_basin_data.name] = shapely.union_all(
                [shapely.Polygon(b) for b in cur_basin_data.boundaries]
            )

    return basin_boundaries
