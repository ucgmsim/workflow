import time
import base64
import enum
import hashlib
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import shapely
import xarray as xr

from qcore import coordinates
from velocity_modelling.constants import get_data_root
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.threshold import compute_station_thresholds

logger = logging.getLogger(__name__)

NZTM_CRS = "EPSG:2193"
WSG84_CRS = "EPSG:4326"


class NZGMDBVersion(enum.StrEnum):
    """NZGMDB Version Enum."""

    V4p3 = "v4.3"


NZGMDB_VERSION_TO_TABLE_NAME = {
    NZGMDBVersion.V4p3: "nzgmdb_4p3_site_table.csv",
}

REGION_CODES = {
    "North Auckland": "NAK",
    "South Auckland": "SAK",
    "Hawkes Bay": "HBY",
    "Gisborne": "GIS",
    "Taranaki": "TAR",
    "Wellington": "WEL",
    "Nelson": "NEL",
    "Marlborough": "MAR",
    "Westland": "WES",
    "Canterbury": "CAN",
    "Otago": "OTA",
    "Southland": "STL",
}

GRID_DATA = pooch.create(
    path=pooch.os_cache("virt_sites"),
    base_url="",
    registry={
        "general_grid.nc": "sha256:06d675c60d90545f29573ab505db17e7662f7958270093d4a8808d6fc379a984",
        "nz_coastline.parquet": "sha256:9a965326690c560d372720b4b64b57480ede10e479b2fe62ba75760707bc521c",
        "nzgmdb_4p3_site_table.csv": "sha256:0144c6e066a461b63441e9c1a1608741c5f056fd3e2926f2cb16748e24296317",
        "nz_regions.parquet": "sha256:c08b250332052126d4e68ef0894d72416791c4184b5bfc23da93adf39582f2f2",
        "nz_territory.parquet": "sha256:afe5a70572d6d42cbcbe4b26291983b03619bd0b567eb6f6cafa20306d03b0f7",
    },
    urls={
        "general_grid.nc": "https://www.dropbox.com/scl/fi/4x987t01kvbz8bbyeid81/general_grid.nc?rlkey=rpztofwihh839500jn2bkw255&st=rpazzciz&dl=1",
        "nz_coastline.parquet": "https://www.dropbox.com/scl/fi/nx7hm82ern7s2b4v7u6zq/nz_coastline.parquet?rlkey=xzvh7gumefaigzhbgyjtuxu2p&st=e4gwa9a9&dl=1",
        "nzgmdb_4p3_site_table.csv": "https://www.dropbox.com/scl/fi/m3l559m2nmjgufipyq03s/nzgmdb_4p3_site_table.csv?rlkey=gg1hn2mm38zxew9fn45ac8p0v&st=iq0kagtv&dl=1",
        "nz_regions.parquet": "https://www.dropbox.com/scl/fi/gvvjf16vpe69zqbglh8ns/nz_regions.parquet?rlkey=14y2exb6wr48h7d0pj7dcvxwa&st=t23pxjsv&dl=1",
        "nz_territory.parquet": "https://www.dropbox.com/scl/fi/3ywi83r8pucp5ajs26do7/nz_territory.parquet?rlkey=viemibogajrtomm0n9xypwq1v&st=ph5pkg2z&dl=1",
    },
)


class GeneralGrid:
    """
    Represents the general base grid of virtual sites.
    Any custom grid is built off this grid.
    """

    def __init__(self, land_mask_grid: xr.DataArray, spacing: int) -> "GeneralGrid":
        """Initialize GeneralGrid."""
        self.land_mask_grid = land_mask_grid
        self.site_ids = np.arange(land_mask_grid.size).reshape(land_mask_grid.shape)
        self.spacing = spacing

        self.grid_lat, self.grid_lon = xr.broadcast(
            self.land_mask_grid.lat, self.land_mask_grid.lon
        )
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
    def shape(self) -> tuple[int, int]:
        """Shape of the grid."""
        return self.land_mask_grid.shape

    @classmethod
    def load(cls, land_mask_grid_ffp: Path) -> "GeneralGrid":
        """Load GeneralGrid from file."""
        logger.info(f"Loading general grid from {land_mask_grid_ffp}...")
        return cls(xr.load_dataarray(land_mask_grid_ffp, engine="h5netcdf"), 50)


class CustomGrid:
    """
    Represents a custom grid of virtual sites
    built from the general grid.
    """

    def __init__(self) -> "CustomGrid":
        """Initialize CustomGrid."""
        logger.info("Loading general grid...")
        self.general_grid = GeneralGrid.load(GRID_DATA.fetch("general_grid.nc"))

        self._and_mask = np.ones(self.general_grid.shape, dtype=bool)
        self._or_mask = np.ones(self.general_grid.shape, dtype=bool)

        self._land_only = None
        self._uniform_spacing = None
        self._region = None
        self._in_basin_mask = None
        self._vel_model_version = None
    
    @property
    def mask(self) -> np.ndarray:
        """
        Site mask that gives the custom grid
        when applied to the the general grid.
        """
        return self._and_mask & self._or_mask

    def add_region_filter(self, region: shapely.Polygon) -> "CustomGrid":
        """
        Adds a region filter. Only sites within the region are kept.
        This is an AND filter, i.e., it removes sites outside the region.
        Can only be applied once.

        Parameters
        ----------
        region : shapely.Polygon
            The region polygon in WGS84 coordinates.
        """
        logger.info("Adding region filter to custom grid...")
        if self._region is not None:
            logger.error("Region filter has already been added.")
            raise
        self._region = region

        # Convert to NZTM
        region = shapely.transform(region, lambda x: coordinates.wgs_depth_to_nztm(x))

        region_mask = shapely.contains_xy(
            region,
            self.general_grid.grid_nztm_y.ravel(),
            self.general_grid.grid_nztm_x.ravel(),
        ).reshape(self.general_grid.shape)
        self._and_mask &= region_mask

        return self

    def add_uniform_spacing_filter(self, spacing: int) -> "CustomGrid":
        """
        Adds a uniform spacing filter.
        This is an OR filter, i.e., it only adds sites.
        Can only be applied once.

        Parameters
        ----------
        spacing : int
            The uniform grid spacing in metres.
        """
        logger.info("Adding uniform spacing filter...")
        if self._uniform_spacing is not None:
            logger.error("Uniform spacing filter has already been added.")
            raise
        if spacing < self.general_grid.spacing:
            logger.error(
                "Uniform spacing must be greater than or"
                " equal to the general grid spacing."
            )
            raise
        if spacing % self.general_grid.spacing != 0:
            logger.error(
                f"Uniform spacing must be a multiple of"
                f" the general grid spacing {self.general_grid.spacing}."
            )
            raise
        self._uniform_spacing = spacing
    
        idx_interval = spacing // self.general_grid.spacing
        spacing_mask = np.zeros(self.general_grid.shape, dtype=bool)
        spacing_mask[::idx_interval, ::idx_interval] = True
        self._or_mask &= spacing_mask

        return self

    def add_land_only_filter(self) -> "CustomGrid":
        """Adds a land only filter, only land sites are kept."""
        logger.info("Adding land only filter...")
        self._land_only = True
        self._and_mask &= self.general_grid.land_mask_grid.values.astype(bool)

        return self

    def add_basin_spacing_filter(
        self, vel_model_version: str, spacing: int
    ) -> "CustomGrid":
        """
        Adds a basin spacing filter.
        Note that this is an OR filter, i.e., it only adds sites

        Parameters
        ----------
        vel_model_version : str
            The velocity model version to use for basin spacing.
        spacing : int
            The grid spacing in metres to use within basins.
        """
        logger.info("Adding basin spacing filter...")
        if (
            self._vel_model_version is not None
            and vel_model_version != self._vel_model_version
        ):
            raise ValueError(
                "Basin spacing filter has already been added with a different velocity model version."
            )
        self._vel_model_version = vel_model_version

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
    
    def get_metadata(self, site_df: pd.DataFrame = None) -> dict:
        """
        Gets the metadata dictionary for the custom grid.
        """
        region = None
        if self._region is not None:
            region = shapely.geometry.mapping(self._region)

        site_metadata = {}
        if site_df is not None:
            site_metadata = {
                "num_sites": len(site_df),
                "num_virtual_sites": len(site_df[site_df.source == "virtual"]),
                "num_real_sites": len(site_df[site_df.source == "real"]),
                "num_basin_sites": len(site_df[site_df.basin.notna()]),
            }

            # Number of sites per basin
            site_metadata["sites_per_basin"] = site_df.groupby("basin").size().to_dict()

        metadata = site_metadata | {
            "land_only": self._land_only,
            "region": region,
            "uniform_spacing": self._uniform_spacing,
            "vel_model_version": self._vel_model_version,
        }
        return metadata

    def get_site_df(
        self,
        nzgmdb_version: NZGMDBVersion | None = None,
        vel_model_version: str | None = None,
    ) -> pd.DataFrame:
        """
        Gets the site dataframe for the custom grid.

        Parameters
        ----------
        nzgmdb_version : NZGMDBVersion, optional
            The NZGMDB version to include real stations from.
        vel_model_version : str, optional
            The velocity model version to use for basin membership
            and Z-values. Is only used if the basin spacing filter
            has not already been added to the custom grid.

        Returns
        -------
        pd.DataFrame
            The site dataframe with columns:
            - site_id: The site ID.
            - lon: The site longitude.
            - lat: The site latitude.
            - nztm_x: The site NZTM X coordinate.
            - nztm_y: The site NZTM Y coordinate.
            - source: "virtual" for virtual sites, "real" for NZGMDB sites.
            - region_name: The name of the region the site is in.
            - region_code: The code of the region the site is in.
            - territory_name: The name of the territory the site is in.
        """
        logger.info("Getting site dataframe...")
        site_df = pd.DataFrame(
            {
                "general_site_id": self.general_grid.site_ids[self.mask],
                "lon": self.general_grid.grid_lon[self.mask],
                "lat": self.general_grid.grid_lat[self.mask],
                "nztm_x": self.general_grid.grid_nztm_x[self.mask],
                "nztm_y": self.general_grid.grid_nztm_y[self.mask],
                "source": "virtual",
            }
        ).set_index("general_site_id")

        # Add region
        logger.info("Adding region information...")
        region_df = (
            gpd.read_parquet(GRID_DATA.fetch("nz_regions.parquet"))
            .to_crs(NZTM_CRS)
            .astype({"name": "category", "code": "category"})
        )

        site_points_df = gpd.GeoDataFrame(
            site_df[["nztm_x", "nztm_y"]],
            geometry=gpd.points_from_xy(site_df.nztm_x, site_df.nztm_y),
            crs=NZTM_CRS,
        )
        joined = gpd.sjoin(site_points_df, region_df, how="left", predicate="within")
        site_df["region_name"] = joined["name"]
        site_df["region_code"] = joined["code"]

        # Add territory
        logger.info("Adding territory information...")
        territory_df = (
            gpd.read_parquet(GRID_DATA.fetch("nz_territory.parquet"))
            .to_crs(NZTM_CRS)
            .astype({"name": "category"})
        )
        joined = gpd.sjoin(
            site_points_df,
            territory_df,
            how="left",
            predicate="within",
        )
        site_df["territory_name"] = joined["name"]

        # Add basin membership & Z-values
        if self._vel_model_version or vel_model_version:
            if (
                self._vel_model_version
                and vel_model_version
                and self._vel_model_version != vel_model_version
            ):
                err_msg = "Basin spacing filter has been added with a different velocity model version."
                logger.error(err_msg)
                raise ValueError(err_msg)
            vel_model_version = self._vel_model_version or vel_model_version

            # Basin membership
            logger.info("Adding basin membership information...")
            basin_boundaries = get_basin_boundaries(vel_model_version)
            basin_df = gpd.GeoDataFrame(
                {"basin": list(basin_boundaries.keys())},
                geometry=list(basin_boundaries.values()),
                crs=WSG84_CRS,
            ).to_crs(NZTM_CRS)
            joined = gpd.sjoin(
                site_points_df,
                basin_df,
                how="left",
                predicate="within",
            )
            # Keep first basin if in multiple basins
            joined = joined.groupby(level=0).first()
            site_df["basin"] = joined["basin"]

            # Get Z-values
            logger.info("Adding Z-values...")
            start = time.time()
            z_values =compute_station_thresholds(site_df, model_version=vel_model_version, show_progress=False)
            logger.info(f"Took: {time.time() - start} to compute Z-values")
            site_df["Z1.0"] = z_values["Z_1.0(km)"]
            site_df["Z2.5"] = z_values["Z_2.5(km)"]

        # Add site ids
        logger.info("Adding site IDs...")
        site_df["site_id"] = np.char.add(
            latlon_ids(site_df.lat.values, site_df.lon.values, length=5),
            site_df.region_code.astype(str).values,
        )
        site_df = site_df.set_index("site_id")

        # Add NZGMDB sites
        if nzgmdb_version is not None:
            logger.info("Adding NZGMDB sites...")
            nzgmdb_site_df = pd.read_csv(
                GRID_DATA.fetch(NZGMDB_VERSION_TO_TABLE_NAME[nzgmdb_version]),
                index_col="sta",
                usecols=["sta", "lat", "lon"],
            )
            nzgmdb_site_df["source"] = "real"
            nzgmdb_nztm_values = coordinates.wgs_depth_to_nztm(
                nzgmdb_site_df[["lat", "lon"]].values
            )
            nzgmdb_site_df["nztm_x"] = nzgmdb_nztm_values[:, 1]
            nzgmdb_site_df["nztm_y"] = nzgmdb_nztm_values[:, 0]

            site_df = pd.concat([site_df, nzgmdb_site_df], axis=0)

        site_df = site_df.astype({"source": "category"})

        # Sanity check
        assert site_df.index.is_unique, "Site IDs are not unique!"
        return site_df


def get_basin_boundaries(vel_model_version: str) -> dict[str, shapely.Polygon]:
    """Gets the basin boundaries for a given velocity model version."""
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


def latlon_ids(lat: np.ndarray, lon: np.ndarray, length: int = 4) -> np.ndarray:
    """Generate a short unique ID from latitude and longitude arrays.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitude values.
    lon : np.ndarray
        Array of longitude values.
    length : int, optional
        Length of the hash ID, by default 4.

    Returns
    -------
    np.ndarray
        Array of hash IDs.
    """

    ids = np.empty(lat.shape, dtype=f"U{length}")
    for i, (lat_val, lon_val) in enumerate(zip(lat.ravel(), lon.ravel())):
        text = f"{lat_val:.6f},{lon_val:.6f}"
        h = hashlib.sha256(text.encode())
        ids[i] = base64.urlsafe_b64encode(h.digest()).decode()[:length]

    return ids
