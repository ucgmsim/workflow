"""Module for generating simulation site grids."""

import enum
import json
import logging
import string
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import pygmt
import shapely
import xarray as xr
from shapely import geometry

from qcore import coordinates
from velocity_modelling.constants import get_data_root
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.threshold import compute_station_thresholds

logger = logging.getLogger(__name__)

NZTM_CRS = "EPSG:2193"
WGS84_CRS = "EPSG:4326"


class NZGMDBVersion(enum.StrEnum):
    """NZGMDB Version Enum."""

    V4p3 = "v4.3"


NZGMDB_VERSION_TO_TABLE_NAME = {
    NZGMDBVersion.V4p3: "nzgmdb_4p3_site_table.csv",
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

    Parameters
    ----------
    land_mask_grid : xr.DataArray
        The land mask grid as an xarray DataArray.
    spacing : int
        The grid spacing in metres.
    """

    def __init__(self, land_mask_grid: xr.DataArray, spacing: int) -> None:
        """
        Initialize GeneralGrid.

        Parameters
        ----------
        land_mask_grid : xr.DataArray
            The land mask grid as an xarray DataArray.
        spacing : int
            The grid spacing in metres.
        """
        self.land_mask_grid = land_mask_grid
        self.site_ids = np.arange(land_mask_grid.size, dtype=np.uint32).reshape(
            land_mask_grid.shape
        )
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
        """
        Shape of the grid.

        Returns
        -------
        tuple[int, int]
            The shape of the grid.
        """
        return self.land_mask_grid.shape  # type: ignore

    @classmethod
    def load(cls, land_mask_grid_ffp: Path) -> "GeneralGrid":
        """
        Load GeneralGrid from file.

        Parameters
        ----------
        land_mask_grid_ffp : Path
            The file path to the general grid file.

        Returns
        -------
        GeneralGrid
            The loaded GeneralGrid instance.
        """
        logger.info(f"Loading general grid from {land_mask_grid_ffp}...")
        da = xr.load_dataarray(land_mask_grid_ffp, engine="h5netcdf")
        return cls(da, int(da.attrs["spacing"]))


def gen_general_land_mask_grid(spacing: int) -> xr.DataArray:
    """
    Generates the general land mask grid with given spacing.

    Parameters
    ----------
    spacing : int
        Grid spacing in metres

    Returns
    -------
    xr.DataArray
        The land mask grid DataArray, with values of 1 for land
        and 0 for ocean.

    Notes
    -----
    Common grid spacing formats:
        - To specify grid spacing of `x` units: `"{x}{unit}/{x}{unit}"`,
        where `unit` can be metres (`e`) or kilometres (`k`).
    """
    land_df = gpd.read_parquet(GRID_DATA.fetch("nz_coastline.parquet"))
    # Combine into a single polygon
    land_polygon = shapely.coverage_union_all(land_df.geometry)
    land_polygon = shapely.transform(
        land_polygon, lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1])
    )

    # Generate grid
    logger.info("Generating grid...")
    land_mask_grid = pygmt.grdlandmask(
        region="NZ", spacing=f"{spacing}e/{spacing}e"
    ).astype(bool)
    land_mask_grid[:] = False
    land_mask_grid.attrs = {"spacing": spacing}

    # Use float32 for coords
    land_mask_grid = land_mask_grid.assign_coords(
        lat=land_mask_grid.lat.astype(np.float32),
        lon=land_mask_grid.lon.astype(np.float32),
    )

    grid_lat, grid_lon = xr.broadcast(land_mask_grid.lat, land_mask_grid.lon)
    grid_nztm = coordinates.wgs_depth_to_nztm(
        np.vstack((grid_lat.values.ravel(), grid_lon.values.ravel())).T
    )

    # Apply land masking
    logger.info("Applying land mask...")
    mask = shapely.contains_xy(land_polygon, grid_nztm[:, 0], grid_nztm[:, 1]).reshape(
        land_mask_grid.shape
    )
    land_mask_grid.values[mask] = 1

    return land_mask_grid


@dataclass
class RegionSpacingConfig:
    """Configuration for region specific spacing."""

    name: str
    """Name of the region."""

    region: shapely.Polygon
    """Region polygon in WGS84 coordinates (lon/lat)."""

    spacing: int
    """Grid spacing in metres within the region."""

    def as_dict(self) -> dict:
        """
        Convert RegionSpacingConfig to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return {
            "name": self.name,
            "region": geometry.mapping(self.region),
            "spacing": self.spacing,
        }

    @classmethod
    def from_config(cls, config_dict: dict) -> "RegionSpacingConfig":
        """
        Create RegionSpacingConfig from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with keys 'name', 'spacing',
            and either 'region' or 'geojson_ffp'.

        Returns
        -------
        RegionSpacingConfig
            The created RegionSpacingConfig instance.
        """
        if "geojson_ffp" not in config_dict and "region" not in config_dict:
            raise ValueError(
                "Either 'geojson_ffp' or 'region' must be provided in the config dictionary."
            )

        if "geojson_ffp" in config_dict:
            with open(config_dict["geojson_ffp"], "r") as f:
                geojson_obj = json.load(f)
                region_polygon = geometry.shape(geojson_obj["features"][0]["geometry"])
        else:
            region_polygon = geometry.shape(config_dict["region"])

        return RegionSpacingConfig(
            name=config_dict["name"],
            region=region_polygon,
            spacing=config_dict["spacing"],
        )


@dataclass
class CustomGridConfig:
    """Configuration for CustomGrid."""

    land_only: bool | None = None
    """Whether to only include land sites."""

    region: shapely.Polygon | None = None
    """Region polygon in WGS84 coordinates (lon/lat)."""

    uniform_spacing: int | None = None
    """Uniform grid spacing in metres."""

    vel_model_version: str | None = None
    """Velocity model version for basin spacing."""

    basin_spacing: int | None = None
    """Grid spacing in metres within basins."""

    per_basin_spacing: dict[str, int] | None = None
    """Per-basin specific spacing in metres."""

    per_region_spacing: list[RegionSpacingConfig] | None = None
    """Per-region specific spacing in metres."""

    nzgmdb_version: NZGMDBVersion | None = None
    """NZGMDB version for real stations."""

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        if (
            self.basin_spacing is not None or self.per_basin_spacing is not None
        ) and self.vel_model_version is None:
            raise ValueError(
                "vel_model_version must be provided if basin_spacing is set."
            )

    @classmethod
    def from_config(cls, config_dict: dict) -> "CustomGridConfig":
        """
        Create CustomGridConfig from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary containing CustomGridConfig settings.

        Returns
        -------
        CustomGridConfig
            The created CustomGridConfig instance.
        """
        # Get the region polygon
        region = config_dict.get("region")
        if region is not None and isinstance(region, dict):
            region = geometry.shape(region)

        # Get the per-region spacing
        per_region_spacing = None
        if (prs_configs := config_dict.get("per_region_spacing")) is not None:
            per_region_spacing = [
                RegionSpacingConfig.from_config(config) for config in prs_configs
            ]

        # Get the NZGMDB version
        nzgmdb_version = config_dict.get("nzgmdb_version")
        if nzgmdb_version is not None:
            nzgmdb_version = NZGMDBVersion(nzgmdb_version)

        return cls(
            land_only=config_dict.get("land_only"),
            region=region,
            uniform_spacing=config_dict.get("uniform_spacing"),
            vel_model_version=config_dict.get("vel_model_version"),
            basin_spacing=config_dict.get("basin_spacing"),
            per_basin_spacing=config_dict.get("per_basin_spacing"),
            per_region_spacing=per_region_spacing,
            nzgmdb_version=nzgmdb_version,
        )

    def as_dict(self) -> dict:
        """
        Convert CustomGridConfig to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        return {
            "land_only": self.land_only,
            "region": (
                geometry.mapping(self.region) if self.region is not None else None
            ),
            "uniform_spacing": self.uniform_spacing,
            "vel_model_version": self.vel_model_version,
            "basin_spacing": self.basin_spacing,
            "per_basin_spacing": self.per_basin_spacing,
            "per_region_spacing": (
                [prs.as_dict() for prs in self.per_region_spacing]
                if self.per_region_spacing is not None
                else None
            ),
            "nzgmdb_version": (
                self.nzgmdb_version.value if self.nzgmdb_version is not None else None
            ),
        }


class CustomGrid:
    """
    Represents a custom grid of virtual sites
    built from the general grid.

    Parameters
    ----------
    general_grid : GeneralGrid, optional
        The general grid to use. If None, it will be loaded from the
        default location.
    """

    def __init__(self, general_grid: GeneralGrid | None = None) -> None:
        """
        Initialize CustomGrid.

        Parameters
        ----------
        general_grid : GeneralGrid, optional
            The general grid to use. If None, it will be loaded from the
            default location.
        """
        if general_grid is None:
            logger.info("Loading general grid...")
            self.general_grid = GeneralGrid.load(GRID_DATA.fetch("general_grid.nc"))
        else:
            self.general_grid = general_grid

        self._reset()

    def _reset(self) -> None:
        """
        Resets the custom grid.
        """
        self._and_mask = np.ones(self.general_grid.shape, dtype=bool)
        self._or_mask = np.zeros(self.general_grid.shape, dtype=bool)
        self.config = None

    @property
    def mask(self) -> np.ndarray:
        """
        Site mask that gives the custom grid
        when applied to the the general grid.

        Returns
        -------
        np.ndarray
            The custom grid mask.
        """
        return self._and_mask & self._or_mask

    def apply_config(self, config: CustomGridConfig) -> "CustomGrid":
        """
        Applies a CustomGridConfig to the CustomGrid.
        Resets any previously applied filters.

        Parameters
        ----------
        config : CustomGridConfig
            The configuration to apply.

        Returns
        -------
        CustomGrid
            Returns self for method chaining.
        """
        self._reset()
        self.config = config

        # Land filter
        if self.config.land_only:
            self._add_land_only_filter()
        # Region filter
        if self.config.region is not None:
            self._add_region_filter(self.config.region)
        # Uniform spacing
        if self.config.uniform_spacing is not None:
            self._add_uniform_spacing_filter(self.config.uniform_spacing)
        # Basin spacing
        if self.config.vel_model_version is not None:
            if self.config.basin_spacing is not None:
                self._add_basin_spacing_filter(
                    self.config.vel_model_version, self.config.basin_spacing
                )
            if self.config.per_basin_spacing is not None:
                # Group by spacing
                basin_spacing_series = pd.Series(self.config.per_basin_spacing)
                spacing_groups = basin_spacing_series.groupby(basin_spacing_series)

                for spacing, basins in spacing_groups.groups.items():
                    self._add_basin_spacing_filter(
                        self.config.vel_model_version,
                        spacing,  # type: ignore
                        basins.tolist(),
                    )
        # Per-region spacing
        if self.config.per_region_spacing is not None:
            for region_spacing_config in self.config.per_region_spacing:
                self._add_region_spacing_filter(region_spacing_config)

        return self

    def _add_region_filter(self, region: shapely.Polygon) -> None:
        """
        Adds a region filter. Only sites within the region are kept.
        This is an AND filter, i.e., it removes sites outside the region.
        Can only be applied once.

        Parameters
        ----------
        region : shapely.Polygon
            The region polygon in WGS84 coordinates (lon/lat).
        """
        logger.info("Adding region filter to custom grid...")
        start_time = time.time()

        # Convert to NZTM
        region_nztm = shapely.transform(
            region, lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1])
        )

        region_mask = shapely.contains_xy(
            region_nztm,
            self.general_grid.grid_nztm_y.ravel(),
            self.general_grid.grid_nztm_x.ravel(),
        ).reshape(self.general_grid.shape)
        self._and_mask &= region_mask
        logger.info(f"Region filter added in {time.time() - start_time} seconds.")

    def _add_uniform_spacing_filter(self, spacing: int) -> None:
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
        start_time = time.time()
        if spacing < self.general_grid.spacing:
            error_msg = (
                "Uniform spacing must be greater than or"
                " equal to the general grid spacing."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if spacing % self.general_grid.spacing != 0:
            error_msg = (
                f"Uniform spacing must be a multiple of"
                f" the general grid spacing {self.general_grid.spacing}."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        idx_interval = spacing // self.general_grid.spacing
        spacing_mask = np.zeros(self.general_grid.shape, dtype=bool)
        spacing_mask[::idx_interval, ::idx_interval] = True
        self._or_mask |= spacing_mask

        logger.info(
            f"Uniform spacing filter added in {time.time() - start_time} seconds."
        )

    def _add_land_only_filter(self) -> None:
        """Adds a land only filter, only land sites are kept."""
        logger.info("Adding land only filter...")
        start_time = time.time()
        self._and_mask &= self.general_grid.land_mask_grid.values.astype(bool)
        logger.info(f"Land only filter added in {time.time() - start_time} seconds.")

    def _add_region_spacing_filter(
        self, region_spacing_config: RegionSpacingConfig
    ) -> None:
        """
        Adds a region specific spacing filter.
        Note that this is an OR filter, i.e., it only adds sites.

        Parameters
        ----------
        region_spacing_config : RegionSpacingConfig
            The region spacing configuration.
        """
        logger.info(
            f"Adding region {region_spacing_config.spacing} spacing filter for {region_spacing_config.name}..."
        )
        start_time = time.time()
        if region_spacing_config.spacing < self.general_grid.spacing:
            raise ValueError(
                "Region spacing must be greater than or equal to the general grid spacing."
            )
        if region_spacing_config.spacing % self.general_grid.spacing != 0:
            raise ValueError(
                f"Region spacing must be a multiple of the general grid spacing {self.general_grid.spacing}."
            )

        region_nztm = shapely.transform(
            region_spacing_config.region,
            lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1]),
        )
        in_region_mask = shapely.contains_xy(
            region_nztm,
            self.general_grid.grid_nztm_y.ravel(),
            self.general_grid.grid_nztm_x.ravel(),
        ).reshape(self.general_grid.shape)

        idx_interval = region_spacing_config.spacing // self.general_grid.spacing
        spacing_mask = np.zeros(self.general_grid.shape, dtype=bool)
        spacing_mask[::idx_interval, ::idx_interval] = True

        self._or_mask |= in_region_mask & spacing_mask
        logger.info(
            f"Region spacing filter added in {time.time() - start_time} seconds."
        )

    def _add_basin_spacing_filter(
        self, vel_model_version: str, spacing: int, basins: list[str] | None = None
    ) -> None:
        """
        Adds a basin spacing filter.
        Note that this is an OR filter, i.e., it only adds sites.

        Parameters
        ----------
        vel_model_version : str
            The velocity model version to use for basin spacing.
        spacing : int
            The grid spacing in metres to use within basins.
        basins : list[str] | None, optional
            The specific basins to apply the spacing to. If None, applies to all basins.
        """
        logger.info(
            f"Adding basin {spacing} spacing filter for {basins if basins is not None else 'all'} basins..."
        )
        start_time = time.time()
        if spacing < self.general_grid.spacing:
            raise ValueError(
                "Basin spacing must be greater than or equal to the general grid spacing."
            )
        if spacing % self.general_grid.spacing != 0:
            raise ValueError(
                f"Basin spacing must be a multiple of the general grid spacing {self.general_grid.spacing}."
            )

        basin_boundaries = get_basin_boundaries(vel_model_version)
        if basins is not None:
            basin_boundaries = {
                k: v for k, v in basin_boundaries.items() if k in basins
            }
        comb_basin_boundaries = shapely.union_all(list(basin_boundaries.values()))
        comb_basin_boundaries = shapely.transform(
            comb_basin_boundaries, lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1])
        )
        in_basin_mask = shapely.contains_xy(
            comb_basin_boundaries,
            self.general_grid.grid_nztm_y.ravel(),
            self.general_grid.grid_nztm_x.ravel(),
        ).reshape(self.general_grid.shape)

        idx_interval = spacing // self.general_grid.spacing
        spacing_mask = np.zeros(self.general_grid.shape, dtype=bool)
        spacing_mask[::idx_interval, ::idx_interval] = True

        self._or_mask |= in_basin_mask & spacing_mask
        logger.info(
            f"Basin spacing filter added in {time.time() - start_time} seconds."
        )

    def get_metadata(self, site_df: pd.DataFrame | None = None) -> dict:
        """
        Gets the metadata dictionary for the custom grid.

        Parameters
        ----------
        site_df : pd.DataFrame, optional
            The site dataframe to generate metadata from.

        Returns
        -------
        dict
            Dictionary containing metadata and configuration information.
        """
        if self.config is None:
            raise ValueError(
                "CustomGridConfig must be applied before getting metadata."
            )

        site_metadata = {}
        if site_df is not None:
            site_metadata = {
                "num_sites": len(site_df),
                "num_virtual_sites": len(site_df[site_df.source == "virtual"]),
                "num_real_sites": len(site_df[site_df.source == "real"]),
            }

            # Number of sites per basin
            if "basin" in site_df.columns:
                site_metadata["num_basin_sites"] = len(site_df[site_df.basin.notna()])
                site_metadata["sites_per_basin"] = (
                    site_df.groupby("basin").size().to_dict()
                )

        metadata = {"site_metadata": site_metadata, "config": self.config.as_dict()}
        return metadata

    def get_site_df(
        self,
    ) -> pd.DataFrame:
        """
        Gets the site dataframe for the custom grid.

        Returns
        -------
        pd.DataFrame
            The site dataframe with columns:
            - general_site_id: The site ID from the general grid.
            - lon: The site longitude.
            - lat: The site latitude.
            - nztm_x: The site NZTM X coordinate.
            - nztm_y: The site NZTM Y coordinate.
            - source: "virtual" for virtual sites, "real" for NZGMDB sites.
            - region_name: The name of the region the site is in.
            - region_code: The code of the region the site is in.
            - territory_name: The name of the territory the site is in.
            - basin: The basin the site is in (if any).
            - Z1.0: The Z1.0 value for the site (if vel_model_version is provided).
            - Z2.5: The Z2.5 value for the site (if vel_model_version is provided).
            - Vs30: Time-averaged shear-wave velocity in the top 30 meters (m/s) (for real sites only).
            - site_code: The site code.

        Notes
        -----
        Basin membership and Z-values are only added if vel_model_version
        is set in the configuration.
        """
        if self.config is None:
            raise ValueError("CustomGridConfig must be applied before getting site df.")

        if self.mask.sum() == 0:
            logger.warning("Custom grid has no sites selected.")
            return pd.DataFrame()

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

        # Add NZGMDB sites
        if self.config.nzgmdb_version is not None:
            start = time.time()
            logger.info("Adding NZGMDB sites...")
            nzgmdb_site_df = pd.read_csv(
                GRID_DATA.fetch(
                    NZGMDB_VERSION_TO_TABLE_NAME[self.config.nzgmdb_version]
                ),
                index_col="sta",
                usecols=["sta", "lat", "lon", "Vs30", "Z1.0", "Z2.5"],
            )
            nzgmdb_site_df["source"] = "real"
            nzgmdb_nztm_values = coordinates.wgs_depth_to_nztm(
                nzgmdb_site_df[["lat", "lon"]].values
            )
            nzgmdb_site_df["nztm_x"] = nzgmdb_nztm_values[:, 1]
            nzgmdb_site_df["nztm_y"] = nzgmdb_nztm_values[:, 0]
            # Convert Z1.0 to km
            nzgmdb_site_df["Z1.0"] /= 1000

            if self.config.region is not None:
                region_nztm = shapely.transform(
                    self.config.region,
                    lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1]),
                )
                region_mask = shapely.contains_xy(
                    region_nztm,
                    nzgmdb_site_df["nztm_y"].values,
                    nzgmdb_site_df["nztm_x"].values,
                )
                nzgmdb_site_df = nzgmdb_site_df[region_mask]

            site_df = pd.concat([site_df, nzgmdb_site_df], axis=0)
            logger.info(f"Took: {time.time() - start} to add NZGMDB sites")

        # Add region
        logger.info("Adding region information...")
        start = time.time()
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

        # Deal with sites that are not in a region
        joined["name"] = joined["name"].cat.add_categories("No Region")
        joined["name"] = joined["name"].fillna("No Region")
        joined["code"] = joined["code"].cat.add_categories("NR")
        joined["code"] = joined["code"].fillna("NR")

        # Keep first region if in multiple regions
        joined = joined.groupby(level=0).first()
        site_df["region_name"] = joined["name"]
        site_df["region_code"] = joined["code"]
        logger.info(f"Took: {time.time() - start} ")

        # Add territory
        logger.info("Adding territory information...")
        start = time.time()
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
        # Keep first territory if in multiple territories
        site_df["territory_name"] = site_df["territory_name"].groupby(level=0).first()
        logger.info(f"Took: {time.time() - start} ")

        # Add basin membership & Z-values
        if self.config.vel_model_version:
            start = time.time()
            # Basin membership
            logger.info("Adding basin membership information...")
            basin_boundaries = get_basin_boundaries(self.config.vel_model_version)
            basin_df = gpd.GeoDataFrame(
                {"basin": list(basin_boundaries.keys())},
                geometry=list(basin_boundaries.values()),
                crs=WGS84_CRS,
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
            logger.info(f"Took: {time.time() - start} to add basin membership")

            # Get Z-values
            start = time.time()
            logger.info("Adding Z-values...")
            logging.getLogger("nzcvm.threshold").setLevel(logging.WARNING)
            z_values = compute_station_thresholds(
                site_df,
                model_version=self.config.vel_model_version,
                show_progress=False,
                include_sigma=False,
                logger=logger,
            )
            site_df["Z1.0"] = z_values["Z1.0(km)"]
            site_df["Z2.5"] = z_values["Z2.5(km)"]
            logger.info(f"Took: {time.time() - start} to add Z-values")

        # Add site ids
        logger.info("Adding site code...")
        virt_mask = site_df.source == "virtual"
        site_df.loc[virt_mask, "site_code"] = np.char.add(
            encode_base62_fixed_array(
                site_df.loc[virt_mask].index.values.astype(int), length=5
            ),
            site_df.loc[virt_mask].region_code.values.astype(str),
        )
        site_df.loc[~virt_mask, "site_code"] = site_df.loc[~virt_mask].index.values
        site_df = site_df.set_index("site_code")
        site_df = site_df.astype({"source": "category"})

        # Sanity check
        assert site_df.index.is_unique, "Site IDs are not unique!"
        return site_df


def get_basin_boundaries(vel_model_version: str) -> dict[str, shapely.Polygon]:
    """
    Gets the basin boundaries for a given velocity model version.

    Parameters
    ----------
    vel_model_version : str
        The velocity model version to load basin data for.

    Returns
    -------
    dict[str, shapely.Polygon]
        Dictionary mapping basin names to their boundary polygons.
    """
    cvm_registry = CVMRegistry(vel_model_version, get_data_root())
    basin_data = cvm_registry.load_basin_data(cvm_registry.global_params["basins"]) # type: ignore

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


def encode_base62_fixed_array(nums: np.ndarray, length: int) -> np.ndarray:
    """
    Vectorized Base62 encoder for many integers.

    Parameters
    ----------
    nums : np.ndarray
        Array of integers to encode.
    length : int
        The fixed length for each encoded string.

    Returns
    -------
    np.ndarray
        Array of Base62 encoded strings of fixed length.
    """
    alphabet = np.array(list(string.digits + string.ascii_letters))
    alphabet_size = len(alphabet)

    if np.any(nums >= alphabet_size**length):
        raise ValueError(f"Some numbers too large for {length}-char Base62 ID.")

    # Create empty char array: shape (N, MAX_LEN)
    out = np.empty((nums.size, length), dtype="<U1")

    n = nums.copy()
    for i in range(length - 1, -1, -1):
        n, rem = divmod(n, alphabet_size)
        out[:, i] = alphabet[rem]

    # Join per row to get array of strings
    return np.apply_along_axis("".join, 1, out)
