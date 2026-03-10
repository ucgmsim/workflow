"""Commands for generating simulation site grids."""
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import shapely
import typer
import yaml

from qcore import cli, coordinates
from workflow import site_gen
from workflow.realisations import DomainParameters, SourceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@cli.from_docstring(app)
def gen_general_grid(grid_spacing: int, output_ffp: Path) -> None:
    """
    Generate the general grid.

    Parameters
    ----------
    grid_spacing : int
        The grid spacing in metres. 
    output_ffp : Path
        The path to save the output netCDF file.
    """
    land_mask_grid = site_gen.gen_general_land_mask_grid(
        grid_spacing,
    )

    logger.info(f"Saving to {output_ffp}...")
    land_mask_grid.to_netcdf(output_ffp)


@cli.from_docstring(app)
def gen_custom_grid(
    output_ffp: Annotated[Path, typer.Argument()],
    config_ffp: Annotated[Path | None, typer.Option()] = None,
    uniform_grid_spacing: Annotated[int | None, typer.Option()] = None,
    basin_spacing: Annotated[int | None, typer.Option()] = None,
    vel_model_version: Annotated[str | None, typer.Option()] = None,
    nzgmdb_version: Annotated[site_gen.NZGMDBVersion | None, typer.Option()] = None,
    rel_ffp: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """
    Generate a NZ-wide custom grid.

    Parameters
    ----------
    output_ffp : Path
        The path to save the output parquet file.
    config_ffp : Path, optional
        The path to the custom grid config file.
        If provided, all other config options will be ignored.
    uniform_grid_spacing : int, optional
        The uniform grid spacing in metres.
        This must be a multiple of the general grid spacing.
        Ignored if a config file is provided.
    basin_spacing : int, optional
        The grid spacing in metres to use within basins.
        This must be a multiple of the general grid spacing.
        If not provided, no basin spacing will be applied.
        Ignored if a config file is provided.
    vel_model_version : str, optional
        The velocity model version to use for basin spacing.
        This must be provided if basin_spacing is specified.
        If basin_spacing is not provided, and this is provided,
        then this velocity model version will be used to set
        basin membership in the output site dataframe.
    nzgmdb_version : NZGMDBVersion, optional
        The NZGMDB version to use for site parameters.
        Ignored if a config file is provided.
    rel_ffp : Path, optional
        The path to the realisation config.
        If provided, the domain will be used to filter the grid.
        Ignored if a config file is provided.
    """
    if config_ffp is not None:
        logger.info(f"Reading custom grid config from {config_ffp}...")
        config_dict = yaml.safe_load(config_ffp.open("r"))
        grid_config = site_gen.CustomGridConfig.from_config(config_dict)
        rel_ffp = config_dict.get("rel_ffp")
    else:
        logger.info("Generating custom grid config from command line options...")
        grid_config = site_gen.CustomGridConfig(
            land_only=True,
            uniform_spacing=uniform_grid_spacing,
            vel_model_version=vel_model_version,
            basin_spacing=basin_spacing,
            nzgmdb_version=nzgmdb_version,
        )

    # If a realisation file is provided, use its domain to filter the grid
    if rel_ffp is not None:
        domain_config = DomainParameters.read_from_realisation(rel_ffp)
        region = shapely.Polygon(domain_config.domain.corners[:, ::-1])
        grid_config.region = region

    custom_grid = site_gen.CustomGrid().apply_config(grid_config)

    # Generate site dataframe and save
    site_df = custom_grid.get_site_df()
    site_df.to_parquet(output_ffp)

    # Save metadata
    metadata = custom_grid.get_metadata(site_df)
    with (output_ffp.parent / f"{output_ffp.stem}_metadata.yaml").open("w") as meta_ffp:
        yaml.dump(metadata, meta_ffp)


@cli.from_docstring(app)
def gen_plot(
    site_df_ffp: Annotated[Path, typer.Argument()],
    output_ffp: Annotated[Path, typer.Argument()],
    rel_ffp: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """
    Generate a plot of the custom grid.

    Parameters
    ----------
    site_df_ffp : Path
        The path to the custom grid site dataframe file (parquet).
    output_ffp : Path
        The path to save the output HTML file.
    rel_ffp : Path, optional
        The path to the realisation config.
        If provided, the domain and source will be plotted.
    """
    site_df = pd.read_parquet(site_df_ffp)

    site_df_meta_ffp = site_df_ffp.parent / f"{site_df_ffp.stem}_metadata.yaml"
    with site_df_meta_ffp.open("r") as f:
        config = site_gen.CustomGridConfig.from_config(
            yaml.load(f, Loader=yaml.FullLoader)["config"]
        )
    
    # Import the following within the function so that it is only required if used
    import plotly.graph_objects as go
    fig = go.Figure()

    virt_sites_mask = site_df.loc[:, "source"] == "virtual"
    fig.add_trace(
        go.Scattermap(
            lon=site_df.loc[virt_sites_mask, "lon"],
            lat=site_df.loc[virt_sites_mask, "lat"],
            mode="markers",
            marker=dict(color="blue", size=4, symbol="circle"),
            hoverinfo="skip",
            hovertemplate=(
                "Site ID: %{customdata[0]}<br>"
                "Lat: %{lat:.6f}<br>"
                "Lon: %{lon:.6f}<br>"
                "Basin: %{customdata[4]}<br>"
                "Z1.0: %{customdata[1]:.3f} km<br>"
                "Z2.5: %{customdata[2]:.3f} km<br>"
                "Vs30: %{customdata[3]:.1f} m/s<br>"
                "<extra></extra>"
            ),
            customdata=np.asarray(
                [
                    site_df.loc[virt_sites_mask].index.values.astype(str),
                    site_df.loc[virt_sites_mask, "Z1.0"].values,
                    site_df.loc[virt_sites_mask, "Z2.5"].values,
                    site_df.loc[virt_sites_mask, "Vs30"].values,
                    site_df.loc[virt_sites_mask, "basin"].values,
                ]
            ).T,
        )
    )

    if "real" in site_df.source.unique():
        real_sites_mask = site_df.loc[:, "source"] == "real"
        fig.add_trace(
            go.Scattermap(
                lon=site_df.loc[real_sites_mask, "lon"],
                lat=site_df.loc[real_sites_mask, "lat"],
                mode="markers",
                marker=dict(color="darkgreen", size=6, symbol="circle"),
                hovertemplate=(
                    "Site ID: %{customdata[0]}<br>"
                    "Lat: %{lat:.6f}<br>"
                    "Lon: %{lon:.6f}<br>"
                    "Basin: %{customdata[4]}<br>"
                    "Z1.0: %{customdata[1]:.3f} km<br>"
                    "Z2.5: %{customdata[2]:.3f} km<br>"
                    "Vs30: %{customdata[3]:.1f} m/s<br>"
                    "<extra></extra>"
                ),
                customdata=np.asarray(
                    [
                        site_df.loc[real_sites_mask].index.values.astype(str),
                        site_df.loc[real_sites_mask, "Z1.0"].values,
                        site_df.loc[real_sites_mask, "Z2.5"].values,
                        site_df.loc[real_sites_mask, "Vs30"].values,
                        site_df.loc[real_sites_mask, "basin"].values,
                    ]
                ).T,
            )
        )

    if config.vel_model_version is not None:
        # Plot the basin boundaries
        basin_boundaries = site_gen.get_basin_boundaries(config.vel_model_version)
        basin_line_properties = dict(color="red", width=1)
        basin_fill_color = "rgba(255,0,0,0.05)"

        for cur_basin_boundary in basin_boundaries.values():
            if isinstance(cur_basin_boundary, shapely.MultiPolygon):
                for poly in cur_basin_boundary.geoms:
                    fig.add_trace(
                        go.Scattermap(
                            lon=np.array(poly.exterior.xy[0]),
                            lat=np.array(poly.exterior.xy[1]),
                            mode="lines",
                            fill="toself",
                            fillcolor=basin_fill_color,
                            line=basin_line_properties,
                            hoverinfo="skip",
                        )
                    )
            else:
                fig.add_trace(
                    go.Scattermap(
                        lon=np.array(cur_basin_boundary.exterior.xy[0]),
                        lat=np.array(cur_basin_boundary.exterior.xy[1]),
                        mode="lines",
                        fill="toself",
                        fillcolor=basin_fill_color,
                        line=basin_line_properties,
                        hoverinfo="skip",
                    )
                )

    if config.per_region_spacing is not None:
        # Plot the regions with different spacing
        for region_spacing in config.per_region_spacing:
            fig.add_trace(
                go.Scattermap(
                    lon=np.array(region_spacing.region.exterior.xy[0]),
                    lat=np.array(region_spacing.region.exterior.xy[1]),
                    mode="lines",
                    line=dict(color="orange", width=1),
                    hoverinfo="skip",
                )
            )

    if rel_ffp is not None:
        # Plot the domain
        domain_corners = DomainParameters.read_from_realisation(rel_ffp).domain.corners
        domain_corners = shapely.Polygon(domain_corners[:, ::-1])
        fig.add_trace(
            go.Scattermap(
                lon=np.array(domain_corners.exterior.xy[0]),
                lat=np.array(domain_corners.exterior.xy[1]),
                mode="lines",
                line=dict(color="black"),
                hoverinfo="skip",
            )
        )

        # Plot the source
        source_config = SourceConfig.read_from_realisation(rel_ffp)
        for _, fault in source_config.source_geometries.items():
            for cur_plane in fault.planes:
                cur_polygon = shapely.transform(
                    cur_plane.geometry,
                    lambda x: coordinates.nztm_to_wgs_depth(x)[:, ::-1],
                )
                fig.add_trace(
                    go.Scattermap(
                        lon=np.array(cur_polygon.exterior.xy[0]),
                        lat=np.array(cur_polygon.exterior.xy[1]),
                        mode="lines",
                        line=dict(color="black"),
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        map=dict(zoom=6, center=dict(lat=site_df.lat.mean(), lon=site_df.lon.mean())),
        showlegend=False,
        title=(
            f"Sites Grid - Velocity Model {config.vel_model_version} - "
            f"Total Sites: {len(site_df)}"
        ),
    )
    fig.write_html(output_ffp)


if __name__ == "__main__":
    app()
