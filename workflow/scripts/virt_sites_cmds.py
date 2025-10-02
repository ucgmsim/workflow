from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pygmt
import shapely
import typer
import xarray as xr

from qcore import cli, coordinates
from workflow.realisations import DomainParameters, SourceConfig
from workflow.virt_sites_grid import GRID_DATA, CustomGrid, get_basin_boundaries

app = typer.Typer()


@app.command("gen-general-grid")
def gen_general_grid(grid_spacing: str, output_ffp: Path):
    """Generate the general grid."""
    x_spacing, y_spacing = grid_spacing.split("/")
    if x_spacing != y_spacing:
        raise ValueError("Currently only supports equal x and y spacing.")
    
    try:
        spacing = int(x_spacing.rstrip("e"))
    except ValueError:
        raise ValueError("Grid spacing must be an integer in metres.")
    
    land_df = gpd.read_parquet(
        GRID_DATA.fetch("nz_coastline.parquet")
    )
    # Combine into a single polygon
    land_polygon = shapely.coverage_union_all(land_df.geometry)
    land_polygon = shapely.transform(
        land_polygon, lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1])
    )

    # Generate grid
    print("Generating grid...")
    land_mask_grid = pygmt.grdlandmask(region="NZ", spacing=grid_spacing).astype(bool)
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
    print("Applying land mask...")
    mask = shapely.contains_xy(land_polygon, grid_nztm[:, 0], grid_nztm[:, 1]).reshape(
        land_mask_grid.shape
    )
    land_mask_grid.values[mask] = 1

    print(f"Saving to {output_ffp}...")
    land_mask_grid.to_netcdf(output_ffp)


@cli.from_docstring(app)
def gen_custom_grid_from_rel(
    rel_ffp: Annotated[Path, typer.Argument()],
    uniform_grid_spacing: Annotated[int, typer.Argument()],
    output_ffp: Annotated[Path, typer.Argument()],
    basin_spacing: Annotated[int | None, typer.Option()] = None,
    vel_model_version: Annotated[str | None, typer.Option()] = None,
) -> None:
    """
    Generate a custom grid from a realisation config.

    Parameters
    ----------
    rel_ffp : Path
        The path to the realisation config.
    uniform_grid_spacing : int
        The uniform grid spacing in metres. 
        This must be a multiple of the general grid spacing.
    output_ffp : Path
        The path to save the output parquet file.
    basin_spacing : int, optional
        The grid spacing in metres to use within basins.
        This must be a multiple of the general grid spacing.
        If not provided, no basin spacing will be applied.
    vel_model_version : str, optional
        The velocity model version to use for basin spacing.
        This must be provided if basin_spacing is specified.
    """
    domain_config = DomainParameters.read_from_realisation(rel_ffp)

    region = shapely.Polygon(domain_config.domain.corners)
    custom_grid = (
        CustomGrid()
        .add_land_only_filter()
        .add_region_filter(region)
        .add_uniform_spacing_filter(uniform_grid_spacing)
    )
    if basin_spacing is not None:
        if vel_model_version is None:
            raise ValueError(
                "vel_model_version must be provided if basin_spacing is provided."
            )
        custom_grid.add_basin_spacing_filter(vel_model_version, basin_spacing)

    site_df = custom_grid.get_site_df()
    site_df.to_parquet(output_ffp)


@cli.from_docstring(app)
def gen_plot(
    site_df_ffp: Annotated[Path, typer.Argument()],
    output_ffp: Annotated[Path, typer.Argument()],
    rel_ffp: Annotated[Path | None, typer.Option()] = None,
    vel_model_version: Annotated[str | None, typer.Option()] = None,
    nzgmdb_site_ffp: Annotated[Path | None, typer.Option()] = None,
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
    vel_model_version : str, optional
        The velocity model version to use for plotting basin boundaries.
        If provided, the basin boundaries will be plotted.
    nzgmdb_site_ffp : Path, optional
        Path to the NZGMDB site file (csv).
        If provided, the NZGMDB sites will be plotted.
    """
    site_df = pd.read_parquet(site_df_ffp)

    fig = go.Figure()

    fig.add_trace(
        go.Scattermap(
            lon=site_df["lon"],
            lat=site_df["lat"],
            mode="markers",
            marker=dict(color="blue", size=4, symbol="circle"),
            hoverinfo="skip",
        )
    )

    if vel_model_version is not None:
        # Plot the basin boundaries
        basin_boundaries = get_basin_boundaries(vel_model_version)
        basin_line_properties = dict(color="red", width=1)

        for cur_basin_boundary in basin_boundaries.values():
            if isinstance(cur_basin_boundary, shapely.MultiPolygon):
                for poly in cur_basin_boundary.geoms:
                    fig.add_trace(
                        go.Scattermap(
                            lon=np.array(poly.exterior.xy[0]),
                            lat=np.array(poly.exterior.xy[1]),
                            mode="lines",
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
                        line=basin_line_properties,
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

    if nzgmdb_site_ffp is not None:
        nzgmdb_site_df = pd.read_csv(
            nzgmdb_site_ffp, index_col="sta", usecols=["sta", "lon", "lat"]
        )
        fig.add_trace(
            go.Scattermap(
                lon=nzgmdb_site_df["lon"],
                lat=nzgmdb_site_df["lat"],
                mode="markers",
                marker=dict(symbol="triangle"),
                hovertext=nzgmdb_site_df.index,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        map=dict(zoom=6, center=dict(lat=site_df.lat.mean(), lon=site_df.lon.mean())),
        showlegend=False,
        title=(
            f"Virtual Sites Grid - Velocity Model {vel_model_version} - "
            f"Total Sites: {len(site_df)} - "
            f"Basin Sites: {site_df['in_basin'].sum()}"
        ),
    )
    fig.write_html(output_ffp)


if __name__ == "__main__":
    app()
    app()
