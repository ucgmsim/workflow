import json
import string
import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely import geometry
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.spatial import distance as sdist

from qcore import coordinates
from workflow import site_gen


@pytest.fixture(scope="module", params=[50_000, 25_000])
def land_mask_grid_spacing_tuple(
    request: pytest.FixtureRequest,
) -> tuple[xr.DataArray, int]:
    spacing = request.param
    return site_gen.gen_general_land_mask_grid(spacing), spacing


@pytest.fixture
def general_grid(
    land_mask_grid_spacing_tuple: tuple[xr.DataArray, int],
) -> site_gen.GeneralGrid:
    return site_gen.GeneralGrid(
        land_mask_grid_spacing_tuple[0], land_mask_grid_spacing_tuple[1]
    )


def test_general_grid(land_mask_grid_spacing_tuple: tuple[xr.DataArray, int]) -> None:
    land_mask_grid, spacing = land_mask_grid_spacing_tuple
    with tempfile.TemporaryDirectory() as tmpdir:
        ffp = Path(f"{tmpdir}/general_grid.nc")
        land_mask_grid.to_netcdf(ffp, engine="h5netcdf")
        loaded_grid = site_gen.GeneralGrid.load(ffp)

        assert loaded_grid.land_mask_grid.spacing == spacing
        assert loaded_grid.shape == land_mask_grid.shape
        assert np.array_equal(loaded_grid.land_mask_grid.values, land_mask_grid.values)
        assert np.array_equal(
            loaded_grid.land_mask_grid.lat.values, land_mask_grid.lat.values
        )
        assert np.array_equal(
            loaded_grid.land_mask_grid.lon.values, land_mask_grid.lon.values
        )


@pytest.fixture(scope="module")
def chch_region_polygon() -> shapely.Polygon:
    polygon = shapely.Polygon(
        [
            (172.60691125905976, -43.50338730757493),
            (172.55414963918173, -43.52695730676956),
            (172.64273442256302, -43.583742177757706),
            (172.68415242515607, -43.5513507340181),
            (172.6440052010451, -43.526126938652155),
            (172.60691125905976, -43.50338730757493),
        ]
    )
    return polygon


@pytest.fixture(scope="module")
def land_region_polygon() -> shapely.Polygon:
    polygon = shapely.Polygon(
        [
            (170.78376767024793, -42.97481408776882),
            (170.78376767024793, -43.68210554829009),
            (172.66566136205932, -43.68210554829009),
            (172.66566136205932, -42.97481408776882),
            (170.78376767024793, -42.97481408776882),
        ]
    )
    return polygon


@pytest.fixture(scope="module")
def canterbury_region_polygon() -> shapely.Polygon:
    polygon = shapely.Polygon(
        [
            (172.86435627311448, -43.338682922734954),
            (172.2989207589734, -43.342255579287965),
            (171.53771742001976, -43.782575480660896),
            (171.7830393264249, -44.18964091597699),
            (173.3801704840081, -43.85958260231439),
            (173.1293617392618, -43.60655002794223),
            (172.86435627311448, -43.338682922734954),
        ]
    )
    return polygon


@pytest.fixture(scope="module")
def water_region_polygon() -> shapely.Polygon:
    polygon = shapely.Polygon(
        [
            (172.95789409381717, -39.97031515444409),
            (172.95789409381717, -40.67305495760141),
            (174.87451261095003, -40.67305495760141),
            (174.87451261095003, -39.97031515444409),
            (172.95789409381717, -39.97031515444409),
        ]
    )
    return polygon


@pytest.fixture(scope="module")
def central_south_island_region_polygon() -> shapely.Polygon:
    polygon = shapely.Polygon(
        [
            (169.07901872038428, -42.23259539489639),
            (169.07901872038428, -44.678482592421624),
            (174.25821453531285, -44.678482592421624),
            (174.25821453531285, -42.23259539489639),
            (169.07901872038428, -42.23259539489639),
        ]
    )
    return polygon


def test_region_spacing_config_region_file(
    chch_region_polygon: shapely.Polygon,
) -> None:
    # Save region as geojson dict
    with tempfile.TemporaryDirectory() as tmpdir:
        geojson_dict = geometry.mapping(chch_region_polygon)
        geojson_ffp = f"{tmpdir}/chch_region.geojson"

        with open(geojson_ffp, "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "properties": {}, "geometry": geojson_dict}
                    ],
                },
                f,
            )

        config_dict = {"name": "chch", "geojson_ffp": geojson_ffp, "spacing": 1000}
        region_spacing_config = site_gen.RegionSpacingConfig.from_config(config_dict)

        assert region_spacing_config.name == "chch"
        assert region_spacing_config.spacing == 1000
        assert shapely.equals_exact(region_spacing_config.region, chch_region_polygon)


def test_region_spacing_config_no_region_specified() -> None:
    # Create region spacing config with no region specified
    config_dict = {"name": "chch", "spacing": 1000}

    with pytest.raises(
        ValueError,
        match="Either 'geojson_ffp' or 'region' must be provided in the config dictionary.",
    ):
        site_gen.RegionSpacingConfig.from_config(config_dict)


def test_region_spacing_config_region_metadata(
    chch_region_polygon: shapely.Polygon,
) -> None:
    # Create region spacing config from metadata dict
    metadata_dict = {
        "name": "chch",
        "region": geometry.mapping(chch_region_polygon),
        "spacing": 1000,
    }

    region_spacing_config = site_gen.RegionSpacingConfig.from_config(metadata_dict)

    assert region_spacing_config.name == "chch"
    assert region_spacing_config.spacing == 1000
    assert shapely.equals_exact(region_spacing_config.region, chch_region_polygon)


def test_custom_grid_config_no_vel_model_version() -> None:
    config_dict = {"basin_spacing": 2500}

    with pytest.raises(
        ValueError,
        match="vel_model_version must be provided if basin_spacing is set.",
    ):
        site_gen.CustomGridConfig.from_config(config_dict)


def test_custom_grid_config(chch_region_polygon: shapely.Polygon) -> None:
    # Create the custom grid config dict
    with tempfile.TemporaryDirectory() as tmpdir:
        geojson_dict = geometry.mapping(chch_region_polygon)
        geojson_ffp = f"{tmpdir}/chch_region.geojson"

        with open(geojson_ffp, "w") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "properties": {}, "geometry": geojson_dict}
                    ],
                },
                f,
            )

        config_dict = {
            "land_only": True,
            "region": chch_region_polygon,
            "uniform_spacing": 5000,
            "vel_model_version": "2.09",
            "basin_spacing": 2500,
            "per_basin_spacing": {"Hanmer_v25p3": 1250},
            "per_region_spacing": [
                {"name": "Christchurch", "geojson_ffp": geojson_ffp, "spacing": 1250}
            ],
            "nzgmdb_version": "v4.3",
        }

        custom_grid_config_1 = site_gen.CustomGridConfig.from_config(config_dict)

        assert custom_grid_config_1.land_only is True
        assert shapely.equals_exact(custom_grid_config_1.region, chch_region_polygon)
        assert custom_grid_config_1.uniform_spacing == 5000
        assert custom_grid_config_1.vel_model_version == "2.09"
        assert custom_grid_config_1.basin_spacing == 2500
        assert custom_grid_config_1.per_basin_spacing == {"Hanmer_v25p3": 1250}
        assert custom_grid_config_1.per_region_spacing is not None
        assert len(custom_grid_config_1.per_region_spacing) == 1
        per_region_config_1 = custom_grid_config_1.per_region_spacing[0]
        assert per_region_config_1.name == "Christchurch"
        assert per_region_config_1.spacing == 1250
        assert shapely.equals_exact(per_region_config_1.region, chch_region_polygon)

        config_dict = custom_grid_config_1.as_dict()
        custom_grid_config_2 = site_gen.CustomGridConfig.from_config(config_dict)

        assert custom_grid_config_1.land_only == custom_grid_config_2.land_only
        assert shapely.equals_exact(
            custom_grid_config_1.region, custom_grid_config_2.region
        )
        assert (
            custom_grid_config_1.uniform_spacing == custom_grid_config_2.uniform_spacing
        )
        assert (
            custom_grid_config_1.vel_model_version
            == custom_grid_config_2.vel_model_version
        )
        assert custom_grid_config_1.basin_spacing == custom_grid_config_2.basin_spacing
        assert (
            custom_grid_config_1.per_basin_spacing
            == custom_grid_config_2.per_basin_spacing
        )
        assert custom_grid_config_2.per_region_spacing is not None
        assert len(custom_grid_config_2.per_region_spacing) == 1
        per_region_config_2 = custom_grid_config_2.per_region_spacing[0]
        assert per_region_config_1.name == per_region_config_2.name
        assert per_region_config_1.spacing == per_region_config_2.spacing
        assert shapely.equals_exact(
            per_region_config_1.region, per_region_config_2.region
        )


def test_custom_grid_init_without_general_grid(
    general_grid: site_gen.GeneralGrid,
) -> None:
    with patch.object(
        site_gen.GeneralGrid,
        "load",
        return_value=general_grid,
    ) as mock_load:
        custom_grid = site_gen.CustomGrid()
        mock_load.assert_called_once()
        assert custom_grid.general_grid == general_grid


def test_custom_grid_land_region_filtering(
    general_grid: site_gen.GeneralGrid, land_region_polygon: shapely.Polygon
) -> None:
    land_only_grid_config = site_gen.CustomGridConfig(
        land_only=True,
        region=land_region_polygon,
        uniform_spacing=general_grid.spacing,
    )
    land_only_grid = site_gen.CustomGrid(general_grid).apply_config(
        land_only_grid_config
    )

    # All points in the custom grid should be land points
    # and therefore all should be included
    # Check via neighbour distances
    site_df = land_only_grid.get_site_df()
    distances = sdist.squareform(sdist.pdist(site_df[["nztm_x", "nztm_y"]].values))
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, general_grid.spacing, rtol=0.05)

    region_config = site_gen.CustomGridConfig(
        region=land_region_polygon,
        uniform_spacing=general_grid.spacing,
    )
    region_grid = site_gen.CustomGrid(general_grid).apply_config(region_config)

    assert site_df.equals(region_grid.get_site_df())


def test_custom_grid_water_region_filtering(
    general_grid: site_gen.GeneralGrid, water_region_polygon: shapely.Polygon
) -> None:
    land_only_grid_config = site_gen.CustomGridConfig(
        land_only=True,
        region=water_region_polygon,
        uniform_spacing=general_grid.spacing,
    )
    land_only_grid = site_gen.CustomGrid(general_grid).apply_config(
        land_only_grid_config
    )

    site_df = land_only_grid.get_site_df()
    assert site_df.shape[0] == 0


def test_uniform_too_small_spacing_error(
    general_grid: site_gen.GeneralGrid,
) -> None:
    custom_grid = site_gen.CustomGrid(general_grid)
    spacing = general_grid.spacing - 1000

    with pytest.raises(ValueError):
        custom_grid._add_uniform_spacing_filter(spacing)


def test_uniform_non_multiple_spacing_error(
    general_grid: site_gen.GeneralGrid,
) -> None:
    custom_grid = site_gen.CustomGrid(general_grid)
    spacing = general_grid.spacing + 3000

    with pytest.raises(ValueError):
        custom_grid._add_uniform_spacing_filter(spacing)


def test_uniform_valid_spacing(
    general_grid: site_gen.GeneralGrid,
) -> None:
    config = site_gen.CustomGridConfig(
        uniform_spacing=general_grid.spacing * 2,
    )

    custom_grid = site_gen.CustomGrid(general_grid).apply_config(config)
    site_df = custom_grid.get_site_df()

    distances = sdist.squareform(sdist.pdist(site_df[["nztm_x", "nztm_y"]].values))
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, general_grid.spacing * 2, rtol=0.12)


def test_basin_spacing(
    general_grid: site_gen.GeneralGrid, canterbury_region_polygon: shapely.Polygon
) -> None:
    uniform_spacing = general_grid.spacing * 2
    config = site_gen.CustomGridConfig(
        region=canterbury_region_polygon,
        uniform_spacing=uniform_spacing,
        basin_spacing=general_grid.spacing,
        vel_model_version="2.09",
    )

    custom_grid = site_gen.CustomGrid(general_grid).apply_config(config)
    site_df = custom_grid.get_site_df()

    distances = sdist.squareform(sdist.pdist(site_df[["nztm_x", "nztm_y"]].values))
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, general_grid.spacing, rtol=0.12)
    assert np.all(min_distances < uniform_spacing)


def test_per_basin_spacing(
    general_grid: site_gen.GeneralGrid,
    central_south_island_region_polygon: shapely.Polygon,
) -> None:
    uniform_spacing = general_grid.spacing * 2
    config = site_gen.CustomGridConfig(
        region=central_south_island_region_polygon,
        uniform_spacing=uniform_spacing,
        per_basin_spacing={"Canterbury_v25p9": general_grid.spacing},
        vel_model_version="2.09",
    )

    custom_grid = site_gen.CustomGrid(general_grid).apply_config(config)
    site_df = custom_grid.get_site_df()

    cant_basin_sites = site_df[site_df.basin == "Canterbury_v25p9"]
    distances = sdist.squareform(
        sdist.pdist(cant_basin_sites[["nztm_x", "nztm_y"]].values)
    )
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, general_grid.spacing, rtol=0.12)

    other_sites = site_df[site_df.basin != "Canterbury_v25p9"]
    distances = sdist.squareform(sdist.pdist(other_sites[["nztm_x", "nztm_y"]].values))
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, uniform_spacing, rtol=0.12)


def test_custom_grid_metadata(general_grid: site_gen.GeneralGrid) -> None:
    config = site_gen.CustomGridConfig(
        uniform_spacing=general_grid.spacing,
    )

    custom_grid = site_gen.CustomGrid(general_grid)
    custom_grid.config = config

    # Create fake site dataframe
    site_df = pd.DataFrame(data=["real", "virtual"], columns=["source"])
    site_df["basin"] = ["basin_test", None]

    metadata = custom_grid.get_metadata(site_df)
    site_metadata = metadata["site_metadata"]
    config = metadata["config"]
    assert site_metadata["num_sites"] == 2
    assert site_metadata["num_real_sites"] == 1
    assert site_metadata["num_virtual_sites"] == 1
    assert site_metadata["num_basin_sites"] == 1
    assert site_metadata["sites_per_basin"]["basin_test"] == 1
    assert config["uniform_spacing"] == general_grid.spacing


def test_per_region_spacing(
    general_grid: site_gen.GeneralGrid,
    central_south_island_region_polygon: shapely.Polygon,
    canterbury_region_polygon: shapely.Polygon,
) -> None:
    uniform_spacing = general_grid.spacing * 2
    config = site_gen.CustomGridConfig(
        region=central_south_island_region_polygon,
        uniform_spacing=uniform_spacing,
        per_region_spacing=[
            site_gen.RegionSpacingConfig(
                name="Canterbury",
                region=canterbury_region_polygon,
                spacing=general_grid.spacing,
            )
        ],
    )

    custom_grid = site_gen.CustomGrid(general_grid).apply_config(config)
    site_df = custom_grid.get_site_df()

    # Get sites within Christchurch region
    canterbury_region_polygon_nztm = shapely.transform(
        canterbury_region_polygon,
        lambda x: coordinates.wgs_depth_to_nztm(x[:, ::-1])[:, ::-1],
    )
    geo_site_df = gpd.GeoDataFrame(
        site_df,
        geometry=gpd.points_from_xy(site_df.nztm_x, site_df.nztm_y),
        crs=site_gen.NZTM_CRS,
    )
    canterbury_sites = geo_site_df[
        geo_site_df.geometry.within(canterbury_region_polygon_nztm)
    ]

    # Check spacing within Canterbury region
    distances = sdist.squareform(
        sdist.pdist(canterbury_sites[["nztm_x", "nztm_y"]].values)
    )
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, general_grid.spacing, rtol=0.12)

    # Check spacing outside Canterbury region
    non_canterbury_sites = geo_site_df.drop(canterbury_sites.index)
    distances = sdist.squareform(
        sdist.pdist(non_canterbury_sites[["nztm_x", "nztm_y"]].values)
    )
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    assert np.allclose(min_distances, uniform_spacing, rtol=0.12)


@pytest.mark.parametrize("uniform_spacing", [5_000, 10_000])
def test_small_uniform_spacing_error(
    general_grid: site_gen.GeneralGrid, uniform_spacing: int
) -> None:
    config = site_gen.CustomGridConfig(uniform_spacing=uniform_spacing)

    with pytest.raises(
        ValueError,
        match="Uniform spacing must be greater than or equal to the general grid spacing",
    ):
        site_gen.CustomGrid(general_grid).apply_config(config)


@pytest.mark.parametrize("factor", [1.2, 1.5])
def test_multiple_error_uniform_spacing(
    general_grid: site_gen.GeneralGrid, factor: float
) -> None:
    config = site_gen.CustomGridConfig(
        uniform_spacing=int(general_grid.spacing * factor),
    )

    with pytest.raises(
        ValueError,
        match="Uniform spacing must be a multiple of the general grid spacing",
    ):
        site_gen.CustomGrid(general_grid).apply_config(config)


@pytest.mark.parametrize("basin_spacing", [5_000, 10_000])
def test_small_basin_spacing_error(
    general_grid: site_gen.GeneralGrid, basin_spacing: int
) -> None:
    uniform_spacing = general_grid.spacing * 2
    config = site_gen.CustomGridConfig(
        uniform_spacing=uniform_spacing,
        basin_spacing=basin_spacing,
        vel_model_version="2.09",
    )

    with pytest.raises(
        ValueError,
        match="Basin spacing must be greater than or equal to the general grid spacing.",
    ):
        site_gen.CustomGrid(general_grid).apply_config(config)


@pytest.mark.parametrize("factor", [1.2, 1.5])
def test_multiple_error_basin_spacing(
    general_grid: site_gen.GeneralGrid, factor: float
) -> None:
    uniform_spacing = general_grid.spacing * 2
    basin_spacing = int(general_grid.spacing * factor)
    config = site_gen.CustomGridConfig(
        uniform_spacing=uniform_spacing,
        basin_spacing=basin_spacing,
        vel_model_version="2.09",
    )

    with pytest.raises(
        ValueError, match="Basin spacing must be a multiple of the general grid spacing"
    ):
        site_gen.CustomGrid(general_grid).apply_config(config)


def test_site_dataframe(general_grid: site_gen.GeneralGrid) -> None:
    config = site_gen.CustomGridConfig(
        uniform_spacing=general_grid.spacing * 2,
        vel_model_version="2.09",
        nzgmdb_version=site_gen.NZGMDBVersion.V4p3,
    )

    custom_grid = site_gen.CustomGrid(general_grid).apply_config(config)
    site_df = custom_grid.get_site_df()

    assert (~site_df.region_name.isnull()).all()
    assert (~site_df.region_code.isnull()).all()
    assert site_df.source.isin(["virtual", "real"]).all()


def test_site_dataframe_basin(
    general_grid: site_gen.GeneralGrid, canterbury_region_polygon: shapely.Polygon
) -> None:
    config = site_gen.CustomGridConfig(
        region=canterbury_region_polygon,
        uniform_spacing=general_grid.spacing * 2,
        vel_model_version="2.09",
        nzgmdb_version=site_gen.NZGMDBVersion.V4p3,
    )

    custom_grid = site_gen.CustomGrid(general_grid).apply_config(config)
    site_df = custom_grid.get_site_df()

    assert (site_df.basin == "Canterbury_v25p9").all()


@given(
    nums=st.lists(
        st.integers(min_value=0, max_value=62**4 - 1), min_size=2, max_size=500
    ),
    length=st.integers(min_value=3, max_value=5),
)
def test_encode_base62_combined_properties(nums: list[int], length: int) -> None:
    """Property: encoded strings should have fixed length, be unique, and use valid base62 characters."""
    nums_array = np.array(nums)
    assume(np.all(nums_array < 62**length))  # Ensure all inputs fit in the length
    assume(
        np.unique(nums_array).size == len(nums_array)
    )  # Ensure all inputs are unique

    result = site_gen.encode_base62_fixed_array(nums_array, length=length)

    # The number of output codes should match the number of input numbers
    assert len(result) == len(nums)

    # All encoded strings should have the specified length
    assert all(len(code) == length for code in result)

    # Different numbers should produce different codes (all inputs are unique)
    assert len(np.unique(result)) == len(nums)

    # Encoded strings should only contain valid base62 characters
    alphabet = set(string.digits + string.ascii_letters)
    for code in result:
        assert all(c in alphabet for c in code)


@given(
    length=st.integers(min_value=1, max_value=6),
)
def test_encode_base62_boundary_values_property(length: int) -> None:
    """Property: test boundary values (0 and max) for a given length."""
    max_val = 62**length - 1
    nums = np.array([0, max_val])

    result = site_gen.encode_base62_fixed_array(nums, length=length)

    assert len(result) == 2
    assert all(len(code) == length for code in result)
    assert result[0] == "0" * length


@given(
    too_large_num=st.integers(min_value=62**3, max_value=62**4),
    length=st.just(3),
)
def test_encode_base62_too_large_error_property(
    too_large_num: int, length: int
) -> None:
    """Property: numbers too large for the specified length should raise ValueError."""
    nums = np.array([too_large_num])

    with pytest.raises(ValueError, match="too large"):
        site_gen.encode_base62_fixed_array(nums, length=length)


@given(
    num=st.integers(min_value=0, max_value=62**3 - 1),
)
def test_encode_base62_same_input_same_output_property(num: int) -> None:
    """Property: encoding the same number multiple times should give the same result."""
    nums = np.array([num, num, num])

    result = site_gen.encode_base62_fixed_array(nums, length=3)

    assert result[0] == result[1] == result[2]
