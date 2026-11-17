import numpy as np
import pytest
from schema import Schema, SchemaError

from workflow import defaults, realisations, schemas


def _point_source(dip: float) -> dict:
    return {
        "type": "point",
        "coordinates": {"latitude": -43.5, "longitude": 172.6, "depth": 10000},
        "length": 1000,
        "width": 1000,
        "strike": 30,
        "dip": dip,
        "dip_dir": 120,
    }


@pytest.mark.parametrize("dip", [0, 90])
def test_point_schema_accepts_valid_dip(dip: float) -> None:
    point = schemas.POINT_SCHEMA.validate(_point_source(dip))
    assert point.dip == dip


@pytest.mark.parametrize("dip", [-1, 91, 170])
def test_point_schema_rejects_invalid_dip(dip: float) -> None:
    with pytest.raises(SchemaError):
        schemas.POINT_SCHEMA.validate(_point_source(dip))


def _rupture_generation(**overrides: object) -> dict:
    table = {"key": 1.0}
    return {
        "spacing_km": 1.0,
        **{
            name: dict(table)
            for name in ("slip", "rise", "rake", "perturbation", "pulse", "jump")
        },
        **overrides,
    }


@pytest.mark.parametrize("propagation", ["predetermined", "sampled", "likeliest"])
@pytest.mark.parametrize("reach", ["sampled", "reference"])
def test_rupture_generation_accepts_propagation_and_reach(
    propagation: str, reach: str
) -> None:
    validated = schemas.RUPTURE_GENERATION_SCHEMA.validate(
        _rupture_generation(propagation=propagation, jump={"reach": reach})
    )
    assert validated["propagation"] == propagation
    assert validated["jump"] == {"reach": reach}


def test_rupture_generation_defaults_to_predetermined() -> None:
    validated = schemas.RUPTURE_GENERATION_SCHEMA.validate(_rupture_generation())
    assert validated["propagation"] == "predetermined"
    assert "reach" not in validated["jump"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"propagation": "tree"},
        {"propagation": None},
        {"jump": {"reach": "mean"}},
        {"jump": {"reach": None}},
    ],
)
def test_rupture_generation_rejects_unknown_propagation_and_reach(
    overrides: dict,
) -> None:
    with pytest.raises(SchemaError):
        schemas.RUPTURE_GENERATION_SCHEMA.validate(_rupture_generation(**overrides))


# --- Coordinate and bearing bounds -------------------------------------------


@pytest.mark.parametrize(
    "latitude, longitude", [(-90, -180), (90, 180), (-90, 180), (90, -180)]
)
def test_lat_lon_schema_accepts_inclusive_bounds(
    latitude: float, longitude: float
) -> None:
    """The extreme latitudes and longitudes are valid coordinates."""
    validated = Schema(schemas.LAT_LON_SCHEMA).validate(
        [{"latitude": latitude, "longitude": longitude}]
    )
    np.testing.assert_array_equal(validated, [[latitude, longitude]])


@pytest.mark.parametrize(
    "latitude, longitude",
    [(-90.5, 0), (90.5, 0), (0, -180.5), (0, 180.5)],
)
def test_lat_lon_schema_rejects_out_of_range(latitude: float, longitude: float) -> None:
    """Coordinates just outside the valid range are rejected."""
    with pytest.raises(SchemaError):
        Schema(schemas.LAT_LON_SCHEMA).validate(
            [{"latitude": latitude, "longitude": longitude}]
        )


@pytest.mark.parametrize("strike, dip_dir", [(0, 360), (360, 0)])
def test_point_schema_accepts_inclusive_bearings(strike: float, dip_dir: float) -> None:
    """Bearings of exactly 0 and 360 degrees are valid."""
    point = schemas.POINT_SCHEMA.validate(
        _point_source(45) | {"strike": strike, "dip_dir": dip_dir}
    )
    assert point.strike == strike
    assert point.dip_dir == dip_dir


@pytest.mark.parametrize("field", ["strike", "dip_dir"])
@pytest.mark.parametrize("bearing", [-0.5, 360.5])
def test_point_schema_rejects_invalid_bearing(field: str, bearing: float) -> None:
    """Bearings just outside [0, 360] are rejected."""
    with pytest.raises(SchemaError):
        schemas.POINT_SCHEMA.validate(_point_source(45) | {field: bearing})


# --- Proportions -------------------------------------------------------------


def _srf_config(**overrides: object) -> dict:
    srf_config = realisations.SRFConfig.read_from_defaults(
        defaults.DefaultsVersion.v24_2_2_1
    ).to_dict()
    return srf_config | overrides


@pytest.mark.parametrize("field", ["side_taper", "top_taper", "alpha_rough"])
@pytest.mark.parametrize("value", [0, 1])
def test_srf_schema_accepts_inclusive_proportions(field: str, value: float) -> None:
    """Proportions of exactly 0 and 1 are valid."""
    validated = schemas.SRF_SCHEMA.validate(_srf_config(**{field: value}))
    assert validated[field] == value


@pytest.mark.parametrize("field", ["side_taper", "top_taper", "alpha_rough"])
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_srf_schema_rejects_invalid_proportions(field: str, value: float) -> None:
    """Proportions outside [0, 1] are rejected."""
    with pytest.raises(SchemaError):
        schemas.SRF_SCHEMA.validate(_srf_config(**{field: value}))


# --- Fault and plane corners -------------------------------------------------


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((1, 4, 3), True),
        ((3, 4, 3), True),
        ((4, 3), False),
        ((12, 3), False),
        ((2, 2, 4, 3), False),
        ((1, 5, 3), False),
        ((1, 4, 4), False),
        ((1, 3, 4), False),
    ],
)
def test_is_correct_fault_corner_shape(shape: tuple[int, ...], expected: bool) -> None:
    """Only (n x 4 x 3) arrays are valid fault corners."""
    assert schemas._is_correct_fault_corner_shape(np.zeros(shape)) is expected


def test_plane_schema_rejects_corners_without_depth() -> None:
    """Plane corners missing depth produce (4 x 2) arrays, which are rejected."""
    corners = [
        {"latitude": -43.0, "longitude": 172.0},
        {"latitude": -43.0, "longitude": 172.1},
        {"latitude": -43.1, "longitude": 172.1},
        {"latitude": -43.1, "longitude": 172.0},
    ]
    with pytest.raises(SchemaError, match="_is_correct_corner_shape"):
        schemas.PLANE_SCHEMA.validate({"type": "plane", "corners": corners})


# --- Stoch bounds ------------------------------------------------------------


_STOCH_CONFIG = {
    "stoch_target_dx": 2.0,
    "stoch_min_dx": 0.0,
    "stoch_max_dx": None,
    "stoch_target_dy": 2.0,
    "stoch_min_dy": 0.0,
    "stoch_max_dy": None,
    "stoch_padding_weight": 0.0,
}


@pytest.mark.parametrize("axis", ["dx", "dy"])
def test_stoch_config_bounds_are_inclusive(axis: str) -> None:
    """A target equal to both the minimum and maximum is valid."""
    config = _STOCH_CONFIG | {
        f"stoch_target_{axis}": 1.5,
        f"stoch_min_{axis}": 1.5,
        f"stoch_max_{axis}": 1.5,
    }
    validated = schemas.STOCH_CONFIG_SCHEMA.validate(config)
    assert validated[f"stoch_target_{axis}"] == 1.5
    assert validated[f"stoch_min_{axis}"] == 1.5
    assert validated[f"stoch_max_{axis}"] == 1.5


@pytest.mark.parametrize("axis", ["dx", "dy"])
def test_stoch_config_rejects_target_below_minimum(axis: str) -> None:
    """A target below the minimum is rejected, even with no maximum."""
    config = _STOCH_CONFIG | {
        f"stoch_target_{axis}": 1.0,
        f"stoch_min_{axis}": 1.5,
    }
    with pytest.raises(SchemaError, match="_stoch_bounds_contain_target"):
        schemas.STOCH_CONFIG_SCHEMA.validate(config)
