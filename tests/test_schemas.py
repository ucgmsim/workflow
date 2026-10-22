import pytest
from schema import SchemaError

from hf_simulation import PathDurationModel, Ray
from workflow import schemas


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


def _hf_config(rayset: list[int], path_duration_model: int) -> dict:
    return {
        "source": {
            "stress_drop_bars": 50.0,
            "corner_frequency_constant": 2.1,
            "corner_frequency_alpha": 0.1,
            "rupture_velocity": {"sigma": 0.1},
        },
        "path": {"rayset": rayset, "path_duration_model": path_duration_model},
        "site": {"fmax_hz": 10.0},
        "record": {"dt": 0.01},
    }


@pytest.mark.parametrize("ray", list(Ray))
@pytest.mark.parametrize("path_duration_model", list(PathDurationModel))
def test_hf_schema_accepts_every_enum_code(
    ray: Ray, path_duration_model: PathDurationModel
) -> None:
    config = _hf_config([int(ray)], int(path_duration_model))
    assert schemas.HF_CONFIG_SCHEMA.validate(config) == config


@pytest.mark.parametrize("ray", [-1, 3, 4])
def test_hf_schema_rejects_unknown_ray(ray: int) -> None:
    with pytest.raises(SchemaError):
        schemas.HF_CONFIG_SCHEMA.validate(_hf_config([ray], 11))


@pytest.mark.parametrize("path_duration_model", [-1, 3, 10, 13])
def test_hf_schema_rejects_unknown_path_duration_model(
    path_duration_model: int,
) -> None:
    with pytest.raises(SchemaError):
        schemas.HF_CONFIG_SCHEMA.validate(_hf_config([1], path_duration_model))
