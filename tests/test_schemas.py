import pytest
from schema import SchemaError

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
