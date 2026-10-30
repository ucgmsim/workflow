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
