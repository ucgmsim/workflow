# NOTE: this fix contains tests that mirror the Realisations wiki. If
# these tests fail, you should update the wiki if necesary to ensure
# it stays consistent with the codebase.
import json
from pathlib import Path

import numpy as np
import pytest
import schema
from velocity_modelling import bounding_box

from source_modelling import rupture_propagation
from workflow import defaults, realisations


def test_bounding_box_example(tmp_path: Path):
    domain_parameters = realisations.DomainParameters(
        resolution=0.1,  # a 0.1km resolution
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=45.0,
            extent_x=100.0,
            extent_y=100.0,
        ),
        depth=40.0,
        duration=60.0,
        dt=0.005,
    )
    realisation_ffp = tmp_path / "realisation.json"
    domain_parameters.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "domain": {
                "resolution": 0.1,
                "domain": [
                    {"latitude": -43.524793866326725, "longitude": 171.76204128885567},
                    {"latitude": -44.16756820707226, "longitude": 172.63312824122775},
                    {"latitude": -43.53034935969409, "longitude": 173.51210368762364},
                    {"latitude": -42.894200350955856, "longitude": 172.64076673694242},
                ],
                "depth": 40.0,
                "duration": 60.0,
                "dt": 0.005,
            }
        }
    domain_parameters_read = realisations.DomainParameters.read_from_realisation(
        realisation_ffp
    )
    assert domain_parameters_read.depth == domain_parameters.depth
    assert domain_parameters_read.duration == domain_parameters.duration
    assert domain_parameters_read.dt == domain_parameters.dt
    assert domain_parameters_read.resolution == domain_parameters.resolution
    assert (
        domain_parameters_read.domain.corners == domain_parameters.domain.corners
    ).all()


def test_domain_parameters_properties():
    domain_parameters = realisations.DomainParameters(
        resolution=0.1,  # a 0.1km resolution
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=45.0,
            extent_x=100.0,
            extent_y=100.0,
        ),
        depth=40.0,
        duration=60.0,
        dt=0.005,
    )
    assert domain_parameters.nx == 1000
    assert domain_parameters.ny == 1000
    assert domain_parameters.nz == 400


def test_srf_config_example(tmp_path):
    domain_parameters = realisations.DomainParameters(
        resolution=0.1,  # a 0.1km resolution
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=45.0,
            extent_x=100.0,
            extent_y=100.0,
        ),
        depth=40.0,
        duration=60.0,
        dt=0.005,
    )
    srf_config = realisations.SRFConfig(
        genslip_dt=1.0,
        genslip_seed=1,
        genslip_version="5.4.2",
        resolution=0.1,
        srfgen_seed=1,
    )

    realisation_ffp = tmp_path / "realisation.json"
    domain_parameters.write_to_realisation(realisation_ffp)
    srf_config.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "domain": {
                "resolution": 0.1,
                "domain": [
                    {"latitude": -43.524793866326725, "longitude": 171.76204128885567},
                    {"latitude": -44.16756820707226, "longitude": 172.63312824122775},
                    {"latitude": -43.53034935969409, "longitude": 173.51210368762364},
                    {"latitude": -42.894200350955856, "longitude": 172.64076673694242},
                ],
                "depth": 40.0,
                "duration": 60.0,
                "dt": 0.005,
            },
            "srf": {
                "genslip_dt": 1.0,
                "genslip_seed": 1,
                "resolution": 0.1,
                "genslip_version": "5.4.2",
                "srfgen_seed": 1,
            },
        }

    assert realisations.SRFConfig.read_from_realisation(realisation_ffp) == srf_config


def test_bad_domain_parameters(tmp_path: Path):
    bad_json = tmp_path / "bad_domain_parameters.json"
    bad_json.write_text(
        json.dumps(
            {
                "domain": {
                    "resolution": 0,  # Set to 0
                    "domain": [
                        {
                            "latitude": -43.524793866326725,
                            "longitude": 171.76204128885567,
                        },
                        {
                            "latitude": -42.894200350955856,
                            "longitude": 172.64076673694242,
                        },
                        {
                            "latitude": -43.53034935969409,
                            "longitude": 173.51210368762364,
                        },
                        {
                            "latitude": -44.16756820707226,
                            "longitude": 172.63312824122775,
                        },
                    ],
                    "depth": 40.0,
                    "duration": 60.0,
                    "dt": 0.005,
                }
            }
        )
    )
    with pytest.raises(schema.SchemaError):
        realisations.DomainParameters.read_from_realisation(bad_json)


def test_bad_config_key(tmp_path: Path):
    bad_json = tmp_path / "bad_domain_parameters.json"
    bad_json.write_text(
        json.dumps(
            {
                "not the correct domain key": {
                    "resolution": 0.1,
                    "domain": [
                        {
                            "latitude": -43.524793866326725,
                            "longitude": 171.76204128885567,
                        },
                        {
                            "latitude": -42.894200350955856,
                            "longitude": 172.64076673694242,
                        },
                        {
                            "latitude": -43.53034935969409,
                            "longitude": 173.51210368762364,
                        },
                        {
                            "latitude": -44.16756820707226,
                            "longitude": 172.63312824122775,
                        },
                    ],
                    "depth": 40.0,
                    "duration": 60.0,
                    "dt": 0.005,
                }
            }
        )
    )
    with pytest.raises(realisations.RealisationParseError):
        realisations.DomainParameters.read_from_realisation(bad_json)


def test_metadata(tmp_path: Path):
    metadata = realisations.RealisationMetadata(
        name="consecutive write test",
        version="1",
        defaults_version=defaults.DefaultsVersion.develop,
    )
    realisation_ffp = tmp_path / "realisation.json"
    metadata.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "metadata": {
                "name": "consecutive write test",
                "version": "1",
                "defaults_version": "develop",
                "tag": None,
            },
        }

    assert (
        realisations.RealisationMetadata.read_from_realisation(realisation_ffp)
        == metadata
    )


def test_velocity_model(tmp_path: Path):
    velocity_model = realisations.VelocityModelParameters(
        min_vs=1.0,
        version="2.06",
        topo_type="SQUASHED_TAPERED",
        dt=0.05,
        ds_multiplier=1.2,
        resolution=0.1,
        vs30=300.0,
        s_wave_velocity=3500.0,
    )
    realisation_ffp = tmp_path / "realisation.json"
    velocity_model.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "velocity_model": {
                "min_vs": 1.0,
                "version": "2.06",
                "topo_type": "SQUASHED_TAPERED",
                "dt": 0.05,
                "ds_multiplier": 1.2,
                "resolution": 0.1,
                "vs30": 300.0,
                "s_wave_velocity": 3500.0,
            }
        }

    assert (
        realisations.VelocityModelParameters.read_from_realisation(realisation_ffp)
        == velocity_model
    )


def test_rupture_prop_config(tmp_path: Path):
    rup_prop = realisations.RupturePropagationConfig(
        rupture_causality_tree={"A": None, "B": "A", "C": "B"},
        jump_points={
            "B": rupture_propagation.JumpPair(
                from_point=np.array([0.0, 1.0]), to_point=np.array([0.0, 0.0])
            ),
            "C": rupture_propagation.JumpPair(
                from_point=np.array([0.25, 0.8]), to_point=np.array([0.5, 0.333])
            ),
        },
        rakes={"A": 100.0, "B": 67.0, "C": 125.0},
        magnitudes={"A": 6.5, "B": 6.7, "C": 6.9},
        hypocentre=np.array([0.0, 0.6]),
    )

    realisation_ffp = tmp_path / "realisation.json"
    rup_prop.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "rupture_propagation": {
                "rupture_causality_tree": {"A": None, "B": "A", "C": "B"},
                "jump_points": {
                    "B": {
                        "from_point": {"s": 0.0, "d": 1.0},
                        "to_point": {"s": 0.0, "d": 0.0},
                    },
                    "C": {
                        "from_point": {"s": 0.25, "d": 0.8},
                        "to_point": {"s": 0.5, "d": 0.333},
                    },
                },
                "rakes": {"A": 100.0, "B": 67.0, "C": 125.0},
                "magnitudes": {"A": 6.5, "B": 6.7, "C": 6.9},
                "hypocentre": {"s": 0.0, "d": 0.6},
            }
        }
    rupture_prop_config = realisations.RupturePropagationConfig.read_from_realisation(
        realisation_ffp
    )
    assert rupture_prop_config.rupture_causality_tree == {"A": None, "B": "A", "C": "B"}
    assert rupture_prop_config.jump_points["B"].from_point.tolist() == [0.0, 1.0]
    assert rupture_prop_config.jump_points["B"].to_point.tolist() == [0.0, 0.0]
    assert rupture_prop_config.jump_points["C"].from_point.tolist() == [0.25, 0.8]
    assert rupture_prop_config.jump_points["C"].to_point.tolist() == [0.5, 0.333]
    assert rupture_prop_config.rakes == {"A": 100.0, "B": 67.0, "C": 125.0}
    assert rupture_prop_config.magnitudes == {"A": 6.5, "B": 6.7, "C": 6.9}
    assert rupture_prop_config.hypocentre.tolist() == [0.0, 0.6]


def test_rupture_prop_properties():
    rup_prop = realisations.RupturePropagationConfig(
        rupture_causality_tree={"A": None, "B": "A", "C": "B"},
        jump_points={
            "B": rupture_propagation.JumpPair(
                from_point=np.array([0.0, 1.0]), to_point=np.array([0.0, 0.0])
            ),
            "C": rupture_propagation.JumpPair(
                from_point=np.array([0.25, 0.8]), to_point=np.array([0.5, 0.333])
            ),
        },
        rakes={"A": 100.0, "B": 67.0, "C": 125.0},
        magnitudes={"A": 6.5, "B": 6.7, "C": 6.9},
        hypocentre=np.array([0.0, 0.6]),
    )
    assert rup_prop.initial_fault == "A"


def test_hf_config(tmp_path):
    test_realisation = tmp_path / "realisation.json"
    test_realisation.write_text("{}")
    hf_config = realisations.HFConfig.read_from_realisation_or_defaults(
        test_realisation, defaults.DefaultsVersion.v24_2_2_1
    )
    hf_config.write_to_realisation(test_realisation)
    assert realisations.HFConfig.read_from_realisation(test_realisation) == hf_config
    # Test that realisation parameters override defaults.
    hf_config.dt = 0.1
    hf_config.write_to_realisation(test_realisation)
    assert (
        realisations.HFConfig.read_from_realisation_or_defaults(
            test_realisation, defaults.DefaultsVersion.v24_2_2_1
        )
        == hf_config
    )


def test_emod3d(tmp_path: Path):
    test_realisation = tmp_path / "realisation.json"
    test_realisation.write_text("{}")
    emod3d = realisations.EMOD3DParameters.read_from_realisation_or_defaults(
        test_realisation, defaults.DefaultsVersion.v24_2_2_1
    )
    emod3d.write_to_realisation(test_realisation)
    assert (
        realisations.EMOD3DParameters.read_from_realisation(test_realisation) == emod3d
    )
    emod3d.dt = 0.1
    emod3d.write_to_realisation(test_realisation)
    assert (
        realisations.EMOD3DParameters.read_from_realisation_or_defaults(
            test_realisation, defaults.DefaultsVersion.v24_2_2_1
        )
        == emod3d
    )


def test_broadband_parameters(tmp_path: Path):
    test_realisation = tmp_path / "realisation.json"
    broadband_parameters = realisations.BroadbandParameters(
        flo=0.5, dt=0.005, fmidbot=0.5, fmin=0.25, site_amp_version="2014"
    )
    broadband_parameters.write_to_realisation(test_realisation)
    with open(test_realisation, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "bb": {
                "flo": 0.5,
                "dt": 0.005,
                "fmidbot": 0.5,
                "fmin": 0.25,
                "site_amp_version": "2014",
            }
        }
    assert (
        realisations.BroadbandParameters.read_from_realisation(test_realisation)
        == broadband_parameters
    )


@pytest.mark.parametrize(
    "realisation_config",
    [
        realisations.EMOD3DParameters,
        realisations.HFConfig,
        realisations.SRFConfig,
        realisations.VelocityModelParameters,
        realisations.BroadbandParameters,
    ],
)
@pytest.mark.parametrize("defaults_version", list(defaults.DefaultsVersion))
def test_defaults_are_loadable(
    tmp_path: Path,
    realisation_config: realisations.RealisationConfiguration,
    defaults_version: defaults.DefaultsVersion,
):
    realisation_config.read_from_defaults(defaults_version)
