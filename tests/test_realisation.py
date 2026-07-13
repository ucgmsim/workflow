# NOTE: this fix contains tests that mirror the Realisations wiki. If
# these tests fail, you should update the wiki if necesary to ensure
# it stays consistent with the codebase.
import json
import struct
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import schema

from IM import im_calculation
from source_modelling import magnitude_scaling, rupture_propagation
from velocity_modelling import bounding_box
from workflow import defaults, realisations, schemas
from workflow.realisations import SourceConfig


def test_bounding_box_example(tmp_path: Path) -> None:
    domain_parameters = realisations.DomainParameters(
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=45.0,
            extent_x=100.0,
            extent_y=100.0,
        ),
        depth=40.0,
        duration=60.0,
    )
    realisation_ffp = tmp_path / "realisation.json"
    domain_parameters.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "domain": {
                "domain": [
                    {"latitude": -43.524793866326725, "longitude": 171.76204128885567},
                    {"latitude": -44.16756820707226, "longitude": 172.63312824122775},
                    {"latitude": -43.53034935969409, "longitude": 173.51210368762364},
                    {"latitude": -42.894200350955856, "longitude": 172.64076673694242},
                ],
                "depth": 40.0,
                "duration": 60.0,
            }
        }
    domain_parameters_read = realisations.DomainParameters.read_from_realisation(
        realisation_ffp
    )
    assert domain_parameters_read.depth == domain_parameters.depth
    assert domain_parameters_read.duration == domain_parameters.duration
    assert (
        domain_parameters_read.domain.corners == domain_parameters.domain.corners
    ).all()


def test_domain_parameters_discretisation() -> None:
    """Test domain parameter discretisation to nx, ny, nz"""
    domain_parameters = realisations.DomainParameters(
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=0.0,
            extent_x=100.0,
            extent_y=100.0,
        ),
        depth=40.0,
        duration=60.0,
    )
    assert domain_parameters.nx(0.1) == 1000
    assert domain_parameters.ny(0.1) == 1000
    assert domain_parameters.nz(0.1) == 400

    domain_parameters = realisations.DomainParameters(
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=0.0,
            extent_x=100.5,
            extent_y=100.5,
        ),
        depth=40.5,
        duration=60.0,
    )
    # NOTE: different behaviour to round()
    assert domain_parameters.nx(1) == 101
    assert domain_parameters.ny(1) == 101
    assert domain_parameters.nz(1) == 41

    domain_parameters = realisations.DomainParameters(
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=0.0,
            extent_x=100.04,
            extent_y=100.09,
        ),
        depth=40.03,
        duration=60.0,
    )

    assert domain_parameters.nx(0.1) == 1000
    assert domain_parameters.ny(0.1) == 1001
    assert domain_parameters.nz(0.1) == 400


def test_srf_config_example(tmp_path: Path) -> None:
    domain_parameters = realisations.DomainParameters(
        domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
            centroid=np.array([-43.53092, 172.63701]),
            bearing=45.0,
            extent_x=100.0,
            extent_y=100.0,
        ),
        depth=40.0,
        duration=60.0,
    )
    srf_config = realisations.SRFConfig(
        resolution=0.1,
        point_source_params=schemas.PointSourceParams(
            stype=schemas.Stype.cos,
            risetime=0.5,
            risetimefac=1.0,
            risetimedep=0.0,
            inittime=0.0,
        ),
        side_taper=0.02,
        bot_taper=0.02,
        top_taper=0.0,
        alpha_rough=0.0,
        gwid=[],
        rvfac_seg=[],
        seg_delay=False,
        slip_sigma=1.0,
        risetime_coef=1.6,
        ymag_exponent=None,
        xmag_exponent=None,
        kx_corner=None,
        ky_corner=None,
        beta_asp=0.3,
        beta_deep=0.13,
        beta_mid=0.13,
        beta_mid_depth=6.5,
        beta_mid_depth_range=1.5,
        beta_shal=0.5,
        beta_shal_depth=2.0,
        beta_shal_depth_range=1.0,
        beta_subevt=0.1,
        deep_risetimedep=17.5,
        deep_risetimedep_range=2.5,
        deep_risetimefac=2.0,
        risetimedep=6.5,
        risetimedep_range=1.5,
        risetimefac=2.0,
        rt_rand=0.0,
        rt_scalefac=1.0,
        stype=None,
        hyb_corlen_deep_wt_end=1.0,
        hyb_corlen_deep_wt_start=0.0,
        hyb_corlen_dep=6.5,
        hyb_corlen_dep_range=1.5,
        hyb_corlen_fac=2.0,
        hyb_corlen_flag=False,
        hyb_corlen_kmodel=schemas.KModel.SUZUKI,
        hyb_corlen_shal_wt_end=0.0,
        hyb_corlen_shal_wt_start=1.0,
        hyb_corlen_side_taper=0.08,
        fdrup_scale_slip=False,
        fdrup_time=False,
        rupture_delay=0.0,
        rvfmax=1.414,
        rvfmin=0.25,
        truncate_zero_slip=True,
        slip_water_level=None,
        rake_sigma=15.0,
        fractal_rake=False,
        tsfac1_scor=0.8,
        tsfac1_sigma=1.0,
        tsfac2_lambda_max=5.0,
        tsfac2_lambda_min=None,
        tsfac2_scor=0.5,
        tsfac2_sigma=1.0,
        tsfac_bzero=-0.1,
        tsfac_coef=1.1,
        tsfac_main=None,
        tsfac_slope=-0.5,
        circular_average=False,
        kmodel=schemas.KModel.MAI,
        kord=4,
        magC=6.3,
        mag_area_Acoef=None,
        mag_area_Bcoef=None,
        mai_wt=0.5,
        modified_corners=False,
        somerville_wt=0.5,
        stretch_kcorner=False,
        use_gaus=True,
        use_median_mag=False,
        lambda_max=None,
        lambda_min=None,
        wavelength_max=None,
        wavelength_min=None,
        asp_taper_fac=0.05,
        extend_fac=None,
        flen_max=None,
        fwid_max=None,
        moment_fraction=None,
        perturb_subfault_location=True,
        rand_rake_degs=60.0,
        rtime1_depth=2.0,
        rtime1_depth_range=1.0,
        rtime1_scor=0.8,
        rtime1_sigma=0.85,
        rtime2_scor=0.5,
        rtime2slip_exp=0.5,
        rtime_rand=None,
        set_rake=None,
        svr_wt=0.0,
        target_savg=None,
        use_Mw=True,
        aseis_flag=False,
        aseis_smooth=False,
        aseis_dep=10.0,
        aseis_fac=None,
        xshift=0.0,
        yshift=0.0,
        read_erf=False,
        read_gsf=False,
        srf_version="1.0",
        write_gsf=False,
        write_srf=False,
        dump_last_seed=False,
        print_command=False,
        print_seed=False,
    )

    realisation_ffp = tmp_path / "realisation.json"
    domain_parameters.write_to_realisation(realisation_ffp)
    srf_config.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "domain": {
                "domain": [
                    {"latitude": -43.524793866326725, "longitude": 171.76204128885567},
                    {"latitude": -44.16756820707226, "longitude": 172.63312824122775},
                    {"latitude": -43.53034935969409, "longitude": 173.51210368762364},
                    {"latitude": -42.894200350955856, "longitude": 172.64076673694242},
                ],
                "depth": 40.0,
                "duration": 60.0,
            },
            "srf": {
                "resolution": 0.1,
                "point_source_params": {
                    "stype": "cos",
                    "risetime": 0.5,
                    "risetimefac": 1.0,
                    "risetimedep": 0.0,
                    "inittime": 0.0,
                },
                "side_taper": 0.02,
                "bot_taper": 0.02,
                "top_taper": 0.0,
                "alpha_rough": 0.0,
                "slip_sigma": 1.0,
                "risetime_coef": 1.6,
                "gwid": [],
                "rvfac_seg": [],
                "seg_delay": False,
                "ymag_exponent": None,
                "xmag_exponent": None,
                "kx_corner": None,
                "ky_corner": None,
                "beta_asp": 0.3,
                "beta_deep": 0.13,
                "beta_mid": 0.13,
                "beta_mid_depth": 6.5,
                "beta_mid_depth_range": 1.5,
                "beta_shal": 0.5,
                "beta_shal_depth": 2.0,
                "beta_shal_depth_range": 1.0,
                "beta_subevt": 0.1,
                "deep_risetimedep": 17.5,
                "deep_risetimedep_range": 2.5,
                "deep_risetimefac": 2.0,
                "risetimedep": 6.5,
                "risetimedep_range": 1.5,
                "risetimefac": 2.0,
                "rt_rand": 0.0,
                "rt_scalefac": 1.0,
                "stype": None,
                "hyb_corlen_deep_wt_end": 1.0,
                "hyb_corlen_deep_wt_start": 0.0,
                "hyb_corlen_dep": 6.5,
                "hyb_corlen_dep_range": 1.5,
                "hyb_corlen_fac": 2.0,
                "hyb_corlen_flag": False,
                "hyb_corlen_kmodel": schemas.KModel.SUZUKI,
                "hyb_corlen_shal_wt_end": 0.0,
                "hyb_corlen_shal_wt_start": 1.0,
                "hyb_corlen_side_taper": 0.08,
                "fdrup_scale_slip": False,
                "fdrup_time": False,
                "rupture_delay": 0.0,
                "rvfmax": 1.414,
                "rvfmin": 0.25,
                "truncate_zero_slip": True,
                "slip_water_level": None,
                "rake_sigma": 15.0,
                "fractal_rake": False,
                "tsfac1_scor": 0.8,
                "tsfac1_sigma": 1.0,
                "tsfac2_lambda_max": 5.0,
                "tsfac2_lambda_min": None,
                "tsfac2_scor": 0.5,
                "tsfac2_sigma": 1.0,
                "tsfac_bzero": -0.1,
                "tsfac_coef": 1.1,
                "tsfac_main": None,
                "tsfac_slope": -0.5,
                "circular_average": False,
                "kmodel": 2,
                "kord": 4,
                "magC": 6.3,
                "mag_area_Acoef": None,
                "mag_area_Bcoef": None,
                "mai_wt": 0.5,
                "modified_corners": False,
                "somerville_wt": 0.5,
                "stretch_kcorner": False,
                "use_gaus": True,
                "use_median_mag": False,
                "lambda_max": None,
                "lambda_min": None,
                "wavelength_max": None,
                "wavelength_min": None,
                "asp_taper_fac": 0.05,
                "extend_fac": None,
                "flen_max": None,
                "fwid_max": None,
                "moment_fraction": None,
                "perturb_subfault_location": True,
                "rand_rake_degs": 60.0,
                "rtime1_depth": 2.0,
                "rtime1_depth_range": 1.0,
                "rtime1_scor": 0.8,
                "rtime1_sigma": 0.85,
                "rtime2_scor": 0.5,
                "rtime2slip_exp": 0.5,
                "rtime_rand": None,
                "set_rake": None,
                "svr_wt": 0.0,
                "target_savg": None,
                "use_Mw": True,
                "aseis_flag": False,
                "aseis_smooth": False,
                "aseis_dep": 10.0,
                "aseis_fac": None,
                "xshift": 0.0,
                "yshift": 0.0,
                "read_erf": False,
                "read_gsf": False,
                "srf_version": "1.0",
                "write_gsf": False,
                "write_srf": False,
                "dump_last_seed": False,
                "print_command": False,
                "print_seed": False,
            },
        }

    assert realisations.SRFConfig.read_from_realisation(realisation_ffp) == srf_config


def test_bad_config_key(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad_domain_parameters.json"
    bad_json.write_text(
        json.dumps(
            {
                "not the correct domain key": {
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
                }
            }
        )
    )
    with pytest.raises(realisations.RealisationParseError):
        realisations.DomainParameters.read_from_realisation(bad_json)


def test_metadata(tmp_path: Path) -> None:
    metadata = realisations.RealisationMetadata(
        name="consecutive write test",
        version="1",
        defaults_version=defaults.DefaultsVersion.v24_2_2_1,
    )
    realisation_ffp = tmp_path / "realisation.json"
    metadata.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "metadata": {
                "name": "consecutive write test",
                "version": "1",
                "defaults_version": "24.2.2.1",
                "tag": None,
            },
        }

    assert (
        realisations.RealisationMetadata.read_from_realisation(realisation_ffp)
        == metadata
    )


def test_velocity_model(tmp_path: Path) -> None:
    velocity_model = realisations.VelocityModelParameters(
        min_vs=1.0,
        version="2.06",
        topo_type="SQUASHED_TAPERED",
        ds_multiplier=1.2,
        vs30=300.0,
        fault_buffer=2000.0,
        s_wave_velocity=3500.0,
        rrup_interpolants=np.ones(shape=(2, 2), dtype=np.float32),
        chunks=dict(),
        layers=None,
        surface=None
    )
    realisation_ffp = tmp_path / "realisation.json"
    velocity_model.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "velocity_model": {
                "min_vs": 1.0,
                "version": "2.06",
                "topo_type": "SQUASHED_TAPERED",
                "ds_multiplier": 1.2,
                "vs30": 300.0,
                "fault_buffer": 2000.0,
                "s_wave_velocity": 3500.0,
                "rrup_interpolants": [[1, 1], [1, 1]],
                'chunks': dict(),
                'layers': None,
                'surface': None
            }
        }

    assert (
        realisations.VelocityModelParameters.read_from_realisation(
            realisation_ffp
        ).to_dict()
        == velocity_model.to_dict()
    )


def test_rupture_prop_config(tmp_path: Path) -> None:
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
    assert rupture_prop_config.hypocentre.tolist() == [0.0, 0.6]


def test_magnitudes(tmp_path: Path) -> None:
    magnitudes = realisations.Magnitudes(
        magnitudes={
            "A": magnitude_scaling.BoldM(6.5),
            "B": magnitude_scaling.BoldM(6.7),
            "C": magnitude_scaling.BoldM(6.9),
        },
    )

    realisation_ffp = tmp_path / "realisation.json"
    magnitudes.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "magnitudes": {
                "magnitudes": {"A": 6.5, "B": 6.7, "C": 6.9},
            }
        }
    magnitudes = realisations.Magnitudes.read_from_realisation(realisation_ffp)
    assert magnitudes.magnitudes == {"A": 6.5, "B": 6.7, "C": 6.9}


def test_rakes(tmp_path: Path) -> None:
    rakes = realisations.Rakes(
        rakes={"A": 100.0, "B": 67.0, "C": 125.0},
    )

    realisation_ffp = tmp_path / "realisation.json"
    rakes.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "rakes": {
                "rakes": {"A": 100.0, "B": 67.0, "C": 125.0},
            }
        }
    rakes = realisations.Rakes.read_from_realisation(realisation_ffp)
    assert rakes.rakes == {"A": 100.0, "B": 67.0, "C": 125.0}


def test_rupture_prop_properties() -> None:
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
        hypocentre=np.array([0.0, 0.6]),
    )
    assert rup_prop.initial_fault == "A"


def test_hf_config(tmp_path: Path) -> None:
    test_realisation = tmp_path / "realisation.json"
    test_realisation.write_text("{}")
    hf_config = realisations.HFConfig.read_from_realisation_or_defaults(
        test_realisation, defaults.DefaultsVersion.v24_2_2_1
    )
    hf_config.write_to_realisation(test_realisation)
    assert realisations.HFConfig.read_from_realisation(test_realisation) == hf_config
    # Test that realisation parameters override defaults.
    hf_config.czero = 2.0
    hf_config.write_to_realisation(test_realisation)
    assert (
        realisations.HFConfig.read_from_realisation_or_defaults(
            test_realisation, defaults.DefaultsVersion.v24_2_2_1
        )
        == hf_config
    )


def test_emod3d(tmp_path: Path) -> None:
    test_realisation = tmp_path / "realisation.json"
    test_realisation.write_text("{}")
    emod3d = realisations.EMOD3DParameters.read_from_realisation_or_defaults(
        test_realisation, defaults.DefaultsVersion.v24_2_2_1
    )
    emod3d.write_to_realisation(test_realisation)
    assert (
        realisations.EMOD3DParameters.read_from_realisation(test_realisation) == emod3d
    )
    emod3d.write_to_realisation(test_realisation)
    assert (
        realisations.EMOD3DParameters.read_from_realisation_or_defaults(
            test_realisation, defaults.DefaultsVersion.v24_2_2_1
        )
        == emod3d
    )


def test_broadband_parameters(tmp_path: Path) -> None:
    test_realisation = tmp_path / "realisation.json"
    broadband_parameters = realisations.BroadbandParameters(
        flo=0.5,
        fmidbot=0.5,
        fmin=0.25,
        fhightop=15.0,
        fmax=25.0,
        site_amp_version=schemas.SiteAmpModel.BA2018,
    )
    broadband_parameters.write_to_realisation(test_realisation)
    with open(test_realisation, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "bb": {
                "flo": 0.5,
                "fmidbot": 0.5,
                "fmin": 0.25,
                "fhightop": 15.0,
                "fmax": 25.0,
                "site_amp_version": "ba2018",
            }
        }
    assert (
        realisations.BroadbandParameters.read_from_realisation(test_realisation)
        == broadband_parameters
    )


def test_logtrail_init_empty() -> None:
    """Test LogTrail initialization with no log provided."""
    trail = realisations.LogTrail([])
    assert trail.log == []
    assert trail._config_key == "log_trail"


def test_logtrail_init_with_log_entries() -> None:
    """Test LogTrail initialization with a list of LogEntry objects."""
    entry1 = realisations.LogEntry(
        utility="util1", args=["a"], version="1", timestamp=datetime.now()
    )
    entry2 = realisations.LogEntry(
        utility="util2", args=["b", "c"], timestamp=datetime.now(), version="1"
    )
    trail = realisations.LogTrail(log=[entry1, entry2])
    assert trail.log == [entry1, entry2]


def test_schemas_validate_float() -> None:
    assert schemas.NUMBER.validate(1.0)


def test_schemas_validate_int() -> None:
    assert schemas.NUMBER.validate(1)


def test_schemas_do_not_validate_strings() -> None:
    with pytest.raises(schema.SchemaError):
        schemas.NUMBER.validate("1")


def test_logtrail_init_with_dicts_post_init() -> None:
    """Test LogTrail post_init conversion of dicts to LogEntry objects."""
    log_data = [
        {
            "utility": "util1",
            "args": ["a"],
            "version": "1",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "utility": "util2",
            "args": ["b"],
            "version": "1",
            "timestamp": datetime.now().isoformat(),
        },
    ]
    # Pass raw list of dicts
    trail = realisations.LogTrail(log=log_data)  # type: ignore
    assert len(trail.log) == 2
    assert isinstance(trail.log[0], realisations.LogEntry)
    assert isinstance(trail.log[1], realisations.LogEntry)
    assert trail.log[0].utility == "util1"
    assert trail.log[0].args == ["a"]
    assert trail.log[1].utility == "util2"
    assert trail.log[1].args == ["b"]


def test_logtrail_log_entry_method() -> None:
    """Test adding an entry using the log_entry method."""
    trail = realisations.LogTrail([])
    trail.log_entry("my_util", ["--flag", "value"])
    assert len(trail.log) == 1
    assert isinstance(trail.log[0], realisations.LogEntry)
    assert trail.log[0].utility == "my_util"
    assert trail.log[0].args == ["--flag", "value"]
    assert isinstance(trail.log[0].timestamp, datetime)


def test_logtrail_to_dict() -> None:
    """Test converting LogTrail to a dictionary."""
    ts = datetime.now()
    entry1 = realisations.LogEntry(
        utility="util1", args=["a"], version="1", timestamp=ts
    )
    entry2 = realisations.LogEntry(
        utility="util2", args=["b", "c"], timestamp=ts, version="1"
    )
    trail = realisations.LogTrail(log=[entry1, entry2])
    trail_dict = trail.to_dict()

    assert isinstance(trail_dict, dict)
    assert "log" in trail_dict
    assert len(trail_dict["log"]) == 2
    # Check if realisations.LogEntry objects were converted back to dicts with ISO timestamps
    assert isinstance(trail_dict["log"][0], dict)
    assert trail_dict["log"][0]["utility"] == "util1"
    assert trail_dict["log"][0]["version"] == "1"
    assert trail_dict["log"][0]["timestamp"] == ts.isoformat()
    assert trail_dict["log"][0]["args"] == ["a"]
    assert isinstance(
        trail_dict["log"][0]["timestamp"], str
    )  # Should be ISO format string
    assert isinstance(trail_dict["log"][1], dict)
    assert trail_dict["log"][1]["utility"] == "util2"
    assert trail_dict["log"][1]["args"] == ["b", "c"]
    assert trail_dict["log"][1]["version"] == "1"
    assert trail_dict["log"][1]["timestamp"] == ts.isoformat()
    assert isinstance(
        trail_dict["log"][1]["timestamp"], str
    )  # Should be ISO format string

    datetime.fromisoformat(trail_dict["log"][0]["timestamp"])


def test_append_log_entry_file_exists_no_key(
    tmp_path: Path,
) -> None:
    """Test append_log_entry when file exists but lacks the 'log_trail' key."""
    realisation_file = tmp_path / "test_realisation.json"
    # Create a file with unrelated content
    initial_content = {"other_key": "some_value"}
    with open(realisation_file, "w") as f:
        json.dump(initial_content, f)

    # Mock realisations.LogEntry.from_utility for the creation case
    with mock.patch("sys.argv", new=["script_name.py", "--flag", "value"]):
        realisations.append_log_entry(realisation_file)

    # Optionally: Check the file content (if not mocking write_to_realisation)
    assert realisation_file.exists()
    with open(realisation_file, "r") as f:
        data = json.load(f)
    assert "log_trail" in data
    assert (
        "other_key" in data
    )  # Check if update worked (depends on write_to_realisation mock/impl)
    assert len(data["log_trail"]["log"]) == 1
    assert data["log_trail"]["log"][0]["utility"] == "script_name.py"


def test_seeds() -> None:
    seeds = realisations.Seeds.random_seeds()
    assert all(
        0 <= seed <= 2 ** (struct.Struct("i").size * 8 - 1) - 1
        for seed in seeds.to_dict().values()
    )


def test_seeds_from_random(tmp_path: Path) -> None:
    """Test read from realisation or random"""
    realisation_file = tmp_path / "test_realisation.json"
    # Create a file with unrelated content
    initial_content = {"other_key": "some_value"}
    with open(realisation_file, "w") as f:
        json.dump(initial_content, f)

    seeds = realisations.Seeds.read_from_realisation_or_random(realisation_file)
    assert all(
        0 <= seed <= 2 ** (struct.Struct("i").size * 8 - 1) - 1
        for seed in seeds.to_dict().values()
    )

    realisation_file = tmp_path / "test_realisation_1.json"
    # Create a file with unrelated content
    seeds = realisations.Seeds(
        nshm_to_realisation_seed=0,
        rupture_propagation_seed=0,
        genslip_seed=0,
        srfgen_seed=0,
        hf_seed=0,
    )
    seeds.write_to_realisation(realisation_file)

    seeds_read = realisations.Seeds.read_from_realisation_or_random(realisation_file)
    assert seeds == seeds_read


def test_velocity_model_1d(tmp_path: Path) -> None:
    velocity_model_1d = realisations.VelocityModel1D(
        model=pd.DataFrame(
            {
                "thickness": [0.1, 10.0, 20.0],
                "Vp": [1500.0, 2000.0, 2500.0],
                "Vs": [800.0, 1000.0, 1200.0],
                "Qp": [1500.0, 2000.0, 2500.0],
                "Qs": [800.0, 1000.0, 1200.0],
                "rho": [1800.0, 2000.0, 2200.0],
            }
        )
    )
    realisation_ffp = tmp_path / "realisation.json"
    velocity_model_1d.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "velocity_model_1d": {
                "model": [
                    {
                        "thickness": 0.1,
                        "Vp": 1500.0,
                        "Vs": 800.0,
                        "Qp": 1500.0,
                        "Qs": 800.0,
                        "rho": 1800.0,
                    },
                    {
                        "thickness": 10.0,
                        "Vp": 2000.0,
                        "Vs": 1000.0,
                        "Qp": 2000.0,
                        "Qs": 1000.0,
                        "rho": 2000.0,
                    },
                    {
                        "thickness": 20.0,
                        "Vp": 2500.0,
                        "Vs": 1200.0,
                        "Qp": 2500.0,
                        "Qs": 1200.0,
                        "rho": 2200.0,
                    },
                ]
            }
        }
    assert (
        realisations.VelocityModel1D.read_from_realisation(realisation_ffp).to_dict()
        == velocity_model_1d.to_dict()
    )


def test_intensity_measure_calculation_parameters(tmp_path: Path) -> None:
    im_calc_params = realisations.IntensityMeasureCalculationParameters(
        ims=[im_calculation.IM("PGA"), im_calculation.IM("PGV")],
        valid_periods=np.array([0.1, 0.2, 0.3]),
        fas_frequencies=np.array([0.5, 1.0, 2.0]),
    )
    realisation_ffp = tmp_path / "realisation.json"
    im_calc_params.write_to_realisation(realisation_ffp)
    with open(realisation_ffp, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "im": {
                "ims": ["PGA", "PGV"],
                "valid_periods": [0.1, 0.2, 0.3],
                "fas_frequencies": [0.5, 1.0, 2.0],
            }
        }
    assert (
        realisations.IntensityMeasureCalculationParameters.read_from_realisation(
            realisation_ffp
        ).to_dict()
        == im_calc_params.to_dict()
    )


def test_resolution(tmp_path: Path) -> None:
    resolution = realisations.Resolution(resolution=0.1)
    assert resolution.dt == 0.005

    resolution_200m = realisations.Resolution(resolution=0.2)
    assert resolution_200m.dt == 0.01

    resolution_400m = realisations.Resolution(resolution=0.4)
    assert resolution_400m.dt == 0.02

    realisation_path = tmp_path / "realisation.json"

    resolution.write_to_realisation(realisation_path)
    with open(realisation_path, "r") as realisation_handle:
        assert json.load(realisation_handle) == {"resolution": {"resolution": 0.1}}

    assert realisations.Resolution.read_from_realisation(realisation_path) == resolution


def test_refinements(tmp_path: Path) -> None:
    refinements = realisations.Refinements(
        refinements=[
            realisations.Refinement(resolution=50.0, bottom=2000.0),
            realisations.Refinement(resolution=100.0, bottom=5000.0),
            realisations.Refinement(resolution=200.0, bottom=25000.0),
        ],
        unbounded_refinement_resolution=400.0,
    )

    realisation_path = tmp_path / "realisation.json"
    refinements.write_to_realisation(realisation_path)
    with open(realisation_path, "r") as realisation_handle:
        assert json.load(realisation_handle) == {
            "refinements": {
                "refinements": [
                    {"resolution": 50.0, "bottom": 2000.0},
                    {"resolution": 100.0, "bottom": 5000.0},
                    {"resolution": 200.0, "bottom": 25000.0},
                ],
                "unbounded_refinement_resolution": 400.0,
            }
        }

    assert (
        realisations.Refinements.read_from_realisation(realisation_path) == refinements
    )


def test_refinements_defaults_loadable() -> None:
    """Refinements should load from v26_7_1Hz defaults and raise for older versions."""
    refinements = realisations.Refinements.read_from_defaults(defaults.DefaultsVersion.v26_7_1Hz)
    assert len(refinements.refinements) == 3
    assert refinements.refinements[0].resolution == 50.0
    assert refinements.refinements[0].bottom == 2000.0
    assert refinements.unbounded_refinement_resolution == 400.0

    for version in defaults.DefaultsVersion:
        if version == defaults.DefaultsVersion.v26_7_1Hz:
            continue
        with pytest.raises(realisations.RealisationParseError):
            realisations.Refinements.read_from_defaults(version)


def test_sources(tmp_path: Path) -> None:
    realisation_ffp = tmp_path / "realisation.json"
    source_json = {
        "sources": {
            "source_geometries": {
                "2016p661400": {
                    "type": "fault",
                    "corners": [
                        {
                            "latitude": -36.86797068168705,
                            "longitude": 179.27706552534542,
                            "depth": 12615.079012268625,
                        },
                        {
                            "latitude": -36.96567964889889,
                            "longitude": 179.24208519658814,
                            "depth": 12615.079012268625,
                        },
                        {
                            "latitude": -36.94379957660565,
                            "longitude": 179.1474674643655,
                            "depth": 13384.920987731375,
                        },
                        {
                            "latitude": -36.84610827130364,
                            "longitude": 179.18256725416285,
                            "depth": 13384.920987731375,
                        },
                    ],
                }
            }
        }
    }
    realisation_ffp = tmp_path / "realisation_expected.json"
    with open(realisation_ffp, "w") as f:
        json.dump(source_json, f)

    sources = SourceConfig.read_from_realisation(realisation_ffp)
    realisation_generated_ffp = tmp_path / "realisation_generated.json"
    sources.write_to_realisation(realisation_generated_ffp)

    with (
        open(realisation_ffp, "r") as f_old,
        open(realisation_generated_ffp, "r") as f_new,
    ):
        assert json.load(f_old) == json.load(f_new)


@pytest.mark.parametrize(
    "realisation_config",
    [
        realisations.EMOD3DParameters,
        realisations.HFConfig,
        realisations.SRFConfig,
        realisations.VelocityModelParameters,
        realisations.BroadbandParameters,
        realisations.VelocityModel1D,
        realisations.IntensityMeasureCalculationParameters,
        realisations.HFVelocityModel1D,
        realisations.Resolution,
        realisations.RuptureVelocity,
    ],
)
@pytest.mark.parametrize("defaults_version", list(defaults.DefaultsVersion))
def test_defaults_are_loadable(
    tmp_path: Path,
    realisation_config: realisations.RealisationConfiguration,
    defaults_version: defaults.DefaultsVersion,
) -> None:
    realisation_config.read_from_defaults(defaults_version)
