import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from source_modelling.moment import BoldM
from source_modelling.sources import Fault, Plane, Point
from workflow import schemas
from workflow.realisations import (
    Magnitudes,
    Rakes,
    RupturePropagationConfig,
    RuptureVelocity,
    Seeds,
    SourceConfig,
    SRFConfig,
    VelocityModel1D,
)
from workflow.scripts import realisation_to_srf


@pytest.fixture
def srf_config() -> SRFConfig:
    return SRFConfig(
        resolution=0.1,
        dt=0.005,
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
        xmag_exponent=1.0,
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
        # Updated IO settings
        read_erf=False,
        read_gsf=True,
        srf_version="2.0",
        write_gsf=False,
        write_srf=True,
        dump_last_seed=False,
        print_command=False,
        print_seed=False,
    )


@pytest.fixture
def rupture_velocity() -> RuptureVelocity:
    return RuptureVelocity(
        rvfrac=1.0,
        rvfrac_shal=0.6,
        rvfrac_slip_sig=None,
        rvfrac_deep=0.7,
        shallow_depth=15.0,
        shallow_transition_range=5.0,
        deep_depth=20.0,
        deep_transition_range=2.5,
    )


def test_build_genslip_command_static_args(
    srf_config: SRFConfig, rupture_velocity: RuptureVelocity
) -> None:
    genslip_path = Path("genslip_v5.6.2")
    gsf_path = Path("/tmp/fault.gsf")
    vel_path = Path("/tmp/velocity.vm")
    cmd = realisation_to_srf._build_genslip_command(
        genslip_path=genslip_path,
        gsf_file_path=gsf_path,
        nx=50,
        ny=25,
        seed=999,
        velocity_model_path=vel_path,
        shypo=10.5,
        dhypo=20.5,
        magnitude=7.8,
        dt=0.01,
        srf_config=srf_config,
        rupture_velocity=rupture_velocity,
    )

    assert cmd[0] == str(genslip_path)

    args = set(cmd[1:])

    assert args == {
        f"infile={gsf_path}",
        f"velfile={vel_path}",
        "write_srf=1",
        "write_gsf=0",
        "resolution=0.1",
        "read_erf=0",
        "srf_version=2.0",
        "read_gsf=1",
        "nstk=50",
        "ndip=25",
        "nh=1",
        "ns=1",
        "seed=999",
        "shypo=10.5",
        "dhypo=20.5",
        "mag=7.8",
        "dt=0.01",
        "side_taper=0.02",
        "bot_taper=0.02",
        "top_taper=0.0",
        "alpha_rough=0.0",
        "seg_delay=0",
        "slip_sigma=1.0",
        "risetime_coef=1.6",
        "xmag_exponent=1.0",
        "rvfrac=1.0",
        "shal_vrup=0.6",
        "shal_vrup_dep=15.0",
        "shal_vrup_deprange=5.0",
        "deep_vrup=0.7",
        "deep_vrup_dep=20.0",
        "deep_vrup_deprange=2.5",
        "beta_asp=0.3",
        "beta_deep=0.13",
        "beta_mid=0.13",
        "beta_mid_depth=6.5",
        "beta_mid_depth_range=1.5",
        "beta_shal=0.5",
        "beta_shal_depth=2.0",
        "beta_shal_depth_range=1.0",
        "beta_subevt=0.1",
        "deep_risetimedep=17.5",
        "deep_risetimedep_range=2.5",
        "deep_risetimefac=2.0",
        "risetimedep=6.5",
        "risetimedep_range=1.5",
        "risetimefac=2.0",
        "rt_rand=0.0",
        "rt_scalefac=1.0",
        "hyb_corlen_deep_wt_end=1.0",
        "hyb_corlen_deep_wt_start=0.0",
        "hyb_corlen_dep=6.5",
        "hyb_corlen_dep_range=1.5",
        "hyb_corlen_fac=2.0",
        "hyb_corlen_flag=0",
        "hyb_corlen_kmodel=5",
        "hyb_corlen_shal_wt_end=0.0",
        "hyb_corlen_shal_wt_start=1.0",
        "hyb_corlen_side_taper=0.08",
        "fdrup_scale_slip=0",
        "fdrup_time=0",
        "rupture_delay=0.0",
        "rvfmax=1.414",
        "rvfmin=0.25",
        "truncate_zero_slip=1",
        "rake_sigma=15.0",
        "fractal_rake=0",
        "tsfac1_scor=0.8",
        "tsfac1_sigma=1.0",
        "tsfac2_lambda_max=5.0",
        "tsfac2_scor=0.5",
        "tsfac2_sigma=1.0",
        "tsfac_bzero=-0.1",
        "tsfac_coef=1.1",
        "tsfac_slope=-0.5",
        "circular_average=0",
        "kmodel=2",
        "kord=4",
        "magC=6.3",
        "mai_wt=0.5",
        "modified_corners=0",
        "somerville_wt=0.5",
        "stretch_kcorner=0",
        "use_gaus=1",
        "use_median_mag=0",
        "asp_taper_fac=0.05",
        "perturb_subfault_location=1",
        "rand_rake_degs=60.0",
        "rtime1_depth=2.0",
        "rtime1_depth_range=1.0",
        "rtime1_scor=0.8",
        "rtime1_sigma=0.85",
        "rtime2_scor=0.5",
        "rtime2slip_exp=0.5",
        "svr_wt=0.0",
        "use_Mw=1",
        "aseis_flag=0",
        "aseis_smooth=0",
        "aseis_dep=10.0",
        "xshift=0.0",
        "yshift=0.0",
        "dump_last_seed=0",
        "print_command=0",
        "print_seed=0",
    }


def _environment(work_directory: Path) -> realisation_to_srf.SRFEnvironmentContext:
    return realisation_to_srf.SRFEnvironmentContext(
        genslip_path=Path("genslip"),
        generic_slip2srf_path=Path("generic_slip2srf"),
        work_directory=work_directory,
        seeds=Seeds(
            nshm_to_realisation_seed=1,
            rupture_propagation_seed=2,
            genslip_seed=3,
            srfgen_seed=4,
            hf_seed=5,
        ),
    )


def test_generate_fault_srf_reraises_on_genslip_failure(
    tmp_path: Path, srf_config: SRFConfig, rupture_velocity: RuptureVelocity
) -> None:
    """Regression test for #164.

    The genslip failure handler used to call ``e.output.decode("utf-8")``, but
    genslip's stdout is written straight to the SRF file handle, so
    ``e.output`` is always None and the handler raised AttributeError before
    it could log genslip's stderr or re-raise the original error.
    """
    name = "fault"
    plane = Plane(
        np.array(
            [
                [1578000.0, 5180000.0, 0.0],
                [1579000.0, 5180000.0, 0.0],
                [1579000.0, 5180000.0, 5000.0],
                [1578000.0, 5180000.0, 5000.0],
            ]
        )
    )
    fault = Fault(planes=[plane])

    params = realisation_to_srf.SRFRealisationContext(
        source_config=SourceConfig({name: fault}),
        rupture_propagation_config=RupturePropagationConfig(
            rupture_causality_tree={name: None},
            jump_points={},
            hypocentre=np.array([0.5, 0.0]),
        ),
        magnitudes=Magnitudes({name: BoldM(7.0)}),
        rakes=Rakes({name: 180.0}),
        velocity_model_1d=VelocityModel1D(pd.DataFrame({"thickness": [1.0, 2.0]})),
        srf_config=srf_config,
        rupture_velocity=rupture_velocity,
    )
    environment = _environment(tmp_path)
    environment.srf_directory.mkdir()

    error = subprocess.CalledProcessError(
        returncode=3, cmd=["genslip"], output=None, stderr=b"genslip stderr boom"
    )

    with (
        patch.object(
            realisation_to_srf,
            "generate_fault_gsf",
            return_value=tmp_path / "fault.gsf",
        ),
        patch.object(realisation_to_srf.subprocess, "run", side_effect=error),
        pytest.raises(subprocess.CalledProcessError),
    ):
        realisation_to_srf.generate_fault_srf(name, params, environment)


def test_generate_point_source_srf_reraises_on_generic_slip2srf_failure(
    tmp_path: Path, srf_config: SRFConfig, rupture_velocity: RuptureVelocity
) -> None:
    """Regression test for #164.

    Same bug as ``test_generate_fault_srf_reraises_on_genslip_failure``, but
    in the generic_slip2srf failure handler used for point sources.
    """
    name = "point"
    point = Point(
        bounds=np.array([1600000.0, 5180000.0, 5000.0]),
        length_m=1000.0,
        width_m=1000.0,
        strike=90.0,
        dip=45.0,
        dip_dir=180.0,
    )

    params = realisation_to_srf.SRFRealisationContext(
        source_config=SourceConfig({name: point}),
        rupture_propagation_config=RupturePropagationConfig(
            rupture_causality_tree={name: None},
            jump_points={},
            hypocentre=np.array([0.5, 0.0]),
        ),
        magnitudes=Magnitudes({name: BoldM(7.0)}),
        rakes=Rakes({name: 180.0}),
        velocity_model_1d=VelocityModel1D(pd.DataFrame({"thickness": [1.0, 2.0]})),
        srf_config=srf_config,
        rupture_velocity=rupture_velocity,
    )
    environment = _environment(tmp_path)

    error = subprocess.CalledProcessError(
        returncode=3,
        cmd=["generic_slip2srf"],
        output=None,
        stderr=b"generic_slip2srf stderr boom",
    )

    with (
        patch.object(
            realisation_to_srf,
            "generate_fault_gsf",
            return_value=tmp_path / "point.gsf",
        ),
        patch.object(
            realisation_to_srf.moment, "magnitude_to_moment", return_value=1e17
        ),
        patch.object(realisation_to_srf.moment, "point_source_slip", return_value=1.0),
        patch.object(realisation_to_srf.subprocess, "run", side_effect=error),
        pytest.raises(subprocess.CalledProcessError),
    ):
        realisation_to_srf.generate_point_source_srf(name, params, environment)
