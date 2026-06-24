from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from workflow import schemas
from workflow.realisations import RuptureVelocity, SRFConfig
from workflow.scripts import realisation_to_srf


def test_build_genslip_command_static_args() -> None:
    srf_config = SRFConfig(
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
    genslip_path = Path("genslip_v5.6.2")
    gsf_path = Path("/tmp/fault.gsf")
    vel_path = Path("/tmp/velocity.vm")
    rupture_velocity = RuptureVelocity(
        rvfrac=1.0,
        rvfrac_shal=0.6,
        rvfrac_slip_sig=None,
        rvfrac_deep=0.7,
        shallow_depth=15.0,
        shallow_transition_range=5.0,
        deep_depth=20.0,
        deep_transition_range=2.5,
    )
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


def test_velocity_model_vs_den() -> None:
    velocity_model_df = pd.DataFrame(
        {
            "thickness": [3.0, 5.0, 5.0, 5.0, 100.0],
            "Vs": [0.73, 1.57, 2.91, 3.64, 4.18],
            "rho": [1.93, 2.34, 2.76, 3.11, 3.42],
        }
    )
    velocity_model_df["depth_km"] = (
        velocity_model_df["thickness"].cumsum() - velocity_model_df["thickness"]
    )

    # Layer tops: [0, 3, 8, 13, 18] km. vs is cm/s (Vs km/s * 1e5); den is g/cm^3 unchanged.
    # An exact-boundary depth (8.0) takes the deeper layer, matching point_source_slip.
    depths_km = np.array([0.0, 5.0, 8.0, 8.06, 25.0])
    vs, den = realisation_to_srf._velocity_model_vs_den(velocity_model_df, depths_km)

    np.testing.assert_allclose(vs, [0.73e5, 1.57e5, 2.91e5, 2.91e5, 4.18e5])
    np.testing.assert_allclose(den, [1.93, 2.34, 2.76, 2.76, 3.42])


def test_rewrite_point_source_srf_as_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    points = pd.DataFrame(
        [
            {
                "lon": 172.8,
                "lat": -43.5,
                "dep": 8.06,
                "stk": 64.0,
                "dip": 58.0,
                "area": 1.0e8,
                "tinit": 0.0,
                "dt": 0.02,
                "rake": 131.0,
                "slip": 12.5,
                "rise": 0.5,
            }
        ]
    )
    fake_srf = SimpleNamespace(points=points, version="1.0")
    monkeypatch.setattr(realisation_to_srf.srf, "read_srf", lambda _ffp: fake_srf)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        realisation_to_srf.srf,
        "write_srf",
        lambda ffp, srf_file: captured.update(ffp=ffp, srf_file=srf_file),
    )

    velocity_model_df = pd.DataFrame(
        {
            "thickness": [3.0, 5.0, 5.0, 5.0, 100.0],
            "Vs": [0.73, 1.57, 2.91, 3.64, 4.18],
            "rho": [1.93, 2.34, 2.76, 3.11, 3.42],
        }
    )
    velocity_model_df["depth_km"] = (
        velocity_model_df["thickness"].cumsum() - velocity_model_df["thickness"]
    )

    realisation_to_srf._rewrite_point_source_srf_as_v2(
        Path("unused.srf"), velocity_model_df
    )

    written = captured["srf_file"]
    assert isinstance(written, SimpleNamespace)
    assert written.version == "2.0"
    assert list(written.points.columns) == [
        "lon",
        "lat",
        "dep",
        "stk",
        "dip",
        "area",
        "tinit",
        "dt",
        "vs",
        "den",
        "rake",
        "slip",
        "rise",
    ]
    assert written.points["vs"].iloc[0] == pytest.approx(2.91e5)
    assert written.points["den"].iloc[0] == pytest.approx(2.76)
