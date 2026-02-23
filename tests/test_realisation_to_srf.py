from pathlib import Path

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
        rvfrac_seg=[],
        seg_delay=False,
        slip_sigma=1.0,
        risetime_coef=1.6,
        ymag_exp=None,
        xmag_exp=1.0,
        kx_corner=None,
        ky_corner=None,
    )
    genslip_path = Path("genslip_v5.6.2")
    gsf_path = Path("/tmp/fault.gsf")
    vel_path = Path("/tmp/velocity.vm")
    rupture_velocity = RuptureVelocity(
        rvfrac=1.0,
        rvfrac_shal=0.6,
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
        "ns=1",
        "write_srf=1",
        "write_gsf=0",
        "resolution=0.1",
        "nh=1",
        "read_erf=0",
        "plane_header=1",
        "srf_version=1.0",
        "read_gsf=1",
        "nstk=50",
        "ndip=25",
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
        "xmag_exp=1.0",
        "rvfrac=1.0",
        "shal_vrup=0.6",
        "shal_vrup_dep=15.0",
        "shal_vrup_deprange=5.0",
        "deep_vrup=0.7",
        "deep_vrup_dep=20.0",
        "deep_vrup_deprange=2.5",
    }
