"""
This script takes converts one realisation CSV file to an SRF output file.

The parameters that are read from the CSV file are documented on the
[wiki](https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM).
"""

import argparse
import os
import subprocess
from logging import Logger
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, TextIO, Tuple, Union

import h5py
import numpy as np
import yaml
from srf_generation import pre_processing_common
from srf_generation.source_parameter_generation.common import (
    DEFAULT_1D_VELOCITY_MODEL_PATH,
)
from srf_generation.source_parameter_generation.uncertainties.common import (
    BB_RUN_PARAMS,
    HF_RUN_PARAMS,
    LF_RUN_PARAMS,
)

from qcore import binary_version, geo, qclogging, srf, utils
from qcore.uncertainties import mag_scaling
from qcore.uncertainties.mag_scaling import MagnitudeScalingRelations

SRF_SUBFAULT_SIZE_KM = 0.1

SRF2STOCH = "srf2stoch"
GENERICSLIP2SRF = "generic_slip2srf"
FAULTSEG2GSFDIPDIR = "fault_seg2gsf_dipdir"

CORNERS_HEADER = (
    """> header line here for specifics
> This is the standard input file format where the hypocenter is first then for each
> Hypocenter (reference??) """,
    "> Below are the corners (first point repeated as fifth to close box)\n",
)


def number_subdivisions(fault_size: float, subdivision_size: float) -> int:
    """
    Calculate the number of meshgrid subdivisions for each fault plane.

    Parameters
    ----------
    fault_size : float
        The size of the fault.
    subdivision_size : float
        The size of each subgrid

    Returns
    -------
    int
        The number of subdivisions
    """

    return round(fault_size / subdivision_size)


def create_stoch(
    stoch_file: str,
    srf_file: str,
    single_segment: bool = False,
    logger: Logger = qclogging.get_basic_logger(),
):
    """Create a stoch file from a SRF file.

    Parameters
    ----------
    stoch_file : str
        The filepath to output the stoch file to.
    srf_file : str
        The filepath of the SRF file.
    single_segment : bool
        True if the stoch file is a single segment.
    logger : Logger
        Optional alternative logger for log output.
    """

    logger.debug("Generating stoch file")
    out_dir = os.path.dirname(stoch_file)
    os.makedirs(out_dir, exist_ok=True)
    dx, dy = 2.0, 2.0
    if not srf.is_ff(srf_file):
        dx, dy = srf.srf_dxy(srf_file)
    logger.debug(f"Saving stoch to {stoch_file}")
    srf2stoch = binary_version.get_unversioned_bin(SRF2STOCH)
    if single_segment:
        command = [srf2stoch, f"target_dx={dx}", f"target_dy={dy}"]
    else:
        command = [srf2stoch, f"dx={dx}", f"dy={dy}"]
    command.extend([f"infile={srf_file}", f"outfile={stoch_file}"])
    logger.debug(f"Creating stoch with command: {command}")
    proc = subprocess.run(command, stderr=subprocess.PIPE, check=True)
    logger.debug(f"{srf2stoch} stderr: {proc.stderr}")


def get_corners_dbottom(
    planes: List[Dict[str, Any]], dip_dir: Union[str, None] = None
) -> Tuple[np.ndarray, List[float]]:
    """Get projected bottom corners of the planes (used for info file output).

    Parameters
    ----------
    planes : List[Dict[str, Any]]
        A list of dictionaries where each dictionary is structured like below.
        ```
            {
                "centre": [float(elon), float(elat)],
                "nstrike": int(nstk),
                "ndip": int(ndip),
                "length": float(ln),
                "width": float(wid),
                "strike": stk,
                "dip": dip,
                "shyp": shyp,
                "dhyp": dhyp,
                "dtop": dtop,
            }
        ```
    """

    dbottom = []
    corners = np.zeros((len(planes), 4, 2))
    for i, p in enumerate(planes):
        # currently only support single dip dir value
        if dip_dir is not None:
            dip_deg = dip_dir
        else:
            dip_deg = p["strike"] + 90

        # projected fault width (along dip direction)
        pwid = p["width"] * np.cos(np.radians(p["dip"]))
        corners[i, 0] = geo.ll_shift(
            p["centre"][1], p["centre"][0], p["length"] / 2.0, p["strike"] + 180
        )[::-1]
        corners[i, 1] = geo.ll_shift(
            p["centre"][1], p["centre"][0], p["length"] / 2.0, p["strike"]
        )[::-1]
        corners[i, 2] = geo.ll_shift(corners[i, 1, 1], corners[i, 1, 0], pwid, dip_deg)[
            ::-1
        ]
        corners[i, 3] = geo.ll_shift(corners[i, 0, 1], corners[i, 0, 0], pwid, dip_deg)[
            ::-1
        ]
        dbottom.append(p["dtop"] + p["width"] * np.sin(np.radians(p["dip"])))

    return (corners, dbottom)


def create_info_file(
    srf_file: str,
    srf_type: int,
    mag: float,
    rake: float,
    dt: float,
    vm: Union[float, None] = None,
    vs: Union[float, None] = None,
    rho: Union[float, None] = None,
    centroid_depth: Union[float, None] = None,
    lon: Union[float, None] = None,
    lat: Union[float, None] = None,
    shypo: Union[float, None] = None,
    dhypo: Union[float, None] = None,
    mwsr: Union[MagnitudeScalingRelations, None] = None,
    tect_type: Union[str, None] = None,
    dip_dir: Union[str, None] = None,
    file_name: Union[str, None] = None,
    logger: Logger = qclogging.get_basic_logger(),
):
    """Store SRF metadat as hdf5.


    Parameters
    ----------
    srf_file : str
        SRF path used as basename for info file and additional metadata.
    srf_type : int
        Realisation type (e.g. 1, 2, 3, or 4). Refer to wiki for details.
    mag : float                                   |
    rake : float                                  |
    dt : float                                    |
    vm : Union[float, None]                       |
    vs : Union[float, None]                       |
    rho : Union[float, None]                      |
    centroid_depth : Union[float, None]           | Refer to wiki (see module documentation for link).
    lon : Union[float, None]                      |
    lat : Union[float, None]                      |
    shypo : Union[float, None]                    |
    dhypo : Union[float, None]                    |
    mwsr : Union[MagnitudeScalingRelations, None] |
    tect_type : Union[str, None]                  |
    dip_dir : Union[str, None]                    |
    file_name : Union[str, None]
        File path to save metadata.
    logger : Logger
        Optional alternative logger for log output.
    """
    logger.debug("Generating srf info file")
    planes = srf.read_header(srf_file, idx=True)
    hlon, hlat, hdepth = srf.get_hypo(srf_file, depth=True)

    corners, dbottom = get_corners_dbottom(planes, dip_dir=dip_dir)

    if file_name is None:
        file_name = srf_file.replace(".srf", ".info")
    logger.debug(f"Saving info file to {file_name}")
    with h5py.File(file_name, "w") as h:
        a = h.attrs
        # only taken from given parameters
        a["type"] = srf_type
        a["dt"] = dt
        a["rake"] = rake
        a["mag"] = mag
        # srf header data
        for k in planes[0].keys():
            a[k] = [p[k] for p in planes]
        # point source has 1 vs/rho, others are taken from vm
        if srf_type == 1:
            a["vs"] = vs
            a["rho"] = rho
            a["hlon"] = lon
            a["hlat"] = lat
            a["hdepth"] = centroid_depth
        else:
            a["vm"] = np.string_(os.path.basename(vm))
            a["hlon"] = hlon
            a["hlat"] = hlat
            a["hdepth"] = hdepth
            a["shyp0"] = shypo
            a["dhyp0"] = dhypo
        if mwsr is not None:
            a["mwsr"] = np.string_(mwsr)
        if tect_type is not None:
            a["tect_type"] = tect_type
        # derived parameters
        a["corners"] = corners
        a["dbottom"] = dbottom


def create_ps_srf(
    realisation_file: str,
    parameter_dictionary: Dict[str, Any],
    stoch_file: Union[None, str] = None,
    logger: Logger = qclogging.get_basic_logger,
):
    """Generate SRF file (point-source finite-fault modeling).

    Parameters
    ----------
    realisation_file : str
        Path to the realisation (a CSV file). The output files (srf, stoch, etc) are
        produced relative to this file path.
    parameter_dictionary : Dict[str, Any]
        Parameters of the realisation. See module documentation for the wiki
        describing these parameters.
    stoch_file : Union[None, str]
        An optional alternative location for the stoch file.
    logger : Logger
        Optional alternative logger for log output.
    """
    latitude = parameter_dictionary.pop("latitude")
    longitude = parameter_dictionary.pop("longitude")
    depth = parameter_dictionary.pop("depth")
    magnitude = parameter_dictionary.pop("magnitude")
    strike = parameter_dictionary.pop("strike")
    rake = parameter_dictionary.pop("rake")
    dip = parameter_dictionary.pop("dip")
    moment = parameter_dictionary.pop("moment", 0)
    dt = parameter_dictionary.pop("dt", 0.005)
    vs = parameter_dictionary.pop("vs", 3.20)
    rho = parameter_dictionary.pop("rho", 2.44)
    target_area_km = parameter_dictionary.pop("target_area_km", None)
    target_slip_cm = parameter_dictionary.pop("target_slip_cm", None)
    tect_type = parameter_dictionary.pop("tect_type", None)
    stype = parameter_dictionary.pop("stype", "cos")
    risetime = parameter_dictionary.pop("risetime", 0.5)
    inittime = parameter_dictionary.pop("inittime", 0.0)

    # srfgen seed is not used by slip2srf (unless we pass rt_rand in),
    # but we also don't need it to be in the sim_params.yaml file
    parameter_dictionary.pop("srfgen_seed", None)

    name = parameter_dictionary.get("name")
    logger.info(f"Generating srf for realisation {name}")

    logger.debug(
        "All srf generation parameters successfully obtained from the realisation file"
    )

    if moment <= 0:
        logger.debug("moment is negative, calculating from magnitude")
        moment = mag_scaling.mag2mom(magnitude)

    # size (dd) and slip
    if target_area_km is not None:
        logger.debug(
            f"target_area_km given ({target_area_km}), using it to calculate fault edge length and slip"
        )
        dd = np.sqrt(target_area_km)
        slip = (moment * 1.0e-20) / (target_area_km * vs * vs * rho)
    elif target_slip_cm is not None:
        logger.debug(
            f"target_slip_cm given ({target_slip_cm}), using it to calculate fault edge length"
        )
        dd = np.sqrt(moment * 1.0e-20) / (target_slip_cm * vs * vs * rho)
        slip = target_slip_cm
    else:
        if tect_type is not None and tect_type == "SUBDUCTION_INTERFACE":
            aa = mag_scaling.mw_to_a_skarlatoudis(magnitude)
        else:
            # Shallow crustal and subduction slab (currently) use Leonard moment-area scaling relation
            aa = np.exp(2.0 / 3.0 * np.log(moment) - 14.7 * np.log(10.0))
        dd = np.sqrt(aa)
        slip = (moment * 1.0e-20) / (aa * vs * vs * rho)

    logger.debug(f"Slip: {slip}, fault plane edge length {dd}")

    ###
    ### file names
    ###
    srf_file = realisation_file.replace(".csv", ".srf")
    logger.debug(f"Srf will be saved to {srf_file}")

    ###
    ### create GSF
    ###
    with NamedTemporaryFile(mode="w+", delete=False) as gsfp:
        gsfp.writelines(
            [
                "# nstk= 1 ndip= 1\n",
                f"# flen= {dd:10.4f} fwid= {dd:10.4f}\n",
                "# LON  LAT  DEP(km)  SUB_DX  SUB_DY  LOC_STK  LOC_DIP  LOC_RAKE  SLIP(cm)  INIT_TIME  SEG_NO\n",
                "1\n",
                f"{longitude:11.5f} {latitude:11.5f} {depth:8.4f} {dd:8.4f} {dd:8.4f} {strike:6.1f} {dip:6.1f} {rake:6.1f} {slip:8.2f} {inittime:8.3f}    0\n",
                "\n",
            ]
        )

    ###
    ### create SRF
    ###
    commands = [
        binary_version.get_unversioned_bin(GENERICSLIP2SRF),
        f"infile={gsfp.name}",
        f"outfile={srf_file}",
        "outbin=0",
        f"stype={stype}",
        f"dt={dt}",
        "plane_header=1",
        f"risetime={risetime}",
        "risetimefac=1.0",
        "risetimedep=0.0",
    ]
    subprocess.run(commands, stderr=subprocess.PIPE, check=True)

    ###
    ### save STOCH
    ###
    if stoch_file is None:
        stoch_file = realisation_file.replace(".csv", ".stoch")
    create_stoch(stoch_file, srf_file, single_segment=True)

    ###
    ### save INFO
    ###
    create_info_file(
        srf_file,
        1,
        magnitude,
        rake,
        dt,
        vs=vs,
        rho=rho,
        centroid_depth=depth,
        lon=longitude,
        lat=latitude,
        tect_type=tect_type,
    )
    logger.info(
        f"Generated srf for realisation {name}. Moving to next available realisation."
    )


def create_ps_ff_srf(
    realisation_file: str,
    parameter_dictionary: Dict[str, Any],
    stoch_file: Union[None, str] = None,
    logger: Logger = qclogging.get_basic_logger,
):
    """Generate SRF file (point-source finite-fault modeling).

    Parameters
    ----------
    realisation_file : str
        Path to the realisation (a CSV file). The output files (srf, stoch, etc) are
        produced relative to this file path.
    parameter_dictionary : Dict[str, Any]
        Parameters of the realisation. See module documentation for the wiki
        describing these parameters.
    stoch_file : Union[None, str]
        An optional alternative location for the stoch file.
    logger : Logger
        Optional alternative logger for log output.
    """

    name = parameter_dictionary.get("name")
    logger.info(f"Generating srf for realisation {name}")

    latitude = parameter_dictionary.pop("latitude")
    longitude = parameter_dictionary.pop("longitude")
    depth = parameter_dictionary.pop("depth")
    magnitude = parameter_dictionary.pop("magnitude")
    strike = parameter_dictionary.pop("strike")
    rake = parameter_dictionary.pop("rake")
    dip = parameter_dictionary.pop("dip")
    dt = parameter_dictionary.pop("dt", 0.005)
    flen = parameter_dictionary.pop("flen")
    dlen = parameter_dictionary.pop("dlen")
    fwid = parameter_dictionary.pop("fwid")
    dwid = parameter_dictionary.pop("dwid")
    dtop = parameter_dictionary.pop("dtop")
    shypo = parameter_dictionary.pop("shypo")
    dhypo = parameter_dictionary.pop("dhypo")
    genslip_version = str(parameter_dictionary.pop("genslip_version"))
    seed = parameter_dictionary.pop("srfgen_seed")
    slip_cov = parameter_dictionary.pop("slip_cov", None)
    risetime_coef = parameter_dictionary.pop("risetime_coef", None)
    rough = parameter_dictionary.pop("rough", 0.0)
    tect_type = parameter_dictionary.pop("tect_type", None)
    vel_mod_1d = parameter_dictionary.pop(
        "srf_vel_mod_1d", DEFAULT_1D_VELOCITY_MODEL_PATH
    )

    asperity_file = parameter_dictionary.pop("asperity_file", None)

    mwsr = MagnitudeScalingRelations(parameter_dictionary.pop("mwsr"))

    rvfac = parameter_dictionary.get("rvfac", None)

    logger.debug(
        "All srf generation parameters successfully obtained from the realisation file"
    )
    srf_file = realisation_file.replace(".csv", ".srf")
    logger.debug(f"Srf will be saved to {srf_file}")
    corners_file = realisation_file.replace(".csv", ".corners")
    logger.debug(f"The corners file will be saved to {corners_file}")

    logger.debug("Getting corners and hypocentre")
    corners = get_corners(latitude, longitude, flen, fwid, dip, strike)
    hypocentre = pre_processing_common.get_hypocentre(
        latitude, longitude, shypo, dhypo, strike, dip
    )
    logger.debug("Saving corners and hypocentre")
    write_corners(corners_file, hypocentre, corners)

    nx = number_subdivisions(flen, dlen)
    ny = number_subdivisions(fwid, dwid)

    with NamedTemporaryFile(mode="w", delete=False) as gsfp:
        gen_gsf(
            gsfp,
            longitude,
            latitude,
            dtop,
            strike,
            dip,
            rake,
            flen,
            fwid,
            nx,
            ny,
            logger=logger,
        )
        gen_srf(
            srf_file,
            gsfp.name,
            2,
            magnitude,
            dt,
            nx,
            ny,
            seed,
            shypo,
            dhypo,
            vel_mod_1d,
            genslip_version=genslip_version,
            rvfac=rvfac,
            slip_cov=slip_cov,
            risetime_coef=risetime_coef,
            rough=rough,
            logger=logger,
            tect_type=tect_type,
            asperity_file=asperity_file,
        )

    if stoch_file is None:
        stoch_file = realisation_file.replace(".csv", ".stoch")
    create_stoch(stoch_file, srf_file, single_segment=True, logger=logger)

    create_info_file(
        srf_file,
        2,
        magnitude,
        rake,
        dt,
        centroid_depth=depth,
        lon=None,
        lat=None,
        tect_type=None,
        dip_dir=None,
        mwsr=mwsr,
        shypo=shypo + 0.5 * flen,
        dhypo=dhypo,
        vm=vel_mod_1d,
        logger=logger,
    )
    logger.info(
        f"Generated srf for realisation {name}. Moving to next available realisation."
    )


def create_multi_plane_srf(
    realisation_file: str,
    parameter_dictionary: Dict[str, Any],
    stoch_file: Union[None, str] = None,
    logger: Logger = qclogging.get_basic_logger(),
):
    """Generate SRF file (multi-plane source modeling)

    Parameters
    ----------
    realisation_file : str
        Path to the realisation (a CSV file). The output files (srf, stoch, etc) are
        produced relative to this file path.
    parameter_dictionary : Dict[str, Any]
        Parameters of the realisation. See module documentation for the wiki
        describing these parameters.
    stoch_file : Union[None, str]
        An optional alternative location for the stoch file.
    logger : Logger
        Optional alternative logger for log output.

    Raises
    ------
    RuntimeError
        If the genslip version is incorrect for the fault type.

    """
    name = parameter_dictionary.get("name")
    rel_logger = qclogging.get_realisation_logger(logger, name)
    rel_logger.info(f"Generating srf for realisation {name}")

    magnitude = parameter_dictionary.pop("magnitude")
    moment = parameter_dictionary.pop("moment")
    rake = parameter_dictionary.pop("rake")
    dip = parameter_dictionary.pop("dip")
    dt = parameter_dictionary.pop("dt", 0.005)
    dtop = parameter_dictionary.pop("dtop")
    length = parameter_dictionary.pop("length")
    shypo = parameter_dictionary.pop("shypo")
    dhypo = parameter_dictionary.pop("dhypo")
    genslip_version = str(parameter_dictionary.pop("genslip_version"))
    dip_dir = parameter_dictionary.pop("dip_dir")
    seed = parameter_dictionary.pop("srfgen_seed")
    tect_type = parameter_dictionary.pop("tect_type")
    fault_type = parameter_dictionary.pop("fault_type")
    plane_count = parameter_dictionary.pop("plane_count")

    rough = parameter_dictionary.pop("rough", 0.0)
    slip_cov = parameter_dictionary.pop("slip_cov", None)

    asperity_file = parameter_dictionary.pop("asperity_file", None)

    strike = [
        parameter_dictionary.pop(f"strike_subfault_{i}") for i in range(plane_count)
    ]
    flen = [
        parameter_dictionary.pop(f"length_subfault_{i}") for i in range(plane_count)
    ]
    fwid = [parameter_dictionary.pop(f"width_subfault_{i}") for i in range(plane_count)]

    clon = [parameter_dictionary.pop(f"clon_subfault_{i}") for i in range(plane_count)]
    clat = [parameter_dictionary.pop(f"clat_subfault_{i}") for i in range(plane_count)]

    for key in list(parameter_dictionary.keys()):
        # Remove all the subfault keys so they don't get passed to the sim_params.yaml file
        # Must make a copy of the keys to be able to remove them.
        if "_subfault_" in key:
            parameter_dictionary.pop(key)

    vel_mod_1d = parameter_dictionary.pop(
        "srf_vel_mod_1d", DEFAULT_1D_VELOCITY_MODEL_PATH
    )
    rvfac = parameter_dictionary.get("rvfac", None)

    logger.debug(
        "All srf generation parameters successfully obtained from the realisation file"
    )

    dlen = dwid = SRF_SUBFAULT_SIZE_KM

    nx = [number_subdivisions(flen[i], dlen) for i in range(plane_count)]
    ny = number_subdivisions(fwid[0], dwid)

    if (
        utils.compare_versions(genslip_version, "5.4.2") < 0
        and tect_type == "SUBDUCTION_INTERFACE"
    ):
        raise RuntimeError(
            "Subduction interface faults are only available for version 5.4.2 and above"
        )

    srf_file = realisation_file.replace(".csv", ".srf")
    rel_logger.debug(f"Srf will be saved to {srf_file}")
    corners_file = realisation_file.replace(".csv", ".corners")
    rel_logger.debug(f"The corners file will be saved to {corners_file}")

    fault_seg_bin = binary_version.get_unversioned_bin(FAULTSEG2GSFDIPDIR)
    with (
        NamedTemporaryFile(mode="w", delete=False) as gsfp,
        NamedTemporaryFile(mode="w", delete=False) as gsf_file,
    ):
        rel_logger.debug(f"Saving segments file to {gsfp.name}")
        gsfp.write(f"{plane_count}\n")
        for f in range(plane_count):
            gsfp.write(
                f"{clon[f]:6f} {clat[f]:6f} {dtop:6f} {strike[f]:.4f} {dip:.4f} {rake:.4f} {flen[f]:.4f} {fwid[f]:.4f} {nx[f]:d} {ny:d}\n"
            )
        # NOTE: This flush call is vital. It ensures that the input file for fault_seg_bin actually contains input for fault_seg_bin to read.
        # without this, genslip will segfault later.
        gsfp.flush()
        rel_logger.debug(f"Gsf will be saved to the temporary file {gsf_file.name}")
        cmd = [
            fault_seg_bin,
            "read_slip_vals=0",
            f"infile={gsfp.name}",
            f"outfile={gsf_file.name}",
        ]
        if dip_dir is not None:
            cmd.append(f"dipdir={dip_dir}")

        rel_logger.info(f"Calling fault_seg2gsf_dipdir with command {' '.join(cmd)}")
        gexec = subprocess.run(cmd, check=True)
        rel_logger.debug(
            f"{fault_seg_bin} finished running with stderr: {gexec.stderr}"
        )
        if int(plane_count > 1):
            rel_logger.debug("Multiple segments detected. Generating xseg argument")
            flen_array = np.asarray(flen)
            xseg = flen_array.cumsum() - flen_array / 2
        else:
            xseg = [-1]

        gen_srf(
            srf_file,
            gsf_file.name,
            4,
            magnitude,
            dt,
            sum(nx),
            ny,
            seed,
            shypo,
            dhypo,
            vel_mod_1d,
            genslip_version=genslip_version,
            rvfac=rvfac,
            rough=rough,
            slip_cov=slip_cov,
            xseg=xseg,
            logger=rel_logger,
            tect_type=tect_type,
            asperity_file=asperity_file,
        )

    rel_logger.info("srf generated, creating stoch")

    if stoch_file is None:
        stoch_file = realisation_file.replace(".csv", ".stoch")
    create_stoch(stoch_file, srf_file, single_segment=(plane_count == 1), logger=logger)

    rel_logger.info("stoch created, making info")

    create_info_file(
        srf_file,
        4,
        magnitude,
        rake,
        dt,
        tect_type=tect_type,
        dip_dir=dip_dir,
        shypo=[shypo],
        dhypo=dhypo,
        vm=vel_mod_1d,
        logger=rel_logger,
    )

    logger.info(
        f"Generated srf for realisation {name}. Moving to next available realisation."
    )


def get_corners(
    lat: float, lon: float, flen: float, fwid: float, dip: float, strike: float
) -> np.ndarray:
    """Get the corners of a finite-fault plane generated from a point-source specified in the WGS84 coordinate system.

    Parameters
    ----------
    lat : float    |
    lon : float    |
    flen : float   | Refer to wiki (see module documentation for link).
    fwid : float   |
    dip : float    |
    strike : float |

    Returns
    -------
    np.ndarray
        A numpy array containing the corners of a fault. The values of the array are
        (lon, lat) pairs  representing coordinates in the WGS84 coordinate system.
        The indices 0, 1, 2, 3 correspond to the corners of the fault in the
        following fashion.

                    flen
              0 +------------+ 1
                |            |
                |            |
                |            | fwid
                |            |
              2 +------------+ 3
    """

    lats, lons = pre_processing_common.calculate_corners(
        dip,
        np.array([-flen / 2.0, flen / 2.0]),
        np.array([-fwid, 0.0]),
        lat,
        lon,
        strike,
    )

    # joined (lon, lat) pairs
    return np.dstack((lons.flat, lats.flat))[0]


def write_corners(filename: str, hypocentre: np.ndarray, corners: np.ndarray):
    """Write a corners text file (used to plot faults).

    The format of the corners file given a hypocentre [lonc, latc] and an array
    of corners [[lon0, lat0], ..., [lon3, lat3]] is as follows

    ```
    OUTPUT_PREHEADER # see CORNERS_HEADER[0]
    lonc latc
    OUTPUT_CORNERS_HEADER # see CORNERS_HEADER[1]
    lon0 lat0
    lon1 lat1
    lon2 lat2
    lon3 lat3
    lon0 lat0
    ```

    The "#"'s are informational comments for documentation and not written to the file.

    Parameters
    ----------
    filename : str
        The filepath to write corners to.
    hypocentre : np.ndarray
        The hypocentre of the eruption.
    corners : np.ndarray
        The bounds of the finite fault (as described in `get_corners`).

    """
    with open(filename, "w", encoding="utf-8") as cf:
        # lines beginning with '>' are ignored
        cf.write(CORNERS_HEADER[0])
        cf.write(f"{hypocentre[0]} {hypocentre[1]}\n")
        cf.write(CORNERS_HEADER[1])
        for i in [0, 1, 3, 2, 0]:
            cf.write(f"{corners[i, 0]:f} {corners[i, 1]:f}\n")


def gen_srf(
    srf_file: str,
    gsf_file: str,
    type: int,
    magnitude: float,
    dt: float,
    nx: float,
    ny: float,
    seed: int,
    shypo: float,
    dhypo: float,
    velocity_model: str,
    genslip_version: str = "3.3",
    rvfac: Union[float, None] = None,
    rough: float = 0.0,
    slip_cov: Union[float, None] = None,
    risetime_coef: Union[float, None] = None,
    tect_type: Union[str, None] = None,
    fault_planes: int = 1,
    asperity_file: Union[str, None] = None,
    xseg: List[float] = [-1],
    logger: Logger = qclogging.get_basic_logger(),
):
    """Wrapper around genslip, which actually generates the SRF files.

    The arguments to this function are validated and passed to genslip as
    command line arguments in a subprocess.

    Parameters
    ----------
    srf_file : str
        The output SRF file
    gsf_file : str
        The input gsf file
    type : int                         |
    magnitude : float                  |
    dt : float                         |
    nx : float                         |
    ny : float                         |
    seed : int                         |
    shypo : float                      |
    dhypo : float                      | Refer to wiki (see module documentation for link).
    velocity_model : str               |
    genslip_version : str              |
    rvfac : Union[float, None]         |
    rough : float                      |
    slip_cov : Union[float, None]      |
    risetime_coef : Union[float, None] |
    tect_type : Union[str, None]       |
    xseg : List[float]                 |
    fault_planes : int
        Number of fault planes.
    asperity_file : Union[str, None]
        Slip file location.
    logger : Logger
        Optional alternative logger for log output.

    Raises
    ------
    AssertionError
        If the genslip version is incorrect for the fault type


    """

    genslip_bin = binary_version.get_genslip_bin(genslip_version)
    if (
        utils.compare_versions(genslip_version, "5.4.2") < 0
        and tect_type == "SUBDUCTION_INTERFACE"
    ):
        raise AssertionError(
            "Cannot generate subduction srfs with genslip version less than 5.4.2"
        )
    if utils.compare_versions(genslip_version, "5") > 0:
        # Positive so version greater than 5
        logger.debug(
            f"Using genslip version {genslip_version}. Using nstk and ndip and rup_delay (for type 4)"
        )
        xstk = "nstk"
        ydip = "ndip"
        rup_name = "rup_delay"
    else:
        # Not positive so version at most 5
        logger.debug(
            f"Using genslip version {genslip_version}. Using nx and ny and rupture_delay (for type 4)"
        )
        xstk = "nx"
        ydip = "ny"
        rup_name = "rupture_delay"
    cmd = [
        genslip_bin,
        "read_erf=0",
        "write_srf=1",
        "read_gsf=1",
        "write_gsf=0",
        f"infile={gsf_file}",
        f"mag={magnitude}",
        f"{xstk}={nx}",
        f"{ydip}={ny}",
        "ns=1",
        "nh=1",
        f"seed={seed}",
        f"velfile={velocity_model}",
        f"shypo={shypo}",
        f"dhypo={dhypo}",
        f"dt={dt}",
        "plane_header=1",
        "srf_version=1.0",
    ]
    if type == 4:
        xseg_array = ",".join(str(seg) for seg in xseg)
        cmd.extend(
            [
                "seg_delay={0}",
                f"nseg={fault_planes}",
                f"nseg_bounds={fault_planes - 1}",
                f"xseg={xseg_array}",
                "rvfac_seg=-1",
                "gwid=-1",
                "side_taper=0.02",
                "bot_taper=0.02",
                "top_taper=0.0",
                f"{rup_name}=0",
            ]
        )

    if tect_type == "SUBDUCTION_INTERFACE":
        cmd.extend(
            [
                "kmodel=-1",
                "xmag_exp=0.5",
                "ymag_exp=0.5",
                "kx_corner=2.5482",
                "ky_corner=2.3882",
                "tsfac_slope=-0.5",
                "tsfac_bzero=-0.1",
            ]
        )
        if risetime_coef is None:
            cmd.append("risetime_coef=1.95")
    if rvfac is not None:
        cmd.append(f"rvfrac={rvfac}")
    if rough is not None:
        cmd.append(f"alpha_rough={rough}")
    if slip_cov is not None:
        cmd.append(f"slip_sigma={slip_cov}")
    if risetime_coef is not None:
        cmd.append(f"risetime_coef={risetime_coef}")
    if asperity_file is not None:
        cmd.append("read_slip_file=1")
        cmd.append(f"init_slip_file={asperity_file}")
    logger.debug(f"Creating SRF with command: {' '.join(cmd)}")
    with open(srf_file, "w", encoding="utf-8") as srfp:
        proc = subprocess.run(cmd, stdout=srfp, stderr=subprocess.PIPE, check=True)
    logger.debug(f"{genslip_bin} stderr: {proc.stderr}")


def gen_gsf(
    gsf_handle: TextIO,
    lon: float,
    lat: float,
    dtop: float,
    strike: float,
    dip: float,
    rake: float,
    flen: float,
    fwid: float,
    nstk: float,
    ndip: float,
    logger: Logger = qclogging.get_basic_logger(),
):
    """Wrapper around the fault_seg2gsf_dipdir binary.

    Parameters
    ----------
    gsf_handle : TextIO
        File handle for the GSF file. Must come from NamedTemporaryFile because
        it must have the `name` property.
    lon : float    |
    lat : float    |
    dtop : float   |
    strike : float |
    dip : float    | Refer to wiki (see module documentation for link).
    rake : float   |
    flen : float   |
    fwid : float   |
    nstk : float   |
    ndip : float   |
    logger : Logger
        Optional alternative logger for log output

    """
    logger.debug(f"Saving gsf to {gsf_handle.name}")
    with subprocess.Popen(
        [binary_version.get_unversioned_bin(FAULTSEG2GSFDIPDIR), "read_slip_vals=0"],
        stdin=subprocess.PIPE,
        stdout=gsf_handle,
    ) as gexec:
        gexec.communicate(
            f"1\n{lon:f} {lat:f} {dtop:f} {strike} {dip} {rake} {flen:f} {fwid:f} {nstk:d} {ndip:d}".encode(
                "utf-8"
            )
        )


def generate_sim_params_yaml(
    sim_params_file: str,
    parameters: Dict[str, Any],
    logger: Logger = qclogging.get_basic_logger(),
):
    """Write simulation parameters to a yaml file.

    Parameters
    ----------
    sim_params_file : str
        The filepath to save the simulation parameters.
    parameters : Dict[str, Any]
        The parameters to save.
    logger : Logger
        optional alternative logger for log output.

    """

    os.makedirs(os.path.dirname(sim_params_file), exist_ok=True)

    logger.debug(f"Raw sim params: {parameters}")
    sim_params = {}
    for key, value in parameters.items():
        if key in HF_RUN_PARAMS:
            if "hf" not in sim_params.keys():
                sim_params["hf"] = {}
            sim_params["hf"][key] = value

        elif key in BB_RUN_PARAMS:
            if "bb" not in sim_params.keys():
                sim_params["bb"] = {}
            sim_params["bb"][key] = value

        elif key in LF_RUN_PARAMS:
            if "emod3d" not in sim_params.keys():
                sim_params["emod3d"] = {}
            sim_params["emod3d"][key] = value
        elif key == "vs30_file_path":
            sim_params["stat_vs_est"] = value
        else:
            sim_params[key] = value

    logger.debug(f"Processed sim params: {sim_params}")
    logger.debug(f"Saving sim params to {sim_params_file}")
    with open(sim_params_file, "w", encoding="utf-8") as spf:
        yaml.dump(sim_params, spf)
    logger.debug("Sim params saved")


def load_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        A Namespace object containing the command line arguments (as specified
        by parse_args).

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("realisation_file", type=str)
    return parser.parse_args()


def main():
    primary_logger = qclogging.get_logger("realisation_to_srf")
    qclogging.add_general_file_handler(primary_logger, "rel2srf.txt")
    args = load_args()
    args.realisation_file = os.path.abspath(args.realisation_file)

    realisation = pre_processing_common.load_realisation_file_as_dict(
        args.realisation_file
    )
    rel_logger = qclogging.get_realisation_logger(primary_logger, realisation["name"])
    if realisation["type"] == 1:
        create_ps_srf(args.realisation_file, realisation, logger=rel_logger)
    elif realisation["type"] == 2:
        create_ps_ff_srf(args.realisation_file, realisation, logger=rel_logger)
    elif realisation["type"] == 4:
        create_multi_plane_srf(args.realisation_file, realisation, logger=rel_logger)
    else:
        raise ValueError(
            f"Type {realisation['type']} faults are not currently supported. "
            "Contact the software team if you believe this is an error."
        )
    sim_params_file = args.realisation_file.replace(".csv", ".yaml")
    generate_sim_params_yaml(sim_params_file, realisation)


if __name__ == "__main__":
    main()
