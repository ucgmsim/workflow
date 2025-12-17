from collections.abc import Callable

import pytest
from typer import Typer
from typer.testing import CliRunner

from workflow.scripts import (
    bb_sim,
    check_domain,
    check_srf,
    copy_velocity_model_parameters,
    create_e3d_par,
    gcmt_auto_simulate,
    gcmt_to_realisation,
    generate_rupture_propagation,
    generate_station_coordinates,
    generate_stoch,
    generate_velocity_model,
    generate_velocity_model_parameters,
    hf_sim,
    im_calc,
    import_realisation,
    lf_to_xarray,
    nshm2022_to_realisation,
    realisation_to_srf,
)


@pytest.mark.parametrize(
    "script",
    [
        bb_sim,
        check_domain,
        check_srf,
        copy_velocity_model_parameters,
        create_e3d_par,
        gcmt_auto_simulate,
        gcmt_to_realisation,
        generate_rupture_propagation,
        generate_station_coordinates,
        generate_stoch,
        generate_velocity_model,
        generate_velocity_model_parameters,
        lf_to_xarray,
        hf_sim,
        im_calc,
        import_realisation,
        nshm2022_to_realisation,
        realisation_to_srf,
    ],
)
def test_invocation_of_script(script: Callable) -> None:
    """Basic check that the scripts can be invoked."""
    runner = CliRunner()
    # The following satisifies the type checker.
    app = getattr(script, "app", None)
    assert isinstance(app, Typer)

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
