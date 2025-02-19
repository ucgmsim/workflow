from pathlib import Path
from typing import Annotated

import typer

from qcore import cli
from workflow import log_utils
from workflow.realisations import DomainParameters, VelocityModelParameters

app = typer.Typer()


@cli.from_docstring(app)
@log_utils.log_call()
def copy_domain(
    from_realisation_ffp: Annotated[Path, typer.Argument()],
    to_realisation_ffp: Annotated[Path, typer.Argument()],
) -> None:
    """Copy domain parameters between two realisations.

    This workflow stage is used so that the median event velocity
    models are able to be reused for subsequent realisations.

    Parameters
    ----------
    from_realisation_ffp : Path
        The realisation to copy domain parameters from.
    to_realisation_ffp : Path
        The realisation to copy domain parameters to.
    """
    from_realisation_domain_parameters = DomainParameters.read_from_realisation(
        from_realisation_ffp
    )
    from_realisation_domain_parameters.write_to_realisation(to_realisation_ffp)
    from_realisation_velocity_model_parameters = (
        VelocityModelParameters.read_from_realisation(from_realisation_ffp)
    )
    from_realisation_velocity_model_parameters.write_to_realisation(to_realisation_ffp)
