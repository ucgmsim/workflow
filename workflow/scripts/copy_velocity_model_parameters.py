"""Copy Domain Parameters.

Description
-----------
Copy domain parameters between two realisations to enable the reuse of median event velocity models for subsequent realisations.

Inputs
------
1. A source realisation file containing domain parameters,
2. A target realisation file to copy the domain parameters to.

Outputs
-------
The target realisation file updated with the domain parameters from the source realisation file.

Environment
-----------
Can be run in the cybershake container. Can also be run from your own computer using the `copy-velocity-model-parameters` command which is installed after running `pip install workflow@git+https://github.com/ucgmsim/workflow`.

Usage
-----
`copy-velocity-model-parameters [OPTIONS] FROM_REALISATION_FFP TO_REALISATION_FFP`

For More Help
-------------
See the output of `copy-domain --help`.
"""

from pathlib import Path
from typing import Annotated

import typer

from qcore import cli
from workflow import log_utils, realisations
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
    realisations.append_log_entry(to_realisation_ffp)
