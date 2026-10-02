"""Regression tests for issue #166: incorrect CLI docstrings.

- `nshm2022_to_realisation.generate_realisation`'s `dhypo` parameter was
  documented as the "strike" coordinate (copy-pasted from `shypo`), when it is
  actually the down-dip coordinate.
- `create_e3d_par`'s module docstring usage line listed a non-existent
  `GRID_FFP` positional argument that `create_e3d_par` does not accept.
"""

import inspect

from typer.models import ArgumentInfo

from workflow.scripts import create_e3d_par, nshm2022_to_realisation


def test_dhypo_docstring_describes_down_dip_coordinate() -> None:
    docstring = inspect.getdoc(nshm2022_to_realisation.generate_realisation)
    assert docstring is not None

    dhypo_doc = docstring.split("dhypo : float, optional")[1].split("\n    ")[1]
    assert "down-dip" in dhypo_doc
    assert "strike" not in dhypo_doc


def test_create_e3d_par_usage_matches_positional_arguments() -> None:
    module_doc = create_e3d_par.__doc__
    assert module_doc is not None

    usage_line = next(
        line
        for line in module_doc.splitlines()
        if line.strip().startswith("`create-e3d-par")
    )

    # After the @cli.from_docstring decorator runs, plain (non-defaulted)
    # parameters become typer.Argument(...) defaults, which is how typer (and
    # the real CLI) tells a positional argument apart from an option.
    signature = inspect.signature(create_e3d_par.create_e3d_par)
    positional_params = [
        name
        for name, param in signature.parameters.items()
        if isinstance(param.default, ArgumentInfo)
    ]

    documented_positionals = [
        token
        for token in usage_line.strip("`").split()
        if token.isupper() and token != "[OPTIONS]"
    ]

    assert documented_positionals == [name.upper() for name in positional_params]
