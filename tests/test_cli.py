import importlib
import pkgutil
from types import ModuleType

from pytest import Metafunc
from typer import Typer
from typer.testing import CliRunner

import workflow.scripts as scripts_package

EXCLUDE_MODULES = {"merge_ts_loop"}


def collect_script_modules() -> list[ModuleType]:
    """
    Dynamically discovers all modules within the workflow.scripts package.
    """
    modules = []
    # Iterates through all modules in the package directory
    for loader, module_name, is_pkg in pkgutil.iter_modules(scripts_package.__path__):
        if module_name in EXCLUDE_MODULES:
            continue

        # Construct full module path (e.g., workflow.scripts.bb_sim)
        full_module_name = f"{scripts_package.__name__}.{module_name}"
        module = importlib.import_module(full_module_name)
        modules.append(module)
    return modules


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """
    Generate tests dynamically based on discovered modules.
    """
    if "script_module" in metafunc.fixturenames:
        found_modules = collect_script_modules()
        # Create readable IDs from the module names (e.g., 'bb_sim')
        ids = [m.__name__.split(".")[-1] for m in found_modules]
        metafunc.parametrize("script_module", found_modules, ids=ids)


def test_invocation_of_script(script_module: ModuleType) -> None:
    """
    Test that each discovered module has a Typer app and responds to --help.
    """
    runner = CliRunner()

    assert hasattr(script_module, "app"), (
        f"Module {script_module.__name__} is missing the 'app' attribute."
    )

    app = getattr(script_module, "app")

    assert isinstance(app, Typer), (
        f"'app' in {script_module.__name__} should be a Typer instance, but got {type(app)}."
    )

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output
