import itertools
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum, auto
from importlib import resources
from pathlib import Path
from string import Template
from typing import Annotated, Any, Generator, Iterable

import jinja2
import networkx as nx
import typer

import workflow
from qcore import cli
from workflow.defaults import DefaultsVersion

app = typer.Typer()


class Parameter(StrEnum):
    event = auto()
    sample = auto()


@dataclass
class Stage:
    id: str
    parameters: list[Parameter] = field(default_factory=list)
    requires_config: set[str] = field(default_factory=set)
    requires_files: set[str] = field(default_factory=set)
    provides_config: set[str] = field(default_factory=set)
    provides_files: set[str] = field(default_factory=set)

    @property
    def provides(self) -> set[str]:
        return self.provides_config | self.provides_files

    @property
    def requires(self) -> set[str]:
        return self.requires_config | self.requires_files

    @property
    def cylc_stage_identifier_template(self) -> str:
        parameters = ", ".join("${" + parameter + "}" for parameter in self.parameters)
        return f"{self.id}<{parameters}>"


def workflow_graph(stages: list[Stage]) -> nx.DiGraph:
    resource_graph = nx.DiGraph()

    for stage in stages:
        resource_graph.add_node(stage.id, type="stage")
        for resource in stage.requires:
            resource_graph.add_node(f"_{resource}", type="resource")
            resource_graph.add_edge(f"_{resource}", stage.id)

        for resource in stage.provides:
            resource_graph.add_node(f"_{resource}", type="resource")
            resource_graph.add_edge(stage.id, f"_{resource}")

    workflow_plan = nx.DiGraph()
    workflow_plan.add_nodes_from(
        [
            node
            for node, data in resource_graph.nodes(data=True)
            if data["type"] == "stage"
        ]
    )
    for node, data in resource_graph.nodes(data=True):
        if data["type"] != "resource":
            continue
        producers = resource_graph.predecessors(node)
        consumers = resource_graph.successors(node)
        workflow_plan.add_edges_from(itertools.product(producers, consumers))

    return workflow_plan


def dfs_paths(workflow_plan: nx.DiGraph, roots: list[str]) -> Generator[list[str]]:
    def aux(path: list[str], visited: set[str]) -> Generator[list[str]]:
        cur = path[-1]
        if cur in visited or workflow_plan.out_degree(cur) == 0:
            visited.add(cur)  # in the case where out-degree == 0, this is helpful
            yield path.copy()
            return
        visited.add(cur)
        for next in workflow_plan.neighbors(cur):
            # As opposed to path + [next], which will create a new list for every call.
            path.append(next)
            yield from aux(path, visited)
            path.pop()

    visited = set()
    for root in roots:
        yield from aux([root], visited)


def dfs_tree_cover(workflow_plan: nx.DiGraph):
    roots = [
        node for node in workflow_plan.nodes() if workflow_plan.in_degree(node) == 0
    ]
    return dfs_paths(workflow_plan, roots)


def workflow_plan_as_cylc_template(
    stages: list[Stage], workflow_plan: nx.DiGraph
) -> Template:
    stage_lookup = {stage.id: stage for stage in stages}
    return Template(
        "\n".join(
            " => ".join(
                stage_lookup[stage].cylc_stage_identifier_template for stage in path
            )
            for path in dfs_tree_cover(workflow_plan)
        )
    )


def load_workflow_stages() -> list[Stage]:
    with resources.open_binary(workflow, "templates", "stages.toml") as f:
        toml_parser = tomllib.load(f)
        return [
            Stage(
                id,
                parameters=kwargs.get("parameters", []),
                requires_files=set(kwargs.get("requires_files", set())),
                requires_config=set(kwargs.get("requires_config", set())),
                provides_files=set(kwargs.get("provides_files", set())),
                provides_config=set(kwargs.get("provides_config", set())),
            )
            for id, kwargs in toml_parser.items()
        ]


def load_host_environment(host: str) -> dict[str, Any]:
    with resources.open_binary(workflow, "templates", host, "environment.toml") as f:
        return tomllib.load(f)


class WorkflowTarget(StrEnum):
    """Enumeration of possible workflow targets."""

    NeSI = auto()
    Hypocentre = auto()
    TACC = auto()


class Source(StrEnum):
    """Realisation source options."""

    GCMT = auto()
    NSHM = auto()


def parse_realisation(realisation_id: str) -> tuple[str, int]:
    """Parse a realisation identifier string from the command line into a realisation identifier.

    Parameters
    ----------
    realisation_id : str
        The realisation identifier string to parse.

    Returns
    -------
    tuple[str, Optional[int]]
        The parsed realisation event and sample number.
    """
    try:
        index = realisation_id.rindex(":")
        return realisation_id[:index], int(realisation_id[index + 1 :])
    except ValueError:
        return (realisation_id, 1)


class GroupIdentifier(StrEnum):
    """Group identifiers to use to bulk target or exclude in workflow planning."""

    Preprocessing = "preprocessing"
    """Alias for all preprocessing stages."""
    HighFrequency = "high_frequency"
    """Alias for the high frequency workflow."""
    LowFrequency = "low_frequency"
    """Alias for the low frequency workflow."""
    Domain = "domain"


GROUP_STAGES = {
    GroupIdentifier.Preprocessing: {
        "generate_velocity_model_parameters",
        "generate_velocity_model",
        "generate_station_coordinates",
        "realisation_to_srf",
        "create_e3d_par",
        "nshm_to_realisation",
        "gcmt_to_realisation",
        "generate_stoch",
    },
    GroupIdentifier.HighFrequency: {"hf_sim"},
    GroupIdentifier.LowFrequency: {"emod3d"},
    GroupIdentifier.Domain: {
        "generate_velocity_model",
        "generate_station_coordinates",
        "generate_velocity_model_parameters",
    },
}


def union_all(sets: Iterable[set[Any]]) -> set[Any]:
    out = set()
    for aset in sets:
        out |= aset
    return out


@cli.from_docstring(app)
def plan_workflow(
    realisation_ids: Annotated[list[str], typer.Argument()],
    flow_file: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    goals: Annotated[
        list[str],
        typer.Option(
            "--goal",
            default_factory=lambda: [],
            rich_help_panel="Planning Workflows",
            show_default=False,
            metavar="STAGE",
        ),
    ],
    group_goal: Annotated[
        list[GroupIdentifier],
        typer.Option(
            default_factory=lambda: [],
            rich_help_panel="Planning Workflows",
            show_default=False,
        ),
    ],
    excluding: Annotated[
        list[str],
        typer.Option(
            default_factory=lambda: [],
            rich_help_panel="Planning Workflows",
            show_default=False,
            metavar="STAGE",
        ),
    ],
    excluding_groups: Annotated[
        list[GroupIdentifier],
        typer.Option(
            "--excluding-group",
            default_factory=lambda: [],
            rich_help_panel="Planning Workflows",
            show_default=False,
        ),
    ],
    # archive: Annotated[
    #     list[StageIdentifier],
    #     typer.Option(
    #         default_factory=lambda: [
    #             StageIdentifier.Broadband,
    #             StageIdentifier.IntensityMeasureCalculation,
    #         ],
    #         rich_help_panel="Archiving",
    #     ),
    # ],
    show_required_files: Annotated[
        bool, typer.Option(rich_help_panel="Visualising Workflows")
    ] = True,
    target_host: Annotated[
        WorkflowTarget, typer.Option(rich_help_panel="Planning Workflows")
    ] = WorkflowTarget.NeSI,
    source: Annotated[Source | None, typer.Option(rich_help_panel="Sources")] = None,
    defaults_version: Annotated[
        DefaultsVersion | None, typer.Option(rich_help_panel="Sources")
    ] = None,
) -> None:
    """Plan a workflow.



    Parameters
    ----------
    realisation_ids : Annotated[list[str], typer.Argument()]
        test
    flow_file : Path
        test
    goals : Annotated[ list[str], typer.Option( "--goal", default_factory=lambda: [], rich_help_panel="Planning Workflows" ), ]
        test
    group_goal : Annotated[ list[GroupIdentifier], typer.Option(default_factory=lambda: [], rich_help_panel="Planning Workflows"), ]
        test
    excluding : Annotated[ list[str], typer.Option(default_factory=lambda: [], rich_help_panel="Planning Workflows"), ]
        test
    excluding_groups : Annotated[ list[GroupIdentifier], typer.Option( "--excluding-group", default_factory=lambda: [], rich_help_panel="Planning Workflows", ), ]
        test
    show_required_files : bool
        test
    target_host : WorkflowTarget
        test
    source : Annotated[Source | None, typer.Option(rich_help_panel="Sources")]
        test
    defaults_version : Annotated[ DefaultsVersion | None, typer.Option(rich_help_panel="Sources") ]
        test

    """
    realisations = [
        parse_realisation(realisation_id) for realisation_id in realisation_ids
    ]
    stages = load_workflow_stages()
    host_environment = load_host_environment(target_host)
    default_environment = {}
    if defaults_version:
        default_environment["DEFAULTS"] = defaults_version

    workflow = workflow_graph(stages)
    source_stage_map = {
        Source.NSHM: {"nshm_to_realisation"},
        Source.GCMT: {"gcmt_to_realisation"},
        None: set(),
    }
    sources_to_remove = {
        "nshm_to_realisation",
        "gcmt_to_realisation",
    } - source_stage_map[source]
    excluding.extend(
        stage for group in excluding_groups for stage in GROUP_STAGES[group]
    )
    workflow.remove_nodes_from(sources_to_remove)
    workflow.remove_nodes_from(excluding)
    reachable = (
        union_all(nx.ancestors(workflow, goal) for goal in goals)
        | set(goals)
        | union_all(GROUP_STAGES[group] for group in group_goal)
    )
    unreachable = set(workflow.nodes()) - reachable
    workflow.remove_nodes_from(unreachable)
    workflow = nx.transitive_reduction(workflow)
    workflow_graph_template = workflow_plan_as_cylc_template(stages, workflow)
    environment = jinja2.Environment(loader=jinja2.PackageLoader("workflow"))
    template = environment.get_template("flow.cylc")
    workflow_graph_string = "\n".join(
        workflow_graph_template.substitute(
            event="event", realisation=f"{event}_realisations"
        )
        for event, _ in realisations
    )
    stage_lookup_map = {stage.id: stage for stage in stages}
    workflow_stages = [stage_lookup_map[stage] for stage in workflow.nodes()]
    template.stream(
        realisations=realisations,
        workflow_graph=workflow_graph_string,
        target_host=target_host,
        stages=workflow_stages,
        environment=default_environment | host_environment,
    ).dump(str(flow_file))


if __name__ == "__main__":
    app()
