"""Workflow planning tool."""

import itertools
import tomllib
from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from importlib import resources
from pathlib import Path, PurePath
from typing import Annotated, Any, BinaryIO, TypeVar

import jinja2
import networkx as nx
import printree
import typer
from jinja2.environment import TemplateStream

import workflow
from qcore import cli
from workflow import realisations
from workflow.defaults import DefaultsVersion

app = typer.Typer()


class Parameter(StrEnum):
    """Parameter names for cylc workflow templates."""

    event = auto()
    """The event parameter."""
    sample = auto()
    """The sample parameter."""


@dataclass
class Stage:
    """A workflow stage."""

    id: str
    """The identifier of the stage, e.g. realisation_to_srf."""
    parameters: list[Parameter] = field(default_factory=list)
    """The cylc parameters that this stage takes."""
    requires_config: set[str] = field(default_factory=set)
    """The configuration blocks required for this workflow stage."""
    requires_files: set[str] = field(default_factory=set)
    """The files required for this workflow stage."""
    provides_config: set[str] = field(default_factory=set)
    """The configuration blocks that this stage produces."""
    provides_files: set[str] = field(default_factory=set)
    """The files that this stage generates."""
    follows: str | None = None
    """An explicit reference to a stage that this stage must follow."""

    @property
    def cylc_stage_identifier_template(self) -> str:
        """Return a template for a cylc stage identifier.

        Returns
        -------
        str
            The cylc stage identifier template. Comes in the form
            {id}<{parameters}>.
        """
        parameters = ", ".join("{" + parameter + "}" for parameter in self.parameters)
        return f"{self.id}<{parameters}>"


@dataclass
class StageConfig:
    """Per-host stage configuration for a workflow stage.

    Hosts create stage configs for each stage to adjust available
    environment variables, slurm or pbs directives, module loading,
    and platforms.
    """

    platform: str | None = None
    """The default platform to target."""
    pre_script: str | None = None
    """The pre-script to run (for loading modules, etc)."""
    directives: dict[str, str] = field(default_factory=dict)
    """The slurm or pbs directives to supply with the job."""
    environment: dict[str, str] = field(default_factory=dict)
    """The environment variables to make available to the script."""
    settings: dict[str, str] = field(default_factory=dict)
    """Additional settings for the stage"""


def build_resource_graph(stages: list[Stage]) -> nx.DiGraph:
    """Build a graph representing workflow stage and resource needs.

    Parameters
    ----------
    stages : list[Stage]
        The stages to build the resource graph from.

    Returns
    -------
    nx.DiGraph
        A directed bipartite graph containing resource and stage
        nodes. An directed edge resource -> stage implies that a stage
        requires a resource to run. A directed edge stage -> resource
        implies that a stage produces a resource.
    """
    resource_graph = nx.DiGraph()

    for stage in stages:
        resource_graph.add_node(stage.id, type="stage")
        for resource in stage.requires_files:
            resource_graph.add_node(
                f"_{resource}", type="resource", resource_type="file"
            )
            resource_graph.add_edge(f"_{resource}", stage.id)
        for resource in stage.requires_config:
            resource_graph.add_node(
                f"_{resource}", type="resource", resource_type="config"
            )
            resource_graph.add_edge(f"_{resource}", stage.id)

        for resource in stage.provides_files:
            resource_graph.add_node(
                f"_{resource}", type="resource", resource_type="file"
            )
            resource_graph.add_edge(stage.id, f"_{resource}")

        for resource in stage.provides_config:
            resource_graph.add_node(
                f"_{resource}", type="resource", resource_type="config"
            )
            resource_graph.add_edge(stage.id, f"_{resource}")
    return resource_graph


def workflow_graph(stages: list[Stage]) -> nx.DiGraph:
    """Build a workflow graph from a list of stages.

    Parameters
    ----------
    stages : list[Stage]
        The stages to build the workflow graph from.

    Returns
    -------
    nx.DiGraph
        A directed acyclic graph over workflow stages, where an edge
        stage a -> stage b implies that stage b depends on the output
        of stage a. The graph is not transitively reduced (hence, some
        edges are redundant).
    """
    resource_graph = build_resource_graph(stages)
    workflow_plan = nx.DiGraph()
    workflow_plan.add_nodes_from([stage.id for stage in stages])
    for node, data in resource_graph.nodes(data=True):
        if data["type"] != "resource":
            continue
        producers = resource_graph.predecessors(node)
        consumers = resource_graph.successors(node)
        workflow_plan.add_edges_from(itertools.product(producers, consumers))
    for stage in stages:
        if stage.follows:
            # This order is required! If we add the stage.follows -> stage.id edge first we get a loop.
            workflow_plan.add_edges_from(
                (stage.id, neighbour)
                for neighbour in workflow_plan.successors(stage.follows)
            )
            workflow_plan.add_edge(stage.follows, stage.id)

    return workflow_plan


def dfs_paths(
    workflow_plan: nx.DiGraph, roots: list[str]
) -> Generator[list[str], None, None]:
    """Yield all DFS paths from a digraph beginning at any listed root.

    Parameters
    ----------
    workflow_plan : nx.DiGraph
        The graph to generate DFS paths for.
    roots : list[str]
        The roots to begin recursion at.

    Yields
    ------
    list[str]
        A DFS path from the graph `workflow_graph` starting from a root in `roots`.
    """

    def aux(path: list[str], visited: set[str]) -> Generator[list[str], None, None]:
        """Auxiliary function to perform DFS recursion.

        Parameters
        ----------
        path : list[str]
            The current path from a root.
        visited : set[str]
            The visited nodes.

        Yields
        ------
        list[str]
            A DFS path from the graph.
        """
        cur = path[-1]
        if cur in visited or workflow_plan.out_degree(cur) == 0:
            visited.add(cur)  # in the case where out-degree == 0, this is helpful

            # Paths must be copied at the end, or else the paths will
            # be modified after yielding.
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


def dfs_tree_cover(workflow_plan: nx.DiGraph) -> Generator[list[str], None, None]:
    """Cover every edge with DFS paths.

    Parameters
    ----------
    workflow_plan : nx.DiGraph
        The graph to cover.

    Yields
    ------
    list[str]
        A DFS path of the graph `workflow_plan`. Every edge of
        `workflow_plan` is guaranteed to be included in at least one
        path.
    """

    roots = [
        node for node in workflow_plan.nodes() if workflow_plan.in_degree(node) == 0
    ]
    yield from dfs_paths(workflow_plan, roots)


def workflow_plan_as_cylc_template(
    stages: list[Stage], workflow_plan: nx.DiGraph
) -> str:
    """Render a workflow plan graph as a workflow plan template.

    Parameters
    ----------
    stages : list[Stage]
        The stages to render.
    workflow_plan : nx.DiGraph
        The workflow plan to render.

    Returns
    -------
    str
        A format-string template for the workflow graph. This
        format-subtring can be substituted with named parameters to
        yield a concrete cylc flow graph to output into a .flow file.
    """
    stage_lookup = {stage.id: stage for stage in stages}
    return "\n".join(
        " => ".join(
            stage_lookup[stage].cylc_stage_identifier_template for stage in path
        )
        for path in dfs_tree_cover(workflow_plan)
    )


def load_workflow_stages() -> list[Stage]:
    """Load workflow stages from the `stages.toml` definition file.

    Returns
    -------
    list[Stage]
        A list of loaded stages.
    """
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
                follows=kwargs.get("follows"),
            )
            for id, kwargs in toml_parser.items()
        ]


def load_host_environment(environment_file: BinaryIO) -> defaultdict[str, StageConfig]:
    """Load a host environment dictionary from its defining toml file.

    Parameters
    ----------
    environment_file : BinaryIO
        The host environment to load.

    Returns
    -------
    defaultdict[str, StageConfig]
        A dictionary containing the loaded stage configs for the host.
    """
    raw = tomllib.load(environment_file)
    host_environment = defaultdict(StageConfig)
    for stage, config in raw.items():
        host_environment[stage] = StageConfig(**config)

    return host_environment


class WorkflowTarget(StrEnum):
    """Enumeration of possible workflow targets."""

    NeSI = auto()
    """Target NeSI HPC environment."""
    Hypocentre = auto()
    """Target Hypocentre or local environment."""
    TACC = auto()
    """Target TACC environment."""
    RCH = auto()
    """Target RCH environment."""


class Source(StrEnum):
    """Realisation source options."""

    GCMT = auto()
    """Source realisation information from GCMT solution."""
    NSHM = auto()
    """Source realisation information from NSHM2022 database."""


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
    """Alias for velocity model generation."""


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


T = TypeVar("T")


def union_all(sets: Iterable[set[T]]) -> set[T]:
    """Union an iterable of sets.

    Parameters
    ----------
    sets : Iterable[set[Any]]
        The sets to union.

    Returns
    -------
    set[Any]
        A set containing the union of all `sets`.
    """
    out = set()
    for aset in sets:
        out |= aset
    return out


def build_filetree(root_path: PurePath, files: set[PurePath]) -> dict[str, Any]:
    """Build a file tree from a set of file paths.

    Parameters
    ----------
    root_path : PurePath
        The root path for the file tree.
    files : set[PurePath]
        The set of files to construct a tree for.

    Returns
    -------
    dict[str, Any]
        A file tree.
    """
    file_descriptions = {
        "realisation.srf": "Contains the slip model for the realisation.",
        "model_params": "Parameters for the model used in the simulation.",
        "grid_file": "Grid file for the model coordinates.",
        "stations.ll": "Station coordinates (lat, lon) in the simulation domain.",
        "stations.statcords": "Station coordinates (x, y) in the simulation domain.",
        "rho3dfile.d": "3D density model file (for first realisation ONLY).",
        "vp3dfile.p": "3D P-wave velocity model file (for first realisation ONLY).",
        "vs3dfile.s": "3D S-wave velocity model file (for first realisation ONLY).",
        "in_basin_mask.b": "In-basin mask file for the velocity model (for first realisation ONLY).",
        "LF": "Directory containing low frequency simulation files.",
        "e3d.par": "EMOD3D parameter file.",
        "realisation.stoch": "Stochastic file for the realisation.",
        "realisation.hf": "High frequency waveform file for the realisation.",
        "realisation.lf": "Low frequency waveform file for the realisation.",
        "realisation.bb": "Broadband waveform file for the realisation.",
        "intensity_measures.parquet": "Parquet file containing intensity measures.",
        "animation.mp4": "Animation of the timeslices.",
        "output.e3d": "Merged XYTS output file from the low frequency simulation.",
    }
    config_descriptions = {
        cls._config_key: cls.__doc__
        for cls in [
            realisations.RealisationMetadata,
            realisations.SRFConfig,
            realisations.RupturePropagationConfig,
            realisations.DomainParameters,
            realisations.VelocityModelParameters,
            realisations.VelocityModel1D,
            realisations.HFConfig,
            realisations.EMOD3DParameters,
            realisations.BroadbandParameters,
            realisations.IntensityMeasureCalculationParameters,
            realisations.Rakes,
            realisations.Magnitudes,
            realisations.SourceConfig,
        ]
    }

    filetree = {}
    root = filetree
    for part in root_path.parts:
        root[part] = {}
        root = root[part]

    for file in sorted(files):
        cur = root
        for part in file.parts[:-1]:
            if part not in cur or isinstance(cur[part], str):
                cur[part] = {}
            cur = cur[part]

        if file.parent.name == "realisation.json":
            cur[file.parts[-1]] = config_descriptions.get(file.name, {})
        else:
            cur[file.parts[-1]] = file_descriptions.get(file.name, {})
    if not root:
        return {}
    return filetree


def print_required_files(stages: list[Stage]) -> None:
    """Print the required files by stages.

    A resource is considered required if no stage in `stages` provides
    that resource.

    Parameters
    ----------
    stages : list[Stage]
        The stages that will be run in the workflow.
    """
    root_path = PurePath("cylc-src") / "WORKFLOW_NAME" / "inputs" / "REALISATION"
    resource_graph = build_resource_graph(stages)
    unmet_resources = {
        PurePath(resource.lstrip("_"))
        if data["resource_type"] == "file"
        else PurePath("realisation.json") / resource.lstrip("_")
        for resource, data in resource_graph.nodes(data=True)
        # Only select nodes that are resources, and have no workflow
        # stage providing them (i.e. in-degree in the resource graph
        # is zero).
        if data["type"] == "resource" and not resource_graph.in_degree(resource)
    }
    filetree = build_filetree(root_path, unmet_resources)
    if not filetree:
        print("You do not require any files (besides the flow.cylc).")
        return
    print("You require the following files for your simulation:")
    print()
    filetree["cylc-src"]["WORKFLOW_NAME"]["flow.cylc"] = "Cylc workflow file."
    printree.ptree(filetree)


def resource_for_target_host(target_host: WorkflowTarget) -> BinaryIO:
    """Open a binary host environment file for reading.

    Parameters
    ----------
    target_host : WorkflowTarget-
         The host to target

    Returns
    -------
    BinaryIO
         A handle to begin reading the environment definition.
    """
    return resources.open_binary(
        workflow, "templates", "environments", f"{target_host}.toml"
    )


def build_targeted_workflow_graph(
    stages: list[Stage], goals: Iterable[str], excluding: set[str]
) -> nx.DiGraph:
    """Build a workflow targetting `goals` while skipping stages in `excluding`.

    Parameters
    ----------
    stages : list[Stage]
        A list of all possible workflow stages.
    goals : Iterable[str]
        The goals for the workflow.
    excluding : set[str]
        The stages to skip.

    Returns
    -------
    nx.DiGraph
        A transitively reduced workflow digraph whose terminal nodes
        are the elements of `goals`. The graph will not contain any
        element of `excluding`.
    """
    workflow = workflow_graph(stages)
    workflow.remove_nodes_from(excluding)
    reachable = union_all(nx.ancestors(workflow, goal) for goal in goals) | set(goals)
    unreachable = set(workflow.nodes()) - reachable
    workflow.remove_nodes_from(unreachable)
    return nx.transitive_reduction(workflow)


def workflow_jinja_template(
    realisations: list[tuple[str, int]],
    workflow: nx.DiGraph,
    stages: list[Stage],
    host_environment: dict[str, StageConfig],
) -> TemplateStream:
    """Construct a jinja template stream for a workflow.



    Parameters
    ----------
    realisations : list[tuple[str, int]]
        The realisations in the workflow.
    workflow : nx.DiGraph
        The workflow digraph to execute.
    stages : list[Stage]
        The stage definitions in the workflow.
    host_environment : dict[str, StageConfig]
        The host environment to target.


    Returns
    -------
    TemplateStream
        A template stream that can be dumped to an output cylc file.
    """
    workflow_graph_template = workflow_plan_as_cylc_template(stages, workflow)
    environment = jinja2.Environment(
        loader=jinja2.PackageLoader("workflow"), trim_blocks=True, lstrip_blocks=True
    )
    template = environment.get_template("flow.cylc")
    workflow_graph_string = "\n".join(
        workflow_graph_template.format(
            event="event", realisation=f"{event}_realisations"
        )
        for event, _ in realisations
    )
    return template.stream(
        realisations=realisations,
        workflow_graph=workflow_graph_string,
        stages=stages,
        host_environment=host_environment,
    )


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
    target_host: Annotated[
        WorkflowTarget, typer.Option(rich_help_panel="Planning Workflows")
    ] = WorkflowTarget.NeSI,
    host_file: Annotated[
        Path | None,
        typer.Option(rich_help_panel="Planning Workflows", exists=True, dir_okay=False),
    ] = None,
    source: Annotated[Source | None, typer.Option(rich_help_panel="Sources")] = None,
    defaults_version: Annotated[
        DefaultsVersion | None, typer.Option(rich_help_panel="Sources")
    ] = None,
    show_required_files: Annotated[
        bool, typer.Option(rich_help_panel="Visualising Workflows")
    ] = True,
) -> None:
    """Plan a workflow.

    Parameters
    ----------
    realisation_ids : list[str]
        List of realisations to generate workflows for. Realisations
        have the format event:realisation_count, such as Darfield:4.
    flow_file : Path
        Path to output flow file (e.g. ~/cylc-src/my-workflow/flow.cylc).
    goals : list[str]
        List of workflow outputs to generate.
    group_goal : list[GroupIdentifier]
        List of group goals to generate.
    excluding : list[str]
        List of stages to exclude.
    excluding_groups : list[GroupIdentifier]
        List of stage groups to exclude.
    show_required_files : bool
        Print the expected directory tree at the start of the simulation.
    target_host : WorkflowTarget
        Select the target host where the workflow will be run.
    host_file : Path | None
        Provide a custom host file for a new host environment.
        Overrides the target host selection if present.
    source : Source | None
        If given, set the source of the realisation. For NSHM and
        GCMT, the realisation id corresponds to the rupture id and
        GCMT PublicID respectively.
    defaults_version : DefaultsVersion | None
        The simulation defaults to apply for all realisations.
        Required if source is specified.
    """
    if source and not defaults_version:
        print("Must specify a defaults version if source is specified.")
        return

    if host_file:
        with open(host_file, "rb") as host_file_handle:
            host_environment = load_host_environment(host_file_handle)
    elif target_host:
        with resource_for_target_host(target_host) as host_file_handle:
            host_environment = load_host_environment(host_file_handle)
    else:
        print("Must specify a host environment or provide a host environment file.")
        return

    if defaults_version:
        host_environment["root"].environment["DEFAULTS"] = defaults_version

    realisations = [
        parse_realisation(realisation_id) for realisation_id in realisation_ids
    ]
    stages = load_workflow_stages()

    source_stage_map = {
        Source.NSHM: {"nshm_to_realisation"},
        Source.GCMT: {"gcmt_to_realisation"},
        None: set(),
    }
    sources_to_remove = {
        "nshm_to_realisation",
        "gcmt_to_realisation",
    } - source_stage_map[source]

    workflow = build_targeted_workflow_graph(
        stages,
        set(goals) | union_all(GROUP_STAGES[group] for group in group_goal),
        set(excluding)
        | sources_to_remove
        | union_all(GROUP_STAGES[group] for group in excluding_groups),
    )

    stage_lookup_map = {stage.id: stage for stage in stages}
    workflow_stages = [stage_lookup_map[stage] for stage in workflow.nodes()]

    workflow_jinja_template(
        realisations, workflow, workflow_stages, host_environment
    ).dump(str(flow_file))

    if show_required_files:
        print_required_files(workflow_stages)


if __name__ == "__main__":
    app()
