"""Create a Cylc workflow plan from a list of goals and stages to exclude.

This is the starting point for most workflow usages, and can be used
to generate a base Cylc workflow to modify and extend.
"""

import tempfile
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NamedTuple, Optional

import jinja2
import networkx as nx
import printree
import tqdm
import typer
from pyvis.network import Network

app = typer.Typer()


class StageIdentifier(StrEnum):
    """Valid stage identifier in the workflow plan."""

    DomainGeneration = "generate_velocity_model_parameters"
    VelocityModelGeneration = "generate_velocity_model"
    StationSelection = "generate_station_coordinates"
    ModelCoordinates = "write_model_coordinates"
    SRFGeneration = "realisation_to_srf"
    CopyDomainParameters = "copy_domain_parameters"
    EMOD3DParameters = "create_e3d_par"
    StochGeneration = "generate_stoch"
    HighFrequency = "hf_sim"
    LowFrequency = "emod3d"
    Broadband = "bb_sim"
    IntensityMeasureCalculation = "im_calc"
    PlotTimeslices = "plot_ts"
    MergeTimeslices = "merge_ts"
    NSHMToRealisation = "nshm_to_realisation"


class GroupIdentifier(StrEnum):
    """Group identifiers to use to bulk target or exclude in workflow planning."""

    Preprocessing = "preprocessing"
    """Alias for all preprocessing stages."""
    HighFrequency = "high_frequency"
    """Alias for the high frequency workflow."""
    LowFrequency = "low_frequency"
    """Alias for the low frequency workflow."""


GROUP_STAGES = {
    GroupIdentifier.Preprocessing: {
        StageIdentifier.DomainGeneration,
        StageIdentifier.VelocityModelGeneration,
        StageIdentifier.StationSelection,
        StageIdentifier.ModelCoordinates,
        StageIdentifier.SRFGeneration,
        StageIdentifier.EMOD3DParameters,
        StageIdentifier.NSHMToRealisation,
        StageIdentifier.StochGeneration,
        StageIdentifier.CopyDomainParameters,
    },
    GroupIdentifier.HighFrequency: {
        StageIdentifier.HighFrequency,
    },
    GroupIdentifier.LowFrequency: {StageIdentifier.LowFrequency},
}

GROUP_GOALS = {
    GroupIdentifier.Preprocessing: {
        StageIdentifier.EMOD3DParameters,
        StageIdentifier.StochGeneration,
    },
    GroupIdentifier.LowFrequency: {StageIdentifier.LowFrequency},
    GroupIdentifier.HighFrequency: {StageIdentifier.HighFrequency},
}


class Stage(NamedTuple):
    """Representation of a workflow stage in the output Cylc file."""

    identifier: StageIdentifier
    """The stage identifier."""
    event: str
    """The event the stage is running for."""
    sample: Optional[int]
    """The sample number of the realisation."""


def stage_inputs(stage: Stage, root: Path) -> set[Path]:
    """Return a list of stage inputs for the stage.

    Parameters
    ----------
    stage : Stage
        The stage to get inputs for.
    root : Path
        The root directory of the simulation.


    Returns
    -------
    set[Path]
    A set of input paths required by the stage.


    """
    realisation_identifier = stage.event
    if stage.sample:
        realisation_identifier += f"_{stage.sample}"
    parent_directory = root / stage.event
    event_directory = root / realisation_identifier
    realisation = {event_directory / "realisation.json"}
    match stage:
        case Stage(identifier=StageIdentifier.NSHMToRealisation):
            return set()
        case Stage(
            identifier=StageIdentifier.SRFGeneration
            | StageIdentifier.VelocityModelGeneration
            | StageIdentifier.StationSelection
            | StageIdentifier.ModelCoordinates
        ):
            return realisation
        case Stage(identifier=StageIdentifier.CopyDomainParameters):
            return {parent_directory / "realisation.json"}
        case Stage(
            identifier=StageIdentifier.EMOD3DParameters
            | StageIdentifier.LowFrequency,
        ):
            return (
                realisation
                | {
                    parent_directory / "stations" / "stations.ll",
                    parent_directory / "stations" / "stations.statcords",
                    parent_directory / "model" / "model_params",
                    parent_directory / "model" / "grid_file",
                    parent_directory / "Velocity_Model" / "rho3dfile.d",
                    parent_directory / "Velocity_Model" / "vp3dfile.p",
                    parent_directory / "Velocity_Model" / "vs3dfile.s",
                    parent_directory / "Velocity_Model" / "in_basin_mask.b",
                    event_directory / "realisation.srf",
                }
                | (
                    set()
                    if stage.identifier == StageIdentifier.EMOD3DParameters
                    else {event_directory / "LF" / "e3d.par"}
                )
            )
        case Stage(identifier=StageIdentifier.StochGeneration):
            return realisation | {event_directory / "realisation.srf"}
        case Stage(identifier=StageIdentifier.HighFrequency):
            return realisation | {
                event_directory / "realisation.stoch",
                parent_directory / "stations" / "stations.ll",
                parent_directory / "Velocity_Model" / "vs3dfile.s",
            }
        case Stage(identifier=StageIdentifier.Broadband):
            return realisation | {
                event_directory / "realisation.stoch",
                parent_directory / "stations" / "stations.ll",
                event_directory / "LF",
                event_directory / "realisation.hf",
                parent_directory / "Velocity_Model" / "vs3dfile.s",
            }
        case Stage(identifier=StageIdentifier.IntensityMeasureCalculation):
            return realisation | {event_directory / "realisation.bb"}
        case Stage(identifier=StageIdentifier.PlotTimeslices):
            return {event_directory / "LF" / "OutBin" / "output.e3d"}
        case Stage(identifier=StageIdentifier.MergeTimeslices):
            return {event_directory / "LF" / "OutBin"}
    return set()


def stage_outputs(stage: Stage, root: Path) -> set[Path]:
    """Return a list of stage inputs for the stage.

    Parameters
    ----------
    stage : Stage
        The stage to get inputs for.
    root : Path
        The root directory of the simulation.


    Returns
    -------
    set[Path]
    A set of input paths required by the stage.


    """
    realisation_identifier = stage.event
    if stage.sample:
        realisation_identifier += f"_{stage.sample}"
    event_directory = root / realisation_identifier
    realisation = {event_directory / "realisation.json"}
    match stage:
        case Stage(identifier=StageIdentifier.NSHMToRealisation):
            return realisation
        case Stage(identifier=StageIdentifier.SRFGeneration):
            return {event_directory / "realisation.srf"}
        case Stage(identifier=StageIdentifier.ModelCoordinates):
            return {
                event_directory / "model" / "model_params",
                event_directory / "model" / "grid_file",
            }
        case Stage(identifier=StageIdentifier.StationSelection):
            return {
                event_directory / "stations" / "stations.ll",
                event_directory / "stations" / "stations.statcords",
            }
        case Stage(identifier=StageIdentifier.VelocityModelGeneration):
            return {
                event_directory / "Velocity_Model" / "rho3dfile.d",
                event_directory / "Velocity_Model" / "vp3dfile.p",
                event_directory / "Velocity_Model" / "vs3dfile.s",
                event_directory / "Velocity_Model" / "in_basin_mask.b",
            }
        case Stage(
            identifier=StageIdentifier.EMOD3DParameters | StageIdentifier.LowFrequency
        ):
            return {event_directory / "LF", event_directory / "LF" / "e3d.par"}
        case Stage(identifier=StageIdentifier.StochGeneration):
            return {event_directory / "realisation.stoch"}
        case Stage(identifier=StageIdentifier.HighFrequency):
            return {event_directory / "realisation.hf"}
        case Stage(identifier=StageIdentifier.Broadband):
            return {event_directory / "realisation.bb"}
        case Stage(identifier=StageIdentifier.IntensityMeasureCalculation):
            return {event_directory / "ims.parquet"}
        case Stage(identifier=StageIdentifier.PlotTimeslices):
            return {event_directory / "animation.mp4"}
        case Stage(identifier=StageIdentifier.MergeTimeslices):
            return {event_directory / "LF" / "OutBin" / "output.e3d"}
        case _:
            return set()


def realisation_workflow(event: str, sample: Optional[int]) -> nx.DiGraph:
    """Add a realisation to a workflow plan.

    Adds all stages for the realisation to run, and links to event
    stages for shared resources (i.e. the velocity model).

    Parameters
    ----------
    workflow_plan : nx.DiGraph
        The current workflow paln.
    event : str
        The event to add.
    sample : Optional[int]
        The sample number (or None, if the original event).
    """
    workflow_plan = nx.from_dict_of_lists(
        {
            Stage(StageIdentifier.NSHMToRealisation, event, sample): [
                Stage(StageIdentifier.SRFGeneration, event, sample)
            ],
            Stage(StageIdentifier.SRFGeneration, event, sample): [
                Stage(StageIdentifier.StochGeneration, event, sample),
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
            ],
            Stage(StageIdentifier.VelocityModelGeneration, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample)
            ],
            Stage(StageIdentifier.StationSelection, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample)
            ],
            Stage(StageIdentifier.ModelCoordinates, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample)
            ],
            Stage(StageIdentifier.EMOD3DParameters, event, sample): [
                Stage(StageIdentifier.LowFrequency, event, sample)
            ],
            Stage(StageIdentifier.LowFrequency, event, sample): [
                Stage(StageIdentifier.Broadband, event, sample),
                Stage(StageIdentifier.MergeTimeslices, event, sample),
            ],
            Stage(StageIdentifier.StochGeneration, event, sample): [
                Stage(StageIdentifier.HighFrequency, event, sample)
            ],
            Stage(StageIdentifier.HighFrequency, event, sample): [
                Stage(StageIdentifier.Broadband, event, sample)
            ],
            Stage(StageIdentifier.Broadband, event, sample): [
                Stage(StageIdentifier.IntensityMeasureCalculation, event, sample)
            ],
            Stage(StageIdentifier.MergeTimeslices, event, sample): [
                Stage(StageIdentifier.PlotTimeslices, event, sample)
            ],
        },
        create_using=nx.DiGraph,
    )
    if not sample:
        workflow_plan.add_edges_from(
            [
                (
                    Stage(StageIdentifier.NSHMToRealisation, event, sample),
                    Stage(StageIdentifier.DomainGeneration, event, sample),
                ),
                (
                    Stage(StageIdentifier.DomainGeneration, event, sample),
                    Stage(StageIdentifier.EMOD3DParameters, event, sample),
                ),
                (
                    Stage(StageIdentifier.DomainGeneration, event, sample),
                    Stage(StageIdentifier.VelocityModelGeneration, event, sample),
                ),
                (
                    Stage(StageIdentifier.DomainGeneration, event, sample),
                    Stage(StageIdentifier.StationSelection, event, sample),
                ),
                (
                    Stage(StageIdentifier.DomainGeneration, event, sample),
                    Stage(StageIdentifier.ModelCoordinates, event, sample),
                ),
            ]
        )
    else:
        workflow_plan.add_edges_from(
            [
                (
                    Stage(StageIdentifier.DomainGeneration, event, None),
                    Stage(StageIdentifier.CopyDomainParameters, event, sample),
                ),
                (
                    Stage(StageIdentifier.CopyDomainParameters, event, sample),
                    Stage(StageIdentifier.EMOD3DParameters, event, sample),
                ),
            ]
        )

    return workflow_plan


def create_abstract_workflow_plan(
    realisations: Sequence[tuple[str, Optional[int]]],
    goals: Iterable[StageIdentifier],
    excluding: Iterable[StageIdentifier],
) -> nx.DiGraph:
    """Create an abstract workflow graph from a list of goals and excluded stages.

    Parameters
    ----------
    goals : Iterable[StageIdentifier]
        The goal stages for the workflow.
    excluding : Iterable[StageIdentifier]
        The excluded stages for the workflow.

    Returns
    -------
    nx.DiGraph
        A abstract workflow plan. This workflow plan contains only
        included stages that are required to reach the goals. If two
        workflow stages depend on each other only through paths
        consisting entirely of excluded nodes, then they are adjacent
        directly in the abstract plan by edges.
    """

    excluding_stages = {
        Stage(excluded, *realisation)
        for excluded in excluding
        for realisation in realisations
    }

    output_graph = nx.DiGraph()
    realisation_iteration = (
        realisations if len(realisations) < 100 else tqdm.tqdm(realisations)
    )

    for realisation in realisation_iteration:
        workflow_plan = realisation_workflow(*realisation)
        workflow_plan = nx.transitive_closure_dag(workflow_plan)

        for goal in goals:
            reduced_graph = nx.transitive_reduction(
                workflow_plan.subgraph(
                    (
                        set(workflow_plan.predecessors(Stage(goal, *realisation)))
                        | {Stage(goal, *realisation)}
                    )
                    - excluding_stages
                )
            )
            output_graph.update(
                edges=reduced_graph.edges(), nodes=reduced_graph.nodes()
            )

    return output_graph


def stage_to_node_string(stage: Stage) -> str:
    r"""Convert a `Stage` into a human readable node identifier string.

    Parameters
    ----------
    stage : Stage
        The stage to render.

    Returns
    -------
    str
        A string of the format
        {stage.identifier}\n{stage.event}_{stage.sample}, if event and
        sample are non-trivial.
    """
    node_string = str(stage.identifier)
    if stage.event:
        node_string += f"\n{stage.event}"
    if stage.sample:
        node_string += f"_{stage.sample}"
    return node_string


def pyvis_graph(workflow_plan: nx.DiGraph) -> Network:
    """Convert a workflow plan into a pyvis diagram for visualisation.

    Parameters
    ----------
    workflow_plan : nx.DiGraph
        The workflow plan to visualise.


    Returns
    -------
    Network
        A pyvis rendering for this workflow plan.
    """
    network = Network(
        width="100%", height="1500px", directed=True, layout="hierarchical"
    )
    network.show_buttons(filter_=["physics"])
    roots = [node for node, degree in workflow_plan.in_degree() if degree == 0]
    reversed_workflow = workflow_plan.reverse()
    stage: Stage
    for stage in workflow_plan.nodes():
        network.add_node(
            stage_to_node_string(stage),
            group=f"{stage.event}_{stage.sample or ''}",
            size=20,
            level=max(
                (
                    len(path) - 1
                    for root in roots
                    for path in nx.all_simple_paths(reversed_workflow, stage, root)
                ),
                default=0,
            ),
        )
    for stage, next_stage in workflow_plan.edges():
        network.add_edge(stage_to_node_string(stage), stage_to_node_string(next_stage))
    return network


REALISATION_ITERATION_RE = r"_rel\d+$"


def parse_realisation(realisation_id: str) -> set[tuple[str, Optional[int]]]:
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
        event, num_samples = realisation_id[:index], realisation_id[index + 1 :]

        return {(event, sample or None) for sample in range(int(num_samples))}
    except ValueError:
        return {(realisation_id, None)}


def build_filetree(files: set[Path]) -> dict[str, Any]:
    filetree: dict[str, Any] = {}
    for file in files:
        cur = filetree
        for part in file.parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        if cur.get(file.parts[-1]) is None:
            cur[file.parts[-1]] = file.parts[-1]
    return filetree


@app.command(
    help="Plan and generate a Cylc workflow file for a number of realisations."
)
def plan_workflow(
    realisation_ids: Annotated[
        list[str],
        typer.Argument(help="List of realisations to generate workflows for."),
    ],
    flow_file: Annotated[
        Path,
        typer.Argument(
            help="Path to output flow file (e.g. ~/cylc-src/my-workflow/flow.cylc)",
            writable=True,
            dir_okay=False,
        ),
    ],
    goal: Annotated[
        list[StageIdentifier],
        typer.Option(
            help="List of workflow outputs to generate",
            default_factory=lambda: [],
        ),
    ],
    group_goal: Annotated[
        list[GroupIdentifier],
        typer.Option(
            help="List of group goals to generate", default_factory=lambda: []
        ),
    ],
    excluding: Annotated[
        list[StageIdentifier],
        typer.Option(help="List of stages to exclude", default_factory=lambda: []),
    ],
    excluding_group: Annotated[
        list[GroupIdentifier],
        typer.Option(
            help="List of stage groups to exclude", default_factory=lambda: []
        ),
    ],
    visualise: Annotated[
        bool, typer.Option(help="Visualise the planned workflow as a graph")
    ] = False,
    show_required_files: Annotated[
        bool,
        typer.Option(
            help="Print the expected directory tree at the start of the simulation."
        ),
    ] = True,
):
    """Plan and generate a Cylc workflow file for a number of realisations.

    Parameters
    ----------
    realisations : list[str]
        The list of realisations to generate the workflow for.
    flow_file : Path
        The output flow file path to write the Cylc workflow to.
    goal : list[StageIdentifier]
        A list of workflow stages to mark as goals. These stages are
        define the endpoints for the workflow.
    group_goal : list[GroupIdentifier]
        A list of workflow groups to target. A workflow group is just
        an alias for a set of workflow stages. Equivalent to adding
        each group member to `goal`.
    excluding : list[StageIdentifier]
        A list of workflow stages to exclude from the flows.
    group_goal : list[GroupIdentifier]
        A list of workflow groups to exclude. A workflow group is just
        an alias for a set of workflow stages. Equivalent to adding
        each group member to `excluding`.
    """
    realisations = set.union(
        *[parse_realisation(realisation_id) for realisation_id in realisation_ids]
    )

    if group_goal:
        goal = set(goal) | set.union(*[GROUP_GOALS[group] for group in group_goal])
    if excluding_group:
        excluding = set(excluding) | set.union(
            *[GROUP_STAGES[group] for group in excluding_group]
        )
    workflow_plan = create_abstract_workflow_plan(realisations, goal, excluding)
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("workflow"),
    )
    template = env.get_template("flow.cylc")
    flow_template = template.render(
        realisations=realisations,
        workflow_plan=nx.to_dict_of_lists(workflow_plan),
    )
    flow_file.write_text(
        # strip empty lines from the output flow template
        "\n".join(line for line in flow_template.split("\n") if line.strip())
    )
    if show_required_files:
        root_path = Path("cylc-src") / "WORKFLOW_NAME" / "input" / "share"
        inputs = set()
        outputs = set()
        for stage in workflow_plan.nodes:
            inputs |= stage_inputs(stage, root_path)
            outputs |= stage_outputs(stage, root_path)
        missing_file_tree = build_filetree(inputs - outputs)

        if missing_file_tree:
            print("You require the following files for your simulation:")
            printree.ptree(missing_file_tree)
    if visualise:
        network = pyvis_graph(workflow_plan)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as graph_render:
            network.show(graph_render.name, notebook=False)
