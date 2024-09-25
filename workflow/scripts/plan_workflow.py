"""Create a Cylc workflow plan from a list of goals and stages to exclude.

This is the starting point for most workflow usages, and can be used
to generate a base Cylc workflow to modify and extend.
"""

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import jinja2
import networkx as nx
import typer

app = typer.Typer()


class StageIdentifier(StrEnum):
    """Valid stage identifier in the workflow plan."""

    CopyInput = "copy_input"
    DomainGeneration = "generate_velocity_model_parameters"
    VelocityModelGeneration = "generate_velocity_model"
    StationSelection = "generate_station_coordinates"
    ModelCoordinates = "write_model_coordinates"
    SRFGeneration = "realisation_to_srf"
    EMOD3DParameters = "create_e3d_par"
    StochGeneration = "generate_stoch"
    HighFrequency = "hf_sim"
    LowFrequency = "emod3d"
    Broadband = "bb_sim"
    IntensityMeasureCalculation = "im_calc"
    PlotTimeslices = "plot_ts"
    MergeTimeslices = "merge_ts"
    NSHMToRealisation = "nshm_to_realisation"


DEFAULT_WORKFLOW_PLAN: nx.DiGraph = nx.from_dict_of_lists(
    {
        StageIdentifier.CopyInput: [StageIdentifier.NSHMToRealisation],
        StageIdentifier.NSHMToRealisation: [
            StageIdentifier.DomainGeneration,
            StageIdentifier.SRFGeneration,
        ],
        StageIdentifier.DomainGeneration: [
            StageIdentifier.VelocityModelGeneration,
            StageIdentifier.StationSelection,
            StageIdentifier.ModelCoordinates,
        ],
        StageIdentifier.SRFGeneration: [
            StageIdentifier.StochGeneration,
            StageIdentifier.EMOD3DParameters,
        ],
        StageIdentifier.VelocityModelGeneration: [StageIdentifier.EMOD3DParameters],
        StageIdentifier.StationSelection: [StageIdentifier.EMOD3DParameters],
        StageIdentifier.ModelCoordinates: [StageIdentifier.EMOD3DParameters],
        StageIdentifier.EMOD3DParameters: [StageIdentifier.LowFrequency],
        StageIdentifier.LowFrequency: [
            StageIdentifier.MergeTimeslices,
            StageIdentifier.Broadband,
        ],
        StageIdentifier.StochGeneration: [StageIdentifier.HighFrequency],
        StageIdentifier.HighFrequency: [StageIdentifier.Broadband],
        StageIdentifier.Broadband: [StageIdentifier.IntensityMeasureCalculation],
        StageIdentifier.MergeTimeslices: [StageIdentifier.PlotTimeslices],
    },
    create_using=nx.DiGraph,
)


def create_abstract_workflow_plan(
    goals: Iterable[StageIdentifier], excluding: Iterable[StageIdentifier]
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
    workflow_plan = nx.transitive_closure_dag(DEFAULT_WORKFLOW_PLAN)
    included_nodes = set.union(
        *[set(workflow_plan.predecessors(goal)) | {goal} for goal in goals]
    ) - set(excluding)
    workflow_plan = workflow_plan.subgraph(included_nodes)
    return nx.transitive_reduction(workflow_plan)


@app.command(
    help="Plan and generate a Cylc workflow file for a number of realisations."
)
def plan_workflow(
    realisations: Annotated[
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
            default_factory=lambda: [StageIdentifier.IntensityMeasureCalculation],
        ),
    ],
    excluding: Annotated[
        list[StageIdentifier],
        typer.Option(help="List of stages to exclude", default_factory=lambda: []),
    ],
):
    """Plan and generate a Cylc workflow file for a number of realisations.

    Parameters
    ----------
    realisations : list[str]
        The list of realisations to generate the workflow for.
    flow_file : Path
        The output flow file path to write the Cylc workflow to.
    goal : list[StageIdentifier]
        A list of workflow stages to mark as goals. These stages are define the endpoints for the workflow.
    excluding : list[StageIdentifier]
        A list of workflow stages to exclude from the flows.
    """
    workflow_plan = create_abstract_workflow_plan(goal, excluding)
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("workflow"),
    )
    template = env.get_template("flow.cylc")
    flow_file.write_text(
        template.render(
            realisations=realisations, workflow_plan=nx.to_dict_of_lists(workflow_plan)
        )
    )
