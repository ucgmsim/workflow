"""Create a Cylc workflow plan from a list of goals and stages to exclude.

This is the starting point for most workflow usages, and can be used
to generate a base Cylc workflow to modify and extend.
"""

import tempfile
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Annotated, Any, NamedTuple, Optional

import jinja2
import networkx as nx
import printree
import tqdm
import typer
from pyvis.network import Network

from workflow import realisations
from workflow.defaults import DefaultsVersion

app = typer.Typer()


class WorkflowTarget(StrEnum):
    """Enumeration of possible workflow targets."""

    NeSI = "nesi"
    Hypocentre = "hypocentre"


class StageIdentifier(StrEnum):
    """Valid stage identifier in the workflow plan."""

    CopyInput = "copy_input"
    GCMTToRealisation = "gcmt_to_realisation"
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


class Source(StrEnum):
    GCMT = "gcmt"
    NSHM = "nshm"


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
    GroupIdentifier.Domain: {
        StageIdentifier.VelocityModelGeneration,
        StageIdentifier.StationSelection,
        StageIdentifier.DomainGeneration,
        StageIdentifier.CopyDomainParameters,
        StageIdentifier.ModelCoordinates,
    },
}

GROUP_GOALS = {
    GroupIdentifier.Preprocessing: {
        StageIdentifier.EMOD3DParameters,
        StageIdentifier.StochGeneration,
    },
    GroupIdentifier.LowFrequency: {StageIdentifier.LowFrequency},
    GroupIdentifier.HighFrequency: {StageIdentifier.HighFrequency},
    GroupIdentifier.Domain: {
        StageIdentifier.VelocityModelGeneration,
        StageIdentifier.StationSelection,
        StageIdentifier.ModelCoordinates,
        StageIdentifier.CopyDomainParameters,
    },
}


class Stage(NamedTuple):
    """Representation of a workflow stage in the output Cylc file."""

    identifier: StageIdentifier
    """The stage identifier."""
    event: str
    """The event the stage is running for."""
    sample: Optional[int]
    """The sample number of the realisation."""


class AnnotatedPath(PurePath):
    """Pure path annotated with description of the file."""

    def __init__(self, path: str | Path, description: Optional[str] = None):
        """Create an annotated path.

        Parameters
        ----------
        path : str | Path
            The path to annotate.
        description : Optional[str]
            The description of the path.
        """
        super().__init__(path)
        self.path = path
        self.description = description

    def __hash__(self):
        """Construct a unique hash for Annotated Paths."""
        if super().name == "realisation.json":
            # Realisations should also have their annotations hashed.
            return hash((self.path, self.description))
        else:
            return super().__hash__()


def stage_inputs(
    stage: Stage,
    root: Path,
    required_realisation_sections: dict[tuple[str, Optional[int]], set[str]],
) -> set[AnnotatedPath]:
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
    realisation_requirements = sorted(
        required_realisation_sections.get((stage.event, stage.sample), set())
    )
    realisation = {
        AnnotatedPath(
            event_directory / "realisation.json",
            f'Realisation file for event containing: {', '.join(realisation_requirements)}.',
        )
    }
    station_ll = AnnotatedPath(
        parent_directory / "stations" / "stations.ll",
        "lat,lon coordinates corresponding to x,y coordinates of stations in domain.",
    )
    s_wave = AnnotatedPath(
        parent_directory / "Velocity_Model" / "vs3dfile.s",
        "s-wave velocity file for velocity model (Section: Velocity Model Files).",
    )
    match stage:
        case Stage(
            identifier=StageIdentifier.SRFGeneration
            | StageIdentifier.VelocityModelGeneration
            | StageIdentifier.StationSelection
            | StageIdentifier.ModelCoordinates
        ):
            return realisation
        case Stage(identifier=StageIdentifier.CopyDomainParameters):
            return {
                AnnotatedPath(
                    parent_directory / "realisation.json",
                    f"Realisation file for event containing: {', '.join(required_realisation_sections.get((stage.event, None), set()))}.",
                )
            }
        case Stage(
            identifier=StageIdentifier.EMOD3DParameters
            | StageIdentifier.LowFrequency,
        ):
            return (
                realisation
                | {
                    station_ll,
                    AnnotatedPath(
                        parent_directory / "stations" / "stations.statcords",
                        "x,y coordinates of stations in domain.",
                    ),
                    AnnotatedPath(
                        parent_directory / "model" / "model_params",
                        "Model centre and corners for EMOD3D.",
                    ),
                    AnnotatedPath(
                        parent_directory / "model" / "grid_file",
                        "Domain extents for EMOD3D.",
                    ),
                    AnnotatedPath(
                        parent_directory / "Velocity_Model" / "rho3dfile.d",
                        "Density component file for velocity model. (Section: Velocity Model Files)",
                    ),
                    AnnotatedPath(
                        parent_directory / "Velocity_Model" / "vp3dfile.p",
                        "p-wave velocity file for velocity model. (Section: Velocity Model Files)",
                    ),
                    s_wave,
                    AnnotatedPath(
                        parent_directory / "Velocity_Model" / "in_basin_mask.b",
                        "Boolean basin mask (Section: Velocity Model Files).",
                    ),
                    AnnotatedPath(
                        event_directory / "realisation.srf",
                        "Slip model of source (Section: SRF Format).",
                    ),
                }
                | (
                    set()
                    if stage.identifier == StageIdentifier.EMOD3DParameters
                    else {
                        AnnotatedPath(
                            event_directory / "LF" / "e3d.par",
                            "EMOD3D Parameters (Section: https://wiki.canterbury.ac.nz/pages/viewpage.action?pageId=100794983)",
                        )
                    }
                )
            )
        case Stage(identifier=StageIdentifier.StochGeneration):
            return realisation | {
                AnnotatedPath(
                    event_directory / "realisation.srf",
                    "Slip model of source (Section: SRF Format).",
                )
            }
        case Stage(identifier=StageIdentifier.HighFrequency):
            return realisation | {
                AnnotatedPath(
                    event_directory / "realisation.stoch",
                    "Downsampled SRF for stochastic source model input (Section: Stoch format).",
                ),
                station_ll,
                s_wave,
            }
        case Stage(identifier=StageIdentifier.Broadband):
            return realisation | {
                station_ll,
                AnnotatedPath(
                    event_directory / "LF", "Low-frequency simulation directory."
                ),
                AnnotatedPath(
                    event_directory / "realisation.hf",
                    "High-frequency waveform file (Section: LF/HF/BB binary format).",
                ),
                s_wave,
            }
        case Stage(identifier=StageIdentifier.IntensityMeasureCalculation):
            return realisation | {
                AnnotatedPath(
                    event_directory / "realisation.bb",
                    "Broadband waveform file (Section: LF/HF/BB binary format).",
                )
            }
        case Stage(identifier=StageIdentifier.PlotTimeslices):
            return {
                AnnotatedPath(
                    event_directory / "LF" / "OutBin" / "output.e3d",
                    "Merged xyts-slices of low-frequency simulation (Section: XYTS.e3d binary format).",
                )
            }
        case Stage(identifier=StageIdentifier.MergeTimeslices):
            return {
                AnnotatedPath(
                    event_directory / "LF" / "OutBin",
                    "Component xyts-slices of low-frequency simulation, one per compute node (Section: XYTS.e3d binary format).",
                )
            }
    return set()


def stage_config_inputs(stage: Stage) -> set[str]:
    """Get the realisation configuration inputs for a given stage.

    Parameters
    ----------
    stage : Stage
        The stage to get inputs for.


    Returns
    -------
    set[str]
        The input config sections for this stage.
    """
    input_dictionary = {
        StageIdentifier.EMOD3DParameters: {
            realisations.DomainParameters._config_key,
            realisations.VelocityModelParameters._config_key,
            realisations.RealisationMetadata._config_key,
        },
        StageIdentifier.Broadband: {
            realisations.RealisationMetadata._config_key,
            realisations.DomainParameters._config_key,
        },
        StageIdentifier.StationSelection: {realisations.DomainParameters._config_key},
        StageIdentifier.VelocityModelGeneration: {
            realisations.DomainParameters._config_key,
            realisations.RealisationMetadata._config_key,
        },
        StageIdentifier.HighFrequency: {
            realisations.DomainParameters._config_key,
            realisations.RealisationMetadata._config_key,
        },
        StageIdentifier.IntensityMeasureCalculation: {
            realisations.RealisationMetadata._config_key,
            realisations.BroadbandParameters._config_key,
        },
        StageIdentifier.DomainGeneration: {
            realisations.RealisationMetadata._config_key,
            realisations.RupturePropagationConfig._config_key,
            realisations.SourceConfig._config_key,
        },
        StageIdentifier.SRFGeneration: {
            realisations.RealisationMetadata._config_key,
            realisations.SourceConfig._config_key,
            realisations.RupturePropagationConfig._config_key,
        },
        StageIdentifier.ModelCoordinates: {realisations.DomainParameters._config_key},
        StageIdentifier.StochGeneration: {realisations.RealisationMetadata._config_key},
        StageIdentifier.StationSelection: {realisations.DomainParameters._config_key},
    }
    return input_dictionary.get(stage.identifier, set())


def stage_config_outputs(stage: Stage) -> set[str]:
    """Get the realisation configuration outputs for a given stage.

    Parameters
    ----------
    stage : Stage
        The stage to get outputs for.


    Returns
    -------
    set[str]
        The output config sections for this stage.
    """
    output_dictionary = {
        StageIdentifier.NSHMToRealisation: {
            realisations.SourceConfig._config_key,
            realisations.RupturePropagationConfig._config_key,
            realisations.RealisationMetadata._config_key,
        },
        StageIdentifier.GCMTToRealisation: {
            realisations.SourceConfig._config_key,
            realisations.RupturePropagationConfig._config_key,
            realisations.RealisationMetadata._config_key,
        },
        StageIdentifier.EMOD3DParameters: {realisations.EMOD3DParameters._config_key},
        StageIdentifier.Broadband: {realisations.BroadbandParameters._config_key},
        StageIdentifier.VelocityModelGeneration: {
            realisations.VelocityModelParameters._config_key
        },
        StageIdentifier.HighFrequency: {realisations.HFConfig._config_key},
        StageIdentifier.IntensityMeasureCalculation: {
            realisations.IntensityMeasureCalcuationParameters._config_key
        },
        StageIdentifier.CopyDomainParameters: {
            realisations.VelocityModelParameters._config_key,
            realisations.DomainParameters._config_key,
        },
        StageIdentifier.DomainGeneration: {
            realisations.VelocityModelParameters._config_key,
            realisations.DomainParameters._config_key,
        },
        StageIdentifier.SRFGeneration: {realisations.SRFConfig._config_key},
        StageIdentifier.StochGeneration: {realisations.HFConfig._config_key},
    }
    return output_dictionary.get(stage.identifier, set())


def realisation_configuration_requirements(
    workflow_plan: nx.DiGraph, realisation: tuple[str, Optional[int]]
) -> set[str]:
    """Get the requirements for the realisation.json in this stage.

    Parameters
    ----------
    workflow_plan : nx.DiGraph
        The overall workflow plane.
    realisation : tuple[str, Optional[int]]
        The realisation to retrieve requirements for.

    Returns
    -------
    set[str]
        The realisation configuration sections that must be present
        for the workflow to run correctly.
    """
    stages = [
        stage
        for stage in workflow_plan.nodes
        if (stage.event, stage.sample) == realisation
    ]
    if stages:
        return set.union(*[stage_config_inputs(stage) for stage in stages]) - set.union(
            *[stage_config_outputs(stage) for stage in stages]
        )
    else:
        return set()


def stage_outputs(stage: Stage, root: Path) -> set[AnnotatedPath]:
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
    realisation = {
        AnnotatedPath(
            event_directory / "realisation.json", "Realisation file for event."
        )
    }
    match stage:
        case Stage(
            identifier=StageIdentifier.NSHMToRealisation
            | StageIdentifier.GCMTToRealisation
        ):
            return realisation
        case Stage(identifier=StageIdentifier.SRFGeneration):
            return {
                AnnotatedPath(
                    event_directory / "realisation.srf",
                    "Slip model of source.",
                )
            }
        case Stage(identifier=StageIdentifier.ModelCoordinates):
            return {
                AnnotatedPath(
                    event_directory / "model" / "model_params",
                    "Model centre and corners for EMOD3D.",
                ),
                AnnotatedPath(
                    event_directory / "model" / "grid_file",
                    "Domain extents for EMOD3D.",
                ),
            }
        case Stage(identifier=StageIdentifier.StationSelection):
            return {
                AnnotatedPath(
                    event_directory / "stations" / "stations.ll",
                    "lat,lon coordinates corresponding to x,y coordinates of stations in domain.",
                ),
                AnnotatedPath(
                    event_directory / "stations" / "stations.statcords",
                    "x,y coordinates of stations in domain.",
                ),
            }
        case Stage(identifier=StageIdentifier.VelocityModelGeneration):
            return {
                AnnotatedPath(
                    event_directory / "Velocity_Model" / "rho3dfile.d",
                    "Density component file for velocity model.",
                ),
                AnnotatedPath(
                    event_directory / "Velocity_Model" / "vp3dfile.p",
                    "p-wave velocity file for velocity model.",
                ),
                AnnotatedPath(
                    event_directory / "Velocity_Model" / "vs3dfile.s",
                    "s-wave velocity file for velocity model.",
                ),
                AnnotatedPath(
                    event_directory / "Velocity_Model" / "in_basin_mask.b",
                    "Boolean basin mask.",
                ),
            }
        case Stage(identifier=StageIdentifier.EMOD3DParameters):
            return {
                AnnotatedPath(
                    event_directory / "LF", "Low-frequency simulation directory."
                ),
                AnnotatedPath(event_directory / "LF" / "e3d.par", "EMOD3D Parameters"),
            }
        case Stage(identifier=StageIdentifier.LowFrequency):
            return {
                AnnotatedPath(
                    event_directory / "LF", "Low-frequency simulation directory."
                )
            }
        case Stage(identifier=StageIdentifier.StochGeneration):
            return {
                AnnotatedPath(
                    event_directory / "realisation.stoch",
                    "Downsampled SRF for stochastic source model input.",
                )
            }
        case Stage(identifier=StageIdentifier.HighFrequency):
            return {
                AnnotatedPath(
                    event_directory / "realisation.hf", "High-frequency waveform file."
                )
            }
        case Stage(identifier=StageIdentifier.Broadband):
            return {
                AnnotatedPath(
                    event_directory / "realisation.bb", "Broadband waveform file."
                )
            }
        case Stage(identifier=StageIdentifier.IntensityMeasureCalculation):
            return {
                AnnotatedPath(
                    event_directory / "ims.parquet", "Intensity measure statistics."
                )
            }
        case Stage(identifier=StageIdentifier.PlotTimeslices):
            return {
                AnnotatedPath(
                    event_directory / "animation.mp4",
                    "Animation of low-frequency waveform.",
                )
            }
        case Stage(identifier=StageIdentifier.MergeTimeslices):
            return {
                AnnotatedPath(
                    event_directory / "LF" / "OutBin" / "output.e3d",
                    "Merged xyts-slices of low-frequency simulation.",
                )
            }
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
            Stage(StageIdentifier.CopyInput, "", None): [
                Stage(StageIdentifier.NSHMToRealisation, event, sample),
                Stage(StageIdentifier.GCMTToRealisation, event, sample),
            ],
            Stage(StageIdentifier.NSHMToRealisation, event, sample): [
                Stage(StageIdentifier.SRFGeneration, event, sample),
            ],
            Stage(StageIdentifier.GCMTToRealisation, event, sample): [
                Stage(StageIdentifier.SRFGeneration, event, sample)
            ],
            Stage(StageIdentifier.SRFGeneration, event, sample): [
                Stage(StageIdentifier.StochGeneration, event, sample),
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
            ],
            Stage(StageIdentifier.VelocityModelGeneration, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
                Stage(StageIdentifier.HighFrequency, event, sample),
            ],
            Stage(StageIdentifier.StationSelection, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
                Stage(StageIdentifier.HighFrequency, event, sample),
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
                    Stage(StageIdentifier.GCMTToRealisation, event, sample),
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


def build_filetree(files: set[AnnotatedPath]) -> dict[str, Any]:
    """Build a file tree from a set of annotated file paths.

    Parameters
    ----------
    files : set[AnnotatedPath]
        The set of files to construct a tree for.


    Returns
    -------
    dict[str, Any]
        A file tree.
    """
    filetree: dict[str, Any] = {}
    for file in files:
        cur = filetree
        for part in file.parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]
        if not cur.get(file.parts[-1]):
            cur[file.parts[-1]] = file.description
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
    target_host: Annotated[
        WorkflowTarget,
        typer.Option(help="Select the target host where the workflow will be run"),
    ] = WorkflowTarget.NeSI,
    source: Annotated[
        Optional[Source],
        typer.Option(
            help="If given, set the source of the realisation. For NSHM and GCMT, the realisation id corresponds to the rupture id and GCMT PublicID respectively."
        ),
    ] = None,
    defaults_version: Annotated[
        Optional[DefaultsVersion],
        typer.Option(
            help="The simulation defaults to apply for all realisations. Required if source is specified."
        ),
    ] = None,
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
    if source and not defaults_version:
        print(
            "You must specify a defaults version if you specify a source. See the help text for options."
        )
        raise typer.Exit(code=1)
    excluding_set = set(excluding)
    goal_set = set(goal)
    if group_goal:
        goal_set |= set.union(*[GROUP_GOALS[group] for group in group_goal])
    if excluding_group:
        excluding_set |= set.union(*[GROUP_STAGES[group] for group in excluding_group])

    excluding_source_map: dict[Optional[Source], set[StageIdentifier]] = {
        Source.GCMT: {StageIdentifier.GCMTToRealisation},
        Source.NSHM: {StageIdentifier.NSHMToRealisation},
    }
    excluding_set |= set.union(
        *excluding_source_map.values()
    ) - excluding_source_map.get(source, set())

    workflow_plan = create_abstract_workflow_plan(realisations, goal_set, excluding_set)
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("workflow"),
    )
    template = env.get_template("flow.cylc")
    flow_template = template.render(
        defaults_version=defaults_version,
        realisations=realisations,
        target_host=target_host,
        workflow_plan=nx.to_dict_of_lists(workflow_plan),
    )
    flow_file.write_text(
        # strip empty lines from the output flow template
        "\n".join(line for line in flow_template.split("\n") if line.strip())
    )
    if show_required_files:
        root_path = Path("cylc-src") / "WORKFLOW_NAME" / "input"
        inputs = {
            AnnotatedPath(
                Path("cylc-src") / "WORKFLOW_NAME" / "flow.cylc",
                f"Your workflow file (the file {flow_file}).",
            )
        }
        outputs = set()
        required_realisation_sections = {
            realisation: realisation_configuration_requirements(
                workflow_plan, realisation
            )
            for realisation in realisations
        }

        for stage in workflow_plan.nodes:
            inputs |= stage_inputs(stage, root_path, required_realisation_sections)
            outputs |= stage_outputs(stage, root_path)

        missing_file_tree = build_filetree(inputs - outputs)

        if missing_file_tree:
            print("You require the following files for your simulation:")
            print()
            printree.ptree(missing_file_tree)
            print()
            print(
                "Refer to the indicated sections in https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation"
            )
        if any(required_realisation_sections.values()):
            print(
                "Refer to the realisation glossary at URL HERE for details on filling in the realisation files."
            )
    if visualise:
        network = pyvis_graph(workflow_plan)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as graph_render:
            network.show(graph_render.name, notebook=False)
