"""Create a Cylc workflow plan from a list of goals and stages to exclude.

This is the starting point for most workflow usages, and can be used
to generate a base Cylc workflow to modify and extend.
"""

import dataclasses
import tempfile
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Annotated, Any, Optional, Self

import jinja2
import networkx as nx
import printree
import tqdm
import typer
from pyvis.network import Network

from qcore import cli
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
    Archive = "archive"
    GCMTToRealisation = "gcmt_to_realisation"
    DomainGeneration = "generate_velocity_model_parameters"
    VelocityModelGeneration = "generate_velocity_model"
    StationSelection = "generate_station_coordinates"
    ModelCoordinates = "write_model_coordinates"
    SRFGeneration = "realisation_to_srf"
    CheckSRF = "check_srf"
    CopyDomainParameters = "copy_domain_parameters"
    EMOD3DParameters = "create_e3d_par"
    CheckDomain = "check_domain"
    StochGeneration = "generate_stoch"
    HighFrequency = "hf_sim"
    LowFrequency = "emod3d"
    Broadband = "bb_sim"
    IntensityMeasureCalculation = "im_calc"
    MergeTimeslices = "merge_ts"
    NSHMToRealisation = "nshm_to_realisation"


class Source(StrEnum):
    """Realisation source options."""

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
        StageIdentifier.GCMTToRealisation,
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

CONTAINER_PATHS = {
    WorkflowTarget.NeSI: Path("/nesi/nobackup/nesi00213/containers/runner_latest.sif"),
    WorkflowTarget.Hypocentre: Path("/mnt/hypo_scratch/containers/runner_latest.sif"),
}

EMOD3D_PATHS = {
    WorkflowTarget.NeSI: Path(
        "/nesi/project/nesi00213/opt/EMOD3D_cylc/tools/emod3d-mpi_v3.0.8"
    ),
    WorkflowTarget.Hypocentre: Path("/mnt/hypo_scratch/EMOD3D/tools/emod3d-mpi_v3.0.8"),
}


@dataclasses.dataclass
class Stage:
    """Representation of a workflow stage in the output Cylc file."""

    identifier: StageIdentifier
    """The stage identifier."""
    event: Optional[str]
    """The event the stage is running for."""
    sample: Optional[int]
    """The sample number of the realisation."""

    @property
    def parent(self) -> Self:  # numpydoc ignore=RT01
        """Stage: the parent stage of this stage."""
        return self.__class__(self.identifier, self.event, None)

    @property
    def directory(self) -> Optional[PurePath]:  # numpydoc ignore=RT01
        """Optional[PurePath]: the directory for this stage."""
        if not self.event:
            return None
        directory = self.event
        if self.sample:
            directory += f"_{self.sample}"
        return PurePath(directory)

    @property
    def outputs(self) -> set[PurePath]:  # numpydoc ignore=RT01
        """set[PurePath]: the outputs for this stage."""
        directory = self.directory
        if not directory:
            return set()
        return {directory / output for output in stage_outputs(self.identifier)}

    @property
    def inputs(self) -> set[PurePath]:  # numpydoc ignore=RT01
        """set[PurePath]: the inputs for this stage."""
        directory = self.directory
        if not self.event or not directory:
            return set()
        workflow_plan = realisation_workflow(self.event, self.sample)
        try:
            input_stages = list(workflow_plan.predecessors(self))
            inputs = set()
            for stage in input_stages:
                inputs |= stage.outputs
            return inputs
        except nx.NetworkXError:
            return set()

    def __hash__(self) -> int:
        """Hash the stage identifier, event and sample number.

        Returns
        -------
        int
            The hash of the stage.
        """
        return hash((self.identifier, self.event, self.sample))

    def __str__(self) -> str:
        """The string representation of the stage.

        Returns
        -------
        str
            The string representation of the stage."""
        _str = str(self.identifier)
        if self.event:
            _str += f"_{self.event}"
        if self.sample:
            _str += f"_{self.sample}"
        return _str


def stage_config_outputs(identifier: StageIdentifier) -> set[str]:
    """Get the realisation configuration outputs for a given stage.

    Parameters
    ----------
    identifier : StageIdentifier
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
            realisations.IntensityMeasureCalculationParameters._config_key
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
    return output_dictionary.get(identifier, set())


def stage_outputs(
    identifier: StageIdentifier, include_config_outputs: bool = True
) -> set[PurePath]:
    """Return a set of stage outputs for the given stage identifier.

    Parameters
    ----------
    identifier : StageIdentifier
        The stage identifier to get outputs for.
    include_config_outputs : bool, optional
        Whether to include configuration outputs (default is True).

    Returns
    -------
    set[PurePath]
        A set of output paths for the stage.
    """
    output_dictionary = {
        StageIdentifier.SRFGeneration: {
            PurePath("realisation.srf"),
        },
        StageIdentifier.ModelCoordinates: {
            PurePath("model") / "model_params",
            PurePath("model") / "grid_file",
        },
        StageIdentifier.StationSelection: {
            PurePath("stations") / "stations.ll",
            PurePath("stations") / "stations.statcords",
        },
        StageIdentifier.VelocityModelGeneration: {
            PurePath("Velocity_Model") / "rho3dfile.d",
            PurePath("Velocity_Model") / "vp3dfile.p",
            PurePath("Velocity_Model") / "vs3dfile.s",
            PurePath("Velocity_Model") / "in_basin_mask.b",
        },
        StageIdentifier.EMOD3DParameters: {
            PurePath("LF"),
            PurePath("LF") / "e3d.par",
        },
        StageIdentifier.LowFrequency: {PurePath("LF")},
        StageIdentifier.StochGeneration: {
            PurePath("realisation.stoch"),
        },
        StageIdentifier.HighFrequency: {PurePath("realisation.hf")},
        StageIdentifier.Broadband: {PurePath("realisation.bb")},
        StageIdentifier.IntensityMeasureCalculation: {
            PurePath("intensity_measures.parquet")
        },
        StageIdentifier.MergeTimeslices: {PurePath("LF") / "OutBin" / "output.e3d"},
    }
    file_outputs = output_dictionary.get(identifier, set())
    if include_config_outputs:
        for output in stage_config_outputs(identifier):
            file_outputs.add(PurePath("realisation.json") / output)

    return file_outputs


def realisation_workflow(event: str, sample: Optional[int]) -> nx.DiGraph:
    """Add a realisation to a workflow plan.

    Adds all stages for the realisation to run, and links to event
    stages for shared resources (i.e. the velocity model).

    Parameters
    ----------
    event : str
        The event to add.
    sample : Optional[int]
        The sample number (or None, if the original event).

    Returns
    -------
    nx.DiGraph
        The workflow plan with the added realisation.
    """
    requires_base = [
        StageIdentifier.SRFGeneration,
        StageIdentifier.StochGeneration,
        StageIdentifier.CheckSRF,
        StageIdentifier.VelocityModelGeneration,
        StageIdentifier.DomainGeneration,
        StageIdentifier.EMOD3DParameters,
        StageIdentifier.IntensityMeasureCalculation,
        StageIdentifier.HighFrequency,
        StageIdentifier.Broadband,
    ]
    requires_domain = [
        StageIdentifier.ModelCoordinates,
        StageIdentifier.Broadband,
        StageIdentifier.StationSelection,
        StageIdentifier.CheckDomain,
        StageIdentifier.CopyDomainParameters,
        StageIdentifier.VelocityModelGeneration,
        StageIdentifier.HighFrequency,
        StageIdentifier.EMOD3DParameters,
    ]
    workflow_plan = nx.from_dict_of_lists(
        {
            Stage(StageIdentifier.NSHMToRealisation, event, sample): [
                Stage(id, event, sample) for id in requires_base
            ],
            Stage(StageIdentifier.GCMTToRealisation, event, sample): [
                Stage(id, event, sample) for id in requires_base
            ],
            Stage(StageIdentifier.SRFGeneration, event, sample): [
                Stage(StageIdentifier.CheckSRF, event, sample),
                Stage(StageIdentifier.LowFrequency, event, sample),
            ],
            Stage(StageIdentifier.CheckSRF, event, sample): [
                Stage(StageIdentifier.StochGeneration, event, sample),
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
            ],
            Stage(StageIdentifier.VelocityModelGeneration, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
                Stage(StageIdentifier.HighFrequency, event, sample),
                Stage(
                    StageIdentifier.Broadband, event, sample
                ),  # This is a transitive dependency, but is useful for determining stage inputs
            ],
            Stage(StageIdentifier.StationSelection, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample),
                Stage(StageIdentifier.HighFrequency, event, sample),
            ],
            Stage(StageIdentifier.ModelCoordinates, event, None): [
                Stage(StageIdentifier.EMOD3DParameters, event, sample)
            ],
            Stage(StageIdentifier.EMOD3DParameters, event, sample): [
                Stage(StageIdentifier.CheckDomain, event, sample),
                Stage(StageIdentifier.LowFrequency, event, sample),
            ],
            Stage(StageIdentifier.CheckDomain, event, sample): [
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
        workflow_plan.add_edges_from(
            [
                (
                    Stage(StageIdentifier.DomainGeneration, event, sample),
                    Stage(id, event, sample),
                )
                for id in requires_domain
            ]
        )
    else:
        workflow_plan.add_edges_from(
            [
                (
                    Stage(StageIdentifier.DomainGeneration, event, None),
                    Stage(StageIdentifier.CopyDomainParameters, event, sample),
                ),
            ]
        )
        workflow_plan.add_edges_from(
            [
                (
                    Stage(StageIdentifier.CopyDomainParameters, event, sample),
                    Stage(id, event, sample),
                )
                for id in requires_domain
                if id != StageIdentifier.CopyDomainParameters
            ]
        )
    return workflow_plan


def create_abstract_workflow_plan(
    realisations: set[tuple[str, Optional[int]]],
    goals: Iterable[StageIdentifier],
    excluding: Iterable[StageIdentifier],
) -> nx.DiGraph:
    """Create an abstract workflow graph from a list of goals and excluded stages.

    Parameters
    ----------
    realisations : set[tuple[str, Optional[int]]]
        The realisations to generate the workflow for.
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

    roots = [node for node, degree in output_graph.in_degree() if degree == 0]
    ends = [node for node, degree in output_graph.out_degree() if degree == 0]
    copy_input_stage = Stage(StageIdentifier.CopyInput, "", None)
    archive_output_stage = Stage(StageIdentifier.Archive, "", None)

    output_graph.add_node(copy_input_stage)
    output_graph.add_node(archive_output_stage)
    for root in roots:
        output_graph.add_edge(copy_input_stage, root)
    for end in ends:
        output_graph.add_edge(end, archive_output_stage)
    return output_graph


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
            str(stage),
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
        network.add_edge(str(stage), str(next_stage))
    return network


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
        "rho3dfile.d": "3D density model file.",
        "vp3dfile.p": "3D P-wave velocity model file.",
        "vs3dfile.s": "3D S-wave velocity model file.",
        "in_basin_mask.b": "In-basin mask file for the velocity model.",
        "LF": "Directory containing low frequency simulation files.",
        "e3d.par": "EMOD3D parameter file.",
        "realisation.stoch": "Stochastic file for the realisation.",
        "realisation.hf": "High frequency waveform file for the realisation.",
        "realisation.bb": "Broadband waveform file for the realisation.",
        "intensity_measures.parquet": "Parquet file containing intensity measures.",
        "animation.mp4": "Animation of the timeslices.",
        "output.e3d": "Merged output file from the low frequency simulation.",
    }
    config_descriptions = {
        realisations.RealisationMetadata._config_key: "Metadata for describing a realisation.",
        realisations.SRFConfig._config_key: "Configuration for SRF generation.",
        realisations.SourceConfig._config_key: "Configuration for defining sources.",
        realisations.RupturePropagationConfig._config_key: "Configuration for rupture propagation.",
        realisations.DomainParameters._config_key: "Parameters defining the spatial and temporal domain for simulation.",
        realisations.VelocityModelParameters._config_key: "Parameters defining the velocity model.",
        realisations.VelocityModel1D._config_key: "1D Velocity Model for SRF and HF.",
        realisations.HFConfig._config_key: "High frequency simulation configuration.",
        realisations.EMOD3DParameters._config_key: "Parameters for EMOD3D LF simulation.",
        realisations.BroadbandParameters._config_key: "Parameters for broadband waveform merger.",
        realisations.IntensityMeasureCalculationParameters._config_key: "Intensity measure calculation parameters.",
    }
    filetree: dict[str, Any] = {}

    root = filetree
    for part in root_path.parts:
        root[part] = {}
        root = root[part]

    for file in sorted(files):
        cur = root
        for part in file.parts[:-1]:
            if part not in cur:
                cur[part] = {}
            cur = cur[part]

        if file.parent.name == "realisation.json":
            cur[file.parts[-1]] = config_descriptions.get(file.name, {})
        else:
            cur[file.parts[-1]] = file_descriptions.get(file.name, {})

    return filetree


@cli.from_docstring(app)
def plan_workflow(
    realisation_ids: Annotated[list[str], typer.Argument()],
    flow_file: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    goal: Annotated[
        list[StageIdentifier],
        typer.Option(default_factory=lambda: [], rich_help_panel="Planning Workflows"),
    ],
    group_goal: Annotated[
        list[GroupIdentifier],
        typer.Option(default_factory=lambda: [], rich_help_panel="Planning Workflows"),
    ],
    excluding: Annotated[
        list[StageIdentifier],
        typer.Option(default_factory=lambda: [], rich_help_panel="Planning Workflows"),
    ],
    excluding_group: Annotated[
        list[GroupIdentifier],
        typer.Option(default_factory=lambda: [], rich_help_panel="Planning Workflows"),
    ],
    archive: Annotated[
        list[StageIdentifier],
        typer.Option(
            default_factory=lambda: [
                StageIdentifier.Broadband,
                StageIdentifier.IntensityMeasureCalculation,
            ],
            rich_help_panel="Archiving",
        ),
    ],
    visualise: Annotated[
        bool, typer.Option(rich_help_panel="Visualising Workflows")
    ] = False,
    show_required_files: Annotated[
        bool, typer.Option(rich_help_panel="Visualising Workflows")
    ] = True,
    target_host: Annotated[
        WorkflowTarget, typer.Option(rich_help_panel="Planning Workflows")
    ] = WorkflowTarget.NeSI,
    source: Annotated[Optional[Source], typer.Option(rich_help_panel="Sources")] = None,
    defaults_version: Annotated[
        Optional[DefaultsVersion], typer.Option(rich_help_panel="Sources")
    ] = None,
    container: Annotated[Optional[Path], typer.Option()] = None,
    emod3d_path: Annotated[Optional[Path], typer.Option()] = None,
):
    """Plan and generate a Cylc workflow file for a number of realisations.

    Parameters
    ----------
    realisation_ids : list[str]
        List of realisations to generate workflows for. Realisations have the format event:realisation_count, such as Darfield:4.
    flow_file : Path
        Path to output flow file (e.g. ~/cylc-src/my-workflow/flow.cylc).
    goal : list[StageIdentifier]
        List of workflow outputs to generate.
    group_goal : list[GroupIdentifier]
        List of group goals to generate.
    excluding : list[StageIdentifier]
        List of stages to exclude.
    excluding_group : list[GroupIdentifier]
        List of stage groups to exclude.
    archive : list[StageIdentifier]
        Add stage outputs to the archive tarball.
    visualise : bool
        Visualise the planned workflow as a graph.
    show_required_files : bool
        Print the expected directory tree at the start of the simulation.
    target_host : WorkflowTarget
        Select the target host where the workflow will be run.
    source : Optional[Source]
        If given, set the source of the realisation. For NSHM and GCMT, the realisation id corresponds to the rupture id and GCMT PublicID respectively.
    defaults_version : Optional[DefaultsVersion]
        The simulation defaults to apply for all realisations. Required if source is specified.
    container : Optional[Path]
        The container to use for the workflow. If not specified, the default container for the target environment will be used.
    emod3d_path : Optional[Path]
        The path to the EMOD3D installation. If not specified, the default path for the target environment will be used.
    """
    container = container or CONTAINER_PATHS[target_host]
    emod3d_path = emod3d_path or EMOD3D_PATHS[target_host]
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
    archiving = set()
    for stage_id in archive:
        archiving |= {
            file.name for file in stage_outputs(stage_id, include_config_outputs=False)
        }

    template = env.get_template("flow.cylc")
    flow_template = template.render(
        container=container,
        emod3d_path=emod3d_path,
        defaults_version=defaults_version,
        realisations=realisations,
        target_host=target_host,
        archiving=archiving,
        workflow_plan={
            node: sorted(dependents, key=lambda stage: str(stage))
            for node, dependents in sorted(
                nx.to_dict_of_lists(workflow_plan).items(),
                key=lambda kv: str(kv[0]),
            )
        },
    )
    flow_file.write_text(
        # strip empty lines from the output flow template
        "\n".join(line for line in flow_template.split("\n") if line.strip())
    )
    if show_required_files:
        root_path = Path("cylc-src") / "WORKFLOW_NAME" / "input"
        inputs = {
            PurePath(
                Path("cylc-src") / "WORKFLOW_NAME" / "flow.cylc",
            )
        }
        outputs = set()
        inputs = set()
        for stage in workflow_plan.nodes:
            inputs |= stage.inputs
            outputs |= stage.outputs

        missing_file_tree = build_filetree(root_path, inputs - outputs)

        if missing_file_tree:
            print("You require the following files for your simulation:")
            print()
            printree.ptree(missing_file_tree)
            print()
            print(
                "You can find documentation for the output files at https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+In+Ground+Motion+Simulation."
            )
    if visualise:
        network = pyvis_graph(workflow_plan)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as graph_render:
            network.show(graph_render.name, notebook=False)
