import itertools
from copy import deepcopy
from dataclasses import dataclass, replace

from workflow.realisations import DomainParameters


@dataclass
class Refinement:
    resolution: float
    bottom: float


THEORETICAL_REFINEMENTS = [
    Refinement(resolution=100.0, bottom=10000.0),
    Refinement(resolution=200.0, bottom=25000.0),
]
UNBOUNDED_REFINEMENT_RESOLUTION = 400.0


def domain_refinements(depth: float) -> list[Refinement]:
    depth_m = depth * 1000.0
    refinements = []
    for refinement in THEORETICAL_REFINEMENTS:
        refinements.append(replace(refinement, bottom=min(refinement.bottom, depth_m)))
        if refinement.bottom > depth_m:
            break
    else:
        # This block only runs when we finish the loop without breaking, i.e. we
        # exhaust the refinement list.
        refinements.append(
            Refinement(resolution=UNBOUNDED_REFINEMENT_RESOLUTION, bottom=depth_m)
        )

    match refinements:
        case [*_, previous_layer, last_layer]:
            # Ensure a minimum amount in the last layer.
            last_layer.bottom = max(
                previous_layer.bottom + last_layer.resolution * 2, last_layer.bottom
            )

    return refinements


def adjust_for_topography(
    refinements: list[Refinement], topography_zmax: float, nzmin: int = 12
) -> tuple[list[Refinement], float]:
    # Ensure no side effects
    refinements = deepcopy(refinements)
    # By shallow copying the refinements before modifying them this view into the refinements will only have the updated refinements, and not the topography and bottom.
    real_refinements = refinements.copy()
    topography_resolution = min(
        (
            refinement
            for refinement in refinements
            if refinement.bottom > topography_zmax
        ),
        key=lambda r: r.bottom,
    ).resolution
    topography = Refinement(bottom=topography_zmax, resolution=topography_resolution)
    refinements.append(topography)
    refinements.sort(key=lambda r: r.bottom)

    for above, below in itertools.pairwise(refinements):
        thickness = below.bottom - above.bottom
        nz = thickness // below.resolution
        cells_needed = nzmin - nz
        if cells_needed > 0:
            below.bottom += cells_needed * below.resolution

    topography_zmax = topography.bottom

    return real_refinements, topography_zmax
