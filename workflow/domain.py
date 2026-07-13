import itertools
from copy import deepcopy

from workflow.realisations import DomainParameters, Refinement, Refinements


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


def gridpoints_from_domain(
    domain_parameters: DomainParameters, refinements: Refinements
) -> int:
    depth = domain_parameters.depth
    area = domain_parameters.domain.area * (1000**2)
    domain_refinements = refinements.refinements_for_depth(depth)
    top = 0.0
    gridpoints = 0
    for refinement in domain_refinements:
        volume = (refinement.bottom - top) * area
        gridpoints += int(volume // (refinement.resolution) ** 3)
        top = refinement.bottom
    return gridpoints
