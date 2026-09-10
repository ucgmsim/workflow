from workflow.realisations import (
    DomainParameters,
    Refinements,
)


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
