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
