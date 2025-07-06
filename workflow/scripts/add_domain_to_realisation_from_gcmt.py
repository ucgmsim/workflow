import numpy as np

from velocity_modelling import bounding_box
from workflow import realisations

domain_parameters = realisations.DomainParameters(
    resolution=0.8,
    domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
        centroid=np.array([-42.367000000000004, 173.7613]),
        bearing=45,
        extent_x=10,
        extent_y=10,
    ),
    depth=10000.0,
    duration=60.0,
    dt=1,
)

# domain_parameters.write_to_realisation(
#     "/home/arr65/data/workflow/test/test_realisation.json"
# )

domain_parameters.write_to_realisation(
    "/home/arr65/data/workflow/test/gcmt_test_Point.json"
)
