from workflow import realisations
from velocity_modelling import bounding_box
from source_modelling import sources
import numpy as np
from workflow.defaults import DefaultsVersion

domain_parameters = realisations.DomainParameters(
    resolution=0.5, 
    domain=bounding_box.BoundingBox.from_centroid_bearing_extents(
        centroid=np.array([-43.53092, 172.63701]),
        bearing=45,
        extent_x=100,
        extent_y=100
    ),
    depth=40.0,
    duration=60.0,
    dt=0.005
)

domain_parameters.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')

srf_config = realisations.SRFConfig(
    genslip_dt = 1.0,
    genslip_version='5.4.2',
    resolution=0.5,
)

srf_config.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')

# test_plane = sources.Plane.from_centroid_strike_dip(
#     centroid=np.array([-43.53092, 172.63701, 1000.0]),
#     dip=90.0,  # vertical fault
#     length=20.0,  # 20km length
#     width=15.0,  # 15km width  
#     strike=45.0  # 45 degree strike
# )
# source_config = realisations.SourceConfig(
#     source_geometries={"test_fault": sources.Fault([test_plane])}
# )

## Point source example
# test_point = sources.Point.from_lat_lon_depth(
#     point_coordinates=np.array([-43.53092, 172.63701, 1000.0]),  # depth in meters
#     length_m=5000.0,  # 5km approximating patch size
#     strike=45.0,      # 45 degree strike
#     dip=90.0,         # vertical dip
#     dip_dir=135.0     # dip direction (strike + 90 for vertical fault)
# )
# source_config = realisations.SourceConfig(
#     source_geometries={"test_fault": test_point}
# )
# source_config.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')

## Plane source example
test_plane = sources.Plane.from_centroid_strike_dip(
    centroid=np.array([-43.53092, 172.63701, 1000.0]),
    dip=90.0,  # vertical fault
    length=20.0,  # 20km length
    width=15.0,  # 15km width
    strike=45.0  # 45 degree strike
)
source_config = realisations.SourceConfig(
    source_geometries={"test_fault": test_plane}
)
source_config.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')





# Add the missing RupturePropagationConfig
rupture_propagation = realisations.RupturePropagationConfig(
    rupture_causality_tree={"test_fault": None},  # Single fault with no parent
    jump_points={},  # No fault jumps
    hypocentre=np.array([0.5, 0.5])  # Center of the fault in s,d coordinates
)

rupture_propagation.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')


# Add the missing Rakes configuration
rakes = realisations.Rakes(
    rakes={"test_fault": 90.0}  # Example rake of 90 degrees (thrust fault)
)

rakes.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')

# Add the missing Magnitudes configuration
magnitudes = realisations.Magnitudes(
    magnitudes={"test_fault": 7.0}  # Example magnitude of 7.0 for test_fault
)

magnitudes.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')


# # Add Seeds configuration
# seeds = realisations.Seeds.random_seeds()  # Generate random seeds
# seeds.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')


# Add metadata to the realisation
metadata = realisations.RealisationMetadata(
    name="test_realisation",
    version="1",
    defaults_version=DefaultsVersion.v24_2_2_4,
    tag="test"  # optional tag
)
metadata.write_to_realisation('/home/arr65/data/workflow/test/test_realisation.json')

# # Debug: Print the contents of the realisation file
# import json
# with open('/home/arr65/data/workflow/test/test_realisation.json', 'r') as f:
#     content = json.load(f)
#     print("Realisation file contents:")
#     print(json.dumps(content, indent=2))

# # Debug: Test reading individual components
# try:
#     test_magnitudes = realisations.Magnitudes.read_from_realisation('/home/arr65/data/workflow/test/test_realisation.json')
#     print("Successfully read magnitudes:", test_magnitudes)
# except Exception as e:
#     print("Failed to read magnitudes:", e)

# try:
#     test_rakes = realisations.Rakes.read_from_realisation('/home/arr65/data/workflow/test/test_realisation.json')
#     print("Successfully read rakes:", test_rakes)
# except Exception as e:
#     print("Failed to read rakes:", e)