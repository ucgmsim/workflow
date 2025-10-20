## Virtual Simulation Sites

### Background

As every set of simulation has different goals and priorities, a single set of virtual stations that meets all requirements is not feasible. On the other hand we want to prevent every simulation set developing their own custom grid of virtual stations, as this would make comparison between simulation sets difficult.

Instead a high density uniform "general" grid has been developed, and the code implemented here allows the generation of custom virtual station grid for each simulation set by sub-sampling this "general" grid as needed. This "general" grid ensures consistency across simulation sets while also allowing for customization.

### Custom Grid Generation

A custom grid can be generated using the 'virt_sites_cmds.py' script using the 'gen-custom-grid-from-rel' command.
For details/help on the command run 'python virt_sites_cmds.py gen-custom-grid-from-rel --help'
**Note that you will need about 16Gb of free memory.**

Running the custom grid generation will produce a parquet file with the following columns:

| Column | Description | Notes |
|--------|-------------|-------|
| `site_id` | Unique identifier for each virtual site | Made up as follows `{4 character lat/lon hash}{2 character region code}` |
| `lon` | Longitude coordinate in decimal degrees | |
| `lat` | Latitude coordinate in decimal degrees | |
| `nztm_x` | X coordinate in New Zealand Transverse Mercator (NZTM) projection | |
| `nztm_y` | Y coordinate in New Zealand Transverse Mercator (NZTM) projection | |
| `source` | Source of the site | Either `virtual` or `real` |
| `basin` | Basin name | None if not in a basin |
| `Z1.0` | Depth to 1.0 km/s shear-wave velocity (meters) | | 
| `Z2.5` | Depth to 2.5 km/s shear-wave velocity (meters) | |
| `vs30` | Time-averaged shear-wave velocity in the top 30 meters (m/s) | |

Additionally, it will also produce a metadata file that contains the setting used to generate the custom grid:

| Setting | Type | Description |
|---------|------|-------------|
| `land_only` | bool | Whether to include only land-based sites |
| `region` | geojson | Geographic region for the grid |
| `uniform_spacing` | int | Spacing between grid points |
| `vel_model_version` | str | Version of the velocity model used |

in addition to some additional information such as:

| Field | Description |
|----------------|-------------|
| `num_sites` | Total number of sites in the grid |
| `num_virtual_sites` | Number of virtual sites |
| `num_real_sites` | Number of real sites |
| `num_basin_sites` | Number of sites located within basins |
| `sites_per_basin` | Number of sites per basin |


### Plot Generation

Visualisation of the custom grid can be done using the 'gen-plot' command.
For details/help 'run python virt_sites_cmds.py gen-plot --help'.

Running this will produce a html file that can be viewed in any browser. 
Note that for large domains and high density grids this might be slow to load.





