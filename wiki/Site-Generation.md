## Virtual Simulation Sites

### Background

As every set of simulation has different goals and priorities, a single set of virtual stations that meets all requirements is not feasible. On the other hand we want to prevent every simulation set developing their own custom grid of virtual stations, as this would make comparison between simulation sets difficult.

Instead a high density uniform "general" grid has been developed, and the code implemented here allows the generation of custom virtual station grid for each simulation set by sub-sampling this "general" grid as needed. This "general" grid ensures consistency across simulation sets while also allowing for customization.

### Custom Grid Generation

A custom grid can be generated using the `site_gen_cmds.py` script using the `gen-custom-grid-from-rel` command.
For details/help on the command run `python site_gen_cmds.py gen-custom-grid-from-rel --help`
There are two options to generate the custom grid via the `gen-custom-grid-from-rel` either specify the spacing via the command line arguments, or pass a configuration yaml file. The configuration allows for more customization, such as specifying spacing per basin or custom defined region.

An example configuration file might look something like this
```yaml
land_only: true
uniform_spacing: 5000

rel_ffp: null

vel_model_version: "2.09"
basin_spacing: 2500
per_basin_spacing: 
  Hanmer_v25p3: 1250
per_region_spacing:
  - name: "Christchurch"
    geojson_ffp: "/home/claudy/dev/tmp_share/virt_sites/chch.json"
    spacing: 1250

nzgmdb_version: v4.3
```
This configuration will select NZ land sites, with a default spacing of 5km, in basin spacing of 2.5km. 
Additionally, 1250m spacing is applied in Hanmer basin, and the custom Christchurch region defined via the geojson file. The geojson needs the following format:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              172.64635288571822,
              -43.46746584851593
            ],
            [
              172.5764348336035,
              -43.46875527988916
            ],
            [
              172.48512720365642,
              -43.510780173947616
            ],
            [
              172.48164483335898,
              -43.581684944518635
            ],
            [
              172.51943709438905,
              -43.605574705482184
            ],
            [
              172.60676846169588,
              -43.61039335808902
            ],
            [
              172.74171895701926,
              -43.575568637023046
            ],
            [
              172.811825205922,
              -43.54170866114695
            ],
            [
              172.7825735380037,
              -43.479415396924466
            ],
            [
              172.75289631792998,
              -43.449932811038906
            ],
            [
              172.68739428988397,
              -43.45477584526233
            ],
            [
              172.6620601942388,
              -43.46177619980023
            ],
            [
              172.64635288571822,
              -43.46746584851593
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}
```

**Note that you will need about 20Gb of free memory to generate custom grids.**


Running the custom grid generation will produce a parquet file with the following columns:

| Column | Description | Notes |
|--------|-------------|-------|
| `site_id` | Unique identifier for each virtual site |  `{4 character lat/lon hash}{2 character region code}`, e.g. "ijSBAOT", is a site in Otago as last two characters are "OT" |
| `lon` | Longitude coordinate in decimal degrees | |
| `lat` | Latitude coordinate in decimal degrees | |
| `nztm_x` | X coordinate in New Zealand Transverse Mercator (NZTM) projection | |
| `nztm_y` | Y coordinate in New Zealand Transverse Mercator (NZTM) projection | |
| `source` | Source of the site | Either `virtual` or `real` |
| `basin` | Basin name | None if not in a basin |
| `Z1.0` | Depth to 1.0 km/s shear-wave velocity (meters) | | 
| `Z2.5` | Depth to 2.5 km/s shear-wave velocity (meters) | |
| `vs30` | Time-averaged shear-wave velocity in the top 30 meters (m/s) | |

Region code mapping:

| Region | Code |
|--------|------|
| North Auckland | NA |
| South Auckland | SA |
| Hawkes Bay | HB |
| Gisborne | GI |
| Taranaki | TA |
| Wellington | WE |
| Nelson | NE |
| Marlborough | MA |
| Westland | WL |
| Canterbury | CA |
| Otago | OT |
| Southland | SL |
| No Region | NR |


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
For details/help run `python site_gen_cmds.py gen-plot --help`.

Running this will produce a html file that can be viewed in any browser. 
Note that for large domains and high density grids this might be slow to load.





