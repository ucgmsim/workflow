## Virtual Simulation Sites

### Background

As every set of simulation has different goals and priorities, a single set of virtual stations that meets all requirements is not feasible. On the other hand we want to prevent every simulation set developing their own custom grid of virtual stations, as this would make comparison between simulation sets difficult.

Instead a high density uniform "general" grid has been developed, and the code implemented here allows the generation of custom virtual station grid for each simulation set by sub-sampling this "general" grid as needed. This "general" grid ensures consistency across simulation sets while also allowing for customization.

### Custom Grid Generation

A custom grid can be generated using the 'virt_sites_cmds.py' script using the 'gen-custom-grid-from-rel' command.
For details/help on the command run 'python virt_sites_cmds.py gen-custom-grid-from-rel --help'
**Note that you will need about 16Gb of free memory.**

Running the custom grid generation will produce a parquet file with the columns 'site_id', 'lon', 'lat' and 'in_basin'.
Note that the 'in_basin' column is False for all sites unless a velocity model version was specified.

### Plot Generation

Visualisation of the custom grid can be done using the 'gen-plot' command.
For details/help 'run python virt_sites_cmds.py gen-plot --help'.

Running this will produce a html file that can be viewed in any browser. 
Note that for large domains and high density grids this might be slow to load.





