# Realisations

This documentation outlines how to create, read and write realisations
using the new realisations workflow. To understand the rationale
behind this module, see [the proposal](Realisations-Proposal.md) on
realisations. Note that the proposal and the implementation differ
slightly as some components of the implementation have changed.

# Creating and Writing Realisations

Realisations are created by instantiating configuration objects, and
then reading or writing these to a realisation file. Realisations can
be incomplete in the sense that they may miss configuration
components at certain steps in realisation generation. To reflect this, the library interface is modular. Let's take a look at an example:

```python
from workflow import realisations
from velocity_modelling import bounding_box
import numpy as np

domain_parameters = realisations.DomainParameters(
    resolution=0.1, # a 0.1km resolution
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

domain_parameters.write_to_realisation('path/to/realisation.json')
```

Inside the `realisation.json` file you would find the following

```json
{
  "domain": {
    "resolution": 0.1,
    "domain": [
      {
        "latitude": -43.524793866326725,
        "longitude": 171.76204128885567
      },
      {
        "latitude": -42.894200350955856,
        "longitude": 172.64076673694242
      },
      {
        "latitude": -43.53034935969409,
        "longitude": 173.51210368762364
      },
      {
        "latitude": -44.16756820707226,
        "longitude": 172.63312824122775
      }
    ],
    "depth": 40.0,
    "duration": 60.0,
    "dt": 0.005
  }
}
```

Everything under the `domain` keyword is the configuration for the domain parameters. Suppose after this we wished to add parameters for SRF generation. Then, we would write something like

```python
srf_config = realisations.SRFConfig(
    genslip_dt = 1.0,
    genslip_seed=1,
    genslip_version='5.4.2',
    resolution=0.1,
    srfgen_seed=1
)

srf_config.write_to_realisation('path/to/realisation.json')
```

And then, inside the `realisation.json` file you'll find

```json
{
  "domain": {
    "resolution": 0.1,
    "domain": [
      {
        "latitude": -43.524793866326725,
        "longitude": 171.76204128885567
      },
      {
        "latitude": -42.894200350955856,
        "longitude": 172.64076673694242
      },
      {
        "latitude": -43.53034935969409,
        "longitude": 173.51210368762364
      },
      {
        "latitude": -44.16756820707226,
        "longitude": 172.63312824122775
      }
    ],
    "depth": 40.0,
    "duration": 60.0,
    "dt": 0.005
  },
  "srf": {
    "genslip_dt": 1.0,
    "genslip_seed": 1,
    "genslip_version": "5.4.2",
    "srfgen_seed": 1
  }
}
```

Notice that the srf config script did not need to read domain parameters to update the realisation file. This is what allows the realisations to be build bit-by-bit, in the same way that our workflow currently generates these files. If we were to have a central `Realisation` object, we'd end up needing to specify a lot of default dummy values everytime we loaded an incomplete realisation file.

# Reading Realisations

To read a realisation, you simply call the `read_from_realisation` classmethod with the filepath of the realisation and from the type of config you wish to read. For example, if the `realisations.json` is as in the previous section

```json
{
  "domain": {
    "resolution": 0.1,
    "domain": [
      {
        "latitude": -43.524793866326725,
        "longitude": 171.76204128885567
      },
      {
        "latitude": -42.894200350955856,
        "longitude": 172.64076673694242
      },
      {
        "latitude": -43.53034935969409,
        "longitude": 173.51210368762364
      },
      {
        "latitude": -44.16756820707226,
        "longitude": 172.63312824122775
      }
    ],
    "depth": 40.0,
    "duration": 60.0,
    "dt": 0.005
  },
  "srf": {
    "genslip_dt": 1.0,
    "genslip_seed": 1,
    "genslip_version": "5.4.2",
    "srfgen_seed": 1
  }
}
```

Then, we may read the domain parameters with the following code

```python
>>> realisations.DomainParameters.read_from_realisation('path/to/realisations.json')
DomainParameters(resolution=0.1, domain=..., depth=40.0, duration=60.0, dt=0.005)
```

At read time, basic input validation checks are made. If we made the `resolution` parameter 0, then we get an error
```python
>>> realisations.DomainParameters.read_from_realisation('path/to/realisations.json')
Traceback (most recent call last):
...
schema.SchemaError: Key 'resolution' error:
is_positive(0.0) should evaluate to True
```

# Creating Your Own Configuration Section

To add your own realisation config object, you need to create a configuration object inheriting from `RealisationConfiguration` and assign the object a matching schema.

We will now walk through an example adding the `DomainParameters` object to the realisation specification.

## Creating a Configuration Object
We first create the class with the all the domain parameters we need for our simulation

```python
@dataclasses.dataclass
class DomainParameters(RealisationConfiguration):
    """
    Parameters defining the spatial and temporal domain for simulation.

    Attributes
    ----------
    resolution : float
        The simulation resolution in kilometres.
    domain : BoundingBox
        The bounding box for the domain.
    depth : float
        The depth of the domain (in metres).
    duration : float
        The simulation duration (in seconds).
    dt : float
        The resolution of the domain in time (in seconds).
    """

    _config_key: ClassVar[str] = "domain"
    _schema: ClassVar[Schema] = schemas.DOMAIN_SCHEMA

    resolution: float
    domain: BoundingBox
    depth: float
    duration: float
    dt: float

    @property
    def nx(self) -> int:
        """int: The number of x coordinate positions in the discretised domain."""
        return int(np.round(self.domain.extent_x / self.resolution))

    @property
    def ny(self) -> int:
        """int: The number of y coordinate positions in the discretised domain."""
        return int(np.round(self.domain.extent_y / self.resolution))

    @property
    def nz(self) -> int:
        """int: The number of z coordinate positions in the discretised domain."""
        return int(np.round(self.depth / self.resolution))

    def to_dict(self) -> dict:
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        param_dict = dataclasses.asdict(self)
        param_dict["domain"] = to_name_coordinate_dictionary(
            self.domain.corners,
        )
        return param_dict
```

We have to write a `to_dict` method for any configuration object we create. This method specifies how to serialise the configuration into a dictionary. The keys and values must be JSON-serialisable python objects. Most of the time, you are fine to just have this method return the output of `dataclasses.asdict`. If you are writing numpy values you will need to write a custom serialisation function. The helper function `to_name_coordinate_dictionary` converts numpy arrays of varying shapes into lists of keyword dictionaries specifying coordinates.

The value of `_config_key` is the key the configuration is read and written to from the realisation. The `_schema` class variable points to the schema to validate input with when loading the realisation.
## Creating the Schema
The next step is specifying the configuration schema. Schemas should validate the types and general bounds of their input variables. They should not perform complex input validation. You should describe each keyword using the `description=` keyword argument and a schema `Literal`. There are a number of prewritten helper functions like `is_positive` to validate certain inputs.

```python
DOMAIN_SCHEMA = Schema(
    {
        Literal("resolution", description="The simulation resolution (in km)"): And(
            float, is_positive
        ),
        Literal("domain", description="The corners of the simulation domain."): And(
            Use(corners_to_array), Use(BoundingBox.from_wgs84_coordinates)
        ),
        Literal("depth", description="The depth of the model (in km)"): And(
            float, is_positive
        ),
        Literal(
            "duration", description="The duration of the simulation (in seconds)"
        ): And(float, is_positive),
        Literal("dt", "The resolution of the domain in time (in seconds)."): And(
            float, is_positive
        ),
    }
)
```
