# Realisations

This documentation outlines how to create, read and write realisations
using the new realisations workflow. To understand the rationale
behind this module, see [the proposal](Realisations-Proposal.md) on
realisations. Note that the proposal and the implementation differ
slightly as some components of the implementation have changed.

# Creating and Writing Realisations

Realisations are created by instantiating configuration objects, and
then reading or writing these to a realisation file. Realisations can
be incompelete in the sense that they may miss configuration
components at certain steps in realisation generation. To reflect this, the library interface is modular. Let's take a look at an example:

```python
from workflow import realisations
import numpy as np

domain_parameters = realisations.DomainParameters(
    resolution=0.1, # a 0.1km resolution
    centroid=np.array([-43.53092, 172.63701]),
    width=100.0,
    length=100.0,
    depth=40.0
)

realisations.write_config_to_realisation(domain_parameters, 'path/to/realisation.json')
```

Inside the `realisation.json` file you would find the following

```json
{
  "domain": {
    "resolution": 0.1,
    "centroid": {
      "latitude": -43.53092,
      "longitude": 172.63701
    },
    "width": 100.0,
    "length": 100.0,
    "depth": 40.0
  }
}
```

Everything under the `domain` keyword is the configuration for the domain parameters. Suppose after this we wished to add parameters for SRF generation. Then, we would write something like

```python
srf_config = realisations.SRFConfig(
    genslip_dt = 1.0,
    genslip_seed=1,
    genslip_version='5.4.2',
    srfgen_seed=1
)

realisations.write_config_to_realisation(srf_config, 'path/to/realisation.json')
```

And then, inside the `realisation.json` file you'll find

```json
{
  "domain": {
    "resolution": 0.1,
    "centroid": {
      "latitude": -43.53092,
      "longitude": 172.63701
    },
    "width": 100.0,
    "length": 100.0,
    "depth": 40.0
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

To read a realisation, you simply call `read_config_from_realisation` with the filepath of the realisation and the type of config you wish to read. For example, if the `realisations.json` is as in the previous section

```json
{
  "domain": {
    "resolution": 0.1,
    "centroid": {
      "latitude": -43.53092,
      "longitude": 172.63701
    },
    "width": 100.0,
    "length": 100.0,
    "depth": 40.0
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
>>> realisations.read_config_from_realisation(realisations.DomainParameters, 'path/to/realisations.json')
DomainParameters(resolution=0.1, centroid=array([-43.53092, 172.63701]), width=100.0, length=100.0, depth=40.0)
```

At read time, basic input validation checks are made. If we made the `resolution` parameter 0, then we get an error
```python
>>> realisations.read_config_from_realisation(realisations.DomainParameters, 'path/to/realisations.json')
Traceback (most recent call last):
...
schema.SchemaError: Key 'resolution' error:
is_positive(0.0) should evaluate to True
```

# Creating Your Own Configuration Section

To add your own realisation config object, you need to:
1. Create a configuration object,
2. Populate `_REALISATION_SCHEMAS` with the configuration schema and,
3. Populate `_REALISATION_KEYS` with the section key your config variables live in.

We will now walk through an example adding the `DomainParameters` object to the realisation specification.

## Creating a Configuration Object
We first create the class with the all the domain parameters we need for our simulation

```python
@dataclasses.dataclass
class DomainParameters:
    """
    Parameters defining the spatial domain for simulation.

    Attributes
    ----------
    resolution : float
        The simulation resolution in kilometers.
    centroid : np.ndarray
        The centroid location of the model in latitude and longitude coordinates.
    width : float
        The width of the model in kilometers.
    length : float
        The length of the model in kilometers.
    depth : float
        The depth of the model in kilometers.
    """

    resolution: float
    centroid: np.ndarray
    width: float
    length: float
    depth: float

    def to_dict(self):
        """
        Convert the object to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the object.
        """
        param_dict = dataclasses.asdict(self)
        param_dict["centroid"] = to_lat_lon_dictionary(self.centroid)
        return param_dict
    
```

We have to write a `to_dict` method for any configuration object we create. This method specifies how to serialise the configuration into a dictionary. The keys and values must be JSON-serialisable python objects. Most of the time, you are fine to just have this method return the output of `dataclasses.asdict`. If you are writing numpy values you will need to write a custom serialisation function. The helper function `to_lat_lon_dictionary` converts numpy arrays of varying shapes into lists of keyword dictionaries specifying coordinates.
## Creating the Schema
The next step is specifying the configuration schema. This is done by populating the `_REALISATION_SCHEMAS` variable with your configuration schema. The key should be the type of your configuration object. The value should be the schema. Schemas should validate the types and general bounds of their input variables. They should not perform complex input validation. You should describe each keyword using the `description=` keyword argument and a schema `Literal`. There are a number of prewritten helper functions like `is_positive` to validate certain inputs.

```python
_REALISATION_SCHEMAS = {
    # ...
    DomainParameters: Schema(
        {
            Literal("resolution", description="The simulation resolution (in km)"): And(
                float, is_positive
            ),
            Literal(
                "centroid", description="The centroid location of the model"
            ): LAT_LON_SCHEMA,
            Literal("width", description="The width of the model (in km)"): And(
                float, is_positive
            ),
            Literal("length", description="The length of the model (in km)"): And(
                float, is_positive
            ),
            Literal("depth", description="The depth of the model (in km)"): And(
                float, is_positive
            ),
        }
    ),
}
```
## Specifying the Configuration Key
We now need to update the `_REALISATION_KEYS` variable to tell the configuration loader where to find our configuration in the realisation file. The domain parameters should be loaded under the `domain` section, so we specify this as the realisation key.

```python
_REALISATION_KEYS = {
    # ...
    DomainParameters: "domain",
}
```
