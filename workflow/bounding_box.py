"""Bounding boxes for simulation domains.

A bounding box is a rotated rectangle in NZTM coordinates. `BoundingBox` is
the type of `DomainParameters.domain` in realisations.

This module is vendored from `velocity_modelling.bounding_box` in
velocity-modelling 2026.8.1 (https://github.com/ucgmsim/velocity_modelling,
MIT licence). Only the parts workflow uses are kept, so that loading a
realisation doesn't import velocity-modelling and its dependency tree. Apart
from small type-checker fixes, the code is unchanged. The bounding box
dimensions are in metres except where otherwise mentioned.

Classes
-------
- BoundingBox: Represents a 2D bounding box with properties and methods for calculations.

Functions
---------
- minimum_area_bounding_box_for_polygons_masked: Returns a bounding box around masked polygons.

References
----------
- BoundingBox wiki page: https://github.com/ucgmsim/qcore/wiki/BoundingBox
"""

from typing import Self

import numpy as np
import numpy.typing as npt
import shapely
from shapely import Polygon

from qcore import coordinates, geo


class BoundingBox:
    """Represents a 2D bounding box with properties and methods for calculations.

    Parameters
    ----------
    bounds : npt.NDArray[np.float64]
        The bounds of the box in NZTM coordinates.

    Attributes
    ----------
    corners : np.ndarray
       The corners of the bounding box in cartesian coordinates. The
       order of the corners should be counter clock-wise from the bottom-left point
       (minimum x, minimum y).
    """

    bounds: npt.NDArray[np.float64]

    def __init__(self, bounds: npt.NDArray[np.float64]) -> None:
        """Create a bounding box from bounds in NZTM coordinates.

        Parameters
        ----------
        bounds : npt.NDArray[np.float64]
            The bounds of the box.
        """
        bottom_left_index = int((bounds - np.mean(bounds, axis=0)).sum(axis=1).argmin())
        bounds = np.copy(bounds)
        bounds[[0, bottom_left_index]] = bounds[[bottom_left_index, 0]]
        angles = np.arctan2(*(bounds[1:] - bounds[0]).T)

        indices = np.argsort(angles, kind="stable") + 1
        self.bounds = np.vstack([bounds[0], bounds[indices]])

    @property
    def corners(self) -> np.ndarray:
        """np.ndarray: the corners of the bounding box in (lat, lon) format."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

    def pad(
        self,
        pad_x: tuple[float, float] = (0.0, 0.0),
        pad_y: tuple[float, float] = (0.0, 0.0),
    ) -> Self:
        """Pad the bounding box by extending it in the x and y directions.

        Parameters
        ----------
        pad_x : tuple[float, float], default (0, 0)
            Padding distances in kilometers for x direction (left, right).
        pad_y : tuple[float, float], default (0, 0)
            Padding distances in kilometers for y direction (bottom, top).

        Returns
        -------
        Self
            A new instance of the bounding box with applied padding
        """
        bounds = self.bounds
        x_direction = bounds[1] - bounds[0]
        x_direction /= np.linalg.norm(x_direction)
        y_direction = bounds[-1] - bounds[0]
        y_direction /= np.linalg.norm(y_direction)
        delta = 1000 * np.array(
            [
                -pad_x[0] * x_direction - pad_y[0] * y_direction,
                pad_x[1] * x_direction - pad_y[0] * y_direction,
                pad_x[1] * x_direction + pad_y[1] * y_direction,
                -pad_x[0] * x_direction + pad_y[1] * y_direction,
            ]
        )

        return self.__class__(bounds + delta)

    @classmethod
    def from_centroid_bearing_extents(
        cls,
        centroid: npt.ArrayLike,
        bearing: float,
        extent_x: float,
        extent_y: float,
    ) -> Self:
        """Create a bounding box from a centroid, bearing, and size.

        The x and y-directions are determined relative to the bearing,

                 N      y-direction = bearing
                 │    /
                 │   /
                 │  /
                 │ /
                 │/
                 ■
                  ╲
                   ╲
                    ╲
                     ╲ x-direction = bearing + 90

        Parameters
        ----------
        centroid : np.ndarray
            The centre of the bounding box (lat, lon).
        bearing : float
            A bearing from north for the bounding box, in degrees.
        extent_x : float
            The length along the x-direction of the bounding box, in
            kilometres.
        extent_y : float
            The length along the y-direction of the bounding box, in
            kilometres.

        Returns
        -------
        Self
            The bounding box with the given centre, bearing, and
            length along the x and y-directions.
        """
        centroid = np.asarray(centroid)
        corner_offset = (
            np.array(
                [[-1 / 2, -1 / 2], [1 / 2, -1 / 2], [1 / 2, 1 / 2], [-1 / 2, 1 / 2]]
            )
            * np.array([extent_y, extent_x])
            * 1000
        ) @ geo.rotation_matrix(np.radians(-bearing))
        return cls(coordinates.wgs_depth_to_nztm(centroid) + corner_offset)

    @classmethod
    def bounding_box_for_geometry(
        cls, geometry: shapely.Geometry, axis_aligned: bool = False
    ) -> Self:
        """Return a bounding box that minimally encloses a geometry.

        Parameters
        ----------
        geometry : shapely.Geometry
            The geometry to enclose.
        axis_aligned : bool
            If True, ensure that the bounding box is axis-aligned.

        Returns
        -------
        Self
            The bounding box for this geometry.

        Raises
        ------
        ValueError
            If the geometry does not have a well-defined bounding box.
            This occurs if the geometry is degenerate (either a line
            or a point).
        """
        if axis_aligned:
            bounding_box_polygon = shapely.envelope(geometry).normalize()
        else:
            bounding_box_polygon = shapely.oriented_envelope(geometry).normalize()
        if not (
            isinstance(bounding_box_polygon, shapely.Polygon)
            and len(bounding_box_polygon.exterior.coords) - 1 == 4
        ):
            raise ValueError("Ill-defined geometry for bounding box.")
        return cls(np.array(bounding_box_polygon.exterior.coords)[:-1])

    @classmethod
    def from_wgs84_coordinates(cls, corner_coordinates: npt.ArrayLike) -> Self:
        """Construct a bounding box from a list of corners.

        Parameters
        ----------
        corner_coordinates : np.ndarray
            The corners in (lat, lon) format.

        Returns
        -------
        Self
            The bounding box represented by these corners.
        """
        return cls(
            np.asarray(coordinates.wgs_depth_to_nztm(np.asarray(corner_coordinates)))
        )

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        """np.ndarray: The origin of the bounding box."""
        return coordinates.nztm_to_wgs_depth(np.mean(self.bounds, axis=0))

    @property
    def extent_x(self) -> np.float64:
        """float: The extent along the x-axis of the bounding box (in km)."""
        return np.linalg.norm(self.bounds[1] - self.bounds[0]) / 1000

    @property
    def extent_y(self) -> np.float64:
        """float: The extent along the y-axis of the bounding box (in km)."""
        return np.linalg.norm(self.bounds[2] - self.bounds[1]) / 1000

    @property
    def bearing(self) -> np.float64:
        """float: The bearing of the bounding box."""
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        vertical_direction = np.append(self.bounds[-1] - self.bounds[0], 0)
        return geo.oriented_bearing_wrt_normal(
            north_direction, vertical_direction, up_direction
        )

    @property
    def great_circle_bearing(self) -> float:
        """float: The great-circle bearing of the bounding box.

        This returns the bearing of the bounding box in WGS84
        coordinate space (as opposed to in the NZTM coordinate space).
        """
        return coordinates.nztm_bearing_to_great_circle_bearing(
            self.origin, self.extent_y / 2, self.bearing
        )

    @property
    def area(self) -> np.float64:
        """float: The area of the bounding box."""
        return self.extent_x * self.extent_y

    @property
    def polygon(self) -> Polygon:
        """Polygon: The shapely geometry for the bounding box."""
        return Polygon(np.append(self.bounds, np.atleast_2d(self.bounds[0]), axis=0))

    def __repr__(self) -> str:
        """A representation of the bounding box."""
        cls = self.__class__.__name__
        return f"{cls}(centre={self.origin}, bearing={self.bearing}, extent_x={self.extent_x}, extent_y={self.extent_y}, corners={self.corners})"


def minimum_area_bounding_box_for_polygons_masked(
    must_include: list[Polygon], may_include: list[Polygon], mask: Polygon
) -> BoundingBox:
    """Find a minimum area bounding box for the points must_include ∪ (may_include ∩ mask).

    Parameters
    ----------
    must_include : list[Polygon]
        List of polygons the bounding box must include.
    may_include : list[Polygon]
        List of polygons the bounding box will include portions of, when inside of mask.
    mask : Polygon
        The masking polygon.

    Returns
    -------
    BoundingBox
        The smallest box containing all the points of `must_include`, and all the
        points of `may_include` that lie within the bounds of `mask`.

    """
    may_include_polygon = shapely.normalize(shapely.union_all(may_include))
    must_include_polygon = shapely.normalize(shapely.union_all(must_include))
    bounding_polygon = shapely.normalize(
        shapely.union(
            must_include_polygon, shapely.intersection(may_include_polygon, mask)
        )
    )
    return BoundingBox.bounding_box_for_geometry(bounding_polygon)
