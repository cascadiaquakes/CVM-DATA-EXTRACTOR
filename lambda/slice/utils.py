import logging
from datetime import datetime
from pyproj import Proj, transform
from datetime import datetime, timezone
import numpy as np
import xarray as xr

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def interpolate_path(
    ds,
    start,
    end,
    num_points=100,
    method="linear",
    grid_ref="latitude_longitude",
    utm_zone=None,
    ellipsoid=None,
):
    """
    Interpolates a dataset along a path defined by start and end coordinates on an irregular grid.

    Parameters:
        ds (xarray.Dataset): The input dataset containing 'latitude' and 'longitude' as coordinates.
        start (tuple): A tuple (latitude, longitude) of the starting point.
        end (tuple): A tuple (latitude, longitude) of the ending point.
        num_points (int): Number of points to interpolate along the path.
        method (str): Interpolation method to use ('linear', 'nearest').

    Returns:
        xarray.Dataset: The interpolated dataset along the path.
    """
    # Create linearly spaced points between start and end
    lat_points = np.linspace(start[0], end[0], num_points)
    lon_points = np.linspace(start[1], end[1], num_points)

    # Define a path dataset for interpolation
    path = xr.Dataset(
        {"latitude": ("points", lat_points), "longitude": ("points", lon_points)}
    )

    # Interpolate the dataset to these points using the specified method
    if grid_ref == "latitude_longitude":
        interpolated_ds = ds.interp(
            latitude=path.latitude, longitude=path.longitude, method=method
        )
    else:
        if None in (utm_zone, ellipsoid):
            message = f"[ERR] for grid_ref: {grid_ref}, utm_zone and ellipsoid are required. Current values: {utm_zone}, {ellipsoid}!"
            logger.error(message)
            raise
        x_points = list()
        y_points = list()
        for index, lat_value in enumerate(lat_points):
            x, y = project_lonlat_utm(
                lon_points[index], lat_points[index], utm_zone, ellipsoid=ellipsoid
            )
            x_points.append(x)
            y_points.append(y)

        # Define a path dataset for interpolation
        path = xr.Dataset({"x": ("points", x_points), "y": ("points", y_points)})
        interpolated_ds = ds.interp(x=path.x, y=path.y, method=method)

    return interpolated_ds, lat_points, lon_points


def subsetter(sliced_data, limits):
    """
    Subset a dataset as a volume.

    Call arguments:
        ds - [required] the xarray dataset
        limits - [required] limits of the volume in all directions.
    """
    geospatial_dict = {
        "latitude": ["geospatial_lat_min", "geospatial_lat_max"],
        "longitude": ["geospatial_lon_min", "geospatial_lon_max"],
        "depth": ["geospatial_vertical_min", "geospatial_vertical_max"],
    }
    # Check if the array has any zero-sized dimensions
    warnings = ""

    try:
        limit_keys = list(limits.keys())
        limit_values = list(limits.values())

        logger.debug(
            f"""[DEBUG] EXTRACTING BASED ON
            ({limit_keys[0]} >= {limit_values[0][0]})
            & ({limit_keys[0]} <= {limit_values[0][1]})
            drop=True, from {sliced_data[limit_keys[0]].data}\n\n\n"""
        )

        # Apply the first filter
        sliced_data = sliced_data.where(
            (sliced_data[limit_keys[0]] >= limit_values[0][0])
            & (sliced_data[limit_keys[0]] <= limit_values[0][1]),
            drop=True,
        )

        logger.debug(
            f"""[DEBUG] EXTRACTING BASED ON
            ({limit_keys[1]} >= {limit_values[1][0]})
            & ({limit_keys[1]} <= {limit_values[1][1]})
            drop=True, from {sliced_data[limit_keys[1]].data}\n\n\n"""
        )
        # Apply the second filter
        sliced_data = sliced_data.where(
            (sliced_data[limit_keys[1]] >= limit_values[1][0])
            & (sliced_data[limit_keys[1]] <= limit_values[1][1]),
            drop=True,
        )

        logger.debug(
            f"""[DEBUG] EXTRACTING BASED ON
            ({limit_keys[2]} >= {limit_values[2][0]})
            & ({limit_keys[2]} <= {limit_values[2][1]})
            drop=True, from {sliced_data[limit_keys[2]].data}\n\n\n"""
        )
        # Apply the third filter
        sliced_data = sliced_data.where(
            (sliced_data[limit_keys[2]] >= limit_values[2][0])
            & (sliced_data[limit_keys[2]] <= limit_values[2][1]),
            drop=True,
        )

        for dim in limit_keys:
            if dim in geospatial_dict:
                #  The dropna method is used to remove coordinates with all NaN values along the specified dimensions
                sliced_data = sliced_data.dropna(dim=dim, how="all")
                if geospatial_dict[dim][0] in sliced_data.attrs:
                    sliced_data.attrs[geospatial_dict[dim][0]] = min(
                        sliced_data[dim].values
                    )
                if geospatial_dict[dim][1] in sliced_data.attrs:
                    sliced_data.attrs[geospatial_dict[dim][1]] = max(
                        sliced_data[dim].values
                    )
    except Exception as ex:
        warnings = ex
        return sliced_data, warnings

    return sliced_data, warnings
    
def closest(lst, value):
    """Find the closest number in a list to the given value

    Keyword arguments:
    lst -- [required] list of the numbers
    value -- [required] value to find the closest list member for.
    """
    arr = np.array(lst)
    # Check if the array is multi-dimensional
    if arr.ndim > 1:
        # Flatten the array and return
        flat_list = list(arr.flatten())
    else:
        # Return the original array if it's already one-dimensional
        flat_list = lst

    return flat_list[
        min(range(len(flat_list)), key=lambda i: abs(flat_list[i] - value))
    ]

def project_lonlat_utm(
    longitude, latitude, utm_zone, ellipsoid, xy_to_latlon=False, preserve_units=False
):
    """
    Performs cartographic transformations. Converts from longitude, latitude to UTM x,y coordinates
    and vice versa using PROJ (https://proj.org).

     Keyword arguments:
    longitude (scalar or array) – Input longitude coordinate(s).
    latitude (scalar or array) – Input latitude coordinate(s).
    xy_to_latlon (bool, default=False) – If inverse is True the inverse transformation from x/y to lon/lat is performed.
    preserve_units (bool) – If false, will ensure +units=m.
    """
    P = Proj(
        proj="utm",
        zone=utm_zone,
        ellps=ellipsoid,
    )
    # preserve_units=preserve_units,

    x, y = P(
        longitude,
        latitude,
        inverse=xy_to_latlon,
    )
    return x, y


def standard_units(unit):
    """Check an input unit and return the corresponding standard unit."""
    unit = unit.strip().lower()
    if unit in ["m", "meter"]:
        return "m"
    elif unit in ["degrees", "degrees_east", "degrees_north"]:
        return "degrees"
    elif unit in ["km", "kilometer"]:
        return "km"
    elif unit in ["g/cc", "g/cm3", "g.cm-3", "grams.centimeter-3"]:
        return "g/cc"
    elif unit in ["kg/m3", "kh.m-3"]:
        return "kg/m3"
    elif unit in ["km/s", "kilometer/second", "km.s-1", "kilometer/s", "km/s"]:
        return "km/s"
    elif unit in ["m/s", "meter/second", "m.s-1", "meter/s", "m/s"]:
        return "m/s"
    elif unit.strip().lower in ["", "none"]:
        return ""

def unit_conversion_factor(unit_in, unit_out):
    """Check input and output unit and return the conversion factor."""

    unit = standard_units(unit_in.strip().lower())
    logger.debug(f"[DEBUG] convert units {unit} to {unit_out}")
    if unit in ["m"]:
        if unit_out == "cgs":
            return standard_units("m"), 1
        else:
            return standard_units("km"), 0.001
    elif unit in ["km"]:
        if unit_out == "cgs":
            return standard_units("m"), 1000
        else:
            return standard_units("km"), 1
    elif unit in ["m/s"]:
        if unit_out == "cgs":
            return standard_units("m/s"), 1
        else:
            return standard_units("km/s"), 0.001
    elif unit in ["g/cc"]:
        if unit_out == "cgs":
            return standard_units("g/cc"), 1
        else:
            return standard_units("kg/m3"), 1000
    elif unit in ["km/s"]:
        if unit_out == "cgs":
            return standard_units("m/s"), 1000
        else:
            return standard_units("km/s"), 1
    elif unit in ["kg/m3"]:
        if unit_out == "cgs":
            return standard_units("g/cc"), 0.001
        else:
            return standard_units("kg/m3"), 1
    elif unit in ["", " "]:
        return standard_units(""), 1    
    elif unit in ["%"]:
        return standard_units("%"), 1
    elif unit in ["degrees"]:
        return standard_units("degrees"), 1

    else:
        logger.error(f"[ERR] Failed to convert units {unit_in} to {unit_out}")
        return unit_in, 1

def utm_to_latlon(x, y, zone, ellipsoid="WGS84"):
    proj_utm = Proj(proj="utm", zone=zone, ellps=ellipsoid)
    proj_latlon = Proj(proj="latlong", datum="WGS84")
    lon, lat = transform(proj_utm, proj_latlon, x, y)
    return lon, lat
