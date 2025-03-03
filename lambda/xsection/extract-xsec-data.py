import os
import json
import boto3
import xarray as xr
import numpy as np
import h5py
from io import BytesIO
import traceback
import logging
import uuid
from datetime import datetime
from pyproj import Proj, transform
import s3fs
import fsspec
from utils import unit_conversion_factor, utm_to_latlon, project_lonlat_utm


# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
file_extensions = {"netcdf": ".nc", "csv": ".csv", "geocsv": ".csv", "hdf5": ".h5", "zarr": ".zarr"}
delimiter = "|"
ndecimal = 4


def subsetter(sliced_data, limits):
    """
    Subset a dataset as a cross-section.

    Call arguments:
        ds - [required] the xarray dataset
        limits - [required] limits of the cross-section in all directions.
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

def interpolate_path(ds, start, end, num_points, method, grid_ref, utm_zone=None, ellipsoid=None):
    """
    Interpolates a dataset along a path defined by start and end coordinates on an irregular grid.
    """

    # Create linearly spaced points between start and end
    lat_points = np.linspace(start[0], end[0], num_points)
    lon_points = np.linspace(start[1], end[1], num_points)

    if grid_ref == "latitude_longitude":
        path = xr.Dataset(
            {"latitude": ("points", lat_points), "longitude": ("points", lon_points)}
        )
        interpolated_ds = ds.interp(latitude=path.latitude, longitude=path.longitude, method=method)
    else:
        if None in (utm_zone, ellipsoid):
            raise ValueError("For non-geographic grids, utm_zone and ellipsoid are required.")
        P = Proj(proj="utm", zone=utm_zone, ellps=ellipsoid)
        x_points, y_points = P(lon_points, lat_points)
        path = xr.Dataset({"x": ("points", x_points), "y": ("points", y_points)})
        interpolated_ds = ds.interp(x=path.x, y=path.y, method=method)

    return interpolated_ds, lat_points, lon_points


def lambda_handler(event, context):
    try:
        logger.info("Received event: %s", json.dumps(event, indent=2))

        # Extract input parameters
        bucket_name = event['bucket_name']
        data_file = event['data_file']
        start_lat = float(event['start_lat'])
        start_lng = float(event['start_lng'])
        end_lat = float(event['end_lat'])
        end_lng = float(event['end_lng'])
        start_depth = float(event['start_depth'])
        end_depth = float(event['end_depth'])
        variables_hidden = event['variables_hidden'].split(',')
        interpolation_method = event.get('interpolation_method', 'none')
        num_points = int(event['num_points'])
        output_format = event.get('output_format', 'csv')
        units = event.get('units', 'km')

        # Log parsed input parameters
        logger.info("Input parameters: bucket_name=%s, data_file=%s, start_lat=%f, start_lng=%f, end_lat=%f, end_lng=%f, start_depth=%f, end_depth=%f, num_points=%d, variables_hidden=%s, numpoints=%d, interpolation_method=%s, output_format=%s",
                    bucket_name, data_file, start_lat, start_lng, end_lat, end_lng, start_depth, end_depth, num_points, variables_hidden, interpolation_method, output_format)

        # Fetch the NetCDF file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=f"data/netcdf/{data_file}")
        netcdf_data = response['Body'].read()
        ds = xr.open_dataset(BytesIO(netcdf_data))

        # Extract global metadata
        global_metadata = ds.attrs
        logger.info(f"Global metadata extracted: {global_metadata}")

        logger.info(f"NetCDF file loaded successfully: {data_file}")

        # Convert depth units
        depth_unit, depth_factor = unit_conversion_factor(ds['depth'].attrs.get('units'), units)
        ds = ds.assign_coords(depth=ds['depth'] * depth_factor)

        ds['depth'].attrs['units'] = depth_unit  # Update metadata
        start_depth *= depth_factor

        logger.info(f"Converted depth units to {units}")

        var_conversion_log = ""
        # Apply unit conversion to selected variables
        for var in variables_hidden:
            if var in ds:
                var_unit, var_conversion_factor = unit_conversion_factor(ds[var].attrs.get("units"), units)
                var_conversion_log = f"{var_conversion_log}\nVar {var} from {ds[var].attrs["units"]} to {var_unit} via {var_conversion_factor}"
                ds[var] *= var_conversion_factor
                ds[var].attrs["units"] = var_unit

        logger.info(f"Unit conversion applied to all variables. {var_conversion_log}")
        
        # We will be working with the variable dataset, so capture the updated metadata for the main dataset now.
        meta = ds.attrs

        # Filter the dataset for the depth range
        ds = ds.where((ds.depth >= start_depth) & (ds.depth <= end_depth), drop=True)

        # Interpolate the dataset along the cross-section path
        interpolated_ds, lat_points, lon_points = interpolate_path(
            ds, 
            start=(start_lat, start_lng), 
            end=(end_lat, end_lng), 
            num_points=num_points, 
            method=interpolation_method, 
            grid_ref=grid_ref, 
            utm_zone=utm_zone, 
            ellipsoid=ellipsoid
        )

        # Extract grid ref metadata
        grid_ref= meta.get("grid_ref", "latitude_longitude")
        logger.debug(f"[DEBUG] Meta: {meta}\n\ngrid_ref:{grid_ref}")

        # If data is not "latitude_longitude", we need to get the start and end in the primary axis.
        if grid_ref != "latitude_longitude":
            utm_zone = meta.get("utm_zone")
            ellipsoid = meta.get("ellipsoid")
            start_x, start_y = project_lonlat_utm(
                start_lng, start_lat, utm_zone, ellipsoid
            )

        # Compute cumulative distances along the cross-section path
        geod = Geod(ellps="WGS84")
        _, _, distances = geod.inv(
            lon_points[:-1], lat_points[:-1], lon_points[1:], lat_points[1:]
        )
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

        # Add distance as a coordinate
        output_data = interpolated_ds.assign_coords(distance=("points", cumulative_distances))
        output_data = output_data.swap_dims({"points": "distance"})

        # Select the required variables
        selected_vars = output_data[variables_hidden]

        # Prepare output buffer
        output = BytesIO()
        unique_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}"

        # Define your S3 bucket and file details
        s3_bucket = bucket_name
        pre = "work_area/"
        file_name = f"{unique_id}_{data_file.split('/')[-1].replace('.nc', f'_xsection{file_extensions[output_format]}')}"
        s3_key = f'{pre}{file_name}'

        # Define the S3 path
        s3_path = f's3://{s3_bucket}/{s3_key}'

        # Save in requested format
        ## **Use Different Buffers for Different Formats**
        # Directly writing NetCDF or HDF5 files to S3 can be challenging due to the 
        # need for file-like objects that support seeking. 
        if output_format == "csv":
            # Convert the Dataset to a DataFrame
            df = output_data.to_dataframe().reset_index().round(ndecimal)
            # Write the DataFrame to CSV directly to S3
            fs = s3fs.S3FileSystem()
            with fs.open(s3_path, 'w') as f:
                df.to_csv(f, sep=",", index=False)
            logger.info("Data written in CSV format.")

        elif output_format == "netcdf":
            # Set the local file path for temporary NetCDF output
            local_file = f"/tmp/{file_name}.nc"

            # Embed global metadata into the output NetCDF
            output_data.attrs = global_metadata

            # Write to NetCDF using the NetCDF4 classic model with compression
            encoding = {var: {"zlib": True, "complevel": 5} for var in output_data.data_vars}

            output_data.to_netcdf(local_file, format="NETCDF4_CLASSIC", encoding=encoding)

            # Upload the file to S3
            s3_client.upload_file(local_file, s3_bucket, s3_key)

            # Delete the local file after successful upload
            if os.path.exists(local_file):
                os.remove(local_file)
                logger.info(f"Deleted local file: {local_file}")
            else:
                logger.warning(f"File not found for cleanup: {local_file}")

            logger.info("Data written in NetCDF4 classic format with compression.")


        elif output_format == "geocsv":
            metadata = f"# dataset: GeoCSV 2.0\n# delimiter: {delimiter}\n"
            for key, value in global_metadata.items():
                metadata += f"# global_{key}: {value}\n"
                
            # Add variable-specific metadata
            for var in output_data.variables:
                metadata += f"# {var}_column: {var}\n"
                for attr_key, attr_value in output_data[var].attrs.items():
                    metadata += f"# {var}_{attr_key}: {attr_value}\n"

            # Convert the Dataset to a DataFrame
            df = output_data.to_dataframe().reset_index().round(ndecimal)

            # Write the DataFrame to CSV directly to S3
            fs = s3fs.S3FileSystem()
            with fs.open(s3_path, 'w') as f:
                f.write(metadata)  # Write metadata as comments at the top
                df.to_csv(f, sep=delimiter, index=False)
            logger.info("Data written in GeoCSV format.")

        elif output_format == "hdf5":
            # Set the local file path for temporary HDF5 output
            local_hdf5_file = f"/tmp/{file_name}"

            # Embed global metadata
            output_data.attrs = global_metadata

            # Write to HDF5 using xarray’s NetCDF backend (which uses HDF5)
            output_data.to_netcdf(local_hdf5_file, format="NETCDF4", engine="h5netcdf")

            # Upload the HDF5 file to S3
            s3_client.upload_file(local_hdf5_file, s3_bucket, s3_key)

            # Clean up local file after upload
            if os.path.exists(local_hdf5_file):
                os.remove(local_hdf5_file)
                logger.info(f"Deleted local file: {local_hdf5_file}")
            else:
                logger.warning(f"Local file not found for cleanup: {local_hdf5_file}")

            logger.info("Data written in HDF5 format (NetCDF4).")

        else:
            raise ValueError("Unsupported output format")

        logger.info(f"Data successfully saved to S3: {s3_path}")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Data processed successfully", "s3_key": s3_key})
        }

    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e), "traceback": traceback.format_exc()})
        }
