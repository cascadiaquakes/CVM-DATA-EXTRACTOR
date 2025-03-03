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
        lat_points = int(event['lat_points'])
        lng_points = int(event['lng_points'])
        variables_hidden = event['variables_hidden'].split(',')
        interpolation_method = event.get('interpolation_method', 'none')
        output_format = event.get('output_format', 'csv')
        units = event.get('units', 'km')

        logger.info(f"Processing file: {data_file} from S3 bucket: {bucket_name}")

        # For volume, no interpolation.
        interpolation_method == "none"

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




"""
        # Generate grid points
        lat_points_list = np.linspace(start_lat, end_lat, lat_points)
        lon_points_list = np.linspace(start_lng, end_lng, lng_points)

        # Slicing or interpolation
        if interpolation_method == "none":
            start_lat = min(ds['latitude'], key=lambda x: abs(x - start_lat))
            end_lat = min(ds['latitude'], key=lambda x: abs(x - end_lat))
            start_lng = min(ds['longitude'], key=lambda x: abs(x - start_lng))
            end_lng = min(ds['longitude'], key=lambda x: abs(x - end_lng))
            start_depth_closest = min(ds['depth'], key=lambda x: abs(x - start_depth))

            sliced_data = ds[variables_hidden].sel(
                latitude=slice(start_lat, end_lat),
                longitude=slice(start_lng, end_lng),
                depth=start_depth_closest
            )
            logger.info("Data slicing completed without interpolation.")
        else:
            interpolated_data = {var: ds[var].interp(
                latitude=lat_points_list,
                longitude=lon_points_list,
                depth=start_depth,
                method=interpolation_method
            ) for var in variables_hidden}
            sliced_data = xr.Dataset(interpolated_data)
            logger.info(f"Interpolation applied using method: {interpolation_method}")
"""
##########################################################

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
            end_x, end_y = project_lonlat_utm(end_lng, end_lat, utm_zone, ellipsoid)
            x = "x"
            y = "y"
        else:
            start_x = start_lng
            start_y = start_lat
            end_x = end_lng
            end_y = end_lat
            x = "longitude"
            y = "latitude"

        subset_limits = dict()
        subset_limits[x] = [start_x, end_x]
        subset_limits[y] = [start_y, end_y]
        subset_limits["depth"] = [start_depth, end_depth]

        sliced_data = ds[variables_to_keep]

        limit_keys = list(subset_limits.keys())
        limit_values = list(subset_limits.values())
        count_after_first_filter = (
            (
                (sliced_data[limit_keys[0]] >= limit_values[0][0])
                & [limit_keys[0]] <= limit_values[0][1])
            )
            .sum()
            .item()
        )
        logger.debug(
            f"Estimated output count_after_first_filter: {count_after_first_filter}"
        )
        count_after_second_filter = (
            (
                (sliced_data[limit_keys[1]] >= limit_values[1][0])
                & (sliced_data[limit_keys[1]] <= limit_values[1][1])
            )
            .sum()
            .item()
        )

        count_after_third_filter = (
            (
                (sliced_data[limit_keys[2]] >= limit_values[2][0])
                & (sliced_data[limit_keys[2]] <= limit_values[2][1])
            )
            .sum()
            .item()
        )

        count_after_all_filters = (
            count_after_first_filter
            * count_after_second_filter
            * count_after_third_filter
            * len(variables_to_keep)
        )

        logger.debug(f"Estimated output grid dimension: {count_after_all_filters}")

        output_data, warnings = subsetter(ds[variables_to_keep], subset_limits)


##########################################################




        # Prepare output buffer
        output = BytesIO()
        unique_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}"
        # Define your S3 bucket and file details
        s3_bucket = bucket_name
        pre = "work_area/"
        file_name = f"{unique_id}_{data_file.split('/')[-1].replace('.nc', f'_volume{file_extensions[output_format]}')}"
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
