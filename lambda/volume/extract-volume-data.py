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
from utils import unit_conversion_factor, utm_to_latlon, project_lonlat_utm, closest, subsetter, interpolate_path
import h5netcdf
import dask

# Configure logging
logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

s3_client = boto3.client("s3")
file_extensions = {"netcdf": ".nc", "csv": ".csv", "geocsv": ".csv", "hdf5": ".h5", "zarr": ".zarr"}
delimiter = "|"
ndecimal = 4



def lambda_handler(event, context):
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
    variable_list = event['variables_hidden'].split(',')
    interpolation_method = event.get('interpolation_method', 'none')
    output_format = event.get('output_format', 'csv')
    units = event.get('units', 'km')

    # For volume, no interpolation.
    interpolation_method == "none"

    logger.info(f"Processing file: {data_file} from S3 bucket: {bucket_name}")
    # Log parsed input parameters
    logger.info(f"Input parameters: {event}")
    data_file_path = f"data/netcdf/{data_file}"
    s3_url = f"s3://{bucket_name}/{data_file_path}"

    try:
        # lazy loading with Dask.
        #ds = xr.open_dataset(os.path.join("static", "netcdf", data_file), chunks={})
        # Open S3 file using fsspec (without storage_options in open_dataset)
        fs = fsspec.filesystem("s3")

        # Use fsspec to open the file in read mode and pass to xarray
        file_obj = fs.open(s3_url, mode="rb")
        ds = xr.open_dataset(file_obj, engine="h5netcdf", chunks="auto")

        # Extract global metadata
        global_metadata = ds.attrs
        logger.info(f"Global metadata extracted: {global_metadata}")        
        meta = ds.attrs

        # Convert start_depth units to the desired standard. start_depth is always in KM.
        _, start_depth_factor = unit_conversion_factor("km", units)

        # Data depth will be updated, we change the start depth too.
        start_depth *= start_depth_factor
        end_depth *= start_depth_factor

        # Convert depth units to the desired standard
        unit_standard, depth_factor = unit_conversion_factor(
            ds["depth"].attrs["units"], units
        )
        logger.debug(
            f"[DEBUG] DEPTH: Convert depth units to the desired standard for output. From: {ds['depth']} {ds['depth'].attrs['units']} To: {units}, unit_standard: {unit_standard}, depth_factor: {depth_factor}. start_depth: {start_depth}, start_depth_factor: {start_depth_factor}"
        )

        if depth_factor != 1:
            # Apply depth conversion factor if needed
            new_depth_values = ds["depth"] * depth_factor
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=ds["depth"].attrs
            )
            new_depth.attrs["units"] = unit_standard
            if "standard_name" in new_depth.attrs:
                new_depth.attrs["long_name"] = (
                    f"{new_depth.attrs['standard_name']} [{unit_standard}]"
                )
            else:
                new_depth.attrs["long_name"] = (
                    f"{new_depth.attrs['variable']} [{unit_standard}]"
                )
            # Assign new depth coordinates with converted units
            ds = ds.assign_coords(depth=new_depth)
        else:
            # If no conversion is needed, just update units
            ds.depth.attrs["units"] = unit_standard
            if "standard_name" in ds.depth.attrs:
                ds.depth.attrs["long_name"] = (
                    f"{ds.depth.attrs['standard_name']} ({unit_standard})"
                )
            else:
                ds.depth.attrs["long_name"] = (
                    f"{new_depth.attrs['variable']} [{unit_standard}]"
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
        for var in variable_list:
            # Convert the variable unit to the desired standard
            _, var_factor = unit_conversion_factor(ds[var].attrs["units"], units)
            logger.debug(
                f"[DEBUG] Convert {var} units to the desired standard for output. From: {ds[var].attrs['units']} To: {units}, unit_standard: {_}, depth_factor: {var_factor}. "
            )
            ds[var].data *= var_factor

        # Initially estimate the output dimension sizes.
        limits = {
            "csv": 10000000,
            "geocsv": 10000000,
            "netcdf": 115000000,
            "hdf5": 115000000,
        }
        sliced_data = ds[variable_list]

        limit_keys = list(subset_limits.keys())
        limit_values = list(subset_limits.values())
        count_after_first_filter = (
            (
                (sliced_data[limit_keys[0]] >= limit_values[0][0])
                & (sliced_data[limit_keys[0]] <= limit_values[0][1])
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
            * len(variable_list)
        )

        logger.debug(f"Estimated output grid dimension: {count_after_all_filters}")

        # Check if the dataset size exceeds the maximum allowed size
        size_help = "\nYou may reduce the request size by following these steps:\n - Reduce the requested model dimensions\n - Limit the number of variables\n - Use km.kg.sec units instead of m.g.sec"
        format_message = ""
        if output_format in ["csv", "geocsv"]:
            format_message = f"Select 'netcdf' as the output format for a larger grid size allowance of {limits['netcdf']:,}."
        if count_after_all_filters > limits[output_format]:
            message =  f"[ERR] Your request has approximately {count_after_all_filters:,} data points, which exceeds the maximum allowed size of {limits[output_format]:,} for {output_format} format.{size_help}"
            logger.error(message)  # Log the error
        
            # Return a properly formatted JSON response
            return {
                "statusCode": 413,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "[ERR] Your request exceeds the allowed size.",
                    "details": [
                        f"Your request has approximately {count_after_all_filters:,} data points.",
                        f"The maximum allowed size for {output_format} is {limits[output_format]:,}.",
                        "You may reduce the request size by: Reducing the requested model dimensions and/or limiting the number of variables",
                        format_message
                    ]
                }),
            }

        output_data, warnings = subsetter(ds[variable_list], subset_limits)


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
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": "Data processed successfully", "s3_key": s3_key})
        }

    except Exception as e:
        logger.error("Error occurred: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        message =f"Error occurred: {e}, {traceback.format_exc()}"
        logger.error(message)  # Log the error
        
        # Return a properly formatted JSON response
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": message}),
        }

