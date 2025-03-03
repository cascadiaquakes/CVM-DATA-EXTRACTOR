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
from utils import unit_conversion_factor, utm_to_latlon, project_lonlat_utm, closest
import h5netcdf
import dask


# Configure logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

s3_client = boto3.client("s3")
file_extensions = {"netcdf": ".nc", "csv": ".csv", "geocsv": ".csv", "hdf5": ".h5", "zarr": ".zarr"}
delimiter = "|"
ndecimal = 4



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
        lat_points = int(event['lat_points'])
        lng_points = int(event['lng_points'])
        variable_list = event['variables_hidden'].split(',')
        interpolation_method = event.get('interpolation_method', 'none')
        output_format = event.get('output_format', 'csv')
        units = event.get('units', 'km')

        logger.info(f"Processing file: {data_file} from S3 bucket: {bucket_name}")

        data_file_path = f"data/netcdf/{data_file}"
        s3_url = f"s3://{bucket_name}/{data_file_path}"

        # Open S3 file using fsspec (without storage_options in open_dataset)
        fs = fsspec.filesystem("s3")

        # Use fsspec to open the file in read mode and pass to xarray
        file_obj = fs.open(s3_url, mode="rb")
        output_data = xr.open_dataset(file_obj, engine="h5netcdf", chunks="auto")

        # Extract global metadata
        global_metadata = output_data.attrs
        logger.info(f"Global metadata extracted: {global_metadata}")

        logger.info(f"NetCDF file loaded successfully: {data_file}")
        try:
            # Convert start_depth units to the desired standard. start_depth is always in KM.
            _, start_depth_factor = unit_conversion_factor("km", units)

            # Convert depth units to the desired standard
            unit_standard, depth_factor = unit_conversion_factor(
                output_data["depth"].attrs["units"], units
            )
            logger.debug(
                f"[DEBUG] DEPTH: Convert depth units to the desired standard for output. From: {output_data['depth'].attrs['units']} To: {units}, unit_standard: {unit_standard}, depth_factor: {depth_factor}. start_depth: {start_depth}, start_depth_factor: {start_depth_factor}"
            )

            # Data depth will be updated, we change the start depth too.
            start_depth *= start_depth_factor

            if depth_factor != 1:
                # Apply depth conversion factor if needed
                new_depth_values = output_data["depth"] * depth_factor
                new_depth = xr.DataArray(
                    new_depth_values, dims=["depth"], attrs=output_data["depth"].attrs
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
                output_data = output_data.assign_coords(depth=new_depth)
            else:
                # If no conversion is needed, just update units
                output_data.depth.attrs["units"] = unit_standard
                if "standard_name" in output_data.depth.attrs:
                    output_data.depth.attrs["long_name"] = (
                        f"{output_data.depth.attrs['standard_name']} ({unit_standard})"
                    )
                else:
                    output_data.depth.attrs["long_name"] = (
                        f"{output_data.depth.attrs['variable']} ({unit_standard})"
                    )
            meta = output_data.attrs  # Extract metadata from the dataset

            # Split the hidden variables list by commas
            selected_data_vars = output_data[
                variable_list
            ]  # Select only the specified variables

            # Apply unit conversion to each selected variable
            for var in variable_list:
                unit_standard, var_factor = unit_conversion_factor(
                    selected_data_vars[var].attrs["units"], units
                )
                logger.debug(
                    f"[DEBUG] VARS: Convert {var} units to the desired standard for output. From: {output_data[var].attrs['units']} To: {units}, unit_standard: {unit_standard}, depth_factor: {depth_factor}"
                )
                # Adjust data by the conversion factor
                selected_data_vars[var].data *= var_factor
                # Update units and display name with the new units
                selected_data_vars[var].attrs["units"] = unit_standard
                selected_data_vars[var].attrs[
                    "display_name"
                ] = f"{selected_data_vars[var].attrs['long_name']} [{unit_standard}]"

        except Exception as ex:
            message = f"[ERR] {ex}"
            logger.error(message)  # Log the error

            # Return a properly formatted JSON response
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": message}),
            }
        logger.info(f"Unit conversion applied to all variables. {variable_list}")
        try:
            # Extract grid ref metadata
            grid_ref = meta.get("grid_ref", "latitude_longitude")
            utm_zone = meta.get("utm_zone")
            ellipsoid = meta.get("ellipsoid")

            # Generate coordinate grids based on grid ref type
            if grid_ref == "latitude_longitude":
                # For lat/lon grid, generate lists of latitudes and longitudes
                lon_list = np.linspace(start_lng, end_lng, lng_points).tolist()
                lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
                x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
                x_grid, y_grid = np.meshgrid(
                    lon_list, lat_list
                )  # Create a meshgrid for lat/lon
            else:
                # For UTM grid, project lat/lon to UTM coordinates
                start_x, start_y = project_lonlat_utm(
                    start_lng, start_lat, utm_zone, ellipsoid
                )
                end_x, end_y = project_lonlat_utm(end_lng, end_lat, utm_zone, ellipsoid)
                x_list = np.linspace(start_x, end_x, lng_points).tolist()
                y_list = np.linspace(start_y, end_y, lat_points).tolist()
                # Convert UTM coordinates back to lat/lon
                lon_list, lat_list = project_lonlat_utm(
                    x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
                )
                x_grid, y_grid = np.meshgrid(x_list, y_list)  # Create a meshgrid for UTM

            interp_type = interpolation_method  # Set interpolation method

            logger.debug(
                f"[DEBUG] Estimated size: {len(x_grid)}x{len(y_grid)}x{len(variable_list)}: {len(x_grid)*len(y_grid)*len(variable_list)}"
            )
            if interp_type == "none":
                # If no interpolation, find the closest grid points for slicing
                start_lat = closest(output_data["latitude"].data, start_lat)
                end_lat = closest(output_data["latitude"].data, end_lat)
                start_lng = closest(output_data["longitude"].data, start_lng)
                end_lng = closest(output_data["longitude"].data, end_lng)

                depth_closest = closest(list(output_data["depth"].data), start_depth)
                logger.debug(
                    f"[DEBUG] Depth: {output_data['depth']}\nLooking for: {start_depth}\nClosest: {depth_closest}"
                )

                # xarray.where() does not support boolean dask arrays for indexing, unless the condition is computed.
                # Ensure all masks are computed first
                depth_mask = (output_data.depth == depth_closest).compute()

                lat_lng_mask = (
                    (output_data.latitude >= start_lat)
                    & (output_data.latitude <= end_lat)
                    & (output_data.longitude >= start_lng)
                    & (output_data.longitude <= end_lng)
                ).compute()

                # Apply computed masks
                selected_data_vars = selected_data_vars.where(depth_mask, drop=True)
                selected_data_vars = selected_data_vars.where(lat_lng_mask, drop=True)
                data_to_return = selected_data_vars  # Return filtered data
                logger.debug(
                    f"[DEBUG] Dataset info: {data_to_return.info()}"
                )

            else:
                # Check for NaN values in the interpolated data
                if any(
                    np.isnan(selected_data_vars[var]).any() and interp_type == "cubic"
                    for var in variable_list
                ):
                    message = "[ERR] Data with NaN values. Can't use the cubic spline interpolation"
                    logger.error(message)  # Log the error

                    # Return a properly formatted JSON response
                    return {
                        "statusCode": 400,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({"error": message}),
                    }
                    
                # If interpolation is required
                interpolated_values_2d = {}
                for var in variable_list:
                    # Interpolate based on latitude/longitude or UTM coordinates
                    if grid_ref == "latitude_longitude":
                        interpolated_values_2d[var] = selected_data_vars[var].interp(
                            latitude=lat_list,
                            longitude=lon_list,
                            depth=start_depth,
                            method=interp_type,
                        )
                    else:
                        interpolated_values_2d[var] = selected_data_vars[var].interp(
                            y=y_list, x=x_list, depth=start_depth, method=interp_type
                        )

                data_to_return = xr.Dataset(
                    interpolated_values_2d
                )  # Return interpolated data as dataset
        except Exception as ex:
            message = f"Error 400 Bad Request: Unsupported output format  {ex}\n{traceback.print_exc()}"
            logger.error(message)  # Log the error

            # Return a properly formatted JSON response
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": message}),
            }

        # Prepare output buffer
        output = BytesIO()
        unique_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}"
        # Define your S3 bucket and file details
        s3_bucket = bucket_name
        pre = "work_area/"
        file_name = f"{unique_id}_{data_file.split('/')[-1].replace('.nc', f'_slice{file_extensions[output_format]}')}"
        s3_key = f'{pre}{file_name}'
        # Define the S3 path
        s3_path = f's3://{s3_bucket}/{s3_key}'

        logger.debug(
            f"[DEBUG] s3_path: {s3_path}, s3_key: {s3_key}"
        )
        # Save in requested format
        ## **Use Different Buffers for Different Formats**
        # Directly writing NetCDF or HDF5 files to S3 can be challenging due to the 
        # need for file-like objects that support seeking. 
        if output_format == "csv":
            logger.debug(f"[DEBUG] Working on CSV")
            # Convert the Dataset to a DataFrame
            df = data_to_return.to_dataframe().reset_index().round(ndecimal)
            # Write the DataFrame to CSV directly to S3
            fs = s3fs.S3FileSystem()
            with fs.open(s3_path, 'w') as f:
                df.to_csv(f, sep=",", index=False)
            logger.info("Data written in CSV format.")

        elif output_format == "netcdf":
            # Set the local file path for temporary NetCDF output
            local_file = f"/tmp/{file_name}.nc"

            # Embed global metadata into the output NetCDF
            data_to_return.attrs = global_metadata

            # Write to NetCDF using the NetCDF4 classic model with compression
            encoding = {var: {"zlib": True, "complevel": 5} for var in data_to_return.data_vars}

            data_to_return.to_netcdf(local_file, format="NETCDF4_CLASSIC", encoding=encoding)

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
            for var in data_to_return.variables:
                metadata += f"# {var}_column: {var}\n"
                for attr_key, attr_value in data_to_return[var].attrs.items():
                    metadata += f"# {var}_{attr_key}: {attr_value}\n"

            # Convert the Dataset to a DataFrame
            df = data_to_return.to_dataframe().reset_index().round(ndecimal)

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
            data_to_return.attrs = global_metadata

            # Write to HDF5 using xarray’s NetCDF backend (which uses HDF5)
            data_to_return.to_netcdf(local_hdf5_file, format="NETCDF4", engine="h5netcdf")

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
            message = f"Error 400 Unsupported output format {output_format}"
            logger.error(message)  # Log the error

            # Return a properly formatted JSON response
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": message}),
            }
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
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "traceback": traceback.format_exc()})
        }