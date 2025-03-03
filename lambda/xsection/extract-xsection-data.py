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
from pyproj import Proj, transform, Geod
import s3fs
import fsspec
from utils import unit_conversion_factor, utm_to_latlon, project_lonlat_utm, closest, interpolate_path
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
        end_depth = float(event['end_depth'])
        variable_list = event['variables_hidden'].split(',')
        interpolation_method = event.get('interpolation_method', 'none')
        num_points = int(event['num_points'])
        output_format = event.get('output_format', 'csv')
        units = event.get('units', 'km')

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

            start = [start_lat, start_lng]
            end = [end_lat, end_lng]

            # Requested depth range is always in km.
            _, depth_range_factor = unit_conversion_factor("km", units)
            depth = [start_depth * depth_range_factor, end_depth * depth_range_factor]

            units = units
            logger.debug(
                f"\n\n[DEBUG] converting the depth units {ds['depth']},\nunits: {ds['depth'].attrs['units']}"
            )
            unit_standard, depth_factor = unit_conversion_factor(
                ds["depth"].attrs["units"], units
            )
            logger.debug(
                f"[DEBUG] depth units: {ds['depth'].attrs['units']},  {depth_factor}"
            )
            ds["depth"].attrs["units"] = unit_standard

            ds["depth"] = ds["depth"] * depth_factor
            ds["depth"].attrs["units"] = units

            ds = ds.where(
                (ds.depth >= float(depth[0])) & (ds.depth <= float(depth[1])),
                drop=True,
            )
        except Exception as ex:
            message = f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            logger.error(message)  # Log the error

            # Return a properly formatted JSON response
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": message}),
            }
        utm_zone = None
        meta = ds.attrs
        dpi = 100

        logger.debug(f"\n\n[DEBUG] META: {meta}")
        if "grid_ref" not in meta:
            logger.debug(
                f"[DEBUG] The 'grid_ref' attribute not found. Assuming geographic coordinate system"
            )
            grid_ref = "latitude_longitude"
        else:
            grid_ref = meta["grid_ref"]

            # Cross-section interpolation type.
            interp_type = interpolation_method

            # Steps in the cross-section.
            steps = num_points
            logger.info(
                f"[INFO] cross_section start:{start}, end: {end}, steps: {steps}, interp_type: {interp_type}, ds: {ds}"
            )
            # Extract the cross-section.
            logger.info(
                f"[INFO] Before cross_section (ds,start,end,steps,interp_type:)\n{ds},{start},{end},{steps},{interp_type}"
            )
            try:
                plot_data, latitudes, longitudes = interpolate_path(
                    ds,
                    start,
                    end,
                    num_points=steps,
                    method=interp_type,
                    grid_ref=grid_ref,
                    utm_zone=meta["utm_zone"],
                    ellipsoid=meta["ellipsoid"],
                )

            except Exception as ex:
                message = f"[ERR] cross_section failed: {ex}\n{traceback.print_exc()}"
                logger.error(message)  # Log the error

                # Return a properly formatted JSON response
                return {
                    "statusCode": 500,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": message}),
                }
                
            # Extract latitude and longitude from your cross-section data
            # latitudes = plot_data['latitude'].values
            # longitudes = plot_data['longitude'].values

            # If the original is not geographic, going outside the model coverage will result in NaN values.
            # Recompute these using the primary coordinates.
            # for _index, _lon in enumerate(plot_data['latitude'].values):
            #     if  np.isnan(_lon):
            #         plot_data['longitude'].values[_index], plot_data['latitude'].values[_index] = project_lonlat_utm(
            #         plot_data['x'].values[_index], plot_data['y'].values[_index], utm_zone, meta["ellipsoid"], xy_to_latlon=True)

            # Geod object for WGS84 (a commonly used Earth model)
            geod = Geod(ellps="WGS84")

            # Calculate distances between consecutive points
            _, _, distances = geod.inv(
                longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:]
            )

            # Compute cumulative distance, starting from 0
            cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

            if units == "mks":
                cumulative_distances = cumulative_distances / 1000.0

            logger.debug(
                f"\n\n[DEBUG] units: {units}, cumulative_distances: {cumulative_distances}"
            )

            # Assuming 'plot_data' is an xarray Dataset or DataArray
            # Create a new coordinate 'distance' based on the cumulative distances
            plot_data = plot_data.assign_coords(distance=("points", cumulative_distances))

            # If you want to use 'distance' as a dimension instead of 'index',
            # you can swap the dimensions (assuming 'index' is your current dimension)
            plot_data = plot_data.swap_dims({"points": "distance"})
            logger.debug(f"\n\n[DEBUG] plot_data:{plot_data}")

            # Split the hidden variables list by commas
            logger.debug(f"\n\n[DEBUG] variable_list:{variable_list}")
            output_data = plot_data[variable_list]  # Select only the specified variables

            for var in variable_list:
                logger.info(
                    f"[INFO] {var} units: {output_data[var].attrs['units']}, units: {units}"
                )
                _, var_factor = unit_conversion_factor(
                    output_data[var].attrs["units"], units
                )
                logger.debug(
                    f"\n\n[DEBUG] plot_var units: {output_data[var].attrs['units']}, units: {units} => unit_standard:{unit_standard}, var_factor: {var_factor}"
                )

                # Copy all attributes from the original variable to the new one
                output_data[var].attrs = ds[var].attrs.copy()
                logger.debug(f"\n\n[DEBUG] var: {var}, attr: {ds[var].attrs}")
                output_data[var] = output_data[var] * var_factor

                output_data.attrs["units"] = unit_standard

            logger.debug(f"\n\n[DEBUG] output_data:{output_data}")

            # Set the depth limits for display.
            logger.debug(f"\n\n[DEBUG] Depth limits:{depth}")        # Prepare output buffer
        
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
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": "Data processed successfully", "s3_key": s3_key})
        }
    except Exception as e:
        logger.error("Error occurred: %s", str(e))
