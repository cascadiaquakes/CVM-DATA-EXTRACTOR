# CRESCENT CVM-DATA-EXTRACTOR

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

AWS Lambda functions that power on-demand data extraction for the [CRESCENT CVM viewer](https://github.com/cascadiaquakes/CRESCENT-CVM-WEB) ([cvm.cascadiaquakes.org](https://cvm.cascadiaquakes.org)). The viewer calls them for requests too large to handle in the web app.

## Functions

| Function | Source | What it extracts |
|---|---|---|
| Volume | `lambda/volume/` | 3D subvolumes |
| Slice | `lambda/slice/` | Horizontal (constant-depth) slices |
| Cross-section | `lambda/xsection/` | Vertical slices along a path |

Each function reads the model netCDF from S3, subsets it to the requested bounds (trimmed to the model's extent), and writes the result back to S3 in the requested format (netCDF, CSV, GeoCSV or HDF5). Missing model values are written as NaN. Each one is packaged as a container image (see its `Dockerfile`).

## Deployment

The functions are deployed with the AWS CDK app in `infra/CvmExtractLambda/`. The GitHub Actions workflow in `.github/workflows/main.yml` deploys the dev functions on every push to `main`, then deploys production after an approval on the `Production` environment.

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

## Contacts

- **William Marfo**, University of California, Davis: [wmarfo@ucdavis.edu](mailto:wmarfo@ucdavis.edu)
- **Loïc Bachelot**, University of Oregon: [lbachelo@uoregon.edu](mailto:lbachelo@uoregon.edu)

## Acknowledgments

This project is supported by the National Science Foundation through the Cascadia Region Earthquake Science Center (CRESCENT), Cooperative Agreement EAR-2225286.
