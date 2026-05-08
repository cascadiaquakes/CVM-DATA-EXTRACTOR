#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cvm_extract_lambda.cvm_extract_lambda_stack import CvmExtractLambdaStack
from lib.deployment_environment_config import DeploymentEnvironmentConfig

AWS_ACCOUNT_ID = '818214664804'
AWS_REGION = 'us-east-2'

# Define configuration based on the deployment environment.
# NOTE: The `deployment-environment` MUST be provided at runtime via CDK context.
# Example: `cdk synth --context deployment-environment=dev`

deployment_environments_configs: dict[str, DeploymentEnvironmentConfig] = {
    "dev": DeploymentEnvironmentConfig(
        bucket_name = "cvm-s3-data-dev-us-east-2-aer1lu3eichu",
        lambda_function_name_slice = "cvm-data-extractor-dev-extract-slice",
        lambda_function_name_xsection = "cvm-data-extractor-dev-extract-xsection",
        lambda_function_name_volume = "cvm-data-extractor-dev-extract-volume"
    ),
    "prod": DeploymentEnvironmentConfig(
        bucket_name = "cvm-s3-data-crescent-us-east-2-aer1lu3eichu",
        lambda_function_name_slice = "cvm-data-extractor-extract-slice",
        lambda_function_name_xsection = "cvm-data-extractor-extract-xsection",
        lambda_function_name_volume = "cvm-data-extractor-extract-volume"
    )
}

app = cdk.App()

# Get runtime deployment environment from CDK context
deployment_environment = app.node.try_get_context("deployment-environment")
if not deployment_environment:
    raise SystemExit(
        "Error: deployment-environment context variable not set."
        " Use '--context deployment-environment=<name of deployment environment>' to set it."
    )
env_config = deployment_environments_configs.get(deployment_environment)
if not env_config:
    raise SystemExit(
        f"Error: No configuration found for deployment environment '{deployment_environment}'."
    )

# CDK uses the stack name as the prefix for nearly all resources created.
# Create a prefix for non-Prod Stacks to avoid naming conflicts between environments.
resource_prefix=f"{deployment_environment}-" if deployment_environment != "prod" else ""


CvmExtractLambdaStack(
    app,
    f"{resource_prefix}CvmExtractLambdaStack",
    env=cdk.Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION),
    config=env_config
    )

app.synth()
