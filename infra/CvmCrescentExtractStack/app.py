#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cvm_crescent_extract_stack.cvm_crescent_extract_stack_stack import CvmCrescentExtractStackStack
from lib.deployment_environment_config import DeploymentEnvironmentConfig

AWS_ACCOUNT_ID = '818214664804'
AWS_REGION = 'us-east-2'

# NOTE: The `deployment-environment` MUST be provided at runtime via CDK context.
# Example: `cdk synth --context deployment-environment=dev`

app = cdk.App()
# Get runtime deployment environment from CDK context
deployment_environment = app.node.try_get_context("deployment-environment")
if not deployment_environment:
    raise SystemExit(
        "Error: deployment-environment context variable not set."
        " Use '--context deployment-environment=<name of deployment environment>' to set it."
    )


# Define configuration based on the deployment environment.
deployment_environments_configs: dict[str, DeploymentEnvironmentConfig] = {
    "dev": DeploymentEnvironmentConfig(
        resource_prefix = f"{deployment_environment}-",
        bucket_name="cvm-s3-data-dev-us-east-2-aer1lu3eichu",
        runtime_environment_name="dev"
    ),
    "prod": DeploymentEnvironmentConfig(
        resource_prefix = "",
        bucket_name="cvm-s3-data-crescent-us-east-2-aer1lu3eichu",
        runtime_environment_name="crescent"
    )
}

env_config = deployment_environments_configs.get(deployment_environment)
if not env_config:
    raise SystemExit(
        f"Error: No configuration found for deployment environment '{deployment_environment}'."
    )


CvmCrescentExtractStackStack(
    app,
    f"{env_config.resource_prefix}CvmCrescentExtractStack",
    env=cdk.Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION),
    config=env_config
    )

app.synth()
