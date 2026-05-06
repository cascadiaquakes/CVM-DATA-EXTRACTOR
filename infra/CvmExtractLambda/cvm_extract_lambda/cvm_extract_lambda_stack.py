from pathlib import Path

from aws_cdk import (
    aws_lambda as _lambda,
    aws_s3 as s3,
    Duration,
    Size,
    Stack,
    Tags,
)
from constructs import Construct


from lib.deployment_environment_config import DeploymentEnvironmentConfig

COMMON_LAMBDA_PROPS = {
    "runtime": _lambda.Runtime.PYTHON_3_12,
    "tracing": _lambda.Tracing.ACTIVE,
    "memory_size": 8192,
    "timeout": Duration.seconds(120),
    "ephemeral_storage_size:": Size.gibibytes(5),
    "environment": {
        "LOG_LEVEL": "INFO",
    },
}

class CvmExtractLambdaStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, config: DeploymentEnvironmentConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("crescent:application:name", "CVM-EXTRACT-LAMBDA")

        project_root = Path(__file__).resolve().parents[3]

        dataBucket = s3.Bucket.from_bucket_arn(self, "DataBucket", bucket_arn=f"arn:aws:s3:::{config.bucket_name}")



    # volume

    # xsection

    # slice