from pathlib import Path

from aws_cdk import (
    aws_lambda as _lambda,
    aws_s3 as s3,
    Duration,
    Size,
    Stack,
    Tags,
    aws_iam as iam
)
from constructs import Construct


from lib.deployment_environment_config import DeploymentEnvironmentConfig

COMMON_LAMBDA_PROPS = {
    "tracing": _lambda.Tracing.ACTIVE,
    "memory_size": 8192,
    "timeout": Duration.seconds(120),
    "ephemeral_storage_size": Size.gibibytes(5),
    "environment": {
        "LOG_LEVEL": "INFO",
    },
}

class CvmExtractLambdaStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, config: DeploymentEnvironmentConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        Tags.of(self).add("crescent:application:name", "CVM-EXTRACT-LAMBDA")

        project_root = Path(__file__).resolve().parents[3]

        data_bucket = s3.Bucket.from_bucket_arn(self, "DataBucket", bucket_arn=f"arn:aws:s3:::{config.bucket_name}")



        # volume
        volume_function = _lambda.DockerImageFunction(
            self,
            "CvmExtractVolumeDataLambda",
            function_name = config.lambda_function_name_volume,
            code = _lambda.DockerImageCode.from_image_asset(str(project_root / "lambda" / "volume")),
            **(COMMON_LAMBDA_PROPS | {"timeout": Duration.seconds(90)})
        )
        data_bucket.grant_read_write(volume_function)
        volume_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[data_bucket.bucket_arn]
            )
        )

        # xsection
        xsection_function = _lambda.DockerImageFunction(
            self,
            "CvmExtractXSectionDataLambda",
            function_name = config.lambda_function_name_xsection,
            code = _lambda.DockerImageCode.from_image_asset(str(project_root / "lambda" / "xsection")),
            **COMMON_LAMBDA_PROPS,
        )
        data_bucket.grant_read_write(xsection_function)
        xsection_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[data_bucket.bucket_arn]
            )
        )

        # slice
        slice_function = _lambda.DockerImageFunction(
            self,
            "CvmExtractSliceDataLambda",
            function_name = config.lambda_function_name_slice,
            code = _lambda.DockerImageCode.from_image_asset(str(project_root / "lambda" / "slice")),
            **COMMON_LAMBDA_PROPS,
        )
        data_bucket.grant_read_write(slice_function)
        slice_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[data_bucket.bucket_arn]
            )
        )
