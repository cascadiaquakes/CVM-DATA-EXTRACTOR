from aws_cdk import Stack
import aws_cdk as cdk
import aws_cdk.aws_iam as iam
import aws_cdk.aws_lambda as aws_lambda
from constructs import Construct
from lib.deployment_environment_config import DeploymentEnvironmentConfig

"""

"""
class CvmCrescentExtractStackStack(Stack):
  def __init__(self, scope: Construct, construct_id: str, config: DeploymentEnvironmentConfig, **kwargs) -> None:
    super().__init__(scope, construct_id, **kwargs)

    # Resources
    cvmExtractSliceDataLambdaServiceRoleF6543ee7 = iam.CfnRole(self, 'CvmExtractSliceDataLambdaServiceRoleF6543EE7',
          assume_role_policy_document = {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          managed_policy_arns = [
            ''.join([
              'arn:',
              self.partition,
              ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            ]),
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'cvm-data-extractor',
            },
          ],
        )

    cvmExtractVolumeDataLambdaServiceRoleAad7232d = iam.CfnRole(self, 'CvmExtractVolumeDataLambdaServiceRoleAAD7232D',
          assume_role_policy_document = {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          managed_policy_arns = [
            ''.join([
              'arn:',
              self.partition,
              ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            ]),
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'cvm-data-extractor',
            },
          ],
        )

    cvmExtractXsectionDataLambdaServiceRoleB62036ca = iam.CfnRole(self, 'CvmExtractXsectionDataLambdaServiceRoleB62036CA',
          assume_role_policy_document = {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          managed_policy_arns = [
            ''.join([
              'arn:',
              self.partition,
              ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            ]),
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'cvm-data-extractor',
            },
          ],
        )

    cvmExtractSliceDataLambdaServiceRoleDefaultPolicyF36f548f = iam.CfnPolicy(self, 'CvmExtractSliceDataLambdaServiceRoleDefaultPolicyF36F548F',
          policy_document = {
            'Statement': [
              {
                'Action': [
                  's3:Abort*',
                  's3:DeleteObject*',
                  's3:GetBucket*',
                  's3:GetObject*',
                  's3:List*',
                  's3:PutObject',
                  's3:PutObjectLegalHold',
                  's3:PutObjectRetention',
                  's3:PutObjectTagging',
                  's3:PutObjectVersionTagging',
                ],
                'Effect': 'Allow',
                'Resource': [
                  f'arn:aws:s3:::{config.bucket_name}',
                  f'arn:aws:s3:::{config.bucket_name}/*',
                ],
              },
              {
                'Action': 's3:ListBucket',
                'Effect': 'Allow',
                'Resource': f'arn:aws:s3:::{config.bucket_name}',
              },
            ],
            'Version': '2012-10-17',
          },
          policy_name = 'CvmExtractSliceDataLambdaServiceRoleDefaultPolicyF36F548F',
          roles = [
            cvmExtractSliceDataLambdaServiceRoleF6543ee7.ref,
          ],
        )

    cvmExtractVolumeDataLambdaServiceRoleDefaultPolicyC628b69f = iam.CfnPolicy(self, 'CvmExtractVolumeDataLambdaServiceRoleDefaultPolicyC628B69F',
          policy_document = {
            'Statement': [
              {
                'Action': [
                  's3:Abort*',
                  's3:DeleteObject*',
                  's3:GetBucket*',
                  's3:GetObject*',
                  's3:List*',
                  's3:PutObject',
                  's3:PutObjectLegalHold',
                  's3:PutObjectRetention',
                  's3:PutObjectTagging',
                  's3:PutObjectVersionTagging',
                ],
                'Effect': 'Allow',
                'Resource': [
                  f'arn:aws:s3:::{config.bucket_name}',
                  f'arn:aws:s3:::{config.bucket_name}/*',
                ],
              },
              {
                'Action': 's3:ListBucket',
                'Effect': 'Allow',
                'Resource': f'arn:aws:s3:::{config.bucket_name}',
              },
            ],
            'Version': '2012-10-17',
          },
          policy_name = 'CvmExtractVolumeDataLambdaServiceRoleDefaultPolicyC628B69F',
          roles = [
            cvmExtractVolumeDataLambdaServiceRoleAad7232d.ref,
          ],
        )

    cvmExtractXsectionDataLambdaServiceRoleDefaultPolicyBa1d8cb4 = iam.CfnPolicy(self, 'CvmExtractXsectionDataLambdaServiceRoleDefaultPolicyBA1D8CB4',
          policy_document = {
            'Statement': [
              {
                'Action': [
                  's3:Abort*',
                  's3:DeleteObject*',
                  's3:GetBucket*',
                  's3:GetObject*',
                  's3:List*',
                  's3:PutObject',
                  's3:PutObjectLegalHold',
                  's3:PutObjectRetention',
                  's3:PutObjectTagging',
                  's3:PutObjectVersionTagging',
                ],
                'Effect': 'Allow',
                'Resource': [
                  f'arn:aws:s3:::{config.bucket_name}',
                  f'arn:aws:s3:::{config.bucket_name}/*',
                ],
              },
              {
                'Action': 's3:ListBucket',
                'Effect': 'Allow',
                'Resource': f'arn:aws:s3:::{config.bucket_name}',
              },
            ],
            'Version': '2012-10-17',
          },
          policy_name = 'CvmExtractXsectionDataLambdaServiceRoleDefaultPolicyBA1D8CB4',
          roles = [
            cvmExtractXsectionDataLambdaServiceRoleB62036ca.ref,
          ],
        )

    cvmExtractSliceDataLambda5A5525a8 = aws_lambda.CfnFunction(self, 'CvmExtractSliceDataLambda5A5525A8',
          code = {
            'imageUri': f"""818214664804.dkr.ecr.us-east-2.{self.url_suffix}/cdk-hnb659fds-container-assets-818214664804-us-east-2:315e784fc9e6f0a71274dfc33acc8d2ff0a9fb65921e68f8e00668beafc522ad""",
          },
          environment = {
            'variables': {
              'APP_VERSION': 'c93d99d5',
              'LOG_LEVEL': 'INFO',
              'ENVIRONMENT': config.runtime_environment_name,
            },
          },
          function_name = f'{config.resource_prefix}cvm-data-extractor-extract-slice',
          memory_size = 8192,
          package_type = 'Image',
          role = cvmExtractSliceDataLambdaServiceRoleF6543ee7.attr_arn,
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'cvm-data-extractor',
            },
          ],
          timeout = 90,
        )
    cvmExtractSliceDataLambda5A5525a8.cfn_options.metadata = {
      'aws:asset:dockerfile-path': 'Dockerfile',
    }
    cvmExtractSliceDataLambda5A5525a8.add_dependency(cvmExtractSliceDataLambdaServiceRoleDefaultPolicyF36f548f)
    cvmExtractSliceDataLambda5A5525a8.add_dependency(cvmExtractSliceDataLambdaServiceRoleF6543ee7)

    cvmExtractVolumeDataLambda3A075bd4 = aws_lambda.CfnFunction(self, 'CvmExtractVolumeDataLambda3A075BD4',
          code = {
            'imageUri': f"""818214664804.dkr.ecr.us-east-2.{self.url_suffix}/cdk-hnb659fds-container-assets-818214664804-us-east-2:bc6636182c99cfa9f6840bd964c82cc4f395ad03f727377bcacf812b8f540d74""",
          },
          environment = {
            'variables': {
              'APP_VERSION': 'c93d99d5',
              'LOG_LEVEL': 'INFO',
              'ENVIRONMENT': config.runtime_environment_name,
            },
          },
          function_name = f'{config.resource_prefix}cvm-data-extractor-extract-volume',
          memory_size = 8192,
          package_type = 'Image',
          role = cvmExtractVolumeDataLambdaServiceRoleAad7232d.attr_arn,
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'cvm-data-extractor',
            },
          ],
          timeout = 120,
        )
    cvmExtractVolumeDataLambda3A075bd4.cfn_options.metadata = {
      'aws:asset:dockerfile-path': 'Dockerfile',
    }
    cvmExtractVolumeDataLambda3A075bd4.add_dependency(cvmExtractVolumeDataLambdaServiceRoleDefaultPolicyC628b69f)
    cvmExtractVolumeDataLambda3A075bd4.add_dependency(cvmExtractVolumeDataLambdaServiceRoleAad7232d)

    cvmExtractXsectionDataLambda4Cd907b2 = aws_lambda.CfnFunction(self, 'CvmExtractXsectionDataLambda4CD907B2',
          code = {
            'imageUri': f"""818214664804.dkr.ecr.us-east-2.{self.url_suffix}/cdk-hnb659fds-container-assets-818214664804-us-east-2:71ec48b4fb4e3526fd7e7e710cf2574a5c73db270780fafafce9391786dcfc26""",
          },
          environment = {
            'variables': {
              'APP_VERSION': 'c93d99d5',
              'LOG_LEVEL': 'INFO',
              'ENVIRONMENT': config.runtime_environment_name,
            },
          },
          function_name = f'{config.resource_prefix}cvm-data-extractor-extract-xsection',
          memory_size = 8192,
          package_type = 'Image',
          role = cvmExtractXsectionDataLambdaServiceRoleB62036ca.attr_arn,
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'cvm-data-extractor',
            },
          ],
          timeout = 120,
        )
    cvmExtractXsectionDataLambda4Cd907b2.cfn_options.metadata = {
      'aws:asset:dockerfile-path': 'Dockerfile',
    }
    cvmExtractXsectionDataLambda4Cd907b2.add_dependency(cvmExtractXsectionDataLambdaServiceRoleDefaultPolicyBa1d8cb4)
    cvmExtractXsectionDataLambda4Cd907b2.add_dependency(cvmExtractXsectionDataLambdaServiceRoleB62036ca)


