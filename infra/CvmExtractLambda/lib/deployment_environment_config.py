from dataclasses import dataclass

@dataclass
class DeploymentEnvironmentConfig:
    bucket_name: str
    lambda_function_name_slice: str
    lambda_function_name_xsection: str
    lambda_function_name_volume: str
