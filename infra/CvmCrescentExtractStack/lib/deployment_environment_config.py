from dataclasses import dataclass

@dataclass
class DeploymentEnvironmentConfig:
    resource_prefix: str
    bucket_name: str
    runtime_environment_name: str