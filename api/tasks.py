from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup,
    Container,
    ContainerGroupRestartPolicy,
    OperatingSystemTypes,
    ResourceRequests,
    ResourceRequirements,
    EnvironmentVariable,
)
import uuid
import os

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
CONTAINER_IMAGE = "yourregistry.azurecr.io/your-training-image:latest"
ACI_REGION = "eastus"

credential = DefaultAzureCredential()
client = ContainerInstanceManagementClient(credential, SUBSCRIPTION_ID)

def run_training_script_in_new_container(kaggle_url: str):
    container_group_name = f"trainjob-{str(uuid.uuid4())[:8]}"

    environment_vars = [EnvironmentVariable(name="KAGGLE_URL", value=kaggle_url)]

    container_resource_requests = ResourceRequests(memory_in_gb=2.0, cpu=1.0)
    container_resource_requirements = ResourceRequirements(requests=container_resource_requests)

    container = Container(
        name="trainer",
        image=CONTAINER_IMAGE,
        resources=container_resource_requirements,
        environment_variables=environment_vars,
    )

    group = ContainerGroup(
        location=ACI_REGION,
        containers=[container],
        os_type=OperatingSystemTypes.linux,
        restart_policy=ContainerGroupRestartPolicy.never,
    )

    deployment = client.container_groups.begin_create_or_update(
        RESOURCE_GROUP, container_group_name, group
    )

    return deployment.result()
