"""
config.py - Configurations for epicapp
"""
from typing import Annotated as A, Literal, Any
import os
from pathlib import Path
from functools import lru_cache
from pydantic import Field, computed_field, AfterValidator, BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """
    Configuration for a model, can be either a local hugging face model, or a OpenAI compatible API
    """
    type: A[Literal["local", "openai"], Field(description = "Type of the model, either a huggingface local model or an openai-compatible API")]
    name: A[str, Field(description = "Hugging face model to use locally")]
    url: A[str|None, Field(description = "URL for an OpenAI-compatible API")] = None
    key: A[str|None, Field(description = "API key or token")] = None
    extra_args: A[dict[str, Any], Field(description = "Extra arguments to pass to the model, e.g. temperature")] = {}

    @model_validator(mode='after')
    def is_valid(self):
        if self.type == "openai" and not self.url:
            raise ValueError('URL is required')
        return self

OptModelConfig = ModelConfig|Literal["disabled"]


class Settings(BaseSettings):
    # Site profile
    site_profile: str = Field(
        default="localdev",
        description="Site profile (e.g., localdev, staging, production)"
    )

    # Application names
    app_name: str = Field(
        default="epicapp",
        description="Name of the application"
    )

    base_path: A[str,
        Field(description="Base URL of the application"),
        AfterValidator(lambda p: p.removesuffix("/")),
    ] = ""

    large_model: ModelConfig = ModelConfig(
        type = "openai",
        name = "gpt-4o",
        url = "https://aoai-eastus2-aaims.openai.azure.com/openai/deployments/gpt-4o?api-version=2025-01-01-preview",
    )
    """ Model to use for the main chat model """
    small_model: OptModelConfig = ModelConfig(
        type = "local",
        name = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    """ Model to use in the lower levels in the hierarchy """
    sql_model: A[OptModelConfig, Field(description = "Model to use for the sql generation")] = ModelConfig(
        type = "local",
        name = "defog/llama-3-sqlcoder-8b",
    )
    """ Model to use for the sql chain """
    db_module: str|None = "src.site.ornl.frontier.db"
    """
    Path to a python module, e.g. "src.site.ornl.frontier.db", which contains two functions
    `create_sql_database()` and `create_sql_prompt_template()`
    """

    @model_validator(mode='before')
    def is_valid(data):
        return data

    # Nvidia GPU Visibility
    cuda_visible_devices: str = Field(
        default="0",
        description="Comma separated list of visible Nvidia GPU instances on the system"
    )

    # Local scratch volume
    scratch_base: str = Field(
        default=os.path.join(os.getcwd(), ".scratch"),
        description="Local scratch space - accepts only POSIX paths"

    )

    shared_base: str = Field(
        default=os.path.join(os.getcwd(), ".shared"),
        description="Local project space - accepts only POSIX paths"
    )

    # Object store access
    data_base: str = Field(
        default=os.path.join(os.getcwd(), ".data", "shared"),
        description="Primary prefix of the data files either POSIX or an s3a:// prefix"
    )
    data_priv_base: str = Field(
        default=os.path.join(os.getcwd(), ".data", "private"),
        description="Primary prefix of sensitive data files POSIX or an s3a:// prefix"
    )

    @computed_field
    @property
    def is_data_base_remote(self) -> bool:
        """Is data base remote?"""
        return self.data_base.startswith("s3a:/")

    @computed_field
    @property
    def is_data_priv_base_remote(self) -> bool:
        """Is data base remote?"""
        return self.data_base.startswith("s3a:/")

    # aws S3 interface
    aws_endpoint_url: str = Field(
        default="",
        description="Custom endpoint URL for AWS S3 (leave empty for default)"
    )
    aws_access_key_id: str = Field(
        default="",
        description="AWS access key ID for S3 access"
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS secret access key for S3 access"
    )

    @computed_field
    @property
    def has_aws_object_store(self) -> bool:
        """Return whether we have an AWS S3 based object store available"""
        return self.aws_endpoint_url != "" and self.aws_access_key_id != "" and self.aws_secret_access_key != ""

    # Kubernetes settings
    kubeconfig: str = Field(
        default="~/.kube/config",
        description="Path to the Kubernetes config file"
    )
    kubernetes_service_host: str = Field(
        default="",
        description="Kubernetes API server host"
    )
    kubernetes_service_port: str = Field(
        default="",
        description="Kubernetes API server port"
    )
    kubernetes_namespace: str = Field(
        default="default",
        description="Default Kubernetes namespace"
    )
    kubernetes_app_namespace: str = Field(
        default="default",
        description="Kubernetes namespace for applications"
    )
    kubernetes_batch_namespace: str = Field(
        default="default",
        description="Kubernetes namespace for batch workflows"
    )

    @computed_field
    @property
    def has_kubernetes(self) -> bool:
        """Returns whether we have a kubernetes cluster in our environment"""
        return self.kubernetes_service_host != ""

    # Metadata server
    metadata_service: str = Field(
        default=str(Path(__file__).resolve().parent.parent),
        description="Metadata service to use (local, remote)"
    )

    @computed_field
    @property
    def is_metadata_service_remote(self) -> bool:
        """Is data base remote?"""
        return self.metadata_service.startswith("s3a:/")

    query_output_limit: int = Field(
        default = 1000,
        description = "Limit to how many results will be kept from a query, as a safety measure if the model tries to dump the whole DB on us.",
    )

    query_output_limit_table: int = Field(
        default = 100,
        description = "Limit to how many results will be displayed in query result tables",
    )

    query_output_limit_chat_model: int = Field(
        default = 20,
        description = "Limit to how many results will passed to the top-level chat model (to avoid eating up too many tokens)",
    )

    # Reading from the .env file and the environment
    # Extra arguments are allowed for now (FIXME: Narrow it down)
    model_config = SettingsConfigDict(
        env_file = ".env",
        extra = "ignore",
        env_nested_delimiter = '__',
    )


@lru_cache
def _get_settings():
    """Return a fully instantiated settings object"""
    return Settings()


def get_settings() -> Settings:
    """Return a fully instantiated settings object

    This function also exports important environment variables to the
    environment, enabling overrides of the environment via the .env file 
    """
    import os
    settings = _get_settings()
    
    #
    # Export important settings to the environment
    #

    # Export CUDA_VISIBAL_DEVICES settings
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

    return settings