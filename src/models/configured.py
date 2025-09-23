"""
src/models - Central repository of common LLM model factories
"""
from .openai import get_openai_model
from .huggingface import get_local_model
from src.config import get_settings, OptModelConfig

settings = get_settings()

def get_configured_model(config: OptModelConfig):
    if config == "disabled":
        return None
    elif config.type == "local":
        return get_local_model(
            name = config.name,
            **config.extra_args,
        )
    elif config.type == "openai":
        return get_openai_model(
            name = config.name,
            url = config.url,
            key = config.key,
            **config.extra_args,
        )
    else:
        raise Exception(f"Invalid model config {config}")


def get_large_model():
    """
    Get the configured large top-level model.
    """
    return get_configured_model(settings.large_model)


def get_small_model():
    """
    Get the configured small model, defaults to Meta-Llama-3.1-8B-Instruct.
    """
    model = get_small_model_opt()
    if not model:
        raise Exception("Small model is not configured")
    return model


def get_small_model_opt():
    """
    Get the configured small model, defaults to Meta-Llama-3.1-8B-Instruct.
    Can be None if small_model is disabled.
    """
    return get_configured_model(settings.small_model)


def get_sql_model():
    """
    Get the configured sql model. Defaults to defog/llama-3-sqlcoder-8b.
    """
    model = get_sql_model_opt()
    if not model:
        raise Exception("sql model is not configured")
    return model


def get_sql_model_opt():
    """
    Get the configured sql model. Defaults to defog/llama-3-sqlcoder-8b.
    Can be None if sql_model is disabled.
    """
    return get_configured_model(settings.sql_model)
