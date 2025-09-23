"""
src/models - Central repository of common Hugging Face LLM model factories
"""
import functools
from langchain_core.messages.utils import get_buffer_string
from weakref import WeakValueDictionary
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


@functools.cache
def get_tokenizer(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    return AutoTokenizer.from_pretrained(model_name)


def get_num_tokens_from_messages(model_name: str, messages: list):
    """
    ChatHuggingFace doesn't return the correct token counts either from get_num_tokens
    so we have to implement that ourselves as well.
    """
    tokenizer = get_tokenizer(model_name)
    return sum(len(tokenizer.encode(get_buffer_string([m]))) for m in messages)

_model_cache = WeakValueDictionary() # weak map won't prevent garbage collection
def get_local_model(
    name: str,
    model_kwargs={},
    pipeline_kwargs={},
):
    """Factory of a local huggingface model defaulting to a generic Llama 3.1 model"""
    cache_key = json.dumps([name, model_kwargs, pipeline_kwargs], sort_keys=True)
    langchain_model = _model_cache.get(cache_key)

    if not langchain_model:
        # Instantiate the tokenizer and the model
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "use_cache": True,
            **model_kwargs,
        }
        model = AutoModelForCausalLM.from_pretrained(
            name,
            **model_kwargs,
        )
        
        # Create a custom pipeline
        tokenizer = get_tokenizer(name)
        pipeline_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "max_new_tokens": 1024,
            "do_sample": False,
            "return_full_text": False, # added return_full_text parameter to prevent splitting issues with prompt
            "num_beams": 5, # do beam search with 5 beams for high quality results,
            **pipeline_kwargs,
        }
        pipe = pipeline(
            "text-generation",
            **pipeline_kwargs,
        )

        # Instantiate a LangChain chat model from the pipeline
        llm = HuggingFacePipeline(pipeline = pipe)
        langchain_model = ChatHuggingFace(
            model_id = name,
            llm = llm,
            metadata = { "model_name": name },
        )
        _model_cache[cache_key] = langchain_model

    return langchain_model
