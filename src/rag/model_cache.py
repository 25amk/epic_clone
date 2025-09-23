"""
model_cache.py - Model cache for RAG
"""
import functools
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
from .llama import LlamaGenerator


@functools.cache
def get_sentence_transformer(model_name) -> SentenceTransformer:
    return SentenceTransformer(model_name, trust_remote_code=True)


@functools.cache
def get_cross_encoder(model_name) -> CrossEncoder:
    return CrossEncoder(model_name)


@functools.cache
def get_embedding_function(model_name) -> HuggingFaceEmbeddingFunction:
    return HuggingFaceEmbeddingFunction(api_key="", model_name=model_name)


@functools.cache
def get_llama_generator(model_name) -> LlamaGenerator:
    return LlamaGenerator(model_name)