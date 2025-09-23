"""
src/rag - RAG (Retrieval Augmented Generation) implementation
"""
from pathlib import Path
from .loader import load_documents, load_local_documents, load_web_documents, create_vectorstore
from .chain import create_rag_chain 

# The embedded database path for the RAG chain
RAG_DB_PATH = Path(__file__).parent / "chroma_db"

from .mxbai import MxbaiRetriever, MxbaiReranker
from .llama import LlamaGenerator