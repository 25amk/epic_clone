"""
src/rag/loader.py - Document loading and processing utilities
"""
from pathlib import Path
import hashlib
import json
import time
from typing import List, Optional, Dict, Union
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class CachedWebLoader:
    """Loader that caches web pages locally"""
    
    def __init__(self, cache_dir: str = ".cache/web"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL"""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"
    
    def _load_cache(self, url: str) -> Optional[Dict]:
        """Load cached content if available and not expired"""
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None
            
        with cache_path.open() as f:
            cached = json.load(f)
            
        # Check if cache is expired (24 hours)
        if time.time() - cached["timestamp"] > 86400:
            return None
            
        return cached

    def _save_cache(self, url: str, content: str, metadata: Dict):
        """Save content and metadata to cache"""
        cache_data = {
            "url": url,
            "content": content,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        cache_path = self._get_cache_path(url)
        with cache_path.open("w") as f:
            json.dump(cache_data, f)

    def _fetch_url(self, url: str) -> tuple[str, Dict]:
        """Fetch URL content and metadata"""
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract main content and remove scripts, styles
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        content = soup.get_text(separator="\n", strip=True)
        
        # Extract metadata
        metadata = {
            "title": soup.title.string if soup.title else "",
            "url": url,
            "domain": urlparse(url).netloc
        }
        
        return content, metadata

    def load(self, urls: List[str]) -> List[Document]:
        """Load documents from URLs with caching"""
        documents = []
        
        for url in urls:
            # Try loading from cache first
            cached = self._load_cache(url)
            
            if cached:
                content = cached["content"]
                metadata = cached["metadata"]
            else:
                # Fetch and cache if not found
                content, metadata = self._fetch_url(url)
                self._save_cache(url, content, metadata)
            
            # Create a LangChain Document object
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
            
        return documents


def load_web_documents(
    urls: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    cache_dir: str = ".cache/web"
) -> List:
    """
    Load and chunk documents from URLs
    
    Args:
        urls: List of URLs to load
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        cache_dir: Directory for caching web content
    """
    # Load documents with caching
    loader = CachedWebLoader(cache_dir=cache_dir)
    documents = loader.load(urls)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)


def load_local_documents(
    source_dir: Union[str, Path],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    glob_pattern: str = "**/*.*"
) -> List[Document]:
    """
    Load and chunk documents from a local directory
    
    Args:
        source_dir: Directory containing documents
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        glob_pattern: Pattern for matching files
    """
    source_dir = Path(source_dir)
    
    # Configure loaders based on file types
    loaders = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PDFMinerLoader,
        # Add more loaders as needed
    }
    
    documents = []
    
    # Recursively process all files
    for file_path in source_dir.glob(glob_pattern):
        if file_path.suffix.lower() in loaders:
            loader_cls = loaders[file_path.suffix.lower()]
            try:
                loader = loader_cls(str(file_path))
                file_docs = loader.load()
                
                # Add source file info to metadata
                for doc in file_docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["file_type"] = file_path.suffix
                    
                documents.extend(file_docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Split documents into chunks
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    return []


def load_documents(
    source: Union[str, Path, List[str]],
    is_web: bool = False,
    **kwargs
) -> List[Document]:
    """
    Universal document loader that handles both local and web documents
    
    Args:
        source: Either a directory path or list of URLs
        is_web: Whether the source is web URLs
        **kwargs: Additional arguments passed to specific loaders
    """
    if is_web:
        return load_web_documents(urls=source, **kwargs)
    else:
        return load_local_documents(source_dir=source, **kwargs)


def create_vectorstore(
    documents: List,
    embedding_model: Optional[str] = "mixedbread-ai/mxbai-embed-large-v1",
    persist_directory: Optional[str] = None
) -> Chroma:
    """
    Create a Chroma vector store from documents
    
    Args:
        documents: List of documents to embed
        embedding_model: Name of embedding model to use (defaults to "mixedbread-ai/mxbai-embed-large-v1")
        persist_directory: Directory to persist vector store
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    if persist_directory:
        # Ensure the persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    # Return a temporary vector store if no persist directory is provided
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    ) 