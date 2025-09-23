"""
Tests for document loading and vector store creation
"""
import pytest
import responses
from pathlib import Path
from .loader import (
    load_web_documents,
    load_local_documents,
    load_documents,
    create_vectorstore,
    CachedWebLoader
)

TEST_URL = "https://test.com/page1"
TEST_CONTENT = """
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Test Content</h1>
    <p>This is a test document one.</p>
    <script>console.log("should be removed")</script>
    <p>This is a test document two.</p>
</body>
</html>
"""

@pytest.fixture
def mock_responses():
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            TEST_URL,
            body=TEST_CONTENT,
            content_type="text/html"
        )
        yield rsps

@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"

def test_web_loader_caching(mock_responses, cache_dir):
    """Test web loading with caching"""
    loader = CachedWebLoader(cache_dir=str(cache_dir))
    
    # First load - should hit network
    docs = loader.load([TEST_URL])
    assert len(docs) == 1
    assert "test document" in docs[0].page_content
    assert docs[0].metadata["title"] == "Test Page"
    
    # Second load - should use cache
    docs = loader.load([TEST_URL])
    assert len(docs) == 1
    assert "test document" in docs[0].page_content
    
    # Verify only one network request was made
    assert len(mock_responses.calls) == 1

def test_load_web_documents(mock_responses, cache_dir):
    """Test loading and chunking web documents"""
    docs = load_web_documents(
        urls=[TEST_URL],
        chunk_size=100,
        chunk_overlap=20,
        cache_dir=str(cache_dir)
    )
    
    assert len(docs) > 0
    assert any("test document" in doc.page_content for doc in docs)
    
    # Metadata should be preserved in chunks
    for doc in docs:
        assert doc.metadata["url"] == TEST_URL
        assert doc.metadata["title"] == "Test Page"

def test_create_vectorstore(mock_responses, cache_dir):
    """Test creating vector store from web documents"""
    docs = load_web_documents(
        urls=[TEST_URL],
        cache_dir=str(cache_dir)
    )
    vectorstore = create_vectorstore(docs)
    
    results = vectorstore.similarity_search("test document")
    assert len(results) > 0
    assert "test document" in results[0].page_content 

@pytest.fixture
def test_docs_dir(tmp_path):
    # Create test documents
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("This is a test document one.")
    
    doc2 = tmp_path / "doc2.txt" 
    doc2.write_text("This is a test document two.")
    
    # Create a markdown file
    doc3 = tmp_path / "doc3.md"
    doc3.write_text("# Test Document Three\nThis is markdown content.")
    
    return tmp_path

def test_load_local_documents(test_docs_dir):
    """Test loading local documents"""
    docs = load_local_documents(
        source_dir=test_docs_dir,
        chunk_size=100,
        chunk_overlap=20
    )
    
    assert len(docs) >= 3  # At least one chunk per file
    assert any("test document one" in doc.page_content for doc in docs)
    assert any("test document two" in doc.page_content for doc in docs)
    assert any("Test Document Three" in doc.page_content for doc in docs)
    
    # Check metadata
    for doc in docs:
        assert "source" in doc.metadata
        assert "file_type" in doc.metadata
        assert doc.metadata["file_type"] in [".txt", ".md"]

def test_universal_loader(test_docs_dir, mock_responses):
    """Test the universal loader with both local and web sources"""
    # Test local loading
    local_docs = load_documents(
        source=test_docs_dir,
        is_web=False,
        chunk_size=1000,
        chunk_overlap=200
    )
    assert len(local_docs) > 0
    assert any("test document" in doc.page_content for doc in local_docs)
    
    # Test web loading
    web_docs = load_documents(
        source=[TEST_URL],
        is_web=True,
        cache_dir=str(test_docs_dir / "cache")
    )
    assert len(web_docs) > 0
    assert any("test document" in doc.page_content for doc in web_docs) 