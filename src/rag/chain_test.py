"""
Tests for the chain module.
"""
import pytest


def test_create_rag_chain(small_model):
    """
    Test the creation of a RAG chain.
    """
    from .chain import create_rag_chain_mxbai

    # Create the RAG chain
    rag_chain = create_rag_chain_mxbai(small_model)

    result = rag_chain.invoke("How many nodes are in the frontier supercomputer?")
    print(result)