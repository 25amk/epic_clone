"""
src/rag/chain.py - RAG chain implementation
"""
from loguru import logger
from typing import Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

# Default prompt templates
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question that captures all necessary context from the conversation.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

ANSWER_TEMPLATE = """Answer the question based on the following context. If you cannot answer the question based on the context, just say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer:"""

def create_rag_chain(
    model,
    vectorstore: Chroma,
    condense_question_prompt: Optional[str] = None,
    answer_prompt: Optional[str] = None,
    return_source_documents: bool = True,
    chain_type: str = "stuff",
    search_kwargs: dict = {"k": 4},
):
    """
    Create a RAG chain using the provided model and vector store
    
    Args:
        model: Language model to use
        vectorstore: Vector store for retrieval
        condense_question_prompt: Custom prompt for question condensing
        answer_prompt: Custom prompt for answering
        return_source_documents: Whether to return source documents
        chain_type: Chain type for combining documents
        search_kwargs: Arguments for vector store search
    """
    # Create prompts
    _condense_template = condense_question_prompt or CONDENSE_QUESTION_TEMPLATE
    _qa_template = answer_prompt or ANSWER_TEMPLATE
    
    condense_prompt = PromptTemplate.from_template(_condense_template)
    answer_prompt = PromptTemplate.from_template(_qa_template)

    # Create retrieval chain
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    
    # Create the conversational chain
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        return_source_documents=return_source_documents,
        chain_type=chain_type,
    ) 


def create_rag_chain_mxbai(model: BaseChatModel) -> Runnable:
    """
    Create a RAG chain using the MixedBread AI model and ChromaDB
    """
    import chromadb
    from . import RAG_DB_PATH
    from .db import initialize_dbs

    logger.info("Initializing RAG chain with MixedBread AI model and ChromaDB")
    client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
    text_db, image_db, table_db = initialize_dbs(client, model)

    def chain(query: str) -> dict:
        """
        Process the query and return the response
        """
        text_result = text_db.answer_query(query)
        image_result = image_db.answer_query(query)
        table_result = table_db.answer_query(query)

        # Combine results from all databases
        combined_result = {
            "text": text_result,
            "image": image_result,
            "table": table_result
        }
        return combined_result
    
    return RunnableLambda(chain)
