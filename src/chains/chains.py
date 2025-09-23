"""
chatui: chainlit based UI
"""
from collections.abc import Iterable
from typing import Any
from textwrap import dedent
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, message_chunk_to_message
from langchain.schema.runnable.utils import AddableDict
from src.rag.loader import load_documents, create_vectorstore
from src.sql.sql import create_sql_qna_chain_as_tool
from src.job_pred.regression import create_job_pred_chain_as_tool
from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel

# Initialize vector store at startup
def initialize_vectorstore(urls: list[str]):
    """Initialize vector store with documents from given URLs"""
    documents = load_documents(urls, is_web=True)
    return create_vectorstore(
        documents=documents,
        persist_directory=".data/vectorstore"
    )


def create_rag_chain(vectorstore, model: BaseChatModel):
    """Create a RAG-based chain for document queries"""
    # RAG prompt template
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""
            You are 'epic', a knowledgeable HPC operational data analytics system.
            Use the following context to answer the question. If you cannot find
            the answer in the context, say so clearly.
            
            Context: {context}
        """)),
        ("human", "{question}")
    ])
    
    # Create RAG chain
    return (
        RunnablePassthrough.assign(
            context=lambda x: vectorstore.similarity_search(x["question"])
        )
        | RunnablePassthrough.assign(
            response=rag_prompt | model | StrOutputParser()
        )
    )


def create_rag_chain_as_tool(
    vectorstore: Any,
    model: BaseChatModel,
):
    """
    Factory for the rag_chain object
    """ 
    rag_chain = create_rag_chain(vectorstore, model)
    return rag_chain.as_tool(
        name="rag_chain",
        description="Use this tool to answer questions about the HPC system, operations, policies, and usage details",
        arg_types={"question": str}
    )


def create_mxbai_rag_chain_as_tool(model: BaseChatModel):
    """
    Factory for the mxbai_rag_chain object
    """
    from src.rag.chain import create_rag_chain_mxbai
    rag_chain = create_rag_chain_mxbai(model)
    return rag_chain.as_tool(
        name="mxbai_rag_chain",
        description="Use this tool to answer questions about the HPC system, operations, policies, and usage details",
        arg_types={"question": str}
    )


def create_execute_tool_call_chain(
    tools: list[BaseTool] = [],
    max_retries: int = 3,
):
    """
    A runnable that just executes any tool calls in the input and returns `ToolMessage`s.
    Does not invoke any llms or "agent loop"
    """
    def execute_tool_call(tool_call: dict[str, Any]):
        """ Execute a tool call with retry logic.  """
        tool_name = tool_call.get("name")
        tool = next(
            (tool for tool in tools if tool.name == tool_name),
            None
        )

        result = None
        error = None
        if not tool:
            error = f"Tool {tool_name} not found"
        else:
            try:
                result = tool.invoke(tool_call)
            except Exception as e:
                error = e
        
        # If the result of the chain is an object like {"error": "..."} assume it failed.
        if isinstance(result, dict) and result.get('error'):
            error = str(result['error'])

        if error:
            if isinstance(error, Exception):
                logger.opt(exception=error).error("Tool call failed")
            else:
                logger.error(f"Tool call failed: {error}")

            result = result if result else {"error": str(error)}
            return ToolMessage(
                name = tool_call['name'],
                tool_call_id = tool_call['id'],
                status = 'error',
                content = json.dumps(result),
            )
        else:
            return result

    return (
        RunnableLambda(lambda msg: msg.tool_calls or []) |
        # RunnableEach does it in parallel, which causes issues when multiple model calls happen
        # concurrently with our current pytorch setup. When/if we switch to using an inference
        # server we can switch back to RunnableEach
        # RunnableEach(bound=RunnableLambda(execute_tool_call))
        RunnableLambda(lambda tool_calls: [execute_tool_call(t) for t in tool_calls])
    )


def create_agent_chain(model, tools, max_retries = 3, system_message: str|None = None):
    """
    Creates a tool call chain that calls the model in an "agent loop"
    Returns a dictionary containing the new messages, indexed by number.
    E.g `{0: AIMessage('', tool_calls=[...]), 1: ToolMessage(...), 2: AIMessage(...)}`
    This is a bit weird, but lets us use AddableDict to stream the results. Use chunk_list_to_dict
    to convert it back to a simple list of messages.
    TODO: It may be cleaner to make a "MessageList" class like AddableDict that handles
    streaming a list of message chunks from multiple messages.
    """
    model = model.bind_tools(tools)
    execute_tool_chain = create_execute_tool_call_chain(tools, max_retries = max_retries)
    MAX_DEPTH = 4

    system_messages = [SystemMessage(system_message)] if system_message else []

    def agent_loop_sync(input_messages: Iterable[list[BaseMessage]]):
        response_messages = []

        ai_msg = None
        depth = 0
        while depth < MAX_DEPTH and (ai_msg is None or ai_msg.tool_calls):
            ai_msg = None
            for chunk in model.stream(system_messages + input_messages + response_messages):
                yield AddableDict({len(response_messages): chunk})
                ai_msg = ai_msg + chunk if ai_msg else chunk
            
            ai_msg = message_chunk_to_message(ai_msg)
            response_messages.append(ai_msg)

            tool_messages = execute_tool_chain.invoke(ai_msg)
            for msg in tool_messages:
                yield AddableDict({len(response_messages): msg})
                response_messages.append(msg)

            depth += 1

    async def agent_loop_async(input_messages: list[BaseMessage]):
        response_messages = []

        ai_msg = None
        depth = 0
        while depth < MAX_DEPTH and (ai_msg is None or ai_msg.tool_calls):
            ai_msg = None
            async for chunk in model.astream(system_messages + input_messages + response_messages):
                yield AddableDict({len(response_messages): chunk})
                ai_msg = ai_msg + chunk if ai_msg else chunk
            
            ai_msg = message_chunk_to_message(ai_msg)
            response_messages.append(ai_msg)

            tool_messages = await execute_tool_chain.ainvoke(ai_msg)
            for msg in tool_messages:
                yield AddableDict({len(response_messages): msg})
                response_messages.append(msg)

            depth += 1

    return RunnableLambda(agent_loop_sync, afunc = agent_loop_async)


def chunk_dict_to_list(data: dict):
    data_list = sorted(data.items(), key = lambda t: t[0])
    return [message_chunk_to_message(chunk) for i, chunk in data_list]


def create_chat_chain_with_tool_router(
    large_model: Runnable,
    small_model: Runnable|None = None,
    sql_model: Runnable|None = None,
    db_module: str|None = None
):
    """ Create a chain that calls the various tools. """
    # Create the necessary subchains as tools based on the availability of the models
    # from the callsite.  Useful for evaluations that doesn't require *all* models to be loaded
    # into memory
    tools = []

    if small_model:
        # RAG Chain availability depends on the large model which is always supplied
        logger.info("Creating RAG chain tool")
        tools.append(create_mxbai_rag_chain_as_tool(small_model))
    else:
        logger.warning("Skipping RAG chain tool")

    # SQL toolchain relies on the sql_model
    if sql_model is not None and db_module is not None:
        logger.info("Creating SQL chain tool")
        tools.append(create_sql_qna_chain_as_tool(
            model = sql_model,
            db_module = db_module,
        ))
    else:
        logger.warning("Skipping SQL chain tool")

    # The Regression model tool
    if small_model is not None:
        logger.info("Creating the regression model tool")
        tools.append(create_job_pred_chain_as_tool(model=small_model))
    else:
        logger.warning("Skipping the regression model tool")

    # The the final top-level chain to return
    chain = create_agent_chain(large_model, tools, system_message=dedent("""
        You are a helpful assistant that can answer questions about the HPC system, operations,
        policies, and usage details. You can call sequence of tools depending on the requirements.
        If a tool result already answers the user’s question (such as a table of results), you do not 
        need to restate the answer unless explicitly asked to explain or summarize it.
    """).strip())
    return chain
