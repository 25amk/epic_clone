"""
toolcall_test.py - Test tool calling chain
"""
import pytest
import functools, operator
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain.schema.runnable.utils import AddableDict
from src.chains.chains import create_execute_tool_call_chain, create_agent_chain, chunk_dict_to_list
from src.models import get_large_model
import asyncio
from unittest.mock import AsyncMock, Mock

# Define decorator-based tools for the integration test
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def test_chunk_dict_to_list():
    chunks = [
        AddableDict({0: AIMessageChunk("hello")}),
        AddableDict({0: AIMessageChunk(" world")}),
        AddableDict({1: ToolMessage("[1]", tool_call_id = '1')}),
    ]
    result = functools.reduce(operator.add, chunks)
    result = chunk_dict_to_list(result)
    assert len(result) == 2
    assert isinstance(result[0], AIMessage)
    assert result[0].content == "hello world"
    assert isinstance(result[1], ToolMessage)
    assert result[1].content == "[1]"

    result = chunk_dict_to_list(AddableDict())
    assert result == []


@pytest.mark.integration
def test_execute_tool_call_basic():
    tools = [add, subtract, multiply, divide]
    chain = create_execute_tool_call_chain(tools)
    
    result = chain.invoke(
        AIMessage(
            content="I'm going to call these tools:",
            tool_calls=[
                {'name': 'divide', 'args': {'a': 5, 'b': 1}, 'id': 'call_1', 'type': 'tool_call'},
            ],
        ),
    )

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].status == 'success'
    assert result[0].content == "5.0"

    # Test multiple tool calls
    result = chain.invoke(
        AIMessage(
            content="I'm going to call these tools:",
            tool_calls=[
                {'name': 'divide', 'args': {'a': 1, 'b': 2}, 'id': 'call_1', 'type': 'tool_call'},
                {'name': 'multiply', 'args': {'a': 1, 'b': 2}, 'id': 'call_2', 'type': 'tool_call'},
            ],
        ),
    )

    assert len(result) == 2
    assert result[0].content == "0.5"
    assert result[1].content == "2.0"


@pytest.mark.integration
def test_execute_tool_call_errors():
    tools = [add, subtract, multiply, divide]
    chain = create_execute_tool_call_chain(tools)
    
    result = chain.invoke(
        AIMessage(
            content="",
            tool_calls=[
                {'name': 'divide', 'args': {'a': 2, 'b': 0}, 'id': 'call_1', 'type': 'tool_call'},
            ],
        ),
    )

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].status == 'error'
    assert 'divide by zero' in result[0].content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_tool_call_async():
    tools = [add, subtract, multiply, divide]
    chain = create_execute_tool_call_chain(tools)
    
    result = await chain.ainvoke(
        AIMessage(
            content="",
            tool_calls=[
                {'name': 'divide', 'args': {'a': 5, 'b': 1}, 'id': 'call_1', 'type': 'tool_call'},
            ],
        ),
    )

    assert len(result) == 1
    assert result[0].content == "5.0"


@pytest.mark.integration
def test_math_tool_chain():
    """Integration test using OpenAI with math tools."""
    # Create LLM with tools
    llm = get_large_model()
    tools = [add, subtract, multiply, divide]
    
    # Create chain
    chain = create_agent_chain(llm, tools=tools)
    
    # Test complex calculation
    messages = [
        HumanMessage(
            content="I need help calculating this: multiply 23 by 4, "
            "then subtract 15 from the result, finally divide by 2. "
            "You must adhere to the dependencies of the tools making sure "
            "the tools are called in the correct order and only when the results are available "
            "Instead of just doing the calculation, you MUST use the provided tools to do it."
            "What's the answer?"
        )
    ]
    
    result = chain.invoke(messages)
    assert isinstance(result, dict)
    result = chunk_dict_to_list(result)
    assert isinstance(result[0], AIMessage)

    # Verify the conversation flow
    assert len(result) > 2  # Should have multiple steps
    
    # Extract the final answer from the last AI message
    final_message = result[-1]
    assert isinstance(final_message, AIMessage)
    
    # We don't expect the exact answer as the tool calls themselves do not
    # depend on the previous tool calls.
    assert "38.5" in final_message.content

    # Verify tool usage
    tool_calls = [msg for msg in result if isinstance(msg, ToolMessage)]
    assert len(tool_calls) >= 3  # Should use at least 3 tools
    
    # Verify tool call order by checking function names
    tool_names = [
        msg.name for msg in result 
        if isinstance(msg, ToolMessage)
    ]
    
    # The tools should be used in this order (though we're flexible about exact order)
    required_tools = {"multiply", "subtract", "divide"}
    assert required_tools.issubset(set(tool_names)), \
        "Not all required tools were used"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_math_tool_chain_async():
    """Integration test using OpenAI with math tools asynchronously."""
    # Create LLM with tools
    llm = get_large_model()
    tools = [add, subtract, multiply, divide]
    
    # Create chain
    chain = create_agent_chain(llm, tools=tools)
    
    # Test complex calculation
    messages = [
        HumanMessage(
            content="I need help calculating this: multiply 23 by 4, "
            "then subtract 15 from the result, finally divide by 2. "
            "You must adhere to the dependencies of the tools making sure "
            "the tools are called in the correct order and only when the results are available "
            "Instead of just doing the calculation, you MUST use the provided tools to do it."
            "What's the answer?"
        )
    ]
    
    result = await chain.ainvoke(messages)
    result = chunk_dict_to_list(result)

    # Verify the conversation flow
    assert len(result) > 2  # Should have multiple steps
    
    # Extract the final answer from the last AI message
    final_message = result[-1]
    assert isinstance(final_message, AIMessage)
    
    assert "38.5" in final_message.content

    # Verify tool usage
    tool_calls = [msg for msg in result if isinstance(msg, ToolMessage)]
    assert len(tool_calls) >= 3  # Should use at least 3 tools
    
    # Verify tool call order by checking function names
    tool_names = [
        msg.name for msg in result 
        if isinstance(msg, ToolMessage)
    ]
    
    # The tools should be used in this order (though we're flexible about exact order)
    required_tools = {"multiply", "subtract", "divide"}
    assert required_tools.issubset(set(tool_names)), \
        "Not all required tools were used"


@pytest.mark.integration
def test_math_tool_chain_stream_sync():
    # Create LLM with tools
    llm = get_large_model()
    tools = [add, subtract, multiply, divide]
    
    # Create chain
    chain = create_agent_chain(llm, tools=tools)
    
    # Test complex calculation
    messages = [
        HumanMessage(
            content="I need help calculating this: multiply 23 by 4, "
            "then subtract 15 from the result, finally divide by 2. "
            "You must adhere to the dependencies of the tools making sure "
            "the tools are called in the correct order and only when the results are available "
            "Instead of just doing the calculation, you MUST use the provided tools to do it."
            "What's the answer?"
        )
    ]
    
    stream_result = list(chain.stream(messages))
    assert isinstance(stream_result[0], dict)
    assert isinstance(stream_result[0][0], AIMessageChunk)
    # Should be streaming by tokens
    assert len(stream_result) > 10

    combined_result = chunk_dict_to_list(functools.reduce(operator.add, stream_result))
    assert len(combined_result) > 2

    assert "38.5" in combined_result[-1].content
    tool_calls = [msg for msg in combined_result if isinstance(msg, ToolMessage)]
    assert len(tool_calls) >= 3  # Should use at least 3 tools


@pytest.mark.integration
@pytest.mark.asyncio
async def test_math_tool_chain_stream_async():
    # Create LLM with tools
    llm = get_large_model()
    tools = [add, subtract, multiply, divide]
    
    # Create chain
    chain = create_agent_chain(llm, tools=tools)
    
    # Test complex calculation
    messages = [
        HumanMessage(
            content="I need help calculating this: multiply 23 by 4, "
            "then subtract 15 from the result, finally divide by 2. "
            "You must adhere to the dependencies of the tools making sure "
            "the tools are called in the correct order and only when the results are available "
            "Instead of just doing the calculation, you MUST use the provided tools to do it."
            "What's the answer?"
        )
    ]
    
    stream_result = [c async for c in chain.astream(messages)]
    assert isinstance(stream_result[0], dict)
    assert isinstance(stream_result[0][0], AIMessageChunk)
    # Should be streaming by tokens
    assert len(stream_result) > 10

    combined_result = chunk_dict_to_list(functools.reduce(operator.add, stream_result))
    assert len(combined_result) > 2

    assert "38.5" in combined_result[-1].content
    tool_calls = [msg for msg in combined_result if isinstance(msg, ToolMessage)]
    assert len(tool_calls) >= 3  # Should use at least 3 tools
