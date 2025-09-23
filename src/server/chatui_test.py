"""
Test the chatui.py file.
"""
import pytest
from typing import Dict, List, Optional, Any, Callable
import sys
from unittest.mock import Mock
from dataclasses import dataclass, field
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.messages import ToolMessage
from functools import wraps

# Mock classes to simulate chainlit behavior
@dataclass
class MockUserSession:
    """Mock implementation of chainlit's UserSession"""
    _store: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

@dataclass
class MockMessage:
    """Mock implementation of chainlit's Message"""
    content: str = ""
    elements: List[Any] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    sent: bool = False

    async def stream_token(self, token: str, is_sequence: bool = False) -> None:
        self.tokens.append(token)
        if not is_sequence:
            self.content += token

    async def send(self) -> None:
        self.sent = True

@dataclass 
class MockChainlit:
    """Mock implementation of chainlit module"""
    user_session: MockUserSession = field(default_factory=MockUserSession)
    messages: List[MockMessage] = field(default_factory=list)
    
    def Message(self, content: str = "", **kwargs) -> MockMessage:
        msg = MockMessage(content=content)
        self.messages.append(msg)
        return msg

    LangchainCallbackHandler = Mock()

    def on_chat_start(self, func: Callable) -> Callable:
        """Mock implementation of the on_chat_start decorator"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper

    def on_message(self, func: Callable) -> Callable:
        """Mock implementation of the on_message decorator"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper

# Create mock chainlit instance
mock_cl = MockChainlit()

# Import and patch the chatui module
sys.modules['chainlit'] = mock_cl

@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_sequence():
    """Test the chat sequence from start to messages"""
    from src.server.chatui import on_chat_start, on_message

    # Start the chat session
    await on_chat_start()

    # Verify chat initialization
    messages = mock_cl.user_session.get("messages")
    assert messages is not None

    # Verify runnable is set
    assert mock_cl.user_session.get("runnable") is not None

    # Test sequence of messages
    # TODO: Add more tests according to the desired testing scenario
    test_messages = [
        "Hello, how are you?",
        "What can you tell me about the HPC system?",
        "Thank you for your help!"
    ]

    # TODO: Add different types of validations according to the desired testing scenario
    for msg_content in test_messages:
        # Create mock incoming message
        mock_message = MockMessage(content=msg_content)
        
        # Process the message
        await on_message(mock_message)

        # Verify response was generated
        assert len(mock_cl.messages) > 0
        latest_message = mock_cl.messages[-1]
        
        # Verify message was sent
        assert latest_message.sent

        # Verify message history was updated
        messages = mock_cl.user_session.get("messages")
        assert messages is not None
        assert len(messages) > 1  # Should have system message + human/ai pairs

        # The second to last message should be a human message or a tool result
        assert (
            isinstance(messages[-2], HumanMessage) or
            isinstance(messages[-2], ToolMessage)
        )

        # If the second to last message is a human message, it should be the same as the input message
        if isinstance(messages[-2], HumanMessage):
            assert messages[-2].content == msg_content

