import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.chatbot import chatbot_response


@pytest.fixture
def mock_qa_chain():
    """Mock the QA chain response."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {"response": "Mocked chatbot response."}
    return mock_chain


def test_chatbot_response(mock_qa_chain):
    """Test chatbot response function."""
    question = "How do I get my child to sleep?"
    response = chatbot_response(question, mock_qa_chain)
    assert isinstance(response, dict)
    assert "response" in response
    assert response["response"] == "Mocked chatbot response."
