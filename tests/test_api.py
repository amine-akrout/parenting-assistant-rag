from fastapi.testclient import TestClient

from src.api.main import app
from src.core.chatbot import create_chatbot_chain, load_retriever


@app.on_event("startup")
async def startup_event():
    """
    Load retriever and chatbot chain at startup.
    """
    global retriever, qa_chain
    retriever = load_retriever()
    qa_chain = create_chatbot_chain(retriever)


def test_chat_endpoint():
    """Test chatbot API response."""
    with TestClient(app) as client:
        response = client.post(
            "/chat/", json={"question": "How do I stop toddler tantrums?"}
        )
        print(response.json())
        assert response.status_code == 200
        assert "answer" in response.json()


def test_invalid_question():
    """Test chatbot API with an empty question."""
    with TestClient(app) as client:
        response = client.post("/chat/", json={"question": ""})
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid question."
