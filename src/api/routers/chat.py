from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.core.chatbot import chatbot_response, create_chatbot_chain, load_retriever
from src.monitoring.monitoring import create_langfuse_handler

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# Load retriever and chain at startup
retriever = None
qa_chain = None
langfuse_handler = None


@router.on_event("startup")
async def startup_event():
    global retriever, qa_chain, langfuse_handler
    retriever = load_retriever()
    qa_chain = create_chatbot_chain(retriever)
    langfuse_handler = create_langfuse_handler()
    if not langfuse_handler:
        logger.error("Failed to create langfuse handler.")


class QuestionRequest(BaseModel):
    question: str


@router.post("/", response_model=dict)
def get_chat_response(request: QuestionRequest):
    if not request.question.strip():
        logger.info("question : ", request.question)
        raise HTTPException(status_code=400, detail="Invalid question.")
    try:
        rag_response = chatbot_response(request.question, qa_chain, langfuse_handler)
        response = rag_response.get(
            "response", "Sorry, I don't have an answer for that."
        )
        return JSONResponse(content={"question": request.question, "answer": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
