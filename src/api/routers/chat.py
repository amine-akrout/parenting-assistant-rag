from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.chatbot import chatbot_response, create_chatbot_chain, load_retriever

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# Load retriever and chain at startup
# retriever = load_retriever()
# qa_chain = create_chatbot_chain(retriever)

retriever = None
qa_chain = None


@router.on_event("startup")
async def startup_event():
    global retriever, qa_chain
    retriever = load_retriever()
    qa_chain = create_chatbot_chain(retriever)


class QuestionRequest(BaseModel):
    question: str


@router.post("/", response_model=dict)
def get_chat_response(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Invalid question.")
    try:
        response = chatbot_response(request.question, qa_chain)
        return JSONResponse(content={"question": request.question, "answer": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
