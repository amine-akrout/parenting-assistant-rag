from fastapi import FastAPI

from src.api.routers import chat

app = FastAPI(title="Parenting Chatbot API", version="1.0")

# Include API routers
app.include_router(chat.router)


@app.get("/")
def root():
    return {"message": "Welcome to the Parenting Chatbot API!"}
