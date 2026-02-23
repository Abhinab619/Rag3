"""
FastAPI wrapper for the RAG Bot.
Exposes the StaticBot as a REST API.
"""
import asyncio
from typing import AsyncGenerator
import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Ensure local imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

bot = None  # type: Optional[object]


async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global bot
    try:
        from static_chat import StaticBot
        print("Initializing RAG Bot...")
        bot = StaticBot()
        print("Bot ready!")
        yield
    except asyncio.CancelledError:
        # Normal during reload / shutdown
        pass
    finally:
        print("Shutting down RAG Bot...")


app = FastAPI(
    title="Citizen Assistant API",
    description="RAG-powered chatbot for government schemes and services",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str] = None


# -------------------------
# Routes
# -------------------------
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Citizen Assistant API is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if bot is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        answer = bot.chat(query)
        intent = bot.detect_focus(query) if hasattr(bot, "detect_focus") else None
        return ChatResponse(
            answer=answer,
            intent=intent
        )
    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
