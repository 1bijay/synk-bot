
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chatbot import ask_question

# --------------------------------------------------
# 1. Configure logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Synk Corp Support API",
    description="Customer support chatbot powered by LangGraph and RAG",
    version="1.0.0",
)

# --------------------------------------------------
# 3. Define the request body
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question",
    )

    session_id: str = Field(
        default="default-session",
        min_length=1,
        max_length=100,
        description="Unique ID for the conversation",
    )


# --------------------------------------------------
# 4. Define the response body
# --------------------------------------------------

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str


# --------------------------------------------------
# 5. Health-check endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Synk Corp Support API is running"
    }


# --------------------------------------------------
# 6. Chat endpoint
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()
    session_id = request.session_id.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty",
        )

    try:
        logger.info(
            "Received chat request for session: %s",
            session_id,
        )

        answer = ask_question(
            question=question,
            session_id=session_id,
        )

        logger.info(
            "Chat request completed for session: %s",
            session_id,
        )

        return ChatResponse(
            session_id=session_id,
            question=question,
            answer=answer,
        )

    except Exception as e:
        # Print the complete traceback in the terminal.
        logger.exception(
            "Chatbot request failed for session: %s",
            session_id,
        )

        # Show the error temporarily while debugging.
        # Remove str(e) from the response before production use.
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot error: {str(e)}",
        ) from e
