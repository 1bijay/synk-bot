
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# --------------------------------------------------
# 1. Configure the OpenRouter chat model
# --------------------------------------------------

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
     max_tokens=1000,
)

# --------------------------------------------------
# 2. Configure the same embedding model used by ingest.py
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 3. Connect to the existing local Qdrant collection
# --------------------------------------------------

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="synk_company_docs",
    path="./qdrant_data",
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

# --------------------------------------------------
# 4. Define the LangGraph state
# --------------------------------------------------

class SupportState(TypedDict):
    question: str
    context: str
    answer: str
    history: list


# --------------------------------------------------
# 5. Retrieve relevant document chunks
# --------------------------------------------------

def retrieve_documents(state: SupportState):
    question = state["question"]

    docs = retriever.invoke(question)

    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Synk Corp documents")

        context_parts.append(
            f"Source: {source}\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    return {"context": context}


# --------------------------------------------------
# 6. Generate an answer grounded in retrieved context
# --------------------------------------------------

def generate_answer(state: SupportState):
    question = state["question"]
    context = state.get("context", "").strip()
    history = state.get("history", [])

    # Avoid making up an answer when retrieval returns nothing.
    if not context:
        answer = (
            "I couldn't find that information in the available "
            "Synk Corp documents. Please contact the company "
            "for confirmation."
        )

    else:
        # Include the last few conversation messages.
        recent_history = history[-6:]

        history_text = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in recent_history
        )

        prompt = f"""
You are the customer support assistant for Synk Corp.

Rules:
1. Answer using only the information in the provided context.
2. Do not invent company policies, services, prices, or contact details.
3. If the context does not contain the answer, say you cannot
   find that information in the available documents.
4. Answer in the language used by the user: English, Nepali,
   or Romanized Nepali.
5. Keep answers clear, polite, and helpful.
6. Treat the conversation history as context, not as company facts.
7. If the user asks a follow-up question, use the history to
   understand it, but verify company facts against the documents.

Conversation history:
{history_text}

Retrieved company context:
{context}

User question:
{question}

Answer:
"""

        response = llm.invoke(prompt)
        answer = response.content

        # Handle models that return non-string content.
        if not isinstance(answer, str):
            answer = str(answer)

    # Save this turn for the same conversation session.
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    return {
        "answer": answer,
        "history": updated_history,
    }


# --------------------------------------------------
# 7. Build and compile the LangGraph workflow
# --------------------------------------------------

builder = StateGraph(SupportState)

builder.add_node("retrieve", retrieve_documents)
builder.add_node("answer", generate_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

# Memory is kept in RAM while this process is running.
graph = builder.compile(checkpointer=InMemorySaver())


# --------------------------------------------------
# 8. Function that FastAPI will call
# --------------------------------------------------

def ask_question(question: str, session_id: str) -> str:
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    # Read the existing conversation history for this session.
    previous_state = graph.get_state(config)

    history = []

    if previous_state and previous_state.values:
        history = previous_state.values.get("history", [])

    # Execute the graph.
    result = graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "history": history,
        },
        config=config,
    )

    return result["answer"]
