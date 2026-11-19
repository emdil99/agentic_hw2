"""Web server for the supplement label agent.

    uv run --no-sync app.py    then open http://127.0.0.1:8000
"""
import os
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agent

INDEX_PATH = Path(__file__).parent / "static" / "index.html"

# session_id -> list of messages. In memory, so sessions reset when the server restarts, and the
# Cloud Run service is capped at one instance so a session's follow-ups reach the same memory.
sessions: dict[str, list[dict]] = {}
# FastAPI runs these handlers on a thread pool, so one session's turns must not interleave.
session_locks: dict[str, threading.Lock] = {}
sessions_guard = threading.Lock()
MAX_SESSIONS = 500


def get_session(session_id: str) -> tuple[list[dict], threading.Lock]:
    """Return a session's history and lock, creating them and evicting the oldest session if full."""
    with sessions_guard:
        if session_id not in sessions:
            if len(sessions) >= MAX_SESSIONS:
                oldest = next(iter(sessions))
                sessions.pop(oldest)
                session_locks.pop(oldest, None)
            sessions[session_id] = [{"role": "system", "content": agent.SYSTEM_PROMPT}]
            session_locks[session_id] = threading.Lock()
        return sessions[session_id], session_locks[session_id]

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(INDEX_PATH)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    history, lock = get_session(session_id)

    with lock:
        turn_start = len(history)
        history.append({"role": "user", "content": request.message})
        try:
            response, tool_calls = agent.run_agent(history)
        except Exception as error:
            # Auth, quota or model errors: show them in the chat instead of a 500, and drop the
            # half-finished turn so the session stays usable.
            del history[turn_start:]
            response, tool_calls = f"Model call failed: {type(error).__name__}: {str(error)[:300]}", []

    for call in tool_calls:
        call["agent"] = "supplement_label_agent"
    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    with sessions_guard:
        sessions.pop(session_id, None)
        session_locks.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    # Cloud Run sets PORT and needs 0.0.0.0. Locally, stay on localhost:8000.
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
