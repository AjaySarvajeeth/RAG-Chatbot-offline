print(">>> Running MAIN.PY from:", __file__)

import os
import asyncio
import threading
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from local_llm import stream_llm
from retriever import get_retriever

# ---------------- CONFIG ---------------- #
INDEX_DIR = Path(os.getenv('INDEX_DIR', './index')).resolve()

# ---------------- APP ---------------- #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

# ---------------- RETRIEVER ---------------- #
retriever = get_retriever()

def build_prompt(user_input: str):
    context_chunks = retriever(user_input)
    context_text = "\n".join(context_chunks)

    prompt = f"""
You are an expert Incident, Problem & Change Management assistant.
Use only the provided documentation to answer.

If the documentation clearly answers the question, respond concisely and accurately.

If the documentation does not directly answer but contains related context, summarize that context and state that the documentation does not explicitly define it.

If nothing relevant is found, reply only with: "I don't know".

Documentation:
{context_text}

Question:
{user_input}

Answer:
"""
    return prompt.strip()

# ---------------- CHAT ENDPOINT ---------------- #
@app.get("/chat")
async def chat(request: Request):
    user_input = request.query_params.get("query", "").strip()
    if not user_input:
        return {"error": "Empty query"}

    prompt = build_prompt(user_input)

    async def event_generator():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()

        # Background thread to run the sync generator
        def worker():
            try:
                for token in stream_llm(prompt):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, f"[Error]: {e}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, "[DONE]")

        threading.Thread(target=worker, daemon=True).start()

        while True:
            token = await queue.get()
            if token == "[DONE]":
                yield {"data": "[DONE]"}
                break
            yield {"data": token}

    return EventSourceResponse(event_generator())

# ---------------- TEMPLATE ROUTE ---------------- #
@app.get("/chat-ui")
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# ---------------- ROOT ---------------- #
@app.get("/")
async def root():
    return {"message": "IPC Chatbot API running."}

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    import uvicorn
    import webbrowser

    def open_browser():
        webbrowser.open("http://127.0.0.1:8000/chat-ui")

    threading.Timer(1.0, open_browser).start()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
