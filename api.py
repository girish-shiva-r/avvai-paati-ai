"""FastAPI backend for Avvaiyar Paatti (deploy on a Hugging Face Docker Space, port 7860)."""
import json, logging, os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
from paatti import Paatti  # noqa: E402

BOT: Optional[Paatti] = None


@asynccontextmanager
async def lifespan(app):
    global BOT
    BOT = Paatti()
    logging.info("Paatti ready: llm=%s embedder=%s", BOT.llm.name,
                 BOT.retriever.embedder.name if BOT.retriever.embedder else "bm25-only")
    yield


app = FastAPI(title="Avvaiyar Paatti API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class Turn(BaseModel):
    role: str
    content: str
    verse_no: Optional[int] = None


class ChatRequest(BaseModel):
    query: str
    history: list[Turn] = []


def _ready():
    if BOT is None:
        raise HTTPException(503, "Paatti is waking up")
    return BOT


@app.get("/health")
def health():
    return {"status": "ok" if BOT else "loading"}


@app.post("/chat")
def chat(req: ChatRequest):
    if len(req.query) > 2000:
        raise HTTPException(400, "message too long")
    return _ready().answer(req.query, [t.model_dump() for t in req.history])


@app.post("/chat_audio")
async def chat_audio(file: UploadFile = File(...), history: str = Form("[]")):
    bot = _ready()
    audio = await file.read()
    if len(audio) > 10 * 1024 * 1024:
        raise HTTPException(400, "audio too large")
    text = bot.llm.transcribe(audio, file.content_type or "audio/wav")
    out = bot.answer(text, json.loads(history or "[]"))
    out["transcribed_text"] = text
    return out


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    if len(audio) > 10 * 1024 * 1024:
        raise HTTPException(400, "audio too large")
    return {"text": _ready().llm.transcribe(audio, file.content_type or "audio/wav").strip()}


@app.get("/verse/{no}")
def verse(no: int):
    v = _ready().kb.get(no)
    if not v:
        raise HTTPException(404, "Aathichoodi has 109 verses")
    return v.card()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
