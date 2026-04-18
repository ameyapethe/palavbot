"""FastAPI entrypoint for palavbot.

Runs under uvicorn locally and under AWS Lambda Web Adapter in production.
The API is stateless; conversation history is supplied by the caller on each
request. The FAISS index is loaded once at process startup and reused across
warm invocations.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from palav.retrieval import (
    ANSWER_MODEL_DEFAULT,
    DEFAULT_INDEX_DIR,
    DEFAULT_LINKS_FILE,
    build_or_load,
    make_answer,
    retrieve,
)

logger = logging.getLogger("palavbot")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

LINKS_FILE = os.getenv("PALAV_LINKS_FILE", DEFAULT_LINKS_FILE)
INDEX_DIR = os.getenv("PALAV_INDEX_DIR", DEFAULT_INDEX_DIR)
ANSWER_MODEL = os.getenv("PALAV_ANSWER_MODEL", ANSWER_MODEL_DEFAULT)
CORS_ORIGINS = [o.strip() for o in os.getenv("PALAV_CORS_ORIGINS", "*").split(",") if o.strip()]


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: List[Message] = Field(default_factory=list, max_length=40)


class Source(BaseModel):
    url: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    external_knowledge: bool = False
    rejected: bool = False


state: dict = {}


def _resolve_openai_key() -> str:
    """Return the OpenAI API key.

    Prefers OPENAI_API_KEY env var (local dev). In Lambda we set
    PALAV_OPENAI_KEY_SSM_PARAM and fetch the SecureString at cold start so
    the secret never lives in Lambda's env config.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    param_name = os.environ.get("PALAV_OPENAI_KEY_SSM_PARAM")
    if not param_name:
        raise RuntimeError(
            "OPENAI_API_KEY not set and PALAV_OPENAI_KEY_SSM_PARAM not configured"
        )
    import boto3  # imported lazily; unused in local dev

    ssm = boto3.client("ssm")
    resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
    return resp["Parameter"]["Value"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    api_key = _resolve_openai_key()

    logger.info("Loading index from %s", INDEX_DIR)
    bundle = build_or_load(links_file=LINKS_FILE, api_key=api_key, index_dir=INDEX_DIR)
    logger.info(
        "Index ready: chunks=%d, from_cache=%s", len(bundle.chunks), bundle.loaded_from_cache
    )
    state["bundle"] = bundle
    state["client"] = OpenAI(api_key=api_key)
    yield
    state.clear()


app = FastAPI(title="Palavbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "chunks": len(state["bundle"].chunks) if "bundle" in state else 0}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    bundle = state.get("bundle")
    client = state.get("client")
    if bundle is None or client is None:
        raise HTTPException(status_code=503, detail="service not initialized")

    retrieved = retrieve(client, bundle.index, bundle.chunks, req.message)
    result = make_answer(
        client=client,
        model=ANSWER_MODEL,
        question=req.message,
        retrieved=retrieved,
        history=[m.model_dump() for m in req.history],
    )
    return ChatResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        external_knowledge=result.external_knowledge,
        rejected=result.rejected,
    )
