"""
ATS Resume Optimizer – FastAPI application entry point.

Environment variables
---------------------
LLM_PROVIDER       "gemini" | "openai" | "azure" | "anthropic" | "openrouter"
                   | "groq" | "ollama"                (default: gemini)
LLM_MODEL          model name                         (default: gemini-flash-latest)
GEMINI_API_KEY     required when LLM_PROVIDER=gemini
OPENAI_API_KEY     required when LLM_PROVIDER=openai
AZURE_OPENAI_API_KEY       ─┐
AZURE_OPENAI_ENDPOINT       ├ required when LLM_PROVIDER=azure
AZURE_OPENAI_DEPLOYMENT     │
AZURE_OPENAI_API_VERSION   ─┘
ANTHROPIC_API_KEY  required when LLM_PROVIDER=anthropic
GROQ_API_KEY       required when LLM_PROVIDER=groq
OLLAMA_BASE_URL    optional when LLM_PROVIDER=ollama  (default: http://localhost:11434/v1)

Run locally
-----------
uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv  # type: ignore

load_dotenv()  # Load .env before any env var is read

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler  # type: ignore
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.middleware import SlowAPIMiddleware  # type: ignore

from db import init_db
from router import limiter, router
from router_v2 import router_v2


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Create DB tables on startup (idempotent — skips existing tables)
    init_db()

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-flash-latest")

    # Warn loudly at startup if the required API key is missing
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        # ollama is local — no API key required
    }
    required_key = key_map.get(provider)
    if required_key and not os.getenv(required_key):
        logger.error(
            "⚠️  %s is not set. LLM calls will fail. "
            "Set it in your .env file or environment.",
            required_key,
        )

    logger.info(
        "ATS Resume Optimizer started | provider=%s | model=%s", provider, model
    )
    yield
    logger.info("ATS Resume Optimizer shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATS Resume Optimizer",
    version="1.0.0",
    lifespan=_lifespan,
    description=(
        "Enterprise-grade AI agent that analyses a candidate's resume against "
        "a target job description, calculates an ATS compatibility score, "
        "identifies gaps, and produces a truthfully tailored ATS-friendly resume."
    ),
    contact={
        "name": "ATS Resume Optimizer",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "Optimizer",
            "description": "Core ATS resume optimization endpoints.",
        },
        {
            "name": "Meta",
            "description": "Health check and service metadata.",
        },
    ],
)


# ---------------------------------------------------------------------------
# CORS – restrict in production via ALLOWED_ORIGINS env var
# ---------------------------------------------------------------------------
# Dev default: the Vite dev server (port 5173).
# Production: set ALLOWED_ORIGINS=https://yourdomain.com in the server env.
# WARNING: allow_credentials=True is incompatible with allow_origins=["*"]
# per the CORS spec.  We detect the wildcard and disable credentials in that
# case to avoid browser errors.
# ---------------------------------------------------------------------------

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
_allow_credentials = "*" not in _origins  # wildcard + credentials is invalid

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Rate limiting (slowapi)
# ---------------------------------------------------------------------------

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(router)
app.include_router(router_v2)
