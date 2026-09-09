"""
FastAPI router for ResumeMatch AI (ATS Resume Optimizer).

Endpoints
---------
POST /optimize         – JSON body (text inputs)
POST /optimize/upload  – multipart form (file uploads)
POST /rescore          – lightweight re-score after Accept/Reject tailoring
GET  /health
GET  /

Security
--------
- X-API-Key header check (enabled when API_KEY env var is set).
  In development, leave API_KEY unset to skip the check.
- Per-IP rate limiting via slowapi:
    /optimize, /optimize/upload, /rescore → 10 req/minute per IP
- Blocking pipeline runs are off-loaded to a thread pool via
  starlette.concurrency.run_in_threadpool so the event loop stays responsive.

File limits
-----------
- 5 MB per uploaded file.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from optimizer import run_optimization, run_rescore
from parser import extract_text_from_bytes, extract_text_from_plain
from schemas import (
    OptimizeRequest,
    OptimizeResponse,
    OutputFormat,
    RescoreRequest,
    RescoreResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _handle_unexpected(exc: Exception, endpoint: str) -> None:
    """
    Log and re-raise unexpected exceptions as HTTPExceptions.
    Authentication errors (401) and rate-limit errors (429) from the LLM
    provider get clear messages instead of the generic "internal server error".
    """
    exc_str = str(exc).lower()
    # Detect rate-limit / quota exhaustion from any LLM provider
    if any(kw in exc_str for kw in ("429", "rate limit", "rate_limit", "quota", "too many requests", "free-models-per-day")):
        logger.error("LLM rate limit exhausted in %s: %s", endpoint, exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "LLM provider rate limit exceeded. "
                "Wait for the daily reset or add credits to your OpenRouter account."
            ),
        ) from exc
    # Detect API key / auth errors from any LLM provider
    if any(kw in exc_str for kw in ("authentication", "401", "api key", "user not found", "unauthorized")):
        logger.error("LLM authentication error in %s: %s", endpoint, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "LLM provider authentication failed. "
                "Check your API key (GEMINI_API_KEY / OPENROUTER_API_KEY / etc.) in .env."
            ),
        ) from exc
    logger.exception("Unexpected error in %s", endpoint)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error. Check server logs.",
    ) from exc

# Maximum upload size: 5 MB per file
_MAX_FILE_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Rate limiter (slowapi) — attached to the router's request objects
# ---------------------------------------------------------------------------

from slowapi import Limiter  # type: ignore
from slowapi.util import get_remote_address  # type: ignore

_limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

# Expose the limiter so main.py can register the exception handler
limiter = _limiter


# ---------------------------------------------------------------------------
# API-key security dependency
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Return the configured API key, or None if auth is disabled."""
    return os.getenv("API_KEY") or None


async def require_api_key(
    request: Request,
    _key: Optional[str] = Depends(_get_api_key),
) -> None:
    """
    Validate the X-API-Key header.
    Skipped entirely when the API_KEY environment variable is not set
    (convenient for local development without a key).
    """
    configured_key = os.getenv("API_KEY")
    if not configured_key:
        # Auth disabled — dev mode
        return
    client_key = request.headers.get("X-API-Key", "")
    if client_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------------------------------------------------------------------------
# Root + Health
# ---------------------------------------------------------------------------

@router.get("/", tags=["Meta"])
def root() -> dict:
    return {
        "service": "ResumeMatch AI",
        "version": "1.0.0",
        "docs": "/docs",
    }


@router.get("/health", tags=["Meta"])
def health() -> dict:
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# POST /optimize  (JSON – plain text inputs)
# ---------------------------------------------------------------------------

@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Optimizer"],
    summary="Optimize a resume against a job description (text input)",
    description=(
        "Accepts plain-text resume and JD. "
        "Returns full ATS analysis, gap analysis, tailored resume, and re-score. "
        "Requires X-API-Key header when API_KEY env var is set."
    ),
    dependencies=[Depends(require_api_key)],
)
@_limiter.limit("10/minute")
async def optimize_text(request: Request, body: OptimizeRequest) -> OptimizeResponse:
    """
    Run the full five-stage ATS optimization pipeline on text inputs.

    The synchronous pipeline is off-loaded to a thread pool so the async
    event loop remains responsive to other requests during the ~30–90 s
    of sequential LLM calls.
    """
    try:
        return await run_in_threadpool(run_optimization, body)
    except (ValueError, RuntimeError) as exc:
        logger.exception("Pipeline error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _handle_unexpected(exc, "/optimize")


# ---------------------------------------------------------------------------
# POST /optimize/upload  (multipart – file uploads)
# ---------------------------------------------------------------------------

@router.post(
    "/optimize/upload",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Optimizer"],
    summary="Optimize a resume against a job description (file upload)",
    description=(
        "Upload resume as PDF/DOCX/TXT and JD as PDF/DOCX/TXT or plain text. "
        "All other parameters are optional. "
        "Requires X-API-Key header when API_KEY env var is set."
    ),
    dependencies=[Depends(require_api_key)],
)
@_limiter.limit("10/minute")
async def optimize_upload(
    request: Request,
    resume_file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    jd_file: Optional[UploadFile] = File(
        None,
        description="Job description file (PDF, DOCX, or TXT). Provide this OR jd_text.",
    ),
    jd_text: Optional[str] = Form(
        None,
        description="Job description as plain text. Provide this OR jd_file.",
    ),
    target_job_title: Optional[str] = Form(None),
    target_company: Optional[str] = Form(None),
    preferred_resume_length: Optional[str] = Form(None),
    candidate_instructions: Optional[str] = Form(None),
    output_format: OutputFormat = Form(OutputFormat.JSON),
) -> OptimizeResponse:
    """
    Run the full five-stage ATS optimization pipeline on uploaded files.

    The synchronous pipeline is off-loaded to a thread pool via
    run_in_threadpool so the async event loop stays responsive.
    """

    # -- Validate mutual exclusivity of jd_file / jd_text -----------------
    if jd_file is None and not jd_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either jd_file or jd_text.",
        )
    if jd_file is not None and jd_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either jd_file or jd_text, not both.",
        )

    # -- Extract resume text -----------------------------------------------
    resume_bytes = await resume_file.read()
    if len(resume_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Resume file exceeds 5 MB limit.",
        )
    try:
        resume_text = extract_text_from_bytes(
            resume_file.filename or "resume.txt", resume_bytes
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    # -- Extract JD text ---------------------------------------------------
    if jd_file is not None:
        jd_bytes = await jd_file.read()
        if len(jd_bytes) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="JD file exceeds 5 MB limit.",
            )
        try:
            jd_raw = extract_text_from_bytes(
                jd_file.filename or "jd.txt", jd_bytes
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
    else:
        jd_raw = extract_text_from_plain(jd_text or "")

    # -- Build request and run pipeline ------------------------------------
    try:
        opt_request = OptimizeRequest(
            candidate_resume=resume_text,
            job_description=jd_raw,
            target_job_title=target_job_title,
            target_company=target_company,
            preferred_resume_length=preferred_resume_length,
            candidate_instructions=candidate_instructions,
            output_format=output_format,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        # ✅ Fixed: off-load blocking I/O-heavy pipeline to thread pool
        return await run_in_threadpool(run_optimization, opt_request)
    except (ValueError, RuntimeError) as exc:
        logger.exception("Pipeline error in /optimize/upload")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _handle_unexpected(exc, "/optimize/upload")


# ---------------------------------------------------------------------------
# POST /rescore  (lightweight re-score after Accept/Reject)
# ---------------------------------------------------------------------------

@router.post(
    "/rescore",
    response_model=RescoreResponse,
    status_code=status.HTTP_200_OK,
    tags=["Optimizer"],
    summary="Re-score a tailored resume without re-running the full pipeline",
    description=(
        "Called after the user accepts or rejects suggestion cards on the frontend. "
        "Skips stages 1–4; runs only the re-score LLM call. "
        "Requires X-API-Key header when API_KEY env var is set."
    ),
    dependencies=[Depends(require_api_key)],
)
@_limiter.limit("10/minute")
async def rescore(request: Request, body: RescoreRequest) -> RescoreResponse:
    """Re-score a (possibly user-modified) tailored resume against the JD."""
    try:
        return await run_in_threadpool(run_rescore, body)
    except (ValueError, RuntimeError) as exc:
        logger.exception("Re-score pipeline error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _handle_unexpected(exc, "/rescore")
