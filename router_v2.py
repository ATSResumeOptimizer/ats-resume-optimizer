"""
V2 API router — async job-based endpoints for B2B / bulk use cases.

Endpoints
---------
POST /v1/jobs              – submit one optimization job (returns 202 immediately)
GET  /v1/jobs/{job_id}     – poll job status / fetch result when completed
POST /v1/batch             – submit N résumés × 1 JD (returns 202 + list of job_ids)
GET  /v1/usage             – per-org token/request usage summary for a period

Auth
----
All v1 endpoints require an X-API-Key header.  The key is looked up in the
api_keys table (SHA-256 hash comparison) and resolves to an Organization.
The single global API_KEY env var is NOT used here; that remains for the legacy
/optimize endpoints.

The existing /optimize, /optimize/upload, and /rescore endpoints are unchanged
and continue to work as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from db import SessionLocal
from models import ApiKey, Job, JobStatus, Organization, UsageRecord
from optimizer import (
    _build_ats_scores,
    _build_final_analysis,
    _build_gap_analysis,
    _build_keyword_analysis,
    _build_optimization_changes,
    _build_requirements,
    _build_tailored_resume,
    _safe_list,
    _safe_str,
    _load_json_col,
)
from parser import extract_text_from_bytes
from schemas import OptimizeRequest, OptimizeResponse, OutputFormat

logger = logging.getLogger(__name__)

router_v2 = APIRouter(prefix="/v1", tags=["V2 Async Jobs"])

# Maximum upload size: 5 MB per file (shared with v1 router)
_MAX_FILE_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth: resolve X-API-Key → Organization
# ---------------------------------------------------------------------------

def _hash_key(raw_key: str) -> str:
    """SHA-256 hex digest of the raw API key string."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def get_org_from_api_key(
    request: Request,
    db: Session = Depends(get_db),
) -> Organization:
    """
    FastAPI dependency: validate X-API-Key, return the owning Organization.

    - Hashes the incoming key and looks it up in api_keys.key_hash.
    - Updates last_used_at on every successful auth.
    - Raises HTTP 401 if the key is missing, unknown, or revoked.
    """
    raw_key = request.headers.get("X-API-Key", "")
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = _hash_key(raw_key)
    api_key_row: ApiKey | None = (
        db.query(ApiKey)
        .filter_by(key_hash=key_hash, is_active=True)
        .first()
    )
    if api_key_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Stamp last_used_at (best-effort — don't fail the request if this errors)
    try:
        api_key_row.last_used_at = datetime.utcnow()
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()

    org: Organization | None = db.query(Organization).filter_by(id=api_key_row.org_id).first()
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Organization not found for this API key.",
        )
    return org


# ---------------------------------------------------------------------------
# Quota check
# ---------------------------------------------------------------------------

def _check_quota(org: Organization, db: Session, count: int = 1) -> None:
    """
    Raise HTTP 429 if the org has consumed >= their monthly_request_quota.
    Counts completed + running jobs this calendar month.
    """
    from sqlalchemy import func, extract  # local import to avoid polluting module namespace
    now = datetime.utcnow()
    used: int = (
        db.query(func.count(Job.id))
        .filter(
            Job.org_id == org.id,
            Job.status.notin_([JobStatus.FAILED]),
            extract("year", Job.created_at) == now.year,
            extract("month", Job.created_at) == now.month,
        )
        .scalar()
        or 0
    )
    if used + count > org.monthly_request_quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Monthly quota exceeded ({used}/{org.monthly_request_quota} requests used). "
                "Upgrade your plan to continue."
            ),
        )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _job_to_dict(job: Job) -> dict:
    """Serialize a Job row to a JSON-safe dict for API responses."""
    base = {
        "job_id": job.id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
    if job.status == JobStatus.FAILED:
        base["error"] = job.error_message
        base["error_stage"] = job.error_stage
    if job.status == JobStatus.COMPLETED:
        base["result"] = _assemble_result(job)
    return base


def _assemble_result(job: Job) -> dict:
    """
    Reconstruct an OptimizeResponse-compatible dict from the Job's checkpoints.
    Mirrors the assembly logic in optimizer.run_optimization().
    """
    ats_data = _load_json_col(job.ats_data)
    rescore_data = _load_json_col(job.rescore_data)
    parsed_jd = _load_json_col(job.parsed_jd)
    parsed_resume = _load_json_col(job.parsed_resume)
    tailored_data = _load_json_col(job.tailored_data)

    original_score = int((ats_data.get("ats_analysis") or {}).get("original_score", 0))
    keyword_analysis_dict = ats_data.get("keyword_analysis") or {}
    if rescore_data.get("keyword_analysis"):
        keyword_analysis_dict = {**keyword_analysis_dict, **rescore_data["keyword_analysis"]}

    response = OptimizeResponse(
        candidate={
            "name": _safe_str(parsed_resume, "name", "Unknown"),
            "target_role": parsed_jd.get("job_title") or "",
        },
        ats_analysis=_build_ats_scores(ats_data),
        requirements=_build_requirements(ats_data),
        gap_analysis=_build_gap_analysis(ats_data),
        optimization_changes=_build_optimization_changes(rescore_data),
        tailored_resume=_build_tailored_resume(tailored_data),
        final_ats_analysis=_build_final_analysis(rescore_data, original_score),
        keyword_analysis=_build_keyword_analysis({"keyword_analysis": keyword_analysis_dict}),
        warnings=[str(w) for w in _safe_list(rescore_data, "warnings")],
    )
    return response.model_dump()


# ---------------------------------------------------------------------------
# POST /v1/jobs — submit a single job
# ---------------------------------------------------------------------------

@router_v2.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit one optimization job (async)",
    description=(
        "Accepts the same payload as POST /optimize but returns immediately with a job_id. "
        "Poll GET /v1/jobs/{job_id} for status and results. "
        "Optionally supply a webhook_url to receive the result via HTTP POST when complete."
    ),
)
async def create_job(
    body: OptimizeRequest,
    webhook_url: Optional[str] = None,
    org: Organization = Depends(get_org_from_api_key),
    db: Session = Depends(get_db),
) -> dict:
    _check_quota(org, db)

    params = body.model_dump(exclude={"candidate_resume", "job_description"})

    job = Job(
        org_id=org.id,
        input_resume_text=body.candidate_resume,
        input_jd_text=body.job_description,
        request_params=json.dumps(params),
        webhook_url=webhook_url,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from worker import enqueue_job  # local import — worker brings in Redis
    enqueue_job(job.id)

    logger.info("Enqueued job %s for org %s", job.id, org.id)
    return {
        "job_id": job.id,
        "status": job.status.value,
        "poll_url": f"/v1/jobs/{job.id}",
    }


# ---------------------------------------------------------------------------
# GET /v1/jobs/{job_id} — poll / fetch result
# ---------------------------------------------------------------------------

@router_v2.get(
    "/jobs/{job_id}",
    summary="Get job status and result",
    description=(
        "Returns the current status of the job. "
        "When status == 'completed', the full OptimizeResponse is included in 'result'. "
        "When status == 'failed', 'error' and 'error_stage' describe what went wrong."
    ),
)
async def get_job(
    job_id: str,
    org: Organization = Depends(get_org_from_api_key),
    db: Session = Depends(get_db),
) -> dict:
    job: Job | None = db.query(Job).filter_by(id=job_id, org_id=org.id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _job_to_dict(job)


# ---------------------------------------------------------------------------
# POST /v1/batch — N résumés × 1 JD
# ---------------------------------------------------------------------------

@router_v2.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch of résumés against one job description",
    description=(
        "Upload up to 50 résumé files (PDF/DOCX/TXT) and one JD text. "
        "Each résumé becomes an independent job. "
        "Returns a list of job_ids to poll individually, or supply webhook_url "
        "to receive each result as it completes."
    ),
)
async def create_batch(
    resume_files: list[UploadFile] = File(..., description="Résumé files (PDF, DOCX, or TXT). Max 50."),
    jd_text: str = Form(..., description="Job description plain text."),
    target_job_title: Optional[str] = Form(None),
    target_company: Optional[str] = Form(None),
    preferred_resume_length: Optional[str] = Form(None),
    candidate_instructions: Optional[str] = Form(None),
    output_format: OutputFormat = Form(OutputFormat.JSON),
    webhook_url: Optional[str] = Form(None),
    org: Organization = Depends(get_org_from_api_key),
    db: Session = Depends(get_db),
) -> dict:
    if len(resume_files) > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Maximum 50 résumé files per batch.",
        )

    _check_quota(org, db, count=len(resume_files))

    params = json.dumps({
        "target_job_title": target_job_title,
        "target_company": target_company,
        "preferred_resume_length": preferred_resume_length,
        "candidate_instructions": candidate_instructions,
        "output_format": output_format.value,
    })

    from worker import enqueue_job  # local import — avoids Redis import at module load

    job_ids: list[str] = []
    errors: list[dict] = []

    for f in resume_files:
        raw_bytes = await f.read()
        if len(raw_bytes) > _MAX_FILE_BYTES:
            errors.append({"filename": f.filename, "error": "File exceeds 5 MB limit."})
            continue
        try:
            resume_text = extract_text_from_bytes(f.filename or "resume.txt", raw_bytes)
        except ValueError as exc:
            errors.append({"filename": f.filename, "error": str(exc)})
            continue

        job = Job(
            org_id=org.id,
            input_resume_text=resume_text,
            input_jd_text=jd_text,
            request_params=params,
            webhook_url=webhook_url,
        )
        db.add(job)
        db.flush()   # get job.id without full commit
        job_ids.append(job.id)

    db.commit()

    for jid in job_ids:
        enqueue_job(jid)

    logger.info("Enqueued batch of %d jobs for org %s", len(job_ids), org.id)
    return {
        "batch_size": len(job_ids),
        "job_ids": job_ids,
        "skipped": errors,
    }


# ---------------------------------------------------------------------------
# GET /v1/usage — per-org usage summary
# ---------------------------------------------------------------------------

@router_v2.get(
    "/usage",
    summary="Per-org token and request usage for a billing period",
    description=(
        "Returns aggregated prompt_tokens, completion_tokens, and request count "
        "for the authenticated org, optionally filtered to a YYYY-MM period. "
        "Defaults to the current calendar month."
    ),
)
async def get_usage(
    period: Optional[str] = None,   # YYYY-MM, e.g. "2025-09"
    org: Organization = Depends(get_org_from_api_key),
    db: Session = Depends(get_db),
) -> dict:
    from sqlalchemy import func, extract  # local import

    now = datetime.utcnow()
    if period:
        try:
            year, month = int(period[:4]), int(period[5:7])
        except (ValueError, IndexError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="period must be in YYYY-MM format, e.g. '2025-09'.",
            )
    else:
        year, month = now.year, now.month

    rows = (
        db.query(
            func.count(UsageRecord.id).label("request_count"),
            func.sum(UsageRecord.prompt_tokens).label("prompt_tokens"),
            func.sum(UsageRecord.completion_tokens).label("completion_tokens"),
        )
        .filter(
            UsageRecord.org_id == org.id,
            extract("year", UsageRecord.created_at) == year,
            extract("month", UsageRecord.created_at) == month,
        )
        .first()
    )

    return {
        "org_id": org.id,
        "period": f"{year:04d}-{month:02d}",
        "request_count": rows.request_count or 0,
        "prompt_tokens": rows.prompt_tokens or 0,
        "completion_tokens": rows.completion_tokens or 0,
        "monthly_quota": org.monthly_request_quota,
        "plan": org.plan,
    }
