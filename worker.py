"""
RQ worker — executes optimization jobs outside the API process.

Start the worker
----------------
    rq worker optimization --url redis://localhost:6379

Each pipeline stage checkpoints its JSON output to the Job row immediately
after completing.  If a stage fails, re-enqueueing the same job_id will skip
all already-completed stages and resume from where it left off.

Environment variables
---------------------
REDIS_URL      Redis broker URL (default: redis://localhost:6379)
DATABASE_URL   SQLAlchemy connection URL (see db.py for defaults)

All LLM provider env vars (LLM_PROVIDER, LLM_MODEL, GEMINI_API_KEY, …)
must also be set in the worker process environment — the same .env file used
by the API server works fine.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv  # type: ignore

load_dotenv()

from redis import Redis  # type: ignore
from rq import Queue  # type: ignore

from db import SessionLocal
from models import Job, JobStatus, UsageRecord
from optimizer import (
    _deterministic_ats_analysis,
    _deterministic_rescore,
    _load_json_col,
    _step_analyse_jd,
    _step_ats_analysis,
    _step_parse_resume,
    _step_rescore,
    _step_tailor_resume,
)
from webhooks import send_webhook

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(_REDIS_URL)
queue = Queue("optimization", connection=redis_conn)

# How long to keep completed job data before the nightly purge task deletes it.
_DATA_RETENTION_DAYS = int(os.getenv("JOB_DATA_RETENTION_DAYS", "30"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dump(obj: dict) -> str:
    """Serialise a dict to a JSON string for storage."""
    return json.dumps(obj, ensure_ascii=False)


def _record_usage(
    db,
    org_id: str,
    job_id: str,
    stage: str,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Append one UsageRecord row to the billing ledger."""
    rec = UsageRecord(
        org_id=org_id,
        job_id=job_id,
        stage=stage,
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    db.add(rec)
    # Caller must commit


# ---------------------------------------------------------------------------
# Main task — executed by the RQ worker process
# ---------------------------------------------------------------------------

def run_job(job_id: str) -> None:
    """
    Execute (or resume) a single optimization job.

    Each ``if job.<stage> is None:`` block is idempotent: it only runs if that
    stage has not already been persisted.  Re-enqueueing a failed job_id will
    skip completed stages and retry from the first incomplete one.

    Usage token recording
    ---------------------
    The step functions in optimizer.py don't yet return token counts because
    _llm_call() returns a plain string.  The UsageRecord rows written here use
    a sentinel value of -1 for token counts until optimizer.py is updated to
    return structured responses.  See the TODO in each _record_usage call.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    db = SessionLocal()
    try:
        job: Job | None = db.query(Job).filter_by(id=job_id).first()
        if job is None:
            logger.error("run_job: job %s not found in DB", job_id)
            return

        params: dict = _load_json_col(job.request_params)

        def _cb(stage: str):
            """Return a usage_callback closure bound to this stage."""
            def _inner(pt: int, ct: int) -> None:
                _record_usage(db, job.org_id, job_id, stage, provider, model, pt, ct)
                db.commit()
            return _inner

        # ── Stage 1 + 2: Parse resume and analyse JD ────────────────────────
        if job.parsed_resume is None or job.parsed_jd is None:
            logger.info("[%s] Stage 1+2: parsing resume and JD", job_id)
            job.status = JobStatus.PARSING
            db.commit()

            parsed_resume = _step_parse_resume(job.input_resume_text, usage_callback=_cb("parse_resume"))
            parsed_resume["raw_text"] = job.input_resume_text

            parsed_jd = _step_analyse_jd(job.input_jd_text, usage_callback=_cb("analyse_jd"))
            parsed_jd["raw_text"] = job.input_jd_text
            if params.get("target_job_title"):
                parsed_jd["job_title"] = params["target_job_title"]

            job.parsed_resume = _dump(parsed_resume)
            job.parsed_jd = _dump(parsed_jd)
            db.commit()

        # ── Stage 3: ATS analysis ────────────────────────────────────────────
        if job.ats_data is None:
            logger.info("[%s] Stage 3: ATS analysis", job_id)
            job.status = JobStatus.ANALYZING
            db.commit()

            parsed_resume = _load_json_col(job.parsed_resume)
            parsed_jd = _load_json_col(job.parsed_jd)
            ats_data = _deterministic_ats_analysis(parsed_resume, parsed_jd)
            job.ats_data = _dump(ats_data)
            db.commit()

        # ── Stage 4: Tailor resume ───────────────────────────────────────────
        if job.tailored_data is None:
            logger.info("[%s] Stage 4: tailoring resume", job_id)
            job.status = JobStatus.TAILORING
            db.commit()

            parsed_resume = _load_json_col(job.parsed_resume)
            parsed_jd = _load_json_col(job.parsed_jd)
            ats_data = _load_json_col(job.ats_data)
            gap_analysis = ats_data.get("gap_analysis") or []

            tailored_data = _step_tailor_resume(
                parsed_resume,
                parsed_jd,
                gap_analysis,
                candidate_instructions=params.get("candidate_instructions") or "",
                preferred_length=params.get("preferred_resume_length") or "",
                usage_callback=_cb("tailor"),
            )
            job.tailored_data = _dump(tailored_data)
            db.commit()

        # ── Stage 5: Re-score ────────────────────────────────────────────────
        if job.rescore_data is None:
            logger.info("[%s] Stage 5: re-scoring", job_id)
            job.status = JobStatus.RESCORING
            db.commit()

            ats_data = _load_json_col(job.ats_data)
            tailored_data = _load_json_col(job.tailored_data)
            parsed_jd = _load_json_col(job.parsed_jd)
            original_score = int((ats_data.get("ats_analysis") or {}).get("original_score", 0))
            gap_analysis = ats_data.get("gap_analysis") or []
            keyword_analysis = ats_data.get("keyword_analysis") or {}

            rescore_data = _deterministic_rescore(
                tailored_data,
                parsed_jd,
                original_score,
            )
            job.rescore_data = _dump(rescore_data)
            db.commit()

        # ── Complete ─────────────────────────────────────────────────────────
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.purge_at = datetime.utcnow() + timedelta(days=_DATA_RETENTION_DAYS)
        # Clear raw inputs now that results are stored (data minimisation)
        job.input_resume_text = ""
        job.input_jd_text = ""
        db.commit()
        logger.info("[%s] Completed", job_id)

        # ── Webhook delivery ─────────────────────────────────────────────────
        if job.webhook_url:
            try:
                send_webhook(job, db)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] Webhook delivery failed: %s", job_id, exc)
                # Non-fatal — job is still COMPLETED

    except Exception as exc:
        logger.exception("[%s] Failed at stage %s: %s", job_id, job.status, exc)
        try:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)[:500]
            job.error_stage = job.status.value if job.status else "unknown"
            db.commit()
        except Exception:  # noqa: BLE001
            pass
        raise  # let RQ's retry policy handle it

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Convenience: enqueue a job from the API process
# ---------------------------------------------------------------------------

def enqueue_job(job_id: str, timeout: str = "15m") -> None:
    """Enqueue *job_id* onto the optimization queue."""
    queue.enqueue(run_job, job_id, job_timeout=timeout)
