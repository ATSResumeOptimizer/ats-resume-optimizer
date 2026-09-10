"""Anonymous-session persistence for the Resume Maker workflow."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

from db import SessionLocal
from models import AnalysisRecord, GeneratedResumeRecord, JobDescriptionRecord, ResumeRecord
from schemas import OptimizeResponse

SESSION_COOKIE = "ats_session_id"


def session_id_from_cookie(value: Optional[str]) -> str | None:
    return value.strip() if value and value.strip() else None


def new_session_id() -> str:
    return str(uuid.uuid4())


def input_hash(resume_text: str, jd_text: str) -> str:
    return hashlib.sha256(f"{resume_text}\n\0{jd_text}".encode("utf-8")).hexdigest()


def get_cached_response(session_id: str | None, resume_text: str, jd_text: str) -> OptimizeResponse | None:
    db = SessionLocal()
    try:
        # Identical inputs are safe to reuse across anonymous sessions. This
        # check happens in the router before any LLM stage is submitted.
        row = db.query(AnalysisRecord).filter_by(
            input_hash=input_hash(resume_text, jd_text),
        ).first()
        if not row or not row.response_json:
            return None
        return OptimizeResponse.model_validate(json.loads(row.response_json))
    except Exception:
        return None
    finally:
        db.close()


def save_analysis(
    session_id: str,
    resume_text: str,
    jd_text: str,
    response: OptimizeResponse,
) -> None:
    db = SessionLocal()
    try:
        digest = input_hash(resume_text, jd_text)
        existing = db.query(AnalysisRecord).filter_by(input_hash=digest).first()
        if existing:
            return

        resume = ResumeRecord(
            session_id=session_id,
            raw_text=resume_text,
            parsed_structure=json.dumps(response.tailored_resume.model_dump()),
        )
        jd = JobDescriptionRecord(
            session_id=session_id,
            raw_text=jd_text,
            parsed_keywords=json.dumps(response.skill_match.model_dump()),
        )
        db.add_all([resume, jd])
        db.flush()

        analysis = AnalysisRecord(
            session_id=session_id,
            resume_id=resume.id,
            job_description_id=jd.id,
            input_hash=digest,
            matched_skills=json.dumps(response.skill_match.matched_skills),
            missing_skills=json.dumps(response.skill_match.missing_skills),
            keyword_match_percentage=response.skill_match.keyword_match_percentage,
            ats_score_breakdown=json.dumps(response.ats_analysis.model_dump()),
            response_json=json.dumps(response.model_dump(mode="json")),
        )
        db.add(analysis)
        db.flush()
        db.add(GeneratedResumeRecord(
            analysis_id=analysis.id,
            template_id=response.selected_template.value,
            tailored_content=json.dumps(response.tailored_resume.model_dump()),
        ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def get_history(session_id: str | None) -> list[OptimizeResponse]:
    if not session_id:
        return []
    db = SessionLocal()
    try:
        rows = (
            db.query(AnalysisRecord)
            .filter_by(session_id=session_id)
            .order_by(AnalysisRecord.created_at.desc())
            .limit(20)
            .all()
        )
        results: list[OptimizeResponse] = []
        for row in rows:
            try:
                results.append(OptimizeResponse.model_validate(json.loads(row.response_json)))
            except Exception:
                continue
        return results
    finally:
        db.close()
