"""
SQLAlchemy ORM models for multi-tenant B2B architecture.

Tables
------
organizations   – one row per tenant (company / team)
api_keys        – per-tenant API keys (replaces single global API_KEY env var)
jobs            – one row per optimization run; stores stage checkpoints
usage_records   – one row per LLM call; the billing/metering ledger

Usage
-----
    from db import SessionLocal
    from models import Organization, Job, JobStatus

    db = SessionLocal()
    job = db.query(Job).filter_by(id=job_id, org_id=org_id).first()
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Organization
# ---------------------------------------------------------------------------

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    plan = Column(String(50), default="trial")          # trial | starter | growth | enterprise
    monthly_request_quota = Column(Integer, default=100)
    webhook_secret = Column(String(128), nullable=True)  # HMAC key for signed callbacks
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    api_keys = relationship("ApiKey", back_populates="organization", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="organization", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Organization id={self.id!r} name={self.name!r} plan={self.plan!r}>"


# ---------------------------------------------------------------------------
# ApiKey
# ---------------------------------------------------------------------------

class ApiKey(Base):
    """
    Per-tenant API key.  Only the SHA-256 hash is stored — never the plaintext.
    Replaces the single global API_KEY env var.
    """
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    org_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    key_hash = Column(String(64), nullable=False, unique=True)   # SHA-256 hex digest
    label = Column(String(100), default="default")               # e.g. "prod", "staging"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    organization = relationship("Organization", back_populates="api_keys")

    __table_args__ = (Index("ix_api_keys_key_hash", "key_hash"),)

    def __repr__(self) -> str:
        return f"<ApiKey id={self.id!r} org_id={self.org_id!r} label={self.label!r}>"


# ---------------------------------------------------------------------------
# Job status enum
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    QUEUED     = "queued"
    PARSING    = "parsing"      # stages 1 + 2 (resume + JD parse)
    ANALYZING  = "analyzing"    # stage 3 (ATS match + gap analysis)
    TAILORING  = "tailoring"    # stage 4
    RESCORING  = "rescoring"    # stage 5
    COMPLETED  = "completed"
    FAILED     = "failed"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class Job(Base):
    """
    One optimization run.  Each pipeline stage writes its JSON output to the
    corresponding column immediately after completing — this is the checkpoint
    mechanism that makes a mid-pipeline crash resumable rather than a total
    loss of all prior LLM calls.
    """
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    org_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED, nullable=False)

    # Raw inputs (stored until purge_at)
    input_resume_text = Column(Text, nullable=False)
    input_jd_text = Column(Text, nullable=False)
    # Scalar optional params from OptimizeRequest (everything except the two texts)
    request_params = Column(Text, default="{}")   # JSON string; avoids dialect-specific JSON col

    # Per-stage checkpoint columns — NULL until that stage completes
    parsed_resume  = Column(Text, nullable=True)   # JSON string
    parsed_jd      = Column(Text, nullable=True)   # JSON string
    ats_data       = Column(Text, nullable=True)   # JSON string
    tailored_data  = Column(Text, nullable=True)   # JSON string
    rescore_data   = Column(Text, nullable=True)   # JSON string

    # Async delivery
    webhook_url = Column(String(2048), nullable=True)

    # Error context (last 500 chars of exception message + stage name)
    error_message = Column(String(500), nullable=True)
    error_stage   = Column(String(50), nullable=True)

    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    purge_at     = Column(DateTime, nullable=True)  # set on completion; nightly task hard-deletes expired rows

    organization = relationship("Organization", back_populates="jobs")

    __table_args__ = (Index("ix_jobs_org_status", "org_id", "status"),)

    def __repr__(self) -> str:
        return f"<Job id={self.id!r} org_id={self.org_id!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# UsageRecord
# ---------------------------------------------------------------------------

class UsageRecord(Base):
    """
    One row per LLM call.  This is the append-only billing/metering ledger.
    Rows are never updated or deleted (they roll up into invoices).
    """
    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    org_id = Column(UUID(as_uuid=False), ForeignKey("organizations.id"), nullable=False)
    job_id = Column(UUID(as_uuid=False), ForeignKey("jobs.id"), nullable=False)
    stage = Column(String(50), nullable=False)    # "parse_resume" | "analyse_jd" | "ats_analysis" | "tailor" | "rescore"
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, default=0, nullable=False)
    completion_tokens = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_usage_org_created", "org_id", "created_at"),)

    def __repr__(self) -> str:
        return (
            f"<UsageRecord job={self.job_id!r} stage={self.stage!r} "
            f"in={self.prompt_tokens} out={self.completion_tokens}>"
        )
