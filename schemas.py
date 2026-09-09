"""
Pydantic models for the ATS Resume Optimizer API.

All input/output shapes are defined here so they can be shared
between the router, the optimizer service, and tests.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RequirementStatus(str, Enum):
    MATCHED = "MATCHED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RequirementClassification(str, Enum):
    REQUIRED = "REQUIRED"
    PREFERRED = "PREFERRED"
    RESPONSIBILITY = "RESPONSIBILITY"
    KEYWORD = "KEYWORD"
    QUALIFICATION = "QUALIFICATION"


class OutputFormat(str, Enum):
    JSON = "json"
    # MARKDOWN variant removed — not yet implemented; will be added when
    # a Markdown export path is wired end-to-end.


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    """Primary request payload sent to POST /optimize."""

    candidate_resume: str = Field(
        ...,
        description="Full text of the candidate's resume (plain text or extracted text).",
        min_length=50,
        max_length=20_000,
    )
    job_description: str = Field(
        ...,
        description="Full text of the target job description.",
        min_length=50,
        max_length=20_000,
    )

    # Optional enrichment inputs
    target_job_title: Optional[str] = Field(
        None, description="Target job title the candidate is applying for."
    )
    target_company: Optional[str] = Field(
        None, description="Target company name."
    )
    preferred_resume_length: Optional[str] = Field(
        None,
        description="Preferred resume length: 'one_page', 'two_page', or 'condensed'.",
        pattern=r"^(one_page|two_page|condensed)$",
    )
    candidate_instructions: Optional[str] = Field(
        None,
        description="Free-form instructions from the candidate (e.g., emphasise leadership).",
    )
    output_format: OutputFormat = Field(
        OutputFormat.JSON,
        description="Preferred output format for the tailored resume content.",
    )


# ---------------------------------------------------------------------------
# Sub-components of the response
# ---------------------------------------------------------------------------

class ATSScores(BaseModel):
    # Overall weighted composite
    original_score: int = Field(..., ge=0, le=100)
    # Sub-scores (weights in parentheses — see prompts.py ATS_ANALYSER_SYSTEM)
    keyword_match: int = Field(..., ge=0, le=100)          # 30 %
    skills_match: int = Field(..., ge=0, le=100)           # 20 %
    experience_match: int = Field(..., ge=0, le=100)       # 15 %
    job_title_match: int = Field(0, ge=0, le=100)          # 10 %
    responsibilities_match: int = Field(0, ge=0, le=100)   # 10 %
    education_match: int = Field(..., ge=0, le=100)        #  5 %
    formatting_score: int = Field(..., ge=0, le=100)       #  5 % (structure + ATS readability)


class RequirementItem(BaseModel):
    requirement: str
    classification: RequirementClassification
    importance: str = Field(
        ..., description="HIGH / MEDIUM / LOW based on JD context."
    )


class Requirements(BaseModel):
    matched: list[RequirementItem] = []
    partially_matched: list[RequirementItem] = []
    missing: list[RequirementItem] = []


class GapItem(BaseModel):
    requirement: str
    classification: RequirementClassification
    status: RequirementStatus
    resume_evidence: str = Field(
        ...,
        description="Verbatim or paraphrased evidence from the original resume, or 'None found'.",
    )
    recommended_action: str


class OptimizationChange(BaseModel):
    section: str
    change_type: str = Field(
        ...,
        description="e.g., 'REWORDED', 'REORDERED', 'KEYWORD_ADDED', 'SECTION_ADDED', 'REMOVED'",
    )
    description: str


class ExperienceEntry(BaseModel):
    company: str
    title: str
    dates: str
    bullets: list[str]


class ProjectEntry(BaseModel):
    name: str
    description: str
    technologies: list[str] = []
    bullets: list[str] = []


class EducationEntry(BaseModel):
    institution: str
    degree: str
    field: str
    dates: str
    details: list[str] = []


class CertificationEntry(BaseModel):
    name: str
    issuer: str
    date: Optional[str] = None


class SkillGroup(BaseModel):
    category: str
    skills: list[str]


class TailoredResume(BaseModel):
    summary: str
    skills: list[SkillGroup] = []
    experience: list[ExperienceEntry] = []
    projects: list[ProjectEntry] = []
    education: list[EducationEntry] = []
    certifications: list[CertificationEntry] = []


class FinalATSAnalysis(BaseModel):
    improved_score: int = Field(..., ge=0, le=100)
    score_change: int
    remaining_gaps: list[str] = []


class KeywordAnalysis(BaseModel):
    matched_keywords: list[str] = []
    added_keywords: list[str] = []
    missing_keywords: list[str] = []


# ---------------------------------------------------------------------------
# Top-level response model
# ---------------------------------------------------------------------------

class OptimizeResponse(BaseModel):
    candidate: dict[str, str]

    ats_analysis: ATSScores

    requirements: Requirements

    gap_analysis: list[GapItem] = []

    optimization_changes: list[OptimizationChange] = []

    tailored_resume: TailoredResume

    final_ats_analysis: FinalATSAnalysis

    keyword_analysis: KeywordAnalysis

    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Minimal parsed resume (internal use, returned by the parser utility)
# ---------------------------------------------------------------------------

class ParsedResume(BaseModel):
    name: str = ""
    contact: dict[str, str] = {}
    summary: str = ""
    skills: list[str] = []
    technical_skills: list[str] = []
    programming_languages: list[str] = []
    tools_and_technologies: list[str] = []
    experience: list[dict[str, Any]] = []
    internships: list[dict[str, Any]] = []
    projects: list[dict[str, Any]] = []
    education: list[dict[str, Any]] = []
    certifications: list[dict[str, Any]] = []
    achievements: list[str] = []
    publications: list[str] = []
    awards: list[str] = []
    other: list[str] = []
    raw_text: str = ""


class ParsedJD(BaseModel):
    job_title: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    technical_skills: list[str] = []
    programming_languages: list[str] = []
    tools: list[str] = []
    cloud_technologies: list[str] = []
    databases: list[str] = []
    frameworks: list[str] = []
    methodologies: list[str] = []
    years_of_experience: Optional[str] = None
    education_requirements: list[str] = []
    certifications: list[str] = []
    responsibilities: list[str] = []
    domain_knowledge: list[str] = []
    soft_skills: list[str] = []
    important_keywords: list[str] = []
    industry_terminology: list[str] = []
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Rescore request/response (lightweight — no re-tailoring)
# ---------------------------------------------------------------------------

class RescoreRequest(BaseModel):
    """
    Re-score a tailored resume against the original JD without re-running
    the full 5-stage pipeline.  Called after the user accepts/rejects
    suggestion cards on the frontend.
    """
    job_description: str = Field(
        ..., min_length=50, max_length=20_000,
        description="The original job description text.",
    )
    tailored_resume: TailoredResume = Field(
        ..., description="The (possibly user-modified) tailored resume."
    )
    original_score: int = Field(
        ..., ge=0, le=100,
        description="The original ATS score before tailoring.",
    )
    gap_analysis: list[GapItem] = Field(
        default_factory=list,
        description="Gap analysis from the original optimization run.",
    )
    keyword_analysis: KeywordAnalysis = Field(
        default_factory=KeywordAnalysis,
        description="Keyword analysis from the original optimization run.",
    )


class RescoreResponse(BaseModel):
    final_ats_analysis: FinalATSAnalysis
    optimization_changes: list[OptimizationChange] = []
    keyword_analysis: KeywordAnalysis
    warnings: list[str] = []
