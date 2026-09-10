"""
Backend test suite for ResumeMatch AI.

Run:
    cd ats-resume-optimizer
    pip install pytest pytest-asyncio httpx
    pytest tests/ -v

Tests use httpx.AsyncClient with a patched _llm_call so no real LLM
calls are made during testing.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

SAMPLE_RESUME = """
John Doe
john@example.com | +1-555-0100 | linkedin.com/in/johndoe

SUMMARY
Software engineer with 4 years of Python and data engineering experience.

TECHNICAL SKILLS
Python, SQL, PySpark, Apache Spark, Databricks, Azure, Git, Linux

EXPERIENCE
Senior Data Engineer — Acme Corp (2021–Present)
• Built ETL pipelines using PySpark and Azure Data Factory
• Optimised SQL queries reducing query time by 40%
• Deployed models to Databricks clusters

Data Engineer — Beta Inc (2019–2021)
• Developed Python scripts for data ingestion from REST APIs
• Maintained PostgreSQL databases

EDUCATION
B.Sc. Computer Science — State University, 2019
""" * 2  # repeat to hit min_length


SAMPLE_JD = """
We are looking for a Senior Data Engineer to join our team.

Required Skills:
- Python (5+ years)
- PySpark / Apache Spark
- Azure cloud services
- SQL and database design
- Databricks

Preferred:
- Kafka
- dbt
- Terraform

Responsibilities:
- Design and implement scalable ETL pipelines
- Optimize data warehouse performance
- Collaborate with data scientists

Education: Bachelor's degree in CS or related field
""" * 2


# ---------------------------------------------------------------------------
# Minimal valid LLM response fixtures
# ---------------------------------------------------------------------------

MOCK_PARSED_RESUME = {
    "name": "John Doe",
    "contact": {"email": "john@example.com"},
    "summary": "Software engineer with Python experience",
    "skills": ["Python", "SQL"],
    "technical_skills": ["PySpark", "Azure"],
    "programming_languages": ["Python", "SQL"],
    "tools_and_technologies": ["Databricks", "Git"],
    "experience": [{"company": "Acme Corp", "title": "Senior Data Engineer", "dates": "2021–Present", "bullets": ["Built ETL pipelines"]}],
    "internships": [],
    "projects": [],
    "education": [{"institution": "State University", "degree": "B.Sc.", "field": "Computer Science", "dates": "2019", "details": []}],
    "certifications": [],
    "achievements": [],
    "publications": [],
    "awards": [],
    "other": [],
}

MOCK_PARSED_JD = {
    "job_title": "Senior Data Engineer",
    "required_skills": ["Python", "PySpark", "Azure", "SQL", "Databricks"],
    "preferred_skills": ["Kafka", "dbt", "Terraform"],
    "technical_skills": ["PySpark", "Spark"],
    "programming_languages": ["Python", "SQL"],
    "tools": ["Databricks", "Git"],
    "cloud_technologies": ["Azure"],
    "databases": ["PostgreSQL"],
    "frameworks": [],
    "methodologies": [],
    "years_of_experience": "5+",
    "education_requirements": ["Bachelor's degree"],
    "certifications": [],
    "responsibilities": ["Design ETL pipelines"],
    "domain_knowledge": [],
    "soft_skills": ["Collaboration"],
    "important_keywords": ["Python", "PySpark", "Azure", "ETL", "Databricks"],
    "industry_terminology": ["Data Warehouse"],
}

MOCK_ATS_ANALYSIS = {
    "ats_analysis": {
        "original_score": 72,
        "keyword_match": 80,
        "skills_match": 75,
        "experience_match": 70,
        "job_title_match": 65,
        "responsibilities_match": 68,
        "education_match": 90,
        "formatting_score": 80,
    },
    "requirements": {
        "matched": [{"requirement": "Python", "classification": "REQUIRED", "importance": "HIGH"}],
        "partially_matched": [{"requirement": "Kafka", "classification": "PREFERRED", "importance": "LOW"}],
        "missing": [{"requirement": "Terraform", "classification": "PREFERRED", "importance": "LOW"}],
    },
    "gap_analysis": [
        {
            "requirement": "Python",
            "classification": "REQUIRED",
            "status": "MATCHED",
            "resume_evidence": "Python listed in skills",
            "recommended_action": "No action needed",
        }
    ],
    "keyword_analysis": {
        "matched_keywords": ["Python", "PySpark", "Azure"],
        "added_keywords": [],
        "missing_keywords": ["Terraform"],
    },
}

MOCK_TAILORED_RESUME = {
    "summary": "Senior Data Engineer with 4 years building scalable ETL pipelines using Python, PySpark, and Azure.",
    "skills": [{"category": "Programming", "skills": ["Python", "SQL"]}],
    "experience": [{"company": "Acme Corp", "title": "Senior Data Engineer", "dates": "2021–Present", "bullets": ["Built ETL pipelines using PySpark"]}],
    "projects": [],
    "education": [{"institution": "State University", "degree": "B.Sc.", "field": "Computer Science", "dates": "2019", "details": []}],
    "certifications": [],
}

MOCK_RESCORE = {
    "final_ats_analysis": {
        "improved_score": 79,
        "score_change": 7,
        "remaining_gaps": ["Terraform"],
    },
    "optimization_changes": [
        {"section": "Summary", "change_type": "REWORDED", "description": "Added ETL focus"}
    ],
    "keyword_analysis": {
        "matched_keywords": ["Python", "PySpark", "Azure", "ETL"],
        "added_keywords": ["ETL"],
        "missing_keywords": ["Terraform"],
    },
    "warnings": [],
}


def _make_llm_side_effect():
    """Return extraction and tailoring fixtures for the reduced pipeline."""
    call_count = [0]
    responses = [
        json.dumps(MOCK_PARSED_RESUME),   # stage 1: parse resume
        json.dumps(MOCK_PARSED_JD),       # stage 2: parse JD
        json.dumps(MOCK_TAILORED_RESUME), # stage 3: tailor
    ]

    def _side_effect(messages, temperature=0.2, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return responses[idx] if idx < len(responses) else json.dumps({})

    return _side_effect


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    with patch("optimizer._llm_call", side_effect=_make_llm_side_effect()):
        from main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_root(client: AsyncClient):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data


@pytest.mark.anyio
async def test_optimize_text_success(client: AsyncClient):
    with patch("optimizer._llm_call", side_effect=_make_llm_side_effect()):
        resp = await client.post(
            "/optimize",
            json={
                "candidate_resume": SAMPLE_RESUME,
                "job_description": SAMPLE_JD,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "ats_analysis" in data
    assert "tailored_resume" in data
    assert data["ats_analysis"]["original_score"] == 77
    assert "job_title_match" in data["ats_analysis"]
    assert "responsibilities_match" in data["ats_analysis"]


@pytest.mark.anyio
async def test_optimize_resume_too_short(client: AsyncClient):
    resp = await client.post(
        "/optimize",
        json={
            "candidate_resume": "too short",
            "job_description": SAMPLE_JD,
        },
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_optimize_resume_too_long(client: AsyncClient):
    resp = await client.post(
        "/optimize",
        json={
            "candidate_resume": "x" * 20_001,
            "job_description": SAMPLE_JD,
        },
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_optimize_upload_jd_text(client: AsyncClient):
    resume_bytes = SAMPLE_RESUME.encode()
    with patch("optimizer._llm_call", side_effect=_make_llm_side_effect()):
        resp = await client.post(
            "/optimize/upload",
            files={"resume_file": ("resume.txt", resume_bytes, "text/plain")},
            data={"jd_text": SAMPLE_JD},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ats_analysis"]["original_score"] == 77


@pytest.mark.anyio
async def test_optimize_upload_missing_jd(client: AsyncClient):
    resume_bytes = SAMPLE_RESUME.encode()
    resp = await client.post(
        "/optimize/upload",
        files={"resume_file": ("resume.txt", resume_bytes, "text/plain")},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_optimize_upload_both_jd_sources(client: AsyncClient):
    """Providing both jd_file and jd_text should fail."""
    resume_bytes = SAMPLE_RESUME.encode()
    jd_bytes = SAMPLE_JD.encode()
    resp = await client.post(
        "/optimize/upload",
        files={
            "resume_file": ("resume.txt", resume_bytes, "text/plain"),
            "jd_file": ("jd.txt", jd_bytes, "text/plain"),
        },
        data={"jd_text": SAMPLE_JD},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_optimize_upload_unsupported_file_type(client: AsyncClient):
    resp = await client.post(
        "/optimize/upload",
        files={"resume_file": ("resume.xlsx", b"fake", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"jd_text": SAMPLE_JD},
    )
    assert resp.status_code == 415


@pytest.mark.anyio
async def test_optimize_upload_file_too_large(client: AsyncClient):
    big_file = b"x" * (5 * 1024 * 1024 + 1)
    resp = await client.post(
        "/optimize/upload",
        files={"resume_file": ("resume.txt", big_file, "text/plain")},
        data={"jd_text": SAMPLE_JD},
    )
    assert resp.status_code == 413


@pytest.mark.anyio
async def test_rescore_endpoint(client: AsyncClient):
    """POST /rescore should return a RescoreResponse."""
    from schemas import TailoredResume, KeywordAnalysis, GapItem

    rescore_body = {
        "job_description": SAMPLE_JD,
        "tailored_resume": MOCK_TAILORED_RESUME,
        "original_score": 72,
        "gap_analysis": MOCK_ATS_ANALYSIS["gap_analysis"],
        "keyword_analysis": MOCK_ATS_ANALYSIS["keyword_analysis"],
    }

    resp = await client.post("/rescore", json=rescore_body)
    assert resp.status_code == 200
    data = resp.json()
    assert "final_ats_analysis" in data
    assert data["final_ats_analysis"]["improved_score"] >= 72


@pytest.mark.anyio
async def test_api_key_required_when_configured(client: AsyncClient):
    """When API_KEY is set, requests without the header should be rejected."""
    import os
    with patch.dict(os.environ, {"API_KEY": "supersecret"}):
        resp = await client.post(
            "/optimize",
            json={"candidate_resume": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_api_key_accepted_when_correct(client: AsyncClient):
    import os
    with patch("optimizer._llm_call", side_effect=_make_llm_side_effect()), \
         patch.dict(os.environ, {"API_KEY": "supersecret"}):
        resp = await client.post(
            "/optimize",
            headers={"X-API-Key": "supersecret"},
            json={"candidate_resume": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_ats_scores_include_new_fields(client: AsyncClient):
    """Verify the two new sub-scores are present in the response."""
    with patch("optimizer._llm_call", side_effect=_make_llm_side_effect()):
        resp = await client.post(
            "/optimize",
            json={"candidate_resume": SAMPLE_RESUME, "job_description": SAMPLE_JD},
        )
    data = resp.json()
    scores = data["ats_analysis"]
    assert "job_title_match" in scores
    assert "responsibilities_match" in scores
    assert isinstance(scores["job_title_match"], int)
    assert isinstance(scores["responsibilities_match"], int)


@pytest.mark.anyio
async def test_run_in_threadpool_does_not_block(client: AsyncClient):
    """
    Verify the event loop stays responsive while a slow pipeline is running.

    We simulate a slow _llm_call and fire two concurrent requests.
    Both should complete (not deadlock), proving run_in_threadpool works.
    """
    import asyncio
    import time

    call_times = []

    def _slow_llm(messages, temperature=0.2, **kwargs):
        time.sleep(0.1)  # simulate short delay
        call_times.append(time.time())
        idx = len(call_times) - 1
        responses = [
            json.dumps(MOCK_PARSED_RESUME),
            json.dumps(MOCK_PARSED_JD),
            json.dumps(MOCK_TAILORED_RESUME),
        ]
        return responses[idx % len(responses)]

    with patch("optimizer._llm_call", side_effect=_slow_llm):
        tasks = [
            client.post("/optimize", json={"candidate_resume": SAMPLE_RESUME, "job_description": SAMPLE_JD}),
            client.get("/health"),  # must respond while pipeline is running
        ]
        results = await asyncio.gather(*tasks)

    assert results[1].status_code == 200  # /health responded despite slow pipeline
    assert results[0].status_code == 200


def test_deterministic_skill_aliases_do_not_infer_missing_skills():
    from optimizer import _deterministic_ats_analysis

    result = _deterministic_ats_analysis(
        {
            "name": "Candidate",
            "skills": ["Postgres", "PySpark", "Python"],
            "experience": [],
            "education": [],
        },
        {
            "job_title": "Data Engineer",
            "required_skills": ["PostgreSQL", "pyspark", "AWS EC2"],
            "important_keywords": ["Python"],
            "responsibilities": [],
        },
    )

    assert set(result["skill_match"]["matched_skills"]) == {"PostgreSQL", "pyspark", "Python"}
    assert result["skill_match"]["missing_skills"] == ["AWS EC2"]


def test_job_title_matching_supports_normalized_partial_titles():
    from optimizer import _deterministic_ats_analysis

    result = _deterministic_ats_analysis(
        {"experience": [{"title": "Senior Software Engineer"}], "skills": [], "education": []},
        {"job_title": "Sr. Software Eng", "required_skills": [], "responsibilities": []},
    )

    assert result["ats_analysis"]["job_title_match"] == 100


def test_json_fence_stripping_and_redacted_parse_diagnostic():
    from optimizer import _parse_json_response

    assert _parse_json_response("```json\n{\"ok\": true}\n```") == {"ok": True}
    with pytest.raises(ValueError, match="redacted-email"):
        _parse_json_response("not json john@example.com +1-555-0100")


def test_template_endpoint_returns_five_distinct_renderable_previews():
    from template_previews import template_previews

    previews = template_previews()
    assert len(previews) == 5
    assert {item["id"] for item in previews} == {
        "classic", "modern", "technical", "executive", "clean"
    }
    assert len({item["thumbnail_svg"] for item in previews}) == 5
    assert all(item["thumbnail_svg"].startswith("<svg") for item in previews)


def test_anthropic_joins_all_text_blocks(monkeypatch):
    import sys
    import types
    from optimizer import _call_anthropic

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(text='{"ok":true}')],
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    fake_module = types.SimpleNamespace(
        Anthropic=lambda **kwargs: types.SimpleNamespace(messages=FakeMessages())
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    _call_anthropic(
        [
            {"role": "system", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
            {"role": "user", "content": [{"type": "text", "text": "three"}, {"type": "text", "text": "four"}]},
        ],
        "test-model",
        0.2,
        100,
    )

    assert captured["system"] == "one\ntwo"
    assert captured["messages"] == [{"role": "user", "content": "three\nfour"}]
