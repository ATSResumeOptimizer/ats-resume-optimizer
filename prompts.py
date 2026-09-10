"""
Prompt templates for every stage of the ATS Resume Optimizer pipeline.

Each function returns a fully-formed system + user message pair ready to be
sent to an LLM.  The templates are intentionally verbose so the model has all
the rules it needs in a single context window.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _system(content: str) -> dict:
    return {"role": "system", "content": content.strip()}


def _user(content: str) -> dict:
    return {"role": "user", "content": content.strip()}


# ---------------------------------------------------------------------------
# Stage 1 – Resume Parser
# ---------------------------------------------------------------------------

RESUME_PARSER_SYSTEM = _system("""
You are a precise resume parser.
Extract every piece of information from the raw resume text and return it as
valid JSON that matches this schema exactly:

{
  "name": "",
  "contact": {"email": "", "phone": "", "linkedin": "", "github": "", "location": ""},
  "summary": "",
  "skills": [],
  "technical_skills": [],
  "programming_languages": [],
  "tools_and_technologies": [],
  "experience": [
    {"company": "", "title": "", "dates": "", "bullets": []}
  ],
  "internships": [
    {"company": "", "title": "", "dates": "", "bullets": []}
  ],
  "projects": [
    {"name": "", "description": "", "technologies": [], "bullets": []}
  ],
  "education": [
    {"institution": "", "degree": "", "field": "", "dates": "", "details": []}
  ],
  "certifications": [
    {"name": "", "issuer": "", "date": ""}
  ],
  "achievements": [],
  "publications": [],
  "awards": [],
  "other": []
}

Rules:
- Extract only what is explicitly present in the resume.
- Do NOT infer or fabricate any information.
- Preserve original dates, job titles, and employer names exactly.
- If a field has no data, use an empty string or empty array.
- Return ONLY valid JSON — no markdown fences, no prose.
""")


def resume_parser_messages(raw_resume: str) -> list[dict]:
    return [
        RESUME_PARSER_SYSTEM,
        _user(f"Parse this resume:\n\n{raw_resume}"),
    ]


# ---------------------------------------------------------------------------
# Stage 2 – JD Analyser
# ---------------------------------------------------------------------------

JD_ANALYSER_SYSTEM = _system("""
You are a precise job description analyser.
Extract every requirement from the job description and return valid JSON matching:

{
  "job_title": "",
  "required_skills": [],
  "preferred_skills": [],
  "technical_skills": [],
  "programming_languages": [],
  "tools": [],
  "cloud_technologies": [],
  "databases": [],
  "frameworks": [],
  "methodologies": [],
  "years_of_experience": "",
  "education_requirements": [],
  "certifications": [],
  "responsibilities": [],
  "domain_knowledge": [],
  "soft_skills": [],
  "important_keywords": [],
  "industry_terminology": []
}

For each item you extract, classify it as one of:
REQUIRED | PREFERRED | RESPONSIBILITY | KEYWORD | QUALIFICATION

Include only items that are explicitly stated or clearly implied by the JD text.
Return ONLY valid JSON — no markdown fences, no prose.
""")


def jd_analyser_messages(raw_jd: str) -> list[dict]:
    return [
        JD_ANALYSER_SYSTEM,
        _user(f"Analyse this job description:\n\n{raw_jd}"),
    ]


# ---------------------------------------------------------------------------
# Stage 3 – ATS Match Analysis + Gap Analysis
# ---------------------------------------------------------------------------

ATS_ANALYSER_SYSTEM = _system("""
You are a strict ATS compatibility analyst.
Given a structured resume JSON and a structured JD JSON, perform a full match
analysis and return valid JSON with this exact schema:

{
  "ats_analysis": {
    "original_score": 0,
    "keyword_match": 0,
    "skills_match": 0,
    "experience_match": 0,
    "job_title_match": 0,
    "responsibilities_match": 0,
    "education_match": 0,
    "formatting_score": 0
  },
  "requirements": {
    "matched": [
      {"requirement": "", "classification": "REQUIRED", "importance": "HIGH"}
    ],
    "partially_matched": [],
    "missing": []
  },
  "gap_analysis": [
    {
      "requirement": "",
      "classification": "REQUIRED",
      "status": "MATCHED",
      "resume_evidence": "",
      "recommended_action": ""
    }
  ],
  "keyword_analysis": {
    "matched_keywords": [],
    "added_keywords": [],
    "missing_keywords": []
  }
}

Sub-score definitions and weights (each sub-score is 0–100; composite is the weighted sum):
- keyword_match           30 % — how many important JD keywords appear in the resume
- skills_match            20 % — how many required + preferred skills are present
- experience_match        15 % — alignment of job duties with past experience
- job_title_match         10 % — similarity of resume titles to the target role
- responsibilities_match  10 % — alignment of listed responsibilities with JD duties
- education_match          5 % — degree + certifications vs. JD requirements
- formatting_score         5 % — ATS-readable structure (standard headings, no tables/graphics)

Composite formula:
  original_score = round(
    keyword_match * 0.30 +
    skills_match  * 0.20 +
    experience_match * 0.15 +
    job_title_match  * 0.10 +
    responsibilities_match * 0.10 +
    education_match  * 0.05 +
    formatting_score * 0.05
  )

Rules:
- Every sub-score must be derived from explicit evidence in the resume and JD.
- Do NOT artificially inflate any score.
- For every missing requirement, check whether transferable evidence exists.
- Use statuses: MATCHED | PARTIAL | MISSING | NOT_APPLICABLE.
- Return ONLY valid JSON — no markdown fences, no prose.
""")


def ats_analyser_messages(parsed_resume: dict, parsed_jd: dict) -> list[dict]:
    import json
    return [
        ATS_ANALYSER_SYSTEM,
        _user(
            "PARSED RESUME:\n"
            + json.dumps(parsed_resume, separators=(",", ":"))
            + "\n\nPARSED JD:\n"
            + json.dumps(parsed_jd, separators=(",", ":"))
        ),
    ]


# ---------------------------------------------------------------------------
# Stage 4 – Resume Tailoring
# ---------------------------------------------------------------------------

TAILORING_SYSTEM = _system("""
You are an expert ATS resume writer. You ONLY work with facts already present
in the candidate's original resume. You never fabricate information.

Given the original parsed resume, the parsed JD, and the gap analysis, produce
a tailored resume as valid JSON:

{
  "contact": {"email": "", "phone": "", "linkedin": "", "github": "", "location": ""},
  "summary": "",
  "skills": [
    {"category": "Programming", "skills": ["Python", "SQL"]}
  ],
  "experience": [
    {
      "company": "",
      "title": "",
      "dates": "",
      "bullets": []
    }
  ],
  "projects": [
    {
      "name": "",
      "description": "",
      "technologies": [],
      "bullets": []
    }
  ],
  "education": [
    {
      "institution": "",
      "degree": "",
      "field": "",
      "dates": "",
      "details": []
    }
  ],
  "certifications": [
    {"name": "", "issuer": "", "date": ""}
  ]
}

Tailoring rules you MUST follow:
1. Never fabricate work experience, projects, certifications, or achievements.
2. Never invent metrics or numbers not present in the original resume.
3. Never change employment dates, employer names, or job titles.
4. Never change education institution, degree, or graduation year.
5. Never claim a technology unless the original resume explicitly mentions it.
6. You MAY reorder skills so JD-relevant skills appear first.
7. You MAY rewrite bullet points for clarity without changing the facts.
8. You MAY rewrite the summary to target the JD.
9. You MAY use JD terminology only when it accurately describes existing experience.
10. You MAY remove content that is irrelevant to the JD.
11. You MUST write naturally — avoid AI-sounding language.
12. Bullet point format: ACTION_VERB + TECHNOLOGY/PROCESS + TASK + RESULT (omit result if not in original).

Return ONLY valid JSON — no markdown fences, no prose.
""")


def tailoring_messages(
    parsed_resume: dict,
    parsed_jd: dict,
    gap_analysis: list,
    candidate_instructions: str = "",
    preferred_length: str = "",
) -> list[dict]:
    import json

    extra = ""
    if candidate_instructions:
        extra += f"\nCandidate instructions: {candidate_instructions}"
    if preferred_length:
        extra += f"\nPreferred resume length: {preferred_length}"

    return [
        TAILORING_SYSTEM,
        _user(
            "ORIGINAL PARSED RESUME:\n"
            + json.dumps(parsed_resume, separators=(",", ":"))
            + "\n\nPARSED JD:\n"
            + json.dumps(parsed_jd, separators=(",", ":"))
            + "\n\nGAP ANALYSIS:\n"
            + json.dumps(gap_analysis, separators=(",", ":"))
            + extra
        ),
    ]


# ---------------------------------------------------------------------------
# Stage 5 – Re-scoring
# ---------------------------------------------------------------------------

RESCORING_SYSTEM = _system("""
You are a strict ATS re-scorer. Given the tailored resume and the original JD,
re-run the ATS analysis and produce valid JSON:

{
  "final_ats_analysis": {
    "improved_score": 0,
    "score_change": 0,
    "remaining_gaps": []
  },
  "optimization_changes": [
    {
      "section": "",
      "change_type": "REWORDED",
      "description": ""
    }
  ],
  "keyword_analysis": {
    "matched_keywords": [],
    "added_keywords": [],
    "missing_keywords": []
  },
  "warnings": []
}

change_type values: REWORDED | REORDERED | KEYWORD_ADDED | SECTION_ADDED | REMOVED

Rules:
- Score honestly based on the tailored resume content.
- If the score is lower than the original, explain why in warnings[].
- Never claim the resume is guaranteed to pass any employer ATS.
- Return ONLY valid JSON — no markdown fences, no prose.
""")


def rescoring_messages(
    tailored_resume: dict,
    parsed_jd: dict,
    original_score: int,
    original_gap_analysis: list,
    original_keyword_analysis: dict,
) -> list[dict]:
    import json

    return [
        RESCORING_SYSTEM,
        _user(
            "TAILORED RESUME:\n"
            + json.dumps(tailored_resume, separators=(",", ":"))
            + "\n\nPARSED JD:\n"
            + json.dumps(parsed_jd, separators=(",", ":"))
            + f"\n\nORIGINAL ATS SCORE: {original_score}"
            + "\n\nORIGINAL GAP ANALYSIS:\n"
            + json.dumps(original_gap_analysis, separators=(",", ":"))
            + "\n\nORIGINAL KEYWORD ANALYSIS:\n"
            + json.dumps(original_keyword_analysis, separators=(",", ":"))
        ),
    ]
