"""
Core ATS Resume Optimizer service.

Pipeline
--------
1. Parse resume (LLM)            → ParsedResume
2. Analyse JD (LLM)              → ParsedJD
3. ATS match + gap analysis (LLM)→ ats_analysis, requirements, gap_analysis, keyword_analysis
4. Tailor resume (LLM)           → TailoredResume
5. Re-score (LLM)                → final_ats_analysis, optimization_changes, warnings
6. Assemble                      → OptimizeResponse

All LLM calls are isolated in _llm_call() so the provider can be swapped
(Gemini, OpenAI, Azure OpenAI, Anthropic, local Ollama, etc.) by changing the
environment variables without touching the pipeline logic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from schemas import (
    ATSScores,
    CertificationEntry,
    EducationEntry,
    ExperienceBulletChange,
    ExperienceEntry,
    FinalATSAnalysis,
    GapItem,
    KeywordAnalysis,
    OptimizationChange,
    OptimizeRequest,
    OptimizeResponse,
    ParsedJD,
    ParsedResume,
    ProjectEntry,
    RequirementClassification,
    RequirementItem,
    RequirementStatus,
    Requirements,
    ResumeComparison,
    SkillMatchAnalysis,
    SkillGroup,
    TailoredResume,
)
from prompts import (
    ats_analyser_messages,
    jd_analyser_messages,
    rescoring_messages,
    resume_parser_messages,
    tailoring_messages,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM backend – configurable via environment variables
# ---------------------------------------------------------------------------
# Required env vars:
#   LLM_PROVIDER  – "gemini" (default) | "openai" | "azure" | "anthropic"
#                   | "openrouter" | "groq" | "ollama"
#   LLM_MODEL     – model name, e.g. "gemini-flash-latest", "gpt-4o",
#                   "claude-3-5-sonnet-20241022", "llama-3.3-70b-versatile",
#                   "llama3.2" (Ollama)
#   GEMINI_API_KEY / OPENAI_API_KEY / AZURE_OPENAI_API_KEY /
#   ANTHROPIC_API_KEY / OPENROUTER_API_KEY
#
# Azure extras:
#   AZURE_OPENAI_ENDPOINT   – https://<resource>.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT – deployment name
#   AZURE_OPENAI_API_VERSION – e.g. "2024-02-01"
#
# OpenRouter extras (all optional):
#   OPENROUTER_BASE_URL     – defaults to https://openrouter.ai/api/v1
#   OPENROUTER_SITE_URL     – sent as HTTP-Referer header (for rankings)
#   OPENROUTER_SITE_NAME    – sent as X-Title header (for rankings)
#
# Groq extras (all optional):
#   GROQ_BASE_URL           – defaults to https://api.groq.com/openai/v1
#
# Ollama extras (all optional):
#   OLLAMA_BASE_URL         – defaults to http://localhost:11434/v1


# A usage callback is an optional callable injected by worker.py so that token
# counts can be recorded in UsageRecord without importing the DB into optimizer.py.
# Signature: (prompt_tokens: int, completion_tokens: int) -> None
_UsageCallback = Any  # callable[[int, int], None]


def _llm_call(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    *,
    usage_callback: _UsageCallback | None = None,
) -> str:
    """
    Send *messages* to the configured LLM and return the raw text response.
    Retries up to 3 times with exponential backoff on empty responses or
    transient rate-limit (429) errors.

    Parameters
    ----------
    usage_callback : callable(prompt_tokens, completion_tokens) | None
        Optional hook called once per successful LLM call with token counts.
        Used by worker.py to write UsageRecord rows without importing the DB
        into this module.  When None (the default), token counts are not recorded.

    Raises RuntimeError if no provider is configured.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-3.6-flash")

    def _dispatch() -> tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens)."""
        if provider == "gemini":
            return _call_gemini(messages, model, temperature, max_tokens)
        elif provider == "openai":
            return _call_openai(messages, model, temperature, max_tokens)
        elif provider == "azure":
            return _call_azure_openai(messages, model, temperature, max_tokens)
        elif provider == "anthropic":
            return _call_anthropic(messages, model, temperature, max_tokens)
        elif provider == "openrouter":
            return _call_openrouter(messages, model, temperature, max_tokens)
        elif provider == "groq":
            return _call_groq(messages, model, temperature, max_tokens)
        elif provider == "ollama":
            return _call_ollama(messages, model, temperature, max_tokens)
        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                "Set LLM_PROVIDER to 'gemini', 'openai', 'azure', 'anthropic', "
                "'openrouter', 'groq', or 'ollama'."
            )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            text, prompt_tokens, completion_tokens = _dispatch()
            if text.strip():
                if usage_callback is not None:
                    try:
                        usage_callback(prompt_tokens, completion_tokens)
                    except Exception:  # noqa: BLE001
                        logger.debug("usage_callback raised; ignoring", exc_info=True)
                return text
            # Empty response — treat like a transient error and retry
            logger.warning("LLM returned empty response (attempt %d/3), retrying…", attempt + 1)
            time.sleep(5 * (attempt + 1))
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate" in exc_str or "quota" in exc_str:
                # Daily/monthly quota exhaustion cannot be resolved by retrying in seconds.
                # Detect known permanent-until-reset limit sources and fail fast.
                if any(kw in exc_str for kw in ("per-day", "per_day", "daily", "monthly", "limit_source")):
                    logger.error("Daily quota exhausted, not retrying: %s", exc)
                    raise
                wait = 10 * (attempt + 1)
                logger.warning("Rate-limited (attempt %d/3), retrying in %ds…", attempt + 1, wait)
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    if last_exc:
        raise last_exc
    return ""


# Return type for all _call_* functions: (text, prompt_tokens, completion_tokens)
# Token counts are 0 when the provider doesn't expose usage (e.g. Ollama).
_LLMResult = tuple[str, int, int]


def _call_gemini(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> _LLMResult:
    """Call Gemini's REST generateContent API with the existing message contract."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    system_parts: list[str] = []
    contents: list[dict[str, object]] = []
    for message in messages:
        role = message["role"]
        content = str(message["content"])
        if role == "system":
            system_parts.append(content)
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": content}],
                }
            )

    payload: dict[str, object] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": {"type": "OBJECT"},
        },
    }
    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}],
        }

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    response = httpx.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    response_data = response.json()
    candidates = response_data.get("candidates") or []
    text = ""
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts)
    usage = response_data.get("usageMetadata") or {}
    return (
        text,
        int(usage.get("promptTokenCount", 0)),
        int(usage.get("candidatesTokenCount", 0)),
    )


def _call_openai(messages: list[dict], model: str, temperature: float, max_tokens: int) -> _LLMResult:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    usage = response.usage
    return (
        response.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _call_azure_openai(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> _LLMResult:
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    usage = response.usage
    return (
        response.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _call_anthropic(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> _LLMResult:
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Install anthropic: pip install anthropic"
        ) from exc

    # Anthropic uses a separate system param
    def _join_content(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for block in value:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(part for part in parts if part)
        return str(value)

    system_parts: list[str] = []
    user_messages: list[dict] = []
    for m in messages:
        content = _join_content(m.get("content", ""))
        if m["role"] == "system":
            system_parts.append(content)
        else:
            user_messages.append({"role": m["role"], "content": content})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="\n\n".join(part for part in system_parts if part).strip(),
        messages=user_messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    text = "".join(b.text for b in response.content if hasattr(b, "text")) if response.content else ""
    usage = response.usage
    return (
        text,
        usage.input_tokens if usage else 0,
        usage.output_tokens if usage else 0,
    )


def _call_openrouter(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> _LLMResult:
    """
    Call any model via OpenRouter's OpenAI-compatible endpoint.

    Required env var : OPENROUTER_API_KEY
    Optional env vars:
      OPENROUTER_BASE_URL  (default: https://openrouter.ai/api/v1)
      OPENROUTER_SITE_URL  (sent as HTTP-Referer, used by OpenRouter rankings)
      OPENROUTER_SITE_NAME (sent as X-Title,     used by OpenRouter rankings)

    OpenRouter supports json_object response_format for most OpenAI models.
    For models that don't support it, we fall back to plain text mode.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set."
        )

    base_url = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    site_url = os.getenv("OPENROUTER_SITE_URL", "")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "ATS Resume Optimizer")

    extra_headers: dict[str, str] = {"X-Title": site_name}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=extra_headers,
    )

    def _attempt(use_json_format: bool) -> _LLMResult:
        kwargs: dict = dict(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    try:
        return _attempt(use_json_format=True)
    except Exception as exc:
        error_str = str(exc).lower()
        if "response_format" in error_str or "json" in error_str or "unsupported" in error_str:
            logger.warning(
                "Model '%s' does not support json_object response_format; "
                "retrying without it. The JSON extraction regex will handle parsing.",
                model,
            )
            return _attempt(use_json_format=False)
        raise


def _call_groq(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> _LLMResult:
    """
    Call any Groq-hosted model via Groq's OpenAI-compatible endpoint.

    Required env var : GROQ_API_KEY
    Optional env vars:
      GROQ_BASE_URL  (default: https://api.groq.com/openai/v1)

    Popular models (as of 2025):
      llama-3.3-70b-versatile  |  llama-3.1-8b-instant
      mixtral-8x7b-32768       |  gemma2-9b-it
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    def _attempt(use_json_format: bool) -> _LLMResult:
        kwargs: dict = dict(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    try:
        return _attempt(use_json_format=True)
    except Exception as exc:
        error_str = str(exc).lower()
        if "response_format" in error_str or "json" in error_str or "unsupported" in error_str:
            logger.warning(
                "Groq model '%s' does not support json_object response_format; "
                "retrying without it.",
                model,
            )
            return _attempt(use_json_format=False)
        raise


def _call_ollama(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> _LLMResult:
    """
    Call a locally-running Ollama instance via its OpenAI-compatible endpoint.

    No API key required (local server).
    Optional env vars:
      OLLAMA_BASE_URL  (default: http://localhost:11434/v1)

    Start Ollama and pull a model before use:
      ollama serve
      ollama pull llama3.2

    Popular models:
      llama3.2  |  llama3.1  |  mistral  |  gemma3  |  phi4  |  qwen2.5
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError("Install openai: pip install openai") from exc

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    client = OpenAI(api_key="ollama", base_url=base_url)

    def _attempt(use_json_format: bool) -> _LLMResult:
        kwargs: dict = dict(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    try:
        return _attempt(use_json_format=True)
    except Exception as exc:
        error_str = str(exc).lower()
        if (
            "response_format" in error_str
            or "json" in error_str
            or "unsupported" in error_str
            or "connection" in error_str
        ):
            if "connection" in error_str:
                raise RuntimeError(
                    f"Cannot connect to Ollama at '{base_url}'. "
                    "Make sure Ollama is running: ollama serve"
                ) from exc
            logger.warning(
                "Ollama model '%s' does not support json_object response_format; "
                "retrying without it.",
                model,
            )
            return _attempt(use_json_format=False)
        raise




# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json_col(col: str | None) -> dict:
    """
    Deserialise a nullable JSON text column to a dict.
    Used by router_v2 and worker to read Job checkpoint columns.
    Returns an empty dict if the column is NULL or unparseable.
    """
    if col is None:
        return {}
    try:
        return json.loads(col)
    except (ValueError, TypeError):
        return {}


def _parse_json_response(raw: str) -> dict:
    """
    Extract a JSON object from *raw*, handling markdown code fences.
    Raises ValueError if no valid JSON is found.
    """
    # Strip one optional outer markdown fence without altering JSON strings.
    clean = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", clean, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        clean = fenced.group(1).strip()
    def _diagnostic_error() -> ValueError:
        diagnostic = re.sub(
            r"(?i)([\w.+-]+)@([\w.-]+\.[a-z]{2,})", "[redacted-email]", raw
        )
        diagnostic = re.sub(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", "[redacted-phone]", diagnostic)
        diagnostic = diagnostic.replace("\r", " ").replace("\n", " ")
        return ValueError(
            "LLM did not return valid JSON "
            f"(response length {len(raw)}; redacted prefix: {diagnostic[:120]!r})"
        )

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise _diagnostic_error()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_parse_resume(raw_text: str, *, usage_callback: _UsageCallback | None = None) -> dict:
    messages = resume_parser_messages(raw_text)
    raw = _llm_call(messages, max_tokens=2500, usage_callback=usage_callback)
    return _parse_json_response(raw)


def _step_analyse_jd(raw_text: str, delay: float = 0.0, *, usage_callback: _UsageCallback | None = None) -> dict:
    if delay:
        time.sleep(delay)
    messages = jd_analyser_messages(raw_text)
    raw = _llm_call(messages, max_tokens=1500, usage_callback=usage_callback)
    return _parse_json_response(raw)


def _step_ats_analysis(parsed_resume: dict, parsed_jd: dict, *, usage_callback: _UsageCallback | None = None) -> dict:
    messages = ats_analyser_messages(parsed_resume, parsed_jd)
    raw = _llm_call(messages, max_tokens=2000, usage_callback=usage_callback)
    return _parse_json_response(raw)


def _step_tailor_resume(
    parsed_resume: dict,
    parsed_jd: dict,
    gap_analysis: list,
    candidate_instructions: str,
    preferred_length: str,
    *,
    usage_callback: _UsageCallback | None = None,
) -> dict:
    messages = tailoring_messages(
        parsed_resume,
        parsed_jd,
        gap_analysis,
        candidate_instructions,
        preferred_length,
    )
    raw = _llm_call(messages, temperature=0.3, max_tokens=2500, usage_callback=usage_callback)
    return _parse_json_response(raw)


def _step_rescore(
    tailored_resume: dict,
    parsed_jd: dict,
    original_score: int,
    gap_analysis: list,
    keyword_analysis: dict,
    *,
    usage_callback: _UsageCallback | None = None,
) -> dict:
    messages = rescoring_messages(
        tailored_resume,
        parsed_jd,
        original_score,
        gap_analysis,
        keyword_analysis,
    )
    raw = _llm_call(messages, max_tokens=1500, usage_callback=usage_callback)
    return _parse_json_response(raw)


# ---------------------------------------------------------------------------
# Deterministic matching
# ---------------------------------------------------------------------------

_SKILL_ALIASES = {
    "pyspark": "pyspark",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "amazon ec2": "aws ec2",
    "aws ec2": "aws ec2",
    "amazon s3": "amazon s3",
    "aws s3": "amazon s3",
}

_TITLE_ALIASES = {
    "swe": "software engineer",
    "sde": "software engineer",
    "sr": "senior",
    "jr": "junior",
    "dev": "developer",
    "eng": "engineer",
    "mgr": "manager",
    "vp": "vice president",
}
_TITLE_STOP_WORDS = {"and", "or", "of", "the", "a", "an", "to", "for"}

_SKILL_VARIANTS = {
    "postgresql": ("postgresql", "postgres"),
    "aws ec2": ("aws ec2", "amazon ec2"),
    "amazon s3": ("amazon s3", "aws s3"),
}


def _normalise_term(value: str) -> str:
    compact = re.sub(r"[^a-z0-9+#.]", " ", value.lower())
    compact = re.sub(r"\s+", " ", compact).strip()
    return _SKILL_ALIASES.get(compact, compact)


def _unique_terms(values: list) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        key = _normalise_term(value)
        if key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _resume_search_text(parsed_resume: dict) -> str:
    return json.dumps(parsed_resume, ensure_ascii=False).lower()


def _jd_skill_terms(parsed_jd: dict) -> list[str]:
    fields = (
        "required_skills", "preferred_skills", "technical_skills",
        "programming_languages", "tools", "cloud_technologies", "databases",
        "frameworks", "methodologies", "certifications", "important_keywords",
        "industry_terminology",
    )
    return _unique_terms([item for field in fields for item in _safe_list(parsed_jd, field)])


def _resume_skill_terms(parsed_resume: dict) -> list[str]:
    fields = (
        "skills", "technical_skills", "programming_languages",
        "tools_and_technologies",
    )
    return _unique_terms([item for field in fields for item in _safe_list(parsed_resume, field)])


def _term_in_text(term: str, text: str) -> bool:
    normalised = _normalise_term(term)
    variants = _SKILL_VARIANTS.get(normalised, (normalised,))
    return bool(normalised) and any(
        re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", text)
        for variant in variants
    )


def _normalise_title(value: str) -> list[str]:
    words = re.findall(r"[a-z0-9+#.]+", value.lower())
    normalised: list[str] = []
    for word in words:
        word = word.strip(".")
        expanded = _TITLE_ALIASES.get(word, word)
        if expanded not in _TITLE_STOP_WORDS:
            normalised.extend(expanded.split())
    return normalised


def _title_match_score(job_title: str, resume_titles: list[str]) -> int:
    target = set(_normalise_title(job_title))
    if not target:
        return 0
    best = 0.0
    for title in resume_titles:
        candidate = set(_normalise_title(title))
        if not candidate:
            continue
        overlap = len(target & candidate) / len(target)
        # A title containing the target terms is a stronger match than a
        # coincidental single-token overlap.
        best = max(best, overlap)
    return round(best * 100)


def _deterministic_ats_analysis(parsed_resume: dict, parsed_jd: dict) -> dict:
    resume_text = _resume_search_text(parsed_resume)
    jd_skills = _jd_skill_terms(parsed_jd)
    resume_skills = _resume_skill_terms(parsed_resume)
    matched = [skill for skill in jd_skills if _term_in_text(skill, resume_text)]
    missing = [skill for skill in jd_skills if skill not in matched]
    keyword_match = round(len(matched) / len(jd_skills) * 100) if jd_skills else 0

    resume_titles = [
        str(item.get("title", ""))
        for item in _safe_list(parsed_resume, "experience")
        if isinstance(item, dict)
    ]
    title_match = _title_match_score(_safe_str(parsed_jd, "job_title"), resume_titles)

    responsibilities = _safe_list(parsed_jd, "responsibilities")
    experience_text = json.dumps(parsed_resume.get("experience", []), ensure_ascii=False).lower()
    responsibility_hits = sum(1 for item in responsibilities if any(
        word in experience_text for word in re.findall(r"[a-z0-9]+", str(item).lower()) if len(word) > 4
    ))
    responsibility_match = round(responsibility_hits / len(responsibilities) * 100) if responsibilities else 0
    formatting_score = 100 if all(_safe_str(parsed_resume, field) or _safe_list(parsed_resume, field) for field in ("name", "skills", "experience", "education")) else 75
    skills_match = keyword_match
    experience_match = min(100, round((responsibility_match + keyword_match) / 2))
    education_match = 100 if _safe_list(parsed_resume, "education") and (_safe_list(parsed_jd, "education_requirements") or not _safe_list(parsed_jd, "education_requirements")) else 0
    original_score = round(
        keyword_match * 0.30 + skills_match * 0.20 + experience_match * 0.15
        + title_match * 0.10 + responsibility_match * 0.10
        + education_match * 0.05 + formatting_score * 0.05
    )

    matched_items = [{"requirement": item, "classification": "REQUIRED", "importance": "HIGH"} for item in matched]
    missing_items = [{"requirement": item, "classification": "REQUIRED", "importance": "HIGH"} for item in missing]
    gaps = [
        {
            "requirement": item,
            "classification": "REQUIRED",
            "status": "MATCHED" if item in matched else "MISSING",
            "resume_evidence": "Found in the extracted resume skills/text." if item in matched else "None found",
            "recommended_action": "Prioritize existing evidence in the tailored resume." if item in matched else "Do not add this skill unless it is genuinely present in the original resume.",
        }
        for item in jd_skills
    ]
    return {
        "ats_analysis": {
            "original_score": original_score,
            "keyword_match": keyword_match,
            "skills_match": skills_match,
            "experience_match": experience_match,
            "job_title_match": title_match,
            "responsibilities_match": responsibility_match,
            "education_match": education_match,
            "formatting_score": formatting_score,
        },
        "requirements": {"matched": matched_items, "partially_matched": [], "missing": missing_items},
        "gap_analysis": gaps,
        "keyword_analysis": {"matched_keywords": matched, "added_keywords": [], "missing_keywords": missing},
        "skill_match": {
            "matched_skills": matched,
            "missing_skills": missing,
            "job_keywords": jd_skills,
            "resume_skills": resume_skills,
            "keyword_match_percentage": keyword_match,
        },
    }


def _deterministic_rescore(tailored_resume: dict, parsed_jd: dict, original_score: int) -> dict:
    parsed_resume = {"skills": [], "technical_skills": [], "raw_text": json.dumps(tailored_resume, ensure_ascii=False)}
    analysis = _deterministic_ats_analysis(parsed_resume, parsed_jd)
    improved = max(original_score, int(analysis["ats_analysis"]["original_score"]))
    return {
        "final_ats_analysis": {"improved_score": improved, "remaining_gaps": analysis["skill_match"]["missing_skills"]},
        "optimization_changes": [],
        "keyword_analysis": analysis["keyword_analysis"],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Assembler helpers – map LLM dicts → Pydantic models safely
# ---------------------------------------------------------------------------

def _safe_str(d: dict, key: str, default: str = "") -> str:
    return str(d.get(key, default) or default)


def _safe_list(d: dict, key: str) -> list:
    val = d.get(key, [])
    return val if isinstance(val, list) else []


def _build_ats_scores(data: dict) -> ATSScores:
    a = data.get("ats_analysis", {})
    return ATSScores(
        original_score=int(a.get("original_score", 0)),
        keyword_match=int(a.get("keyword_match", 0)),
        skills_match=int(a.get("skills_match", 0)),
        experience_match=int(a.get("experience_match", 0)),
        job_title_match=int(a.get("job_title_match", 0)),
        responsibilities_match=int(a.get("responsibilities_match", 0)),
        education_match=int(a.get("education_match", 0)),
        formatting_score=int(a.get("formatting_score", 0)),
    )


def _build_req_item(raw: dict) -> RequirementItem:
    classification = raw.get("classification", "KEYWORD")
    try:
        cls = RequirementClassification(classification.upper())
    except ValueError:
        cls = RequirementClassification.KEYWORD

    importance = str(raw.get("importance", "MEDIUM")).upper()
    if importance not in ("HIGH", "MEDIUM", "LOW"):
        importance = "MEDIUM"

    return RequirementItem(
        requirement=_safe_str(raw, "requirement"),
        classification=cls,
        importance=importance,
    )


def _build_requirements(data: dict) -> Requirements:
    reqs = data.get("requirements", {})
    return Requirements(
        matched=[_build_req_item(r) for r in _safe_list(reqs, "matched")],
        partially_matched=[
            _build_req_item(r) for r in _safe_list(reqs, "partially_matched")
        ],
        missing=[_build_req_item(r) for r in _safe_list(reqs, "missing")],
    )


def _build_gap_analysis(data: dict) -> list[GapItem]:
    items = []
    for raw in _safe_list(data, "gap_analysis"):
        try:
            status = RequirementStatus(
                str(raw.get("status", "MISSING")).upper()
            )
        except ValueError:
            status = RequirementStatus.MISSING

        classification = raw.get("classification", "KEYWORD")
        try:
            cls = RequirementClassification(classification.upper())
        except ValueError:
            cls = RequirementClassification.KEYWORD

        items.append(
            GapItem(
                requirement=_safe_str(raw, "requirement"),
                classification=cls,
                status=status,
                resume_evidence=_safe_str(raw, "resume_evidence", "None found"),
                recommended_action=_safe_str(raw, "recommended_action"),
            )
        )
    return items


def _build_keyword_analysis(data: dict) -> KeywordAnalysis:
    ka = data.get("keyword_analysis", {})
    return KeywordAnalysis(
        matched_keywords=_safe_list(ka, "matched_keywords"),
        added_keywords=_safe_list(ka, "added_keywords"),
        missing_keywords=_safe_list(ka, "missing_keywords"),
    )


def _build_tailored_resume(data: dict) -> TailoredResume:
    skills: list[SkillGroup] = []
    for sg in _safe_list(data, "skills"):
        skills.append(
            SkillGroup(
                category=_safe_str(sg, "category", "Other"),
                skills=_safe_list(sg, "skills"),
            )
        )

    experience: list[ExperienceEntry] = []
    for exp in _safe_list(data, "experience"):
        experience.append(
            ExperienceEntry(
                company=_safe_str(exp, "company"),
                title=_safe_str(exp, "title"),
                dates=_safe_str(exp, "dates"),
                bullets=_safe_list(exp, "bullets"),
            )
        )

    projects: list[ProjectEntry] = []
    for proj in _safe_list(data, "projects"):
        projects.append(
            ProjectEntry(
                name=_safe_str(proj, "name"),
                description=_safe_str(proj, "description"),
                technologies=_safe_list(proj, "technologies"),
                bullets=_safe_list(proj, "bullets"),
            )
        )

    education: list[EducationEntry] = []
    for edu in _safe_list(data, "education"):
        education.append(
            EducationEntry(
                institution=_safe_str(edu, "institution"),
                degree=_safe_str(edu, "degree"),
                field=_safe_str(edu, "field"),
                dates=_safe_str(edu, "dates"),
                details=_safe_list(edu, "details"),
            )
        )

    certifications: list[CertificationEntry] = []
    for cert in _safe_list(data, "certifications"):
        certifications.append(
            CertificationEntry(
                name=_safe_str(cert, "name"),
                issuer=_safe_str(cert, "issuer"),
                date=cert.get("date"),
            )
        )

    return TailoredResume(
        contact={str(k): str(v) for k, v in (data.get("contact") or {}).items() if v},
        summary=_safe_str(data, "summary"),
        skills=skills,
        experience=experience,
        projects=projects,
        education=education,
        certifications=certifications,
    )


def _build_comparison(original: dict, tailored: dict) -> ResumeComparison:
    changes: list[ExperienceBulletChange] = []
    original_experience = _safe_list(original, "experience")
    tailored_experience = _safe_list(tailored, "experience")
    for index, after_entry in enumerate(tailored_experience):
        if not isinstance(after_entry, dict):
            continue
        before_entry = (
            original_experience[index]
            if index < len(original_experience) and isinstance(original_experience[index], dict)
            else {}
        )
        before_bullets = _safe_list(before_entry, "bullets")
        after_bullets = _safe_list(after_entry, "bullets")
        for bullet_index in range(max(len(before_bullets), len(after_bullets))):
            before = str(before_bullets[bullet_index]) if bullet_index < len(before_bullets) else ""
            after = str(after_bullets[bullet_index]) if bullet_index < len(after_bullets) else ""
            if before != after:
                changes.append(
                    ExperienceBulletChange(
                        company=_safe_str(after_entry, "company", _safe_str(before_entry, "company")),
                        title=_safe_str(after_entry, "title", _safe_str(before_entry, "title")),
                        before=before,
                        after=after,
                    )
                )
    return ResumeComparison(
        original_summary=_safe_str(original, "summary"),
        tailored_summary=_safe_str(tailored, "summary"),
        changed_experience_bullets=changes,
    )


def _build_final_analysis(data: dict, original_score: int) -> FinalATSAnalysis:
    fa = data.get("final_ats_analysis", {})
    improved = int(fa.get("improved_score", original_score))
    return FinalATSAnalysis(
        improved_score=improved,
        score_change=improved - original_score,
        remaining_gaps=[str(g) for g in _safe_list(fa, "remaining_gaps")],
    )


def _build_optimization_changes(data: dict) -> list[OptimizationChange]:
    changes = []
    for raw in _safe_list(data, "optimization_changes"):
        changes.append(
            OptimizationChange(
                section=_safe_str(raw, "section", "Unknown"),
                change_type=_safe_str(raw, "change_type", "REWORDED"),
                description=_safe_str(raw, "description"),
            )
        )
    return changes


# ---------------------------------------------------------------------------
# Lightweight re-score entry point (Phase 2b)
# ---------------------------------------------------------------------------

def run_rescore(request: "RescoreRequest") -> "RescoreResponse":
    """
    Re-score a (possibly user-modified) tailored resume against the JD.
    Skips stages 1–4; runs only stage 5.

    Used by the frontend after the user accepts/rejects suggestion cards.
    """
    from schemas import RescoreRequest, RescoreResponse  # local import avoids circularity

    logger.info("Starting lightweight re-score")

    tailored_dict = request.tailored_resume.model_dump()
    parsed_jd = {
        "important_keywords": [
            *request.keyword_analysis.matched_keywords,
            *request.keyword_analysis.missing_keywords,
        ],
        "raw_text": request.job_description,
    }
    rescore_data = _deterministic_rescore(tailored_dict, parsed_jd, request.original_score)
    keyword_analysis_dict = request.keyword_analysis.model_dump()

    merged_ka = {
        **keyword_analysis_dict,
        **(rescore_data.get("keyword_analysis") or {}),
    }

    return RescoreResponse(
        final_ats_analysis=_build_final_analysis(rescore_data, request.original_score),
        optimization_changes=_build_optimization_changes(rescore_data),
        keyword_analysis=_build_keyword_analysis({"keyword_analysis": merged_ka}),
        warnings=[str(w) for w in _safe_list(rescore_data, "warnings")],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_optimization(
    request: OptimizeRequest,
    progress_callback: Callable[[str], None] | None = None,
) -> OptimizeResponse:
    """
    Execute the full five-stage ATS optimization pipeline.

    Parameters
    ----------
    request : OptimizeRequest
        Validated request payload from the API layer.

    Returns
    -------
    OptimizeResponse
        Fully structured JSON-serialisable response.
    """
    logger.info("Starting ATS optimization pipeline")
    if progress_callback:
        progress_callback("reading_resume")

    # -------------------------------------------------------------------
    # Stage 1 + 2 – Parse resume AND analyse JD in parallel (independent)
    # -------------------------------------------------------------------
    logger.info("Stage 1+2: Parsing resume and analysing JD in parallel")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_resume = pool.submit(_step_parse_resume, request.candidate_resume)
        # Stagger by 2 s so both calls don't hit the free-tier endpoint simultaneously
        future_jd = pool.submit(_step_analyse_jd, request.job_description)
        parsed_resume_dict = future_resume.result()
        parsed_jd_dict = future_jd.result()

    parsed_resume_dict["raw_text"] = request.candidate_resume
    parsed_jd_dict["raw_text"] = request.job_description
    if progress_callback:
        progress_callback("analyzing_job_description")

    # Override job title from optional input
    if request.target_job_title:
        parsed_jd_dict["job_title"] = request.target_job_title

    # -------------------------------------------------------------------
    # Stage 3 – ATS match + gap analysis
    # -------------------------------------------------------------------
    logger.info("Stage 3: Running ATS analysis")
    ats_data = _deterministic_ats_analysis(parsed_resume_dict, parsed_jd_dict)
    if progress_callback:
        progress_callback("matching_skills")

    original_score = int(
        (ats_data.get("ats_analysis") or {}).get("original_score", 0)
    )
    gap_analysis_list = _safe_list(ats_data, "gap_analysis")
    keyword_analysis_dict = ats_data.get("keyword_analysis", {})

    # -------------------------------------------------------------------
    # Stage 4 – Tailor resume
    # -------------------------------------------------------------------
    logger.info("Stage 4: Tailoring resume")
    if progress_callback:
        progress_callback("optimizing_resume")
    tailored_dict = _step_tailor_resume(
        parsed_resume_dict,
        parsed_jd_dict,
        gap_analysis_list,
        candidate_instructions=request.candidate_instructions or "",
        preferred_length=request.preferred_resume_length or "",
    )
    if progress_callback:
        progress_callback("generating_final_resume")

    # -------------------------------------------------------------------
    # Stage 5 – Re-score
    # -------------------------------------------------------------------
    logger.info("Stage 5: Re-scoring")
    rescore_data = _deterministic_rescore(
        tailored_dict,
        parsed_jd_dict,
        original_score,
    )
    if progress_callback:
        progress_callback("calculating_ats_score")

    # Merge keyword_analysis from rescore if available
    if rescore_data.get("keyword_analysis"):
        merged_ka = {
            **keyword_analysis_dict,
            **rescore_data["keyword_analysis"],
        }
    else:
        merged_ka = keyword_analysis_dict

    # -------------------------------------------------------------------
    # Assemble response
    # -------------------------------------------------------------------
    logger.info("Assembling response")

    candidate_name = _safe_str(parsed_resume_dict, "name", "Unknown")
    target_role = parsed_jd_dict.get("job_title", "") or request.target_job_title or ""

    response = OptimizeResponse(
        candidate={"name": candidate_name, "target_role": target_role},
        selected_template=request.resume_template,
        ats_analysis=_build_ats_scores(ats_data),
        requirements=_build_requirements(ats_data),
        gap_analysis=_build_gap_analysis(ats_data),
        optimization_changes=_build_optimization_changes(rescore_data),
        tailored_resume=_build_tailored_resume(tailored_dict),
        comparison=_build_comparison(parsed_resume_dict, tailored_dict),
        final_ats_analysis=_build_final_analysis(rescore_data, original_score),
        keyword_analysis=_build_keyword_analysis(
            {"keyword_analysis": merged_ka}
        ),
        skill_match=SkillMatchAnalysis(**(ats_data.get("skill_match") or {})),
        warnings=[str(w) for w in _safe_list(rescore_data, "warnings")],
    )

    logger.info(
        "Pipeline complete. Original score: %d → Improved score: %d",
        original_score,
        response.final_ats_analysis.improved_score,
    )
    return response
