"""Server-generated previews for the five supported resume templates."""

from __future__ import annotations

from schemas import ResumeTemplate


_THEMES = {
    ResumeTemplate.CLASSIC: ("#1f2937", "#6b7280", "Classic"),
    ResumeTemplate.MODERN: ("#2563eb", "#dbeafe", "Modern"),
    ResumeTemplate.TECHNICAL: ("#047857", "#d1fae5", "Technical"),
    ResumeTemplate.EXECUTIVE: ("#7c3aed", "#ede9fe", "Executive"),
    ResumeTemplate.CLEAN: ("#111827", "#f3f4f6", "Clean"),
}


def _svg(template: ResumeTemplate) -> str:
    primary, accent, label = _THEMES[template]
    sidebar = (
        f'<rect x="24" y="58" width="112" height="206" rx="4" fill="{accent}"/>'
        if template in (ResumeTemplate.MODERN, ResumeTemplate.TECHNICAL)
        else ""
    )
    heading_x = 154 if sidebar else 32
    lines = "".join(
        f'<rect x="{heading_x}" y="{88 + i * 30}" width="{250 - (i % 3) * 35}" height="8" rx="4" fill="#d1d5db"/>'
        for i in range(5)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 280" role="img" '
        f'aria-label="{label} resume template preview">'
        f'<rect width="320" height="280" rx="8" fill="white" stroke="{primary}" stroke-width="3"/>'
        f'<rect x="24" y="24" width="272" height="18" rx="5" fill="{primary}"/>'
        f"{sidebar}"
        f'<rect x="{heading_x}" y="62" width="110" height="10" rx="5" fill="{primary}"/>'
        f"{lines}"
        f'<rect x="{heading_x}" y="246" width="90" height="8" rx="4" fill="{primary}"/>'
        "</svg>"
    )


def template_previews() -> list[dict[str, str]]:
    return [
        {
            "id": template.value,
            "name": template.value.title(),
            "thumbnail_svg": _svg(template),
            "preview_svg": _svg(template),
        }
        for template in ResumeTemplate
    ]
