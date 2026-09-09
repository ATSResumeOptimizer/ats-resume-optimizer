"""
Webhook delivery for async job completion notifications.

When a Job is created with a webhook_url, send_webhook() is called by the
worker after the job reaches COMPLETED or FAILED status.

Signature scheme
----------------
The payload is signed with HMAC-SHA256 using the org's webhook_secret.
The signature is sent in the X-Signature-SHA256 header (same convention as
GitHub and Stripe webhooks), so partners can verify authenticity:

    import hmac, hashlib
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(f"sha256={expected}", request.headers["X-Signature-SHA256"])

If the org has no webhook_secret configured, the header is omitted and
the payload is delivered unsigned (partners should set a secret in production).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging

import httpx  # type: ignore

logger = logging.getLogger(__name__)

_WEBHOOK_TIMEOUT_SECONDS = 10


def send_webhook(job, db) -> None:
    """
    POST the job result to job.webhook_url, signed with the org's webhook_secret.

    Parameters
    ----------
    job : models.Job
        The completed (or failed) job row.  Must have org relationship loaded
        (or be in a live session so lazy-loading works).
    db  : sqlalchemy Session
        Used to load the org's webhook_secret if not already present on the
        relationship.

    Raises httpx.HTTPError on delivery failure — callers should catch and log.
    """
    from models import Organization  # local import avoids circular at module level
    from router_v2 import _job_to_dict  # local import

    org: Organization | None = db.query(Organization).filter_by(id=job.org_id).first()
    if org is None:
        logger.warning("send_webhook: org %s not found, skipping webhook", job.org_id)
        return

    payload_dict = _job_to_dict(job)
    body = json.dumps(payload_dict, ensure_ascii=False, default=str).encode("utf-8")

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Job-Id": job.id,
        "X-Job-Status": job.status.value,
    }

    if org.webhook_secret:
        sig = hmac.new(
            org.webhook_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        headers["X-Signature-SHA256"] = f"sha256={sig}"
    else:
        logger.debug(
            "send_webhook: org %s has no webhook_secret — delivering unsigned", org.id
        )

    response = httpx.post(
        job.webhook_url,
        content=body,
        headers=headers,
        timeout=_WEBHOOK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    logger.info(
        "Webhook delivered for job %s → %s (HTTP %d)",
        job.id,
        job.webhook_url,
        response.status_code,
    )
