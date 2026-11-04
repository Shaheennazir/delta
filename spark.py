"""Shared transport for muse-spark-1.3 unary Responses calls (planner + reviewer).

OpenAI-compatible: POST {BASE}/responses with Bearer $META_API_KEY.
Vision via input_image data URLs; structured JSON via text.format json_schema.
Any transport/model error propagates — callers decide fallback.
"""
from __future__ import annotations

import base64
import logging

import httpx

from sam_client import BASE_URL, FILE_TIMEOUT, api_key

log = logging.getLogger("delta.spark")
SPARK_MODEL = "muse-spark-1.3-contributor"


def jpeg_data_url(jpeg: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"


def extract_text(body: dict) -> str:
    """First output_text of a unary Responses body. Raises KeyError/ValueError."""
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                return part.get("text", "")
    raise ValueError("no output_text in Responses body")


async def post_unary(payload: dict, timeout: float = 120.0) -> dict:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=httpx.Timeout(connect=30.0, read=timeout, write=60.0, pool=30.0)) as c:
        r = await c.post(
            "/responses",
            headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        body = r.json()
    if body.get("status") not in ("completed",):
        raise RuntimeError(f"spark response status: {body.get('status')}")
    return body
