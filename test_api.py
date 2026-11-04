import time

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

PAYLOAD = {
    "title": "Test clip",
    "focus": "Summarize what happens in 2 sentences.",
    "video_url": "https://storage.pitchmind.io/clips/session_9821.mp4",
    "fps": 30.0,
}


def test_api_rejects_bad_payload():
    r = client.post("/api/v1/sessions", json={"title": "x"})
    assert r.status_code == 422
    r = client.post("/api/v1/sessions", json={**PAYLOAD, "evil": "print(1)"})
    assert r.status_code == 422  # sandboxing: no extras


def _make_mp4(frames: int = 8, w: int = 160, h: int = 128) -> bytes:
    import io

    import imageio.v3 as iio
    import numpy as np

    buf = io.BytesIO()
    iio.imwrite(buf, np.zeros((frames, h, w, 3), dtype=np.uint8),
                extension=".mp4", fps=10)
    return buf.getvalue()


def test_upload_rejects_non_video():
    r = client.post("/api/v1/uploads", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_upload_accepts_mp4():
    r = client.post("/api/v1/uploads",
                    files={"file": ("clip.mp4", _make_mp4(), "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["video_url"].startswith("upload://")
    assert body["meta"]["width"] == 160


def test_create_session_real_pipeline_completes():
    """Real transcribe+spark on a tiny silent clip (no mocks)."""
    up = client.post("/api/v1/uploads",
                     files={"file": ("clip.mp4", _make_mp4(), "video/mp4")})
    assert up.status_code == 200
    payload = {**PAYLOAD, "video_url": up.json()["video_url"]}
    r = client.post("/api/v1/sessions", json=payload)
    assert r.status_code == 202
    sid = r.json()["session_id"]
    body = {}
    for _ in range(200):  # real models take minutes
        g = client.get(f"/api/v1/sessions/{sid}")
        assert g.status_code == 200
        body = g.json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(3)
    assert body["status"] == "completed", body
    assert body["summary"], "real spark summary must be non-empty"
    assert isinstance(body["moments"], list)
    assert body["has_audio"] is False  # synthetic clip has no audio track


def test_sse_stream_completes_on_real_path():
    up = client.post("/api/v1/uploads",
                     files={"file": ("clip.mp4", _make_mp4(), "video/mp4")})
    payload = {**PAYLOAD, "video_url": up.json()["video_url"]}
    r = client.post("/api/v1/sessions", json=payload)
    sid = r.json()["session_id"]
    g = client.get(f"/api/v1/sessions/{sid}/stream")
    assert g.status_code == 200
    assert '"done": true' in g.text
