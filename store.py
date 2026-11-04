"""SQLite persistence for the Video Intelligence Engine (prod store).

Single-file DB at ./data/delta.db (WAL). Replaces the old in-memory
_store/_progress dicts so sessions, frames, entities, events, OCR, audio
events and QA runs survive restarts. No mocks: rows are only written from
real perception outputs or deterministic computation.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "delta.db")

_lock = threading.Lock()
os.makedirs(DATA_DIR, exist_ok=True)


def _connect() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def init_db() -> None:
    with _lock, _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(
          session_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'processing',
          title TEXT DEFAULT '', focus TEXT DEFAULT '', video_url TEXT DEFAULT '',
          answer TEXT DEFAULT '', summary TEXT DEFAULT '', themes_json TEXT DEFAULT '[]',
          transcript TEXT DEFAULT '', turns_json TEXT DEFAULT '[]',
          moments_json TEXT DEFAULT '[]', on_screen_text TEXT DEFAULT '',
          has_audio INTEGER DEFAULT 0,
          video_json TEXT DEFAULT '{}',
          error TEXT DEFAULT '', updated_at REAL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS progress(
          session_id TEXT PRIMARY KEY, stage TEXT, detail TEXT, error TEXT, updated_at REAL);
        CREATE TABLE IF NOT EXISTS frames(
          session_id TEXT, frame_index INTEGER, t REAL,
          path TEXT, motion REAL DEFAULT 0, keyframe INTEGER DEFAULT 0,
          PRIMARY KEY(session_id, frame_index));
        CREATE TABLE IF NOT EXISTS entities(
          session_id TEXT, entity_id TEXT, global_id TEXT DEFAULT '',
          label TEXT DEFAULT '', kind TEXT DEFAULT 'object', confidence REAL DEFAULT 0,
          PRIMARY KEY(session_id, entity_id));
        CREATE TABLE IF NOT EXISTS entity_gallery(
          global_id TEXT PRIMARY KEY, label TEXT DEFAULT '', kind TEXT DEFAULT 'object',
          descriptor_json TEXT DEFAULT '[]', n_obs INTEGER DEFAULT 0,
          first_seen REAL DEFAULT 0, last_seen REAL DEFAULT 0, confidence REAL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS observations(
          session_id TEXT, entity_id TEXT, t REAL,
          frame_index INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
          track_conf REAL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_obs ON observations(session_id, entity_id, t);
        CREATE TABLE IF NOT EXISTS events(
          session_id TEXT, event_id TEXT, action TEXT DEFAULT '',
          entity_ids_json TEXT DEFAULT '[]',
          start_t REAL DEFAULT 0, end_t REAL DEFAULT 0,
          evidence_json TEXT DEFAULT '[]', confidence REAL DEFAULT 0,
          state_before_json TEXT DEFAULT '{}', state_after_json TEXT DEFAULT '{}',
          detail TEXT DEFAULT '',
          PRIMARY KEY(session_id, event_id));
        CREATE TABLE IF NOT EXISTS ocr_texts(
          session_id TEXT, t REAL, frame_index INTEGER,
          text TEXT, confidence REAL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_ocr ON ocr_texts(session_id, t);
        CREATE TABLE IF NOT EXISTS audio_events(
          session_id TEXT, event_id TEXT, kind TEXT,
          start_t REAL, end_t REAL, confidence REAL DEFAULT 0,
          detail TEXT DEFAULT '',
          PRIMARY KEY(session_id, event_id));
        CREATE TABLE IF NOT EXISTS qa_runs(
          session_id TEXT, qa_id TEXT, question TEXT, answer TEXT,
          evidence_json TEXT DEFAULT '[]', confidence REAL DEFAULT 0,
          raw_confidence REAL DEFAULT 0,
          verification TEXT DEFAULT 'unknown', created_at REAL,
          PRIMARY KEY(session_id, qa_id));
        CREATE TABLE IF NOT EXISTS trace(
          session_id TEXT, seq INTEGER, tool TEXT, event TEXT DEFAULT '',
          detail TEXT DEFAULT '', created_at REAL,
          PRIMARY KEY(session_id, seq));
        """)
        # Backfill: events table predates detail column on old DBs.
        try:
            c.execute("ALTER TABLE events ADD COLUMN detail TEXT DEFAULT ''")
        except Exception:
            pass
        # Backfill: sessions predates the direct answer column on old DBs.
        try:
            c.execute("ALTER TABLE sessions ADD COLUMN answer TEXT DEFAULT ''")
        except Exception:
            pass
        # Backfill: capability columns added for evidence/alignment/calibration.
        for col, ddl in (
            ("answer_evidence_json", "TEXT DEFAULT '[]'"),
            ("answer_confidence", "REAL DEFAULT 0"),
            ("answer_verification", "TEXT DEFAULT 'unknown'"),
            ("answer_calibration", "TEXT DEFAULT 'uncalibrated'"),
            ("av_offset_s", "REAL DEFAULT 0"),
            ("av_sync_confidence", "REAL DEFAULT 0"),
        ):
            try:
                c.execute(f"ALTER TABLE sessions ADD COLUMN {col} {ddl}")
            except Exception:
                pass
        try:
            c.execute("ALTER TABLE entities ADD COLUMN global_id TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            c.execute("ALTER TABLE qa_runs ADD COLUMN raw_confidence REAL DEFAULT 0")
        except Exception:
            pass
        c.execute("""CREATE TABLE IF NOT EXISTS calibration_points(
          x REAL PRIMARY KEY, y REAL NOT NULL, n INTEGER DEFAULT 1, updated_at REAL DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS outcomes(
          session_id TEXT, item_id TEXT, raw_confidence REAL, correct INTEGER,
          created_at REAL, PRIMARY KEY(session_id, item_id))""")


def _now() -> float:
    return time.time()


# ---- sessions / progress (compat with old orchestrator/app API) ----

def upsert_session(session_id: str, **fields) -> None:
    init_db()
    fields["session_id"] = session_id
    fields["updated_at"] = _now()
    cols = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "session_id")
    with _lock, _connect() as c:
        c.execute(f"INSERT INTO sessions({cols}) VALUES({placeholders}) "
                  f"ON CONFLICT(session_id) DO UPDATE SET {updates}",
                  tuple(fields.values()))


def get_session(session_id: str) -> dict | None:
    init_db()
    with _lock, _connect() as c:
        r = c.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(r) if r else None


def list_sessions(limit: int = 50) -> list[dict]:
    """Newest-first session index for the chat sidebar. Survives restarts."""
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT session_id,title,status,focus,video_url,updated_at FROM sessions "
            "WHERE session_id NOT GLOB 'bench_*' "
            "ORDER BY updated_at DESC LIMIT ?", (limit,))]


def save_progress(session_id: str, stage: str, detail: str = "", error: str | None = None) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("INSERT INTO progress(session_id,stage,detail,error,updated_at) VALUES(?,?,?,?,?) "
                  "ON CONFLICT(session_id) DO UPDATE SET stage=excluded.stage, detail=excluded.detail, "
                  "error=excluded.error, updated_at=excluded.updated_at",
                  (session_id, stage, detail, error, _now()))


def get_progress(session_id: str) -> dict | None:
    init_db()
    with _lock, _connect() as c:
        r = c.execute("SELECT * FROM progress WHERE session_id=?", (session_id,)).fetchone()
        return dict(r) if r else None


# ---- frame-indexed temporal database ----

def save_frames(session_id: str, rows: list[dict]) -> None:
    init_db()
    with _lock, _connect() as c:
        c.executemany("INSERT OR REPLACE INTO frames(session_id,frame_index,t,path,motion,keyframe) "
                      "VALUES(?,?,?,?,?,?)",
                      [(session_id, r["frame_index"], r["t"], r.get("path", ""),
                        r.get("motion", 0.0), int(bool(r.get("keyframe", False)))) for r in rows])


def get_frames(session_id: str) -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM frames WHERE session_id=? ORDER BY frame_index", (session_id,))]


def save_entities(session_id: str, entities: list[dict], observations: list[dict]) -> None:
    init_db()
    with _lock, _connect() as c:
        c.executemany("INSERT OR REPLACE INTO entities(session_id,entity_id,global_id,label,kind,confidence) "
                      "VALUES(?,?,?,?,?,?)",
                      [(session_id, e["entity_id"], e.get("global_id") or e["entity_id"],
                        e.get("label", ""), e.get("kind", "object"),
                        e.get("confidence", 0.0)) for e in entities])
        c.execute("DELETE FROM observations WHERE session_id=?", (session_id,))
        c.executemany("INSERT INTO observations(session_id,entity_id,t,frame_index,x1,y1,x2,y2,track_conf) "
                      "VALUES(?,?,?,?,?,?,?,?,?)",
                      [(session_id, o["entity_id"], o["t"], o.get("frame_index", 0),
                        o["bbox"][0], o["bbox"][1], o["bbox"][2], o["bbox"][3],
                        o.get("track_conf", 0.0)) for o in observations])


def get_entities(session_id: str) -> tuple[list[dict], list[dict]]:
    init_db()
    with _lock, _connect() as c:
        ents = [dict(r) for r in c.execute(
            "SELECT * FROM entities WHERE session_id=?", (session_id,))]
        obs = []
        for r in c.execute("SELECT * FROM observations WHERE session_id=? ORDER BY t",
                           (session_id,)):
            d = dict(r)
            d["bbox"] = [d.get("x1", 0), d.get("y1", 0), d.get("x2", 0), d.get("y2", 0)]
            obs.append(d)
        return ents, obs


def get_gallery(kind: str | None = None) -> list[dict]:
    """Cross-session appearance gallery. Descriptors are unit vectors."""
    init_db()
    with _lock, _connect() as c:
        if kind:
            rows = c.execute("SELECT * FROM entity_gallery WHERE kind=?", (kind,))
        else:
            rows = c.execute("SELECT * FROM entity_gallery")
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["descriptor"] = json.loads(d.pop("descriptor_json") or "[]")
            except Exception:
                d["descriptor"] = []
            out.append(d)
        return out


def upsert_gallery(global_id: str, label: str, kind: str, descriptor: list[float],
                   confidence: float) -> dict:
    """Running-mean appearance descriptor; count-weighted so one frame cannot dominate."""
    init_db()
    with _lock, _connect() as c:
        row = c.execute("SELECT * FROM entity_gallery WHERE global_id=?", (global_id,)).fetchone()
        now = _now()
        if not row:
            c.execute("INSERT INTO entity_gallery(global_id,label,kind,descriptor_json,n_obs,"
                      "first_seen,last_seen,confidence) VALUES(?,?,?,?,?,?,?,?)",
                      (global_id, label, kind, json.dumps(descriptor), 1, now, now, confidence))
            return {"global_id": global_id, "label": label, "kind": kind,
                    "descriptor": descriptor, "n_obs": 1, "confidence": confidence}
        old = json.loads(row["descriptor_json"] or "[]")
        n = int(row["n_obs"] or 0)
        if old and descriptor and len(old) == len(descriptor):
            merged = [(o * n + d) / (n + 1) for o, d in zip(old, descriptor)]
            norm = sum(v * v for v in merged) ** 0.5 or 1.0
            merged = [v / norm for v in merged]
        else:
            merged = descriptor or old
        c.execute("UPDATE entity_gallery SET label=?, kind=?, descriptor_json=?, n_obs=?, "
                  "last_seen=?, confidence=? WHERE global_id=?",
                  (label or row["label"], kind or row["kind"], json.dumps(merged), n + 1,
                   now, max(float(row["confidence"] or 0), confidence), global_id))
        return {"global_id": global_id, "label": label or row["label"],
                "kind": kind or row["kind"], "descriptor": merged, "n_obs": n + 1,
                "confidence": max(float(row["confidence"] or 0), confidence)}


def save_calibration(points: list[dict]) -> None:
    """Replace the isotonic calibration curve: [{x, y, n}]."""
    init_db()
    with _lock, _connect() as c:
        c.execute("DELETE FROM calibration_points")
        c.executemany("INSERT OR REPLACE INTO calibration_points(x,y,n,updated_at) "
                      "VALUES(?,?,?,?)",
                      [(float(p["x"]), float(p["y"]), int(p.get("n", 1)), _now())
                       for p in points])


def get_calibration() -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT x,y,n FROM calibration_points ORDER BY x")]


def save_outcome(session_id: str, item_id: str, raw_confidence: float, correct: bool) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("INSERT OR REPLACE INTO outcomes(session_id,item_id,raw_confidence,correct,"
                  "created_at) VALUES(?,?,?,?,?)",
                  (session_id, item_id, float(raw_confidence), int(bool(correct)), _now()))


def get_outcomes() -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT session_id,item_id,raw_confidence,correct FROM outcomes ORDER BY created_at")]


def save_events(session_id: str, events: list[dict]) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("DELETE FROM events WHERE session_id=?", (session_id,))
        c.executemany(
            "INSERT INTO events(session_id,event_id,action,entity_ids_json,start_t,end_t,"
            "evidence_json,confidence,state_before_json,state_after_json,detail) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [(session_id, e["event_id"], e.get("action", ""),
              json.dumps(e.get("entity_ids", [])), float(e.get("start_t", 0)),
              float(e.get("end_t", 0)), json.dumps(e.get("evidence", [])),
              float(e.get("confidence", 0.0)), json.dumps(e.get("state_before", {})),
              json.dumps(e.get("state_after", {})), e.get("detail", "")) for e in events])


def get_events(session_id: str) -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        rows = c.execute("SELECT * FROM events WHERE session_id=? ORDER BY start_t",
                         (session_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k, nk in (("entity_ids_json", "entity_ids"), ("evidence_json", "evidence"),
                      ("state_before_json", "state_before"), ("state_after_json", "state_after")):
            try:
                d[nk] = json.loads(d.pop(k) or ("{}" if "state" in k else "[]"))
            except Exception:
                d[nk] = {} if "state" in k else []
        out.append(d)
    return out


def save_ocr(session_id: str, rows: list[dict]) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("DELETE FROM ocr_texts WHERE session_id=?", (session_id,))
        c.executemany("INSERT INTO ocr_texts(session_id,t,frame_index,text,confidence) VALUES(?,?,?,?,?)",
                      [(session_id, r["t"], r.get("frame_index", 0),
                        r.get("text", ""), r.get("confidence", 0.0)) for r in rows])


def get_ocr(session_id: str) -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM ocr_texts WHERE session_id=? ORDER BY t", (session_id,))]


def save_audio_events(session_id: str, rows: list[dict]) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("DELETE FROM audio_events WHERE session_id=?", (session_id,))
        c.executemany("INSERT INTO audio_events(session_id,event_id,kind,start_t,end_t,confidence,detail) "
                      "VALUES(?,?,?,?,?,?,?)",
                      [(session_id, r["event_id"], r.get("kind", ""), r.get("start_t", 0),
                        r.get("end_t", 0), r.get("confidence", 0.0), r.get("detail", "")) for r in rows])


def get_audio_events(session_id: str) -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM audio_events WHERE session_id=? ORDER BY start_t", (session_id,))]


def save_qa(session_id: str, qa_id: str, question: str, answer: str,
            evidence: list, confidence: float, verification: str,
            raw_confidence: float = 0.0) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("INSERT OR REPLACE INTO qa_runs(session_id,qa_id,question,answer,evidence_json,"
                  "confidence,raw_confidence,verification,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                  (session_id, qa_id, question, answer, json.dumps(evidence),
                   confidence, raw_confidence or confidence, verification, _now()))


def get_qa(session_id: str) -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        rows = c.execute("SELECT * FROM qa_runs WHERE session_id=? ORDER BY created_at",
                         (session_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["evidence"] = json.loads(d.pop("evidence_json") or "[]")
        except Exception:
            d["evidence"] = []
        out.append(d)
    return out


def append_trace(session_id: str, tool: str, event: str, detail: str = "") -> dict:
    """Append one agent-tool trace row. event: sent|waiting|received|failed."""
    init_db()
    with _lock, _connect() as c:
        row = c.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM trace WHERE session_id=?",
                        (session_id,)).fetchone()
        seq = int(row["m"] or 0) + 1
        now = _now()
        c.execute("INSERT INTO trace(session_id,seq,tool,event,detail,created_at) VALUES(?,?,?,?,?,?)",
                  (session_id, seq, tool, event, detail[:500], now))
    return {"session_id": session_id, "seq": seq, "tool": tool,
            "event": event, "detail": detail[:500], "created_at": now}


def get_trace(session_id: str) -> list[dict]:
    init_db()
    with _lock, _connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM trace WHERE session_id=? ORDER BY seq", (session_id,))]


def clear_trace(session_id: str) -> None:
    init_db()
    with _lock, _connect() as c:
        c.execute("DELETE FROM trace WHERE session_id=?", (session_id,))
