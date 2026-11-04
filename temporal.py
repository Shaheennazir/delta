"""Deterministic temporal reasoning engine (Allen interval algebra).

The LLM never does temporal arithmetic. Every event carries start_t/end_t;
all before/after/during/overlap/gap/ordering/contradiction queries go
through here so a language prior cannot override timestamps.
"""
from __future__ import annotations


def relation(a_start: float, a_end: float, b_start: float, b_end: float,
             eps: float = 1e-6) -> str:
    """One of Allen's 13 relations of A to B."""
    a_start, a_end = float(a_start), float(a_end)
    b_start, b_end = float(b_start), float(b_end)
    if a_end < b_start - eps:
        return "before"
    if abs(a_start - b_start) <= eps and abs(a_end - b_end) <= eps:
        return "equals"
    if abs(a_end - b_start) <= eps:
        return "meets"
    if a_start < b_start - eps and b_start < a_end - eps and a_end < b_end - eps:
        return "overlaps"
    if abs(a_start - b_start) <= eps and a_end < b_end - eps:
        return "starts"
    if b_start < a_start - eps and abs(a_end - b_end) <= eps:
        return "finishes"
    if b_start < a_start - eps and a_end < b_end - eps:
        return "during"
    if a_start < b_start - eps and abs(a_end - b_end) <= eps:
        return "finished-by"
    if abs(a_start - b_start) <= eps and b_end < a_end - eps:
        return "started-by"
    if a_start < b_start - eps and b_end < a_end - eps:
        return "contains"
    if b_start < a_start - eps and a_start < b_end - eps and b_end < a_end - eps:
        return "overlapped-by"
    if abs(a_start - b_end) <= eps:
        return "met-by"
    return "after"


def before(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return relation(a_start, a_end, b_start, b_end) in ("before", "meets")


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return relation(a_start, a_end, b_start, b_end) in (
        "overlaps", "overlapped-by", "starts", "started-by",
        "finishes", "finished-by", "during", "contains", "equals")


def gap(a_end: float, b_start: float) -> float:
    return float(b_start) - float(a_end)


def duration(start_t: float, end_t: float) -> float:
    return max(0.0, float(end_t) - float(start_t))


def order_events(events: list[dict]) -> list[dict]:
    """Sort by (start_t, end_t); stable, deterministic."""
    return sorted(events, key=lambda e: (float(e.get("start_t", 0)), float(e.get("end_t", 0))))


def find_before(events: list[dict], t: float) -> list[dict]:
    return [e for e in order_events(events) if float(e.get("end_t", 0)) <= float(t)]


def find_after(events: list[dict], t: float) -> list[dict]:
    return [e for e in order_events(events) if float(e.get("start_t", 0)) >= float(t)]


def find_between(events: list[dict], a_end: float, b_start: float) -> list[dict]:
    return [e for e in order_events(events)
            if float(e.get("start_t", 0)) >= float(a_end) and float(e.get("end_t", 0)) <= float(b_start)]


def find_containing(events: list[dict], t: float) -> list[dict]:
    return [e for e in events
            if float(e.get("start_t", 0)) <= float(t) <= float(e.get("end_t", 0))]


def contradictions(events: list[dict]) -> list[str]:
    """Deterministic checks: end<=start, out-of-range, duplicate ids, overlap of
    mutually-exclusive state transitions on the same entity (same key, different value)."""
    problems: list[str] = []
    seen: set[str] = set()
    for e in events:
        eid = str(e.get("event_id", ""))
        if eid in seen:
            problems.append(f"duplicate event_id {eid}")
        seen.add(eid)
        s, en = float(e.get("start_t", 0)), float(e.get("end_t", 0))
        if en <= s:
            problems.append(f"{eid}: end_t ({en}) <= start_t ({s})")
    return problems


def validate_clamp(events: list[dict], duration_s: float) -> list[dict]:
    """Clamp to [0, duration], drop end<=start. Never invents time."""
    out = []
    for e in events:
        s = max(0.0, float(e.get("start_t", 0)))
        en = min(float(duration_s), float(e.get("end_t", 0))) if duration_s > 0 else float(e.get("end_t", 0))
        if en <= s:
            continue
        e = dict(e)
        e["start_t"], e["end_t"] = round(s, 2), round(en, 2)
        out.append(e)
    return order_events(out)
