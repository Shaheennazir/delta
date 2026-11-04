"""Deterministic temporal reasoning engine (Allen interval algebra).

The LLM never does temporal arithmetic. Every event carries start_t/end_t;
all before/after/during/overlap/gap/ordering/contradiction queries go
through here so a language prior cannot override timestamps.
"""
from __future__ import annotations


def _norm_ts(x) -> float:
    """Normalised absolute time: ms-rounded float. All ordering and Allen
    relations go through here so input precision never leaks into decisions."""
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return 0.0


def _eid(e: dict) -> str:
    return str(e.get("event_id") or e.get("candidate_id") or "")


def relation(a_start: float, a_end: float, b_start: float, b_end: float,
             eps: float = 1e-6) -> str:
    """One of Allen's 13 relations of A to B (timestamps normalised first)."""
    a_start, a_end = _norm_ts(a_start), _norm_ts(a_end)
    b_start, b_end = _norm_ts(b_start), _norm_ts(b_end)
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
    """Sort by (normalised start, normalised end, stable id). Pure function of
    timestamps + ids: never input position. The id tie-break makes every
    permutation of the same timestamped events order identically."""
    return sorted(events, key=lambda e: (_norm_ts(e.get("start_t", 0)),
                                         _norm_ts(e.get("end_t", 0)), _eid(e)))


def find_before(events: list[dict], t: float) -> list[dict]:
    t = _norm_ts(t)
    return [e for e in order_events(events) if _norm_ts(e.get("end_t", 0)) <= t]


def find_after(events: list[dict], t: float) -> list[dict]:
    t = _norm_ts(t)
    return [e for e in order_events(events) if _norm_ts(e.get("start_t", 0)) >= t]


def find_between(events: list[dict], a_end: float, b_start: float) -> list[dict]:
    a_end, b_start = _norm_ts(a_end), _norm_ts(b_start)
    return [e for e in order_events(events)
            if _norm_ts(e.get("start_t", 0)) >= a_end and _norm_ts(e.get("end_t", 0)) <= b_start]


def find_containing(events: list[dict], t: float) -> list[dict]:
    t = _norm_ts(t)
    return [e for e in events
            if _norm_ts(e.get("start_t", 0)) <= t <= _norm_ts(e.get("end_t", 0))]


def contradictions(events: list[dict], duration: float = 0.0) -> list[str]:
    """Deterministic checks: malformed spans, duplicate ids, out-of-range, and
    mutually-exclusive state values asserted by overlapping events on one entity."""
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
        if duration > 0 and (s < -1e-6 or en > duration + 1e-6):
            problems.append(f"{eid}: span [{s},{en}] outside [0,{duration}]")
    # Same entity + same state key + same time + different value = contradiction.
    claims: list[dict] = []
    for e in events:
        s, en = float(e.get("start_t", 0)), float(e.get("end_t", 0))
        for eid in e.get("entity_ids") or []:
            for key, value in (e.get("state_after") or {}).items():
                claims.append({"event_id": str(e.get("event_id", "")), "entity_id": eid,
                               "key": key, "value": value, "start": s, "end": en})
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            if (a["entity_id"] == b["entity_id"] and a["key"] == b["key"]
                    and a["value"] != b["value"]
                    and min(a["end"], b["end"]) > max(a["start"], b["start"]) + 1e-6):
                problems.append(
                    f"{a['entity_id']}: {a['key']} is both {a['value']!r} "
                    f"({a['event_id']}) and {b['value']!r} ({b['event_id']})")
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
