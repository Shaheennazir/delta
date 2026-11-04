"""Probability calibration + selective abstention (no fixed confidence fudge).

Calibration is fitted from labelled outcomes (isotonic regression, pool-adjacent-
violators). Until outcomes exist the raw score is returned with method
"uncalibrated" — never presented as calibrated. Call fit_from_outcomes() after
recording outcomes; every subsequent answer uses the stored curve.

Selective prediction: choose the lowest threshold whose cumulative error on the
labelled set stays at/below the requested risk, maximising coverage. This is the
standard risk-coverage trade-off, not a hand-picked magic number.
"""
from __future__ import annotations


def _mean(block: dict) -> float:
    return block["wy"] / block["w"] if block["w"] else 0.0


def fit_isotonic(pairs: list[tuple[float, int]]) -> list[dict]:
    """PAVA over (score, correct). Returns monotonically non-decreasing points."""
    clean = sorted((max(0.0, min(1.0, float(x))), 1 if y else 0) for x, y in pairs)
    if not clean:
        return []
    # Aggregate identical raw scores first: x is the table's primary key and a
    # score repeated N times carries N times the weight.
    grouped: dict[float, list[float]] = {}
    for x, y in clean:
        g = grouped.setdefault(x, [0.0, 0.0])
        g[0] += 1.0
        g[1] += float(y)
    blocks: list[dict] = []
    for x in sorted(grouped):
        w, wy = grouped[x]
        blocks.append({"w": w, "wx": x * w, "wy": wy})
        while len(blocks) > 1 and _mean(blocks[-2]) > _mean(blocks[-1]):
            b = blocks.pop()
            blocks[-1]["w"] += b["w"]
            blocks[-1]["wx"] += b["wx"]
            blocks[-1]["wy"] += b["wy"]
    return [{"x": round(b["wx"] / b["w"], 6), "y": round(_mean(b), 6),
             "n": int(b["w"])} for b in blocks]


def apply_curve(points: list[dict], x: float) -> float:
    """Piecewise-linear interpolation of the fitted isotonic curve."""
    if not points:
        return max(0.0, min(1.0, float(x)))
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    x = max(0.0, min(1.0, float(x)))
    if x <= xs[0]:
        return max(0.0, min(1.0, ys[0]))
    if x >= xs[-1]:
        return max(0.0, min(1.0, ys[-1]))
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            t = 0.0 if span <= 0 else (x - xs[i - 1]) / span
            return round(ys[i - 1] + t * (ys[i] - ys[i - 1]), 6)
    return ys[-1]


def brier(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return 0.0
    return round(sum((float(p) - float(y)) ** 2 for p, y in pairs) / len(pairs), 6)


def expected_calibration_error(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    """ECE over equal-width confidence bins."""
    if not pairs:
        return 0.0
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(bins - 1, max(0, int(float(p) * bins)))].append((float(p), int(y)))
    ece = 0.0
    for b in buckets:
        if not b:
            continue
        acc = sum(y for _, y in b) / len(b)
        conf = sum(p for p, _ in b) / len(b)
        ece += (len(b) / len(pairs)) * abs(acc - conf)
    return round(ece, 6)


def selective_threshold(pairs: list[tuple[float, int]], max_risk: float = 0.1) -> dict:
    """Lowest confidence threshold with empirical error <= max_risk; max coverage."""
    ordered = sorted(((float(p), int(y)) for p, y in pairs), reverse=True)
    best = {"threshold": 1.0, "coverage": 0.0, "risk": 0.0, "n": 0}
    errors = 0
    for i, (p, y) in enumerate(ordered, 1):
        errors += 0 if y else 1
        risk = errors / i
        if risk <= max_risk:
            best = {"threshold": round(p, 6), "coverage": round(i / len(ordered), 6),
                    "risk": round(risk, 6), "n": i}
        else:
            break
    return best


def fit_from_outcomes(max_risk: float = 0.1) -> dict:
    """Refit + persist from recorded human outcomes. Returns curve + metrics."""
    import store
    rows = store.get_outcomes()
    pairs = [(float(r["raw_confidence"]), int(r["correct"])) for r in rows]
    points = fit_isotonic(pairs)
    store.save_calibration(points)
    return {"n": len(pairs), "points": points, "brier": brier(pairs),
            "ece": expected_calibration_error(pairs),
            "selective": selective_threshold(pairs, max_risk)}


def apply_saved(x: float) -> tuple[float, str]:
    """Apply the persisted curve. Never claims calibration without outcomes."""
    import store
    points = store.get_calibration()
    if not points:
        return max(0.0, min(1.0, float(x))), "uncalibrated"
    return apply_curve(points, x), "isotonic"


def select_threshold_from_saved(max_risk: float = 0.1) -> float:
    import store
    rows = store.get_outcomes()
    if not rows:
        return 0.5  # Bayesian prior: only accept a coin-flip-or-better claim
    pairs = [(float(r["raw_confidence"]), int(r["correct"])) for r in rows]
    return float(selective_threshold(pairs, max_risk)["threshold"])
