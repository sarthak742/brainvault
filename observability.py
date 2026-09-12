"""
Lightweight observability for BrainVault.

Session 1 (Observability) had three pillars: logs, metrics, traces. Logging you
already have. This module adds the other two cheaply:

- METRICS: every answered query appends one JSON line to logs/metrics.jsonl.
- TRACES : each record carries the timing of the query (latency_ms), so you can
  see where time goes.

It is deliberately just a JSONL file + an aggregator — no Prometheus/Grafana
needed to start. "You cannot improve what you cannot measure" begins here.
"""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("brainvault.observability")

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = LOG_DIR / "metrics.jsonl"

_write_lock = threading.Lock()


def record(event: Dict[str, Any]) -> None:
    """Append one metrics event (a dict) as a JSON line."""
    event = {"ts": time.time(), **event}
    try:
        with _write_lock, open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        logger.exception("Failed to write metrics event")


class Timer:
    """`with Timer() as t: ...` then read t.ms for elapsed milliseconds."""
    def __enter__(self):
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000.0


def summary() -> Dict[str, Any]:
    """Aggregate the metrics file into the numbers you'd put on a dashboard."""
    if not METRICS_FILE.exists():
        return {"queries": 0}

    n = 0
    grounded = 0
    refused = 0
    cache_hits = 0
    latencies = []

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "query":
                continue
            n += 1
            if e.get("grounded"):
                grounded += 1
            else:
                refused += 1  # not grounded == system declined to answer from context
            if e.get("cache_hit"):
                cache_hits += 1
            if isinstance(e.get("latency_ms"), (int, float)):
                latencies.append(e["latency_ms"])

    latencies.sort()

    def pct(p):
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(p / 100 * len(latencies)))
        return round(latencies[idx], 1)

    return {
        "queries": n,
        "grounded_rate": round(grounded / n, 3) if n else 0.0,
        "refusal_rate": round(refused / n, 3) if n else 0.0,
        "cache_hit_rate": round(cache_hits / n, 3) if n else 0.0,
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p50": pct(50),
            "p95": pct(95),
        },
    }
