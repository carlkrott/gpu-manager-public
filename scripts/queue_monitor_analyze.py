#!/usr/bin/env python3
"""Summarize JSONL emitted by queue_monitor.py without fixed service names."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "gpu-manager-semantic-monitor.v1"


def load_records(path: Path, *, hours: float | None) -> tuple[list[Mapping[str, Any]], int]:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
        if hours is not None
        else None
    )
    records: list[Mapping[str, Any]] = []
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                timestamp = datetime.fromisoformat(record["timestamp"])
                if record.get("schema_version") != SCHEMA:
                    raise ValueError("unsupported schema")
                if cutoff is not None and timestamp < cutoff:
                    continue
                records.append(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
    return records, invalid


def summarize(records: list[Mapping[str, Any]], *, invalid: int) -> dict[str, Any]:
    if not records:
        return {
            "schema_version": "gpu-manager-semantic-monitor-summary.v1",
            "sample_count": 0,
            "invalid_record_count": invalid,
            "status": "no_samples",
        }

    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    queue_depths: dict[str, list[int]] = {}
    oldest_waits: dict[str, list[float]] = {}
    unavailable_counts: Counter[str] = Counter()
    delegated_counts: Counter[str] = Counter()
    for record in records:
        queues = record.get("queues")
        if not isinstance(queues, Mapping):
            continue
        for name, queue in queues.items():
            if not isinstance(queue, Mapping):
                continue
            if queue.get("queue_state") == "delegated":
                delegated_counts[str(name)] += 1
                continue
            if queue.get("queue_state") != "observed":
                unavailable_counts[str(name)] += 1
                continue
            depth = queue.get("queue_depth")
            if isinstance(depth, int) and not isinstance(depth, bool) and depth >= 0:
                queue_depths.setdefault(str(name), []).append(depth)
            wait = queue.get("oldest_wait_seconds")
            if isinstance(wait, (int, float)) and not isinstance(wait, bool) and wait >= 0:
                oldest_waits.setdefault(str(name), []).append(float(wait))

    queues = {}
    for name in sorted(set(queue_depths) | set(unavailable_counts) | set(delegated_counts)):
        depths = queue_depths.get(name, [])
        waits = oldest_waits.get(name, [])
        queues[name] = {
            "observed_samples": len(depths),
            "unavailable_samples": unavailable_counts[name],
            "delegated_samples": delegated_counts[name],
            "average_depth": round(sum(depths) / len(depths), 2) if depths else None,
            "maximum_depth": max(depths) if depths else None,
            "maximum_oldest_wait_seconds": round(max(waits), 3) if waits else None,
        }

    return {
        "schema_version": "gpu-manager-semantic-monitor-summary.v1",
        "sample_count": len(records),
        "invalid_record_count": invalid,
        "first_timestamp": records[0].get("timestamp"),
        "last_timestamp": records[-1].get("timestamp"),
        "status_counts": dict(sorted(statuses.items())),
        "queues": queues,
        "latest_metrics": records[-1].get("metrics"),
        "latest_controller": records[-1].get("controller"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="semantic monitor JSONL file")
    parser.add_argument("--hours", type=float)
    args = parser.parse_args(argv)
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours must be positive")
    try:
        records, invalid = load_records(args.log, hours=args.hours)
    except OSError as exc:
        print(json.dumps({"error": f"cannot read monitor log: {type(exc).__name__}"}))
        return 2
    print(json.dumps(summarize(records, invalid=invalid), indent=2, sort_keys=True))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
