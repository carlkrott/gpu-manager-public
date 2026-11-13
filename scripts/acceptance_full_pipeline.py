#!/usr/bin/env python3
"""Live acceptance gate for one real generation pipeline job.

Separate from the idle-restore proof (tests/publication/test_idle_restore_http.py):
that suite exercises the handlers against fakes; this gate drives a *running*
manager end to end. It submits one generation job (by default one that covers
the qwen-correct stage) and passes only if

  1. the job reaches ``completed`` (never merely accepted or queued),
  2. the job record shows the qwen-correct stage ran (marker search),
  3. the generation bundle (default ``comfyui``) is released after the idle TTL,
  4. both dGPU idle baselines are restored (VRAM per card within tolerance),
  5. the manager is not left ``llm_evicted`` / degraded.

The API token is read from a file (never argv/env value) and is only ever sent
as an Authorization header; it is never printed or written to the report.

    acceptance_full_pipeline.py --base-url http://127.0.0.1:8091 \
        --token-file /run/credentials/gpu-manager.service/api-token \
        [--payload job.json] [--baseline card0=14.3 --baseline card1=16.6]

Exit status: 0 pass, 1 gate failed, 2 usage/transport error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import aiohttp

DEFAULT_BASELINES_GB = {"card0": 14.3, "card1": 16.6}
DEFAULT_TOLERANCE_GB = 1.5
DEFAULT_GENERATION_BUNDLE = "comfyui"
DEFAULT_STAGE_MARKER = "qwen"
DEFAULT_PAYLOAD: dict[str, Any] = {
    "type": "image-edit-q5-benchmark",
    "params": {"prompt": "acceptance gate: tiny qwen-correct generation"},
    "priority": 50,
    "metadata": {"source": "acceptance_full_pipeline"},
}
TERMINAL_OK = frozenset({"completed"})
TERMINAL_BAD = frozenset({"failed", "cancelled", "dead_letter", "timeout"})


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class GateResult:
    checks: list[Check] = field(default_factory=list)
    job_id: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, ok, detail))
        return ok

    def report(self) -> dict[str, Any]:
        return {
            "verdict": "pass" if self.ok else "fail",
            "job_id": self.job_id,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks
            ],
        }


def parse_baselines(items: list[str] | None) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_BASELINES_GB)
    out: dict[str, float] = {}
    for item in items:
        card, sep, gb = item.partition("=")
        if not sep or not card.strip():
            raise ValueError(f"baseline must be CARD=GB, got {item!r}")
        out[card.strip()] = float(gb)
    return out


def _contains_marker(value: Any, marker: str) -> bool:
    marker = marker.lower()
    if isinstance(value, Mapping):
        return any(
            marker in str(k).lower() or _contains_marker(v, marker)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_marker(v, marker) for v in value)
    return marker in str(value).lower()


def _loaded_bundle_names(health: Mapping[str, Any]) -> set[str]:
    rows = health.get("loaded_runtime_bundles")
    if rows is None:
        rows = (health.get("vram") or {}).get("loaded_runtime_bundles")
    return {
        str(r.get("bundle")) for r in (rows or []) if isinstance(r, Mapping) and r.get("bundle")
    }


def _card_used_gb(vram: Mapping[str, Any]) -> dict[str, float]:
    used: dict[str, float] = {}
    for gpu in vram.get("gpus") or []:
        if not isinstance(gpu, Mapping):
            continue
        mib = gpu.get("vram_used_mib", gpu.get("used_mib"))
        if mib is None:
            continue
        for key in (gpu.get("card"), gpu.get("gpu_id")):
            if key:
                used[str(key)] = float(mib) / 1024.0
    return used


async def _body(response: aiohttp.ClientResponse) -> Any:
    """JSON body, or the raw text for non-JSON errors (a plain 401/502)."""
    text = await response.text()
    try:
        return json.loads(text)
    except ValueError:
        return {"raw": text[:200]}


class Gate:
    def __init__(self, base_url: str, token: str, session: aiohttp.ClientSession):
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._session = session

    async def _get(self, path: str) -> tuple[int, Any]:
        async with self._session.get(self._base + path, headers=self._headers) as r:
            return r.status, await _body(r)

    async def _post(self, path: str, body: Mapping[str, Any]) -> tuple[int, Any]:
        async with self._session.post(
            self._base + path, headers=self._headers, json=body
        ) as r:
            return r.status, await _body(r)

    async def _until(self, probe, deadline: float, interval: float):
        """Poll ``probe`` (async -> (done, value)) until done or deadline."""
        value: Any = None
        while True:
            done, value = await probe()
            if done or time.monotonic() >= deadline:
                return done, value
            await asyncio.sleep(interval)

    async def run(
        self,
        payload: Mapping[str, Any],
        *,
        baselines_gb: Mapping[str, float],
        tolerance_gb: float,
        generation_bundle: str,
        stage_marker: str,
        job_timeout_s: float,
        release_timeout_s: float,
        poll_interval_s: float,
    ) -> GateResult:
        result = GateResult()
        status, body = await self._post("/v1/submit/generation", payload)
        job_id = body.get("job_id") if isinstance(body, Mapping) else None
        result.job_id = job_id
        if not result.add(
            "job_submitted", status in (200, 201, 202) and bool(job_id),
            f"http {status} job_id={job_id}",
        ):
            return result

        async def job_probe():
            st, doc = await self._get(f"/v1/job/{job_id}")
            if st == 200 and isinstance(doc, Mapping) and doc.get("status") in (
                TERMINAL_OK | TERMINAL_BAD
            ):
                return True, doc
            return False, doc

        done, job = await self._until(
            job_probe, time.monotonic() + job_timeout_s, poll_interval_s
        )
        job = job if isinstance(job, Mapping) else {}
        job_status = job.get("status")
        if not done:
            result.add("job_completed", False, f"not terminal before timeout; status={job_status}")
            return result
        if not result.add(
            "job_completed", job_status in TERMINAL_OK,
            f"status={job_status} error_type={job.get('error_type')}",
        ):
            return result

        if stage_marker:
            result.add(
                "qwen_correct_stage_ran",
                _contains_marker(
                    {k: v for k, v in job.items() if k not in ("params", "payload")},
                    stage_marker,
                ),
                f"marker={stage_marker!r} in job record",
            )

        async def bundle_probe():
            st, doc = await self._get("/health")
            if st != 200 or not isinstance(doc, Mapping):
                return False, {"http": st}
            loaded = _loaded_bundle_names(doc)
            return generation_bundle not in loaded, {"doc": doc, "loaded": sorted(loaded)}

        deadline = time.monotonic() + release_timeout_s
        released, state = await self._until(bundle_probe, deadline, poll_interval_s)
        result.add(
            "generation_bundle_released", released,
            f"bundle={generation_bundle} loaded={state.get('loaded')}",
        )

        async def baseline_probe():
            st, doc = await self._get("/vram")
            if st != 200 or not isinstance(doc, Mapping):
                return False, {"http": st}
            used = _card_used_gb(doc)
            off = {
                card: used.get(card)
                for card, want in baselines_gb.items()
                if used.get(card) is None or abs(used[card] - want) > tolerance_gb
            }
            return not off, {"used": used, "off": off}

        restored, vstate = await self._until(baseline_probe, deadline, poll_interval_s)
        used = vstate.get("used") or {}
        result.add(
            "dgpu_baselines_restored", restored,
            "; ".join(
                f"{card} used={used.get(card, 'n/a') if not isinstance(used.get(card), float) else round(used[card], 2)}GB"
                f" want={want}±{tolerance_gb}GB"
                for card, want in baselines_gb.items()
            ),
        )

        st, health = await self._get("/health")
        health = health if isinstance(health, Mapping) else {}
        evicted = bool((health.get("vram") or {}).get("llm_evicted"))
        mode = health.get("runtime_mode")
        contradictions = health.get("contradictions") or []
        result.add(
            "manager_not_degraded",
            st == 200 and not evicted and mode != "degraded" and not contradictions,
            f"llm_evicted={evicted} runtime_mode={mode} contradictions={contradictions}",
        )
        return result


async def run_gate(
    base_url: str,
    token: str,
    payload: Mapping[str, Any] | None = None,
    *,
    baselines_gb: Mapping[str, float] | None = None,
    tolerance_gb: float = DEFAULT_TOLERANCE_GB,
    generation_bundle: str = DEFAULT_GENERATION_BUNDLE,
    stage_marker: str = DEFAULT_STAGE_MARKER,
    job_timeout_s: float = 900.0,
    release_timeout_s: float = 300.0,
    poll_interval_s: float = 3.0,
) -> GateResult:
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await Gate(base_url, token, session).run(
            payload or DEFAULT_PAYLOAD,
            baselines_gb=baselines_gb or DEFAULT_BASELINES_GB,
            tolerance_gb=tolerance_gb,
            generation_bundle=generation_bundle,
            stage_marker=stage_marker,
            job_timeout_s=job_timeout_s,
            release_timeout_s=release_timeout_s,
            poll_interval_s=poll_interval_s,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--token-file", required=True, type=Path)
    ap.add_argument("--payload", type=Path, help="JSON body for /v1/submit/generation")
    ap.add_argument("--baseline", action="append", metavar="CARD=GB")
    ap.add_argument("--tolerance-gb", type=float, default=DEFAULT_TOLERANCE_GB)
    ap.add_argument("--generation-bundle", default=DEFAULT_GENERATION_BUNDLE)
    ap.add_argument("--stage-marker", default=DEFAULT_STAGE_MARKER,
                    help="substring the completed job record must contain ('' skips)")
    ap.add_argument("--job-timeout-s", type=float, default=900.0)
    ap.add_argument("--release-timeout-s", type=float, default=300.0,
                    help="idle-hold TTL plus slack for release + baseline restore")
    ap.add_argument("--poll-interval-s", type=float, default=3.0)
    args = ap.parse_args(argv)
    try:
        token = args.token_file.read_text().strip()
        payload = json.loads(args.payload.read_text()) if args.payload else None
        baselines = parse_baselines(args.baseline)
        result = asyncio.run(run_gate(
            args.base_url, token, payload,
            baselines_gb=baselines, tolerance_gb=args.tolerance_gb,
            generation_bundle=args.generation_bundle, stage_marker=args.stage_marker,
            job_timeout_s=args.job_timeout_s, release_timeout_s=args.release_timeout_s,
            poll_interval_s=args.poll_interval_s,
        ))
    except (OSError, ValueError, aiohttp.ClientError, asyncio.TimeoutError) as exc:
        print(json.dumps({"verdict": "error", "error": type(exc).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(result.report(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
