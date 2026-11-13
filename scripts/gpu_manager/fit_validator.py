"""Fit validator (plan v3, phase 4b): one model component per GPU. REPORT-ONLY.

Pure functions over a services config and an optional live VRAM snapshot (the
dict ``_poll_vram()`` returns).  Nothing here blocks an activation: callers log
or serve the findings.  Stdlib only, so it also runs as a CLI:

    python3 scripts/gpu_manager/fit_validator.py --config ~/.config/gpu-manager/services.json [--live] [--json]

Units: GB means GiB (MiB / 1024).  Rules (severity ``error`` = activation
would be refused once this gate is enforced):

- ``over_budget``: a component's VRAM peak is over its card's budget.  Budget =
  card VRAM, minus a 2.5 GB display reserve on the ``primary`` card, minus
  protected residents and residents of higher-priority bundles (the ones a
  load cannot evict), measured live when a snapshot is given.
- ``split_offload``: a non-MoE component splits or offloads (ComfyUI DisTorch /
  ``virtual_vram_gb`` / ``donor_device``, llama ``-ts`` / ``--cpu-moe`` /
  ``--lowvram`` ...).  ``moe: true`` is exempt.
- ``weights_off_models_ssd``: a service/revision ``weights_path`` (or
  ``model_path``) does not resolve under $GPU_MANAGER_MODELS_ROOT (rule off
  when unset).  ``moe: true`` is exempt.
- ``unsized`` / ``workflow_missing`` (info): nothing to measure against.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = os.environ.get("GPU_MANAGER_MODELS_ROOT", "")
MODEL_ROOTS = ("~/ComfyUI/models", "~/models")
DISPLAY_RESERVE_GB = 2.5
FALLBACK_CARD_GB = 16.0
# Used only when no live snapshot covers a card (budget_source: fallback).
# Measured by poll_vram 2026-10-22 (plan v3): d1 3929 MiB; Qwen-Dual 7.7 / 12.9 GB.
FALLBACK_RESIDENT_GB = {
    "D1-MI50": {"mi50": 3929 / 1024},
    "Qwen-Dual": {"mi50": 7.7, "rx6950xt": 12.9},
}
ERROR_RULES = ("over_budget", "split_offload", "weights_off_models_ssd")
DISK_EXEMPT = {"Qwen-Dual"}  # plan v3: lives on /srv/strata-data by design
_LLAMA_SPLIT_FLAGS = {"-ts", "--tensor-split", "--cpu-moe", "-cmoe", "--n-cpu-moe",
                      "-ncmoe", "--lowvram"}
_COMFY_FILE_KEYS = ("unet_name", "ckpt_name", "clip_name", "clip_name1", "clip_name2",
                    "clip_name3", "vae_name", "model_name")
_SHARD = re.compile(r"-0*1-of-(\d+)\.gguf$")
_OBSERVED = re.compile(r"^observed_(.+?)_peak_vram_mib$")


# ── sizes ────────────────────────────────────────────────────────────────────

def _path_gb(path: str) -> float | None:
    files = [path]
    m = _SHARD.search(path)
    if m:
        files = glob.glob(_SHARD.sub(f"-*-of-{m.group(1)}.gguf", path)) or files
    try:
        return sum(os.path.getsize(f) for f in files) / 1024 ** 3
    except OSError:
        return None


def default_size_lookup(roots=MODEL_ROOTS) -> Callable[[str], float | None]:
    """Size in GB of an absolute path, or of a bare file name found under *roots*."""
    index: dict[str, str] = {}

    def lookup(name: str) -> float | None:
        if os.path.isabs(name):
            return _path_gb(name)
        if not index:
            for root in roots:
                for d, _, files in os.walk(os.path.expanduser(root), followlinks=True):
                    for f in files:
                        index.setdefault(f, os.path.join(d, f))
            index.setdefault("", "")  # walked, even if empty
        found = index.get(os.path.basename(name))
        return _path_gb(found) if found else None

    return lookup


# ── devices and budgets ──────────────────────────────────────────────────────

def _resolve_device(config: Mapping, device: Any) -> str | None:
    """Map 'rx' / 'mi50' / 'cuda:N' / a gpu id to the configured gpu id."""
    if device in (None, "", "cpu"):
        return None
    devices = config.get("gpu_devices") or {}
    text = str(device)
    if text in devices:
        return text
    if text.startswith("cuda:"):
        for gid, cfg in devices.items():
            if str((cfg or {}).get("rocm_device")) == text[5:]:
                return gid
        return text
    for gid in devices:
        if gid.startswith(text):
            return gid
    return text


def _snapshot_ok(snapshot: Any) -> bool:
    return isinstance(snapshot, Mapping) and not snapshot.get("error") and bool(snapshot.get("gpus"))


def _live_gpu(snapshot: Any, gid: str) -> Mapping | None:
    if not _snapshot_ok(snapshot):
        return None
    for gpu in snapshot.get("gpus") or []:
        if isinstance(gpu, Mapping) and gpu.get("gpu_id") == gid:
            return gpu
    return None


def _live_service_mib(snapshot: Mapping, gpu: Mapping, gid: str, svc: Mapping) -> float:
    tenants = {t.get("pid"): t for t in (snapshot.get("vulkan_tenants") or {}).get(gid, [])
               if isinstance(t, Mapping)}
    total = 0.0
    for p in gpu.get("processes") or []:
        if not isinstance(p, Mapping):
            continue
        port_hit = svc.get("port") and p.get("service_port") == svc.get("port")
        path_hit = svc.get("model_path") and p.get("model_path") == svc.get("model_path")
        if port_hit or path_hit:
            vulkan = (tenants.get(p.get("pid")) or {}).get("amdgpu_vram_mib") or 0
            total += max(float(p.get("vram_mib") or 0), float(vulkan))
    return total


def _foreign_mib(config: Mapping, snapshot: Mapping, gpu: Mapping, gid: str) -> float:
    """VRAM held on gid by processes no configured service owns (Breccia, desktop, ad-hoc
    llama.cpp): gpu-manager cannot evict them, so they shrink every budget (audit B7)."""
    services = [s for s in (config.get("services") or {}).values() if isinstance(s, Mapping)]

    def managed(p: Mapping) -> bool:
        return any((s.get("port") and p.get("service_port") == s.get("port")) or
                   (s.get("model_path") and p.get("model_path") == s.get("model_path"))
                   for s in services)

    procs = {p.get("pid"): p for p in gpu.get("processes") or [] if isinstance(p, Mapping)}
    by_pid: dict = {pid: float(p.get("vram_mib") or 0) for pid, p in procs.items() if not managed(p)}
    for t in (snapshot.get("vulkan_tenants") or {}).get(gid, []):
        pid = t.get("pid") if isinstance(t, Mapping) else None
        if pid is None or (pid in procs and managed(procs[pid])):
            continue
        by_pid[pid] = max(by_pid.get(pid, 0.0), float(t.get("amdgpu_vram_mib") or 0))
    return sum(by_pid.values())


def _bundle_of(config: Mapping) -> dict[str, tuple[str, int]]:
    out = {}
    for bname, b in (config.get("bundles") or {}).items():
        for s in (b or {}).get("services") or []:
            out[s] = (bname, int((b or {}).get("priority", 5)))
    return out


def card_budget(config: Mapping, snapshot: Any, gid: str, *, owner_priority: int = 5,
                owner_services=()) -> tuple[float, str, dict[str, float]]:
    """(budget_gb, budget_source, {resident: gb}) for one card and one owner."""
    dev = (config.get("gpu_devices") or {}).get(gid) or {}
    total = float(dev.get("vram_mib") or 0) / 1024 or FALLBACK_CARD_GB
    reserve = DISPLAY_RESERVE_GB if dev.get("role") == "primary" else 0.0
    gpu = _live_gpu(snapshot, gid)
    source = "live" if gpu is not None and dev.get("vram_mib") else "fallback"
    bundle_of = _bundle_of(config)
    residents: dict[str, float] = {}
    for name, svc in (config.get("services") or {}).items():
        if not isinstance(svc, Mapping) or name in owner_services or svc.get("enabled") is False:
            continue
        # shortcut: mirrors _evict_then_load_bundle by priority only (no idle/pin/lease
        # nuance), upgrade when the gate becomes enforcing.
        unevictable = svc.get("protected") is True or (
            name in bundle_of and bundle_of[name][1] > owner_priority)
        if not unevictable:
            continue
        if gpu is not None:
            gb = _live_service_mib(snapshot, gpu, gid, svc) / 1024
        else:
            gb = (FALLBACK_RESIDENT_GB.get(name) or {}).get(gid) or (
                float(svc.get("vram_peak_gb") or 0) if svc.get("gpu_id") == gid else 0.0)
        if gb:
            residents[name] = round(gb, 2)
    if gpu is not None:
        # the display reserve already covers desktop tenants on the primary card
        foreign = _foreign_mib(config, snapshot, gpu, gid) / 1024 - reserve
        if foreign > 0.05:
            residents["foreign"] = round(foreign, 2)
    return round(total - reserve - sum(residents.values()), 2), source, residents


# ── components ───────────────────────────────────────────────────────────────

def _flag_split(args) -> list[str]:
    hits = []
    for a in args or []:
        flag = str(a).split("=", 1)[0]
        if flag in _LLAMA_SPLIT_FLAGS:
            hits.append(flag)
    return hits


def _service_component(name: str, svc: Mapping, default_device) -> dict | None:
    weights = svc.get("weights_path") or svc.get("model_path") or None
    split = _flag_split(svc.get("launch_args"))
    if not (weights or split or svc.get("vram_peak_gb") is not None):
        return None
    return {"component": name, "device": svc.get("gpu_id") or default_device,
            "weights": weights, "lookup": weights, "split": split,
            "moe": svc.get("moe") is True, "vram_peak_gb": svc.get("vram_peak_gb"),
            "measured": svc.get("measured_vram_peak_gb"), "disk_rule": bool(weights)}


def _workflow_components(workflow: Mapping, default_device) -> list[dict]:
    comps = []
    for node in workflow.values():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs") if isinstance(node.get("inputs"), Mapping) else {}
        cls = str(node.get("class_type") or "")
        split = []
        if "DisTorch" in cls:
            split.append(cls)
        if float(inputs.get("virtual_vram_gb") or 0) > 0:
            split.append(f"virtual_vram_gb={inputs['virtual_vram_gb']}")
        if inputs.get("donor_device"):
            split.append(f"donor_device={inputs['donor_device']}")
        if "--lowvram" in str(inputs.get("args") or ""):
            split.append("--lowvram")
        files = [inputs[k] for k in _COMFY_FILE_KEYS if isinstance(inputs.get(k), str)]
        for f in files:
            comps.append({"component": f, "lookup": f,
                          "device": inputs.get("compute_device") or inputs.get("device")
                          or default_device, "split": split})
    return comps


def _load_workflow(path: str, project_root: Path) -> Mapping | None:
    p = Path(path) if os.path.isabs(path) else Path(project_root) / path
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, Mapping) else None


def _bundle_components(config: Mapping, bname: str, b: Mapping, project_root: Path) -> list[dict]:
    if isinstance(b.get("components"), list):
        return [{**c, "component": c.get("name") or c.get("component") or "?",
                 "lookup": c.get("weights_path"), "weights": c.get("weights_path"),
                 "split": _flag_split(c.get("launch_args")) + list(c.get("split") or []),
                 "measured": c.get("measured_vram_peak_gb"), "moe": c.get("moe") is True,
                 "disk_rule": bool(c.get("weights_path"))}
                for c in b["components"] if isinstance(c, Mapping)]
    services = config.get("services") or {}
    gpu = b.get("gpu_id")
    comps = []
    units = set()
    for s in b.get("services") or []:
        svc = services.get(s) or {}
        if svc.get("enabled") is False:
            continue
        units.add(svc.get("systemd_unit"))
        c = _service_component(s, svc, gpu)
        if c:
            comps.append(c)
    profiles = config.get("runtime_profiles") or {}
    model_set_names = {(profiles.get(b.get("runtime_profile")) or {}).get("model_set")}
    model_set_names |= {p.get("model_set") for p in profiles.values()
                        if isinstance(p, Mapping) and (p.get("process") or {}).get("unit") in units - {None}}
    workflows = {t.get("workflow_template") or id(t.get("workflow")): t
                 for t in (config.get("generation_templates") or {}).values()
                 if isinstance(t, Mapping) and t.get("bundle") == bname
                 and (t.get("workflow_template") or isinstance(t.get("workflow"), Mapping))}
    for ms_name in sorted(n for n in model_set_names if n):
        ms = (config.get("model_sets") or {}).get(ms_name) or {}
        for c in ms.get("components") or []:
            if isinstance(c, Mapping) and c.get("device") not in (None, "cpu"):
                comps.append({"component": c.get("artifact") or c.get("id"), "lookup": c.get("artifact"),
                              "device": c.get("device"), "vram_peak_gb": c.get("vram_peak_gb"),
                              "measured": c.get("measured_vram_peak_gb")})
        meta = ms.get("metadata") or {}
        for key, mib in meta.items():
            m = _OBSERVED.match(key)
            if m and isinstance(mib, (int, float)):
                comps.append({"component": f"{ms_name} (measured {m.group(1)} peak)",
                              "device": m.group(1), "measured": mib / 1024})
        if meta.get("workflow"):
            workflows.setdefault(meta["workflow"], {"workflow_template": meta["workflow"]})
    for key, t in workflows.items():
        wf = t.get("workflow") if isinstance(t.get("workflow"), Mapping) else _load_workflow(
            t["workflow_template"], project_root)
        if wf is None:
            comps.append({"component": str(key), "info": "workflow_missing"})
        else:
            comps.extend(_workflow_components(wf, gpu))
    return comps


def _owners(config: Mapping, project_root: Path):
    """Yield (owner, priority, owner_services, components)."""
    bundles = config.get("bundles") or {}
    bundle_of = _bundle_of(config)
    for bname, b in bundles.items():
        if isinstance(b, Mapping):
            yield bname, int(b.get("priority", 5)), set(b.get("services") or []), \
                _bundle_components(config, bname, b, project_root)
    for name, svc in (config.get("services") or {}).items():
        if isinstance(svc, Mapping) and name not in bundle_of and svc.get("enabled") is not False:
            c = _service_component(name, svc, None)
            if c and c["device"]:
                yield name, int(svc.get("priority", 5)), {name}, [c]
    catalog = ((config.get("service_revision_catalog") or {}).get("services")) or {}
    for name, revisions in catalog.items():
        for rev in revisions if isinstance(revisions, list) else []:
            d = (rev or {}).get("definition") or {}
            comps = []
            c = _service_component(name, d, None)
            if c:
                comps.append(c)
            te_split = ((d.get("resource_requirements") or {}).get("text_encoder_gpu_split")) or []
            if len(te_split) > 1:
                comps.append({"component": "text_encoder", "device": te_split[0],
                              "split": [f"text_encoder_gpu_split={'+'.join(te_split)}"]})
            if comps:
                yield f"{name}@r{rev.get('revision')}", int(d.get("priority", 5)), {name}, comps


# ── evaluation ───────────────────────────────────────────────────────────────

def _under(path: str, root: str) -> bool:
    real, root = os.path.realpath(path), os.path.realpath(root)
    try:
        return os.path.commonpath([real, root]) == root
    except ValueError:
        return False


# shortcut: each component is checked alone against its card budget, not summed per card
# (plan v3 rule is one component per GPU); upgrade if co-resident components appear.
def evaluate(config: Mapping, snapshot: Any = None, *, project_root: Path | str = REPO_ROOT,
             size_lookup: Callable[[str], float | None] | None = None,
             models_root: str = MODELS_ROOT, owners=None) -> list[dict]:
    """Findings for every bundle/service/revision (or only *owners*)."""
    lookup = size_lookup or default_size_lookup()
    findings: list[dict] = []
    seen = set()
    for owner, priority, owner_services, comps in _owners(config, Path(project_root)):
        if owners is not None and owner not in owners:
            continue
        for c in comps:
            gid = _resolve_device(config, c.get("device"))
            row = {"bundle": owner, "component": c["component"], "device": gid,
                   "vram_peak_gb": None, "budget_gb": None, "estimate": False,
                   "budget_source": None}

            def add(rule, severity, detail):
                key = (owner, row["component"], gid, rule, detail)
                if key not in seen:
                    seen.add(key)
                    findings.append({**row, "rule": rule, "severity": severity, "detail": detail})

            if c.get("info"):
                add(c["info"], "info", "workflow file not found")
                continue
            moe = c.get("moe") is True
            peak, estimate, residents = c.get("measured"), False, {}
            if peak is None:
                peak = c.get("vram_peak_gb")
            if peak is None and c.get("lookup"):
                peak, estimate = lookup(c["lookup"]), True
            if gid is not None:
                budget, source, residents = card_budget(
                    config, snapshot, gid, owner_priority=priority, owner_services=owner_services)
                row.update(budget_gb=budget, budget_source=source)
            if peak is not None:
                row.update(vram_peak_gb=round(float(peak), 2), estimate=estimate)
            if c.get("split") and not moe:
                add("split_offload", "error", ", ".join(c["split"]))
            disk_exempt = moe or c["component"] in DISK_EXEMPT
            if models_root and c.get("disk_rule") and c.get("weights") \
                    and not disk_exempt and not _under(c["weights"], models_root):
                add("weights_off_models_ssd", "error",
                    f"{c['weights']} -> {os.path.realpath(c['weights'])}")
            if moe or gid is None:
                continue
            if peak is None:
                add("unsized", "info", "no vram_peak_gb and no weight file to estimate from")
            elif float(peak) > row["budget_gb"]:
                add("over_budget", "error", f"residents {residents}" if residents else "")
    return findings


def report(config: Mapping, snapshot: Any = None, **kw) -> dict:
    findings = evaluate(config, snapshot, **kw)
    return {
        "report_only": True,
        "budget_source": "live" if _snapshot_ok(snapshot) else "fallback",
        "would_refuse": sorted({f["bundle"] for f in findings if f["severity"] == "error"}),
        "findings": findings,
    }


def warn_if_unfit(log, config: Mapping, snapshot: Any, owners=(), *, service: str | None = None,
                  **kw) -> list[dict]:
    """Activation hook: log what the gate would refuse.  Never raises, never blocks.

    *service* adds the bundles holding that service (or the service itself).
    """
    try:
        owners = set(owners)
        if service:
            owners |= {b for b, cfg in (config.get("bundles") or {}).items()
                       if service in ((cfg or {}).get("services") or [])} or {service}
        errors = [f for f in evaluate(config, snapshot, owners=owners, **kw)
                  if f["severity"] == "error"]
        for f in errors:
            log.warning("FitValidator (report-only): activation of %s would be refused: "
                        "%s %s on %s (%s)", f["bundle"], f["rule"], f["component"],
                        f["device"], f["detail"])
        return errors
    except Exception as exc:  # report-only: a validator bug must never break a load
        log.warning("FitValidator (report-only) failed: %s", exc)
        return []


# ── CLI ──────────────────────────────────────────────────────────────────────

def _fetch_live(url: str) -> Mapping | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)
    except Exception as exc:
        print(f"live snapshot unavailable ({exc}); using fallback residents", file=sys.stderr)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Report-only GPU fit validator")
    ap.add_argument("--config", default="~/.config/gpu-manager/services.json")
    ap.add_argument("--live", action="store_true", help="read poll_vram data from --vram-url")
    ap.add_argument("--vram-url", default="http://127.0.0.1:8091/vram")
    ap.add_argument("--project-root", default=str(REPO_ROOT))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    config = json.loads(Path(os.path.expanduser(a.config)).read_text())
    snapshot = _fetch_live(a.vram_url) if a.live else None
    out = report(config, snapshot, project_root=a.project_root)
    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"budget_source={out['budget_source']}  would_refuse={out['would_refuse']}")
    for f in out["findings"]:
        print(f"{f['severity']:5} {f['rule']:22} {f['bundle']:28} {str(f['device']):9} "
              f"peak={f['vram_peak_gb']}{'~' if f['estimate'] else ''} budget={f['budget_gb']} "
              f"{f['component']}  {f['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
