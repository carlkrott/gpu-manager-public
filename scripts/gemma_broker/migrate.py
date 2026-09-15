"""Fail-closed migration of unclaimed legacy Gemma stream work."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

from .contracts import JobRecord, Priority, ReasonCode


class MigrationBlocked(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MigrationReport:
    dry_run: bool
    migrated_entry_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    ignored_legacy_slot_keys: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("migrated_entry_ids", "job_ids", "ignored_legacy_slot_keys"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MigrationReport":
        return cls(
            dry_run=bool(data["dry_run"]),
            migrated_entry_ids=tuple(data["migrated_entry_ids"]),
            job_ids=tuple(data["job_ids"]),
            ignored_legacy_slot_keys=tuple(data["ignored_legacy_slot_keys"]),
        )


class LegacyMigration:
    STREAM = "queue:Gemma"
    GROUP = "workers"
    LEGACY_INGRESS_KEY = "legacy:Gemma:authoritative"

    def __init__(
        self,
        redis_client,
        repository,
        *,
        candidate_prefix: str,
        now: Callable[[], float],
        in_flight_count: Callable[[], int] | None = None,
    ) -> None:
        if not candidate_prefix or candidate_prefix in {"gemma:", "combined-gemma:"}:
            raise ValueError("isolated candidate prefix required")
        self.redis = redis_client
        self.repository = repository
        self.prefix = candidate_prefix
        self.now = now
        self.in_flight_count = in_flight_count or (lambda: 0)

    @staticmethod
    def _text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def _pending_count(self) -> int:
        try:
            result = self.redis.xpending(self.STREAM, self.GROUP)
        except Exception as exc:
            if "NOGROUP" in str(exc).upper():
                return 0
            raise MigrationBlocked(ReasonCode.REDIS_STATE_UNKNOWN.value) from exc
        if isinstance(result, dict):
            raw = result.get("pending", result.get(b"pending", 0))
        elif isinstance(result, (tuple, list)):
            raw = result[0] if result else 0
        else:
            raw = result or 0
        return int(raw)

    def _entries(self) -> list[tuple[str, dict]]:
        entries = []
        for entry_id, fields in self.redis.xrange(self.STREAM, min="-", max="+"):
            cooked = {
                str(self._text(key)): self._text(value)
                for key, value in dict(fields).items()
            }
            entries.append((str(self._text(entry_id)), cooked))
        return entries

    def _slot_keys(self) -> tuple[str, ...]:
        if hasattr(self.redis, "hashes"):
            keys = getattr(self.redis, "hashes").keys()
        elif hasattr(self.redis, "scan_iter"):
            keys = self.redis.scan_iter("slots:*:Gemma*")
        else:
            keys = ()
        return tuple(sorted(str(self._text(key)) for key in keys if "Gemma" in str(self._text(key))))

    def dry_run(self) -> MigrationReport:
        entries = self._entries()
        return MigrationReport(
            dry_run=True,
            migrated_entry_ids=tuple(entry_id for entry_id, _ in entries),
            job_ids=tuple(self._job_id(entry_id) for entry_id, _ in entries),
            ignored_legacy_slot_keys=self._slot_keys(),
        )

    @staticmethod
    def _job_id(entry_id: str) -> str:
        return "legacy-" + hashlib.sha256(entry_id.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _request(fields: dict) -> dict:
        value = fields.get("request", fields.get("request_body", fields.get("payload", {})))
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise MigrationBlocked("LEGACY_REQUEST_INVALID")
        return value

    def execute(self) -> MigrationReport:
        report_key = f"{self.prefix}migration:report"
        prior = self.redis.get(report_key)
        if prior:
            return MigrationReport.from_dict(json.loads(self._text(prior)))
        if self._pending_count() != 0:
            raise MigrationBlocked(ReasonCode.MIGRATION_PEL_NOT_EMPTY.value)
        if self.in_flight_count() != 0:
            raise MigrationBlocked("MIGRATION_IN_FLIGHT_NOT_EMPTY")
        fence_key = f"{self.prefix}migration:fence"
        if not self.redis.set(fence_key, "claimed", nx=True):
            prior = self.redis.get(report_key)
            if prior:
                return MigrationReport.from_dict(json.loads(self._text(prior)))
            raise MigrationBlocked("MIGRATION_FENCE_HELD")

        entries = self._entries()
        job_ids: list[str] = []
        for sequence, (entry_id, fields) in enumerate(entries, start=1):
            request = self._request(fields)
            encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
            priority_raw = fields.get("priority", int(Priority.NORMAL))
            try:
                priority = Priority(int(priority_raw))
            except (TypeError, ValueError):
                priority = Priority.NORMAL
            job = JobRecord.new(
                job_id=self._job_id(entry_id),
                idempotency_key=f"legacy:{entry_id}",
                request_sha256=hashlib.sha256(encoded).hexdigest(),
                request_body=request,
                submitted_at=self.now(),
                enqueue_sequence=sequence,
                base_priority=priority,
                input_tokens_estimate=None,
                max_output_tokens=int(request.get("max_tokens", 0) or 0),
                required_capabilities=("chat",),
            )
            stored = self.repository.submit(job, caller_scope="legacy-migration")
            job_ids.append(stored.job_id)

        rollback_key = f"{self.prefix}migration:legacy-ingress-preimage"
        preimage = self.redis.get(self.LEGACY_INGRESS_KEY)
        self.redis.set(rollback_key, "" if preimage is None else self._text(preimage))
        self.redis.set(self.LEGACY_INGRESS_KEY, "false")
        report = MigrationReport(
            dry_run=False,
            migrated_entry_ids=tuple(entry_id for entry_id, _ in entries),
            job_ids=tuple(job_ids),
            ignored_legacy_slot_keys=self._slot_keys(),
        )
        self.redis.set(report_key, json.dumps(report.to_dict(), sort_keys=True))
        return report

    def rollback_ingress(self) -> None:
        rollback_key = f"{self.prefix}migration:legacy-ingress-preimage"
        preimage = self.redis.get(rollback_key)
        if preimage is None:
            raise MigrationBlocked("MIGRATION_ROLLBACK_SNAPSHOT_MISSING")
        restored = self._text(preimage)
        if restored == "":
            self.redis.delete(self.LEGACY_INGRESS_KEY)
        else:
            self.redis.set(self.LEGACY_INGRESS_KEY, restored)
