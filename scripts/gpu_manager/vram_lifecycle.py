"""Facade-bound VRAM lifecycle manager: GPU state machine and default LLM tenant.

``VRAMLifecycleManager`` is extracted verbatim from ``scripts/gpu-manager.py``.
Every method reads facade globals (``GpuState``, ``event_bus``, ``scheduler``,
``logger``, service helpers and so on) at call time, so ``bind(facade_globals)``
rebuilds each function with ``types.FunctionType`` against the supplied facade
dictionary and builds a fresh class whose methods, static methods, class
methods and ``__class__`` cell are rebuilt the same way.  No ``sys.modules``
entry, shared facade pointer or copied singleton state is kept here: the
``vram`` singleton, ``session``, ``scheduler`` and ``event_bus`` stay
facade-owned.  The class below is a template: constructing it directly would
resolve names in this module.

Two definition-time defaults are import-time snapshots, not call-time lookups:
``drain_llm(timeout=DRAIN_TIMEOUT)`` and
``_wait_for_llm_healthy(timeout=LLM_RESTORE_TIMEOUT)``.  The constants below
seed the template with the facade's import-time values (60 and 90) and
rebuilding retains them, exactly as the original ``class`` statement did.
"""
from __future__ import annotations

from types import CellType, FunctionType

NAMES = ("VRAMLifecycleManager",)
DRAIN_TIMEOUT = 60  # facade import-time value, snapshotted by drain_llm's default
LLM_RESTORE_TIMEOUT = 90  # facade import-time value, snapshotted by _wait_for_llm_healthy's default


class VRAMLifecycleManager:
    """Manages the GPU state machine and default LLM tenant lifecycle.

    GPU state machine:
      LLM_LOADED → DRAINING → GPU_WORK → RESTORING → LLM_LOADED
      GPU_WORK → LLM_RELIEF → DRAINING → GPU_WORK

    Qwen3.6 is the default GPU tenant. ComfyUI and ACE-Step evict it
    when they need the GPU, and it's restored when they're done.
    """

    def __init__(self, session: ClientSession, logger: logging.Logger):
        self.session = session
        self.logger = logger
        self.gpu_state = GpuState.LLM_LOADED
        self._llm_evicted = False
        # B4: a failed restore keeps _llm_evicted and retries with backoff.
        self._restore_failures = 0
        self._restore_retry_at = 0.0
        # C3 BUGFIX: per-GPU eviction flag — eviction is per-GPU,
        # not global. A second call to evict a different GPU must
        # not short-circuit because GPU A was already evicted.
        self._gpu_evicted: dict[str, bool] = {}
        self._comfyui_stopped = False
        # ComfyUI idle auto-stop (VRAM conservation)
        self._comfyui_idle_since: float | None = None
        self._comfyui_auto_stopped: bool = False
        # Monotonic time of the last "stopped, explicit start needed" notice.
        self._comfyui_stopped_notice_at: float | None = None
        self._evict_lock = asyncio.Lock()

    def mark_llm_evicted(self, value: bool, source: str = "unknown") -> None:
        """Single-source setter for self._llm_evicted.

        No-op when value is unchanged (avoids spurious event noise).
        """
        value = bool(value)
        if value == self._llm_evicted:
            return
        old = self._llm_evicted
        self._llm_evicted = value
        if self.logger:
            self.logger.info(
                f"VRAM: _llm_evicted {old} -> {self._llm_evicted} (source={source})"
            )

    def status(self) -> dict:
        runtime = _get_loaded_runtime_snapshot()
        configured_idle_service = _get_idle_service_name()
        return {
            "gpu_state": self.gpu_state.value,
            "llm_evicted": self._llm_evicted,
            # Legacy alias: this is the configured/default idle service unit,
            # not proof that the unit is currently loaded.
            "llm_service": configured_idle_service,
            "configured_idle_service": configured_idle_service,
            "loaded_runtime_services": runtime["services"],
            "loaded_runtime_bundles": runtime["bundles"],
            "llm_service_semantics": "configured_idle_service_legacy_alias",
            "comfyui_stopped": self._comfyui_stopped,
            "comfyui_auto_stopped": self._comfyui_auto_stopped,
            "comfyui_idle_seconds": (
                time.time() - self._comfyui_idle_since
                if self._comfyui_idle_since else 0
            ),
        }

    # ── Drain before evict ─────────────────────────────────────────

    async def drain_llm(self, timeout: float = DRAIN_TIMEOUT) -> bool:
        """Wait for active LLM requests to finish before evicting.

        Returns True if all requests drained, False if timeout.
        New LLM requests are already queued (not forwarded) because
        the proxy checks _llm_healthy which is set to False during drain.
        """
        if llm.active_requests == 0:
            self.logger.info("Drain: no active LLM requests — immediate")
            # Mark unhealthy so new requests queue instead of forwarding
            llm.update_health(False)
            return True

        self.logger.info(
            f"Drain: waiting for {llm.active_requests} active LLM request(s) "
            f"(timeout={timeout}s)"
        )
        # Mark unhealthy so new requests queue instead of forwarding
        llm.update_health(False)

        start = time.time()
        while time.time() - start < timeout:
            if llm.active_requests == 0:
                self.logger.info(
                    f"Drain: all requests finished in {time.time() - start:.1f}s"
                )
                return True
            self.logger.debug(f"Drain: {llm.active_requests} still active")
            await asyncio.sleep(2)

        self.logger.warning(
            f"Drain: timeout after {timeout}s — {llm.active_requests} still active"
        )
        return False

    # ── LLM evict/restore ──────────────────────────────────────────

    async def evict_llm(self, reason: str = "gpu_work_start", target_gpu: str = None,
                        skip_bundle_name: str | None = None,
                        admission_authorized: bool = False) -> bool:
        """Stop ALL GPU-using LLM services to free VRAM.

        BUGFIX (2026-05-20):
        1. Previously only stopped the idle_service (gemma, CPU-only).
           Now stops ALL services with HIP_VISIBLE_DEVICES=0 to free VRAM.
        2. systemctl stop sometimes leaves orphan processes (PyTorch cache).
           Force-kill by port if systemctl doesn't free the port.
        3. Restart acestep-api after eviction to clear PyTorch VRAM cache.

        MAINTENANCE MODE: Skips eviction entirely — LLM stays loaded.
        """
        if bundle_lifecycle_controller is None:
            # A direct or queue-owned eviction must never fall through to the
            # legacy systemd implementation when the lifecycle owner is gone.
            # Queue admission still passes the explicit authorization flag,
            # but controller availability remains a hard prerequisite here.
            self.logger.error(
                "VRAM: refusing LLM eviction without lifecycle controller (%s)",
                reason,
            )
            return False
        if not admission_authorized:
            return await _run_standalone_llm_eviction(
                self,
                reason=reason,
                target_gpu=target_gpu,
                skip_bundle_name=skip_bundle_name,
            )
        config = _load_services_config()
        if _maintenance_mode_enabled(config.get("scheduling", {})):
            self.logger.info(f"VRAM: Maintenance mode — skipping LLM eviction ({reason})")
            return False

        # ── Phase 2: Per-GPU priority-based eviction ─────────────────
        # If target_gpu is specified, check if the GPU's current tenant
        # has pending work or is higher priority than idle.
        if target_gpu and target_gpu in scheduler._gpu_state:
            gpu_st = scheduler._gpu_state[target_gpu]
            idle_bundle = _get_idle_bundle(target_gpu)
            idle_priority = idle_bundle.get("priority", 0) if idle_bundle else 0

            # If the GPU is already draining, wait for it to finish
            if gpu_st.get("draining"):
                self.logger.info(
                    f"VRAM: GPU {target_gpu} already draining — waiting for drain to complete"
                )
                gpu_bundle_state = _get_gpu_state(target_gpu)
                await gpu_bundle_state.wait_drain(timeout=DRAIN_TIMEOUT)
                gpu_st["draining"] = False

            # Direct callers preserve the conservative priority fence.  The
            # linked-bundle admission transaction has already fenced every
            # target GPU, drained the relevant logical group, and selected a
            # real durable head job; it must not re-interpret an idle/static
            # priority as permanent residency.
            if not admission_authorized and gpu_st.get("current_priority", 0) > idle_priority:
                pending = scheduler._get_pending_for_gpu(target_gpu)
                higher = [j for j in pending if j.priority > gpu_st["current_priority"]]
                if not higher:
                    # Current work is higher priority — don't evict, just return
                    self.logger.info(
                        f"VRAM: GPU {target_gpu} has higher-priority bundle "
                        f"(priority={gpu_st['current_priority']}) — not evicting"
                    )
                    return False

        async with self._evict_lock:
            # C3 BUGFIX: resolve work_gpu BEFORE the early-return so we can
            # check the per-GPU eviction flag, not the global one. (Eviction
            # is per-GPU: evicting GPU A must not block eviction of GPU B.)
            work_gpu = (
                target_gpu
                or (getattr(self, "_evicted_gpu_id", None) or None)
            )
            # BUGFIX: _current_backend lives on GPUScheduler, not on
            # VRAMLifecycleManager. Use getattr() to be defensive.
            if not work_gpu and getattr(self, '_current_backend', None):
                # Try to infer from the active backend's service config.
                _cfg = _load_services_config()
                _svc_map = _cfg.get("services", {})
                _cur = self._current_backend
                _prefix = _cur.split("-")[0] if "-" in _cur else _cur
                for _sk, _sc in _svc_map.items():
                    if (
                        _sc.get("type") == "generation_backend"
                        and _sc.get("systemd_unit", "").startswith(_prefix)
                    ):
                        work_gpu = _sc.get("gpu_id")
                        break
            if not work_gpu:
                # Fallback: assume work is on the primary GPU (gpu_primary)
                work_gpu = "gpu_primary"

            if self._gpu_evicted.get(work_gpu):
                # Phase 5.2.50: stale-flag fix. Verify against actual VRAM
                # before short-circuiting. The flag can be set during a prior
                # crash-detection cycle that didn't actually free VRAM.
                # If GPU is still occupied, clear the stale flag and proceed
                # with real eviction.
                try:
                    vram_data = await _poll_vram()
                    matched = _get_vram_entry_for_gpu(vram_data, work_gpu)
                    if matched and matched.get("vram_used_mib", 0) > 2048:
                        self.logger.warning(
                            f"VRAM: stale _gpu_evicted[{work_gpu}]=True "
                            f"but {work_gpu} still using "
                            f"{matched.get('vram_used_mib')} MiB — "
                            f"clearing flag and proceeding with eviction"
                        )
                        self._gpu_evicted[work_gpu] = False
                except Exception as e:
                    self.logger.debug(f"evict_llm stale-flag check failed: {e}")

                if self._gpu_evicted.get(work_gpu):
                    self.logger.info(
                        f"VRAM: LLM already evicted on GPU {work_gpu} ({reason})"
                    )
                    return True

            self.logger.info(f"VRAM: Evicting GPU LLM services on {work_gpu or 'unknown'} ({reason})")
            self._evicted_gpu_id = work_gpu  # Track for targeted restore

            # BUGFIX (2026-05-20b): Dynamic service discovery.
            # Previous code hardcoded specific service names
            # and missed gemma-e4b-pi (port 8101, 7.3 GB VRAM!) → OOM.
            # Now we stop ALL services with priority < 10 (LLMs/assistants).
            config = _load_services_config()
            svc_map = config.get("services", {})
            gpu_services = []  # (unit_name, port, svc_key)
            for svc_key, svc_cfg in svc_map.items():
                if not svc_cfg.get("enabled", True):
                    continue
                if svc_cfg.get("priority", 99) >= 10:
                    continue  # Skip GPU work backends (ComfyUI=10, ACE-Step=20)
                if svc_key == "gpu-manager":
                    continue  # NEVER stop ourselves!
                if svc_cfg.get("cpu_only"):
                    continue  # Skip CPU-only services (gemma-vision)
                # MULTI-GPU FIX: Only stop services on the same GPU as the work
                svc_gpu = svc_cfg.get("gpu_id", "gpu_primary")  # Default to primary GPU
                if svc_gpu and work_gpu and svc_gpu != work_gpu:
                    self.logger.info(f"VRAM: Skipping {svc_key} (on {svc_gpu}, work on {work_gpu})")
                    continue
                unit = svc_cfg.get("systemd_unit", f"{svc_key}.service")
                port = svc_cfg.get("port", 0)
                gpu_services.append((unit, port, svc_key))

            # Phase 3: Use _unload_bundle to stop bundles on the target GPU
            bundles = _resolve_bundles()
            gpu_bundles = [
                (name, b) for name, b in bundles.items()
                if b.get("gpu_id") == work_gpu
            ]
            # Phase 5.1.6: skip the bundle we're about to (re)load.
            # Without this, _evict_then_load_bundle races itself:
            # unload stops the service, then _load_bundle starts it,
            # and the start can be killed by a queued stop from the
            # unload path. The bundle load then times out waiting for
            # a port that keeps getting killed.
            if skip_bundle_name is not None:
                before = len(gpu_bundles)
                gpu_bundles = [
                    (n, b) for n, b in gpu_bundles if n != skip_bundle_name
                ]
                if before != len(gpu_bundles):
                    self.logger.info(
                        f"VRAM: skip_bundle_name={skip_bundle_name!r} "
                        f"({before} -> {len(gpu_bundles)} bundles to unload)"
                    )
            if gpu_bundles:
                self.logger.info(
                    f"VRAM: Unloading {len(gpu_bundles)} bundle(s) on GPU {work_gpu}: "
                    f"{[n for n, _ in gpu_bundles]}"
                )
                # B3: a reservation-only share (no services, e.g. Qwen-Dual-mi50) is
                # held by its owner's process; evicting this GPU means stopping the owner.
                targets: list[str] = []
                for name, b in gpu_bundles:
                    owners = [] if b.get("services") else [
                        o for o in (b.get("linked_from") or [])
                        if o in bundles and o != skip_bundle_name
                    ]
                    for target in owners or [name]:
                        if target not in targets:
                            targets.append(target)
                for bundle_name in targets:
                    # B1: drain broker-owned members on every eviction path, not only the fallback.
                    for svc_key in bundles[bundle_name].get("services") or []:
                        if not await _drain_combined_gemma_member_before_stop(svc_key):
                            return False
                    # B2: a failed unload leaves VRAM occupied; never mark the GPU evicted.
                    if not await _unload_bundle(bundle_name):
                        self.logger.error(
                            f"VRAM: unload of bundle {bundle_name} failed; eviction on {work_gpu} aborted"
                        )
                        return False
            else:
                # Fallback: no bundles defined — stop services directly
                self.logger.info(
                    f"VRAM: No bundles for GPU {work_gpu}, "
                    f"stopping {len(gpu_services)} service(s) directly"
                )
                for unit, port, svc_key in gpu_services:
                    try:
                        if not await _drain_combined_gemma_member_before_stop(svc_key):
                            return False
                        # Phase 5.1.5: this stop is intentional (we are
                        # evicting the LLM to free VRAM). Stamp it so the
                        # Supervisor doesn't tear down an unrelated pool
                        # whose service briefly looked unhealthy during
                        # the cascade.
                        if event_bus is not None:
                            event_bus.mark_planned_stop(svc_key)
                        result = await asyncio.create_subprocess_exec(
                            "sudo", "systemctl", "stop", unit,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await result.wait()
                        self.logger.info(f"VRAM: systemctl stop {unit}")
                    except Exception as e:
                        self.logger.warning(f"VRAM: Failed to stop {unit}: {e}")
                    finally:
                        # Phase 5.2.6: caller-side clear. Once the
                        # stop returns (success or failure), release
                        # the supervisor's deferral so a real crash
                        # is not silently swallowed.
                        if event_bus is not None:
                            event_bus.clear_planned_stop(svc_key)

            # Wait for ports to free, force-kill if needed
            # (Still needed for services not managed by bundles, or stubborn processes)
            for unit, port_num, svc_name in gpu_services:
                if not port_num:
                    continue
                freed = await self._wait_for_port_free(timeout=15, port=port_num)
                if freed:
                    self.logger.info(f"VRAM: {svc_name} port {port_num} freed")
                else:
                    # BUGFIX (2026-05-20c): Must use sudo to kill system service processes.
                    # gpu-manager runs as operator, service processes run as root.
                    # Sequence: stop → kill → stop (to block Restart=on-failure) → reset-failed.
                    # Without the second stop, Restart=on-failure (default 100ms) would restart
                    # the service before we can reset-failed.
                    self.logger.warning(f"VRAM: Force-killing {svc_name} (unit={unit}, port {port_num})")
                    try:
                        # Step 1: SIGKILL the process through systemd
                        kill_result = await asyncio.create_subprocess_exec(
                            "sudo", "systemctl", "kill", unit,
                            "--signal=SIGKILL",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await kill_result.wait()
                        # Step 2: Immediately issue 'stop' to block Restart=on-failure trigger.
                        # systemd won't restart a stopped service even if Restart= is set.
                        stop_result = await asyncio.create_subprocess_exec(
                            "sudo", "systemctl", "stop", unit,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await stop_result.wait()
                        await asyncio.sleep(1)
                        # Step 3: Clear the failed state so future starts work cleanly
                        reset_result = await asyncio.create_subprocess_exec(
                            "sudo", "systemctl", "reset-failed", unit,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await reset_result.wait()
                        self.logger.info(f"VRAM: {svc_name} killed and blocked from restart")
                    except Exception:
                        pass

            # BUGFIX (2026-05-20c): Wait for VRAM to actually free before starting ACE-Step.
            # Even after fuser -k, the GPU driver takes a few seconds to reclaim VRAM.
            # ACE-Step needs ~10 GB, so we need VRAM used to drop below ~4 GB.
            self.logger.info("VRAM: Waiting for GPU VRAM to free after service stops...")
            vram_freed = await self._wait_for_vram_below(
                target_mib=4096, timeout=30, gpu_id=work_gpu
            )
            if not vram_freed:
                self.logger.warning("VRAM: GPU VRAM did not free within timeout — proceeding anyway")

            # Cache ownership belongs to the drained runtime. Never start a
            # retired ACE engine as a side effect of switching other services.

            self.mark_llm_evicted(True, source="evict_llm")
            # C3 BUGFIX: also mark this GPU as evicted per-GPU so the
            # early-return check in evict_llm knows about it, and so a
            # future call for a DIFFERENT GPU isn't short-circuited.
            if work_gpu:
                self._gpu_evicted[work_gpu] = True
            llm.update_health(False)
            # Phase 2/Phase 0: Update per-GPU state after eviction.
            # GpuBundleState is the single source of truth — setters sync to
            # the legacy scheduler._gpu_state dict automatically.
            if work_gpu:
                bundle_state = _get_gpu_state(work_gpu)
                bundle_state.loaded_bundle = None
                bundle_state.bundle_priority = 0
                bundle_state.set_evicting(True, source="evict_llm")
                bundle_state.set_draining(False, source="evict_llm")
                bundle_state.set_loaded_bundles([], source="evict_llm")
                bundle_state.set_current_priority(0, source="evict_llm")
            self.logger.info(f"VRAM: LLM evicted ({reason}) — GPU {work_gpu} cleared")
            return True


    async def restore_llm(self, reason: str = "gpu_work_done") -> bool:
        """Restart ALL GPU LLM services after GPU work completes."""
        async with _RestoreAdmissionGuard(self):
            if not self._llm_evicted:
                self.logger.info(f"VRAM: LLM not evicted, skip restore ({reason})")
                return True
            if time.monotonic() < self._restore_retry_at:
                self.logger.info(
                    f"VRAM: restore backing off after {self._restore_failures} failure(s) ({reason})"
                )
                return False

            self.logger.info(f"VRAM: Restoring all GPU LLM services ({reason})")

            # This call is made by the scheduler-owned restoration
            # transaction. Its own guard must not deadlock the nested VRAM
            # restore, while all real generation/admission blockers remain
            # authoritative.
            release_blockers = _generation_release_blockers(
                ignore_scheduler_restoring=True
            )
            if release_blockers:
                self.logger.info(
                    "VRAM: restore deferred until generation release: %s",
                    ",".join(release_blockers),
                )
                return False

            # The generic release fence above established that no generation
            # owner remains. Do not wait on a hard-coded backend port here:
            # an unrelated stopped service would otherwise add a full timeout
            # to every idle restore.
            await asyncio.sleep(2)

            # MULTI-GPU FIX: restore every configured per-GPU idle bundle,
            # not only the most recently evicted GPU.  Dual-GPU generation
            # paths (Krea2 = comfyui on gpu_primary + encoder reservation on
            # gpu_secondary) call evict_llm once per target GPU; _evicted_gpu_id is a
            # scalar and can only remember one of them.  Restoring only that
            # GPU left the other idle service to crash-recovery, producing a
            # long unhealthy window and stale scheduler state.  The user's
            # intended post-generation state is the combined Gemma endpoint,
            # so bring all configured idle GPU services back atomically here.
            config = _load_services_config()
            svc_map = config.get("services", {})
            restored_gpus: set[str] = set()
            restored_names: list[str] = []
            restored_member_names: list[str] = []
            healthy = True

            idle_entries = _get_all_idle_services()
            if idle_entries:
                for idle_entry in idle_entries:
                    idle_gpu = idle_entry.get("gpu_id")
                    idle_svc_name = idle_entry.get("service_name")
                    idle_bundle = _get_idle_bundle(idle_gpu) if idle_gpu else None
                    if idle_bundle:
                        idle_name = idle_bundle.get("bundle_name", idle_svc_name or "idle")
                        self.logger.info(
                            f"VRAM: Restoring idle bundle '{idle_name}' on GPU {idle_gpu}"
                        )
                        load_ok = await _run_bundle_transition(
                            idle_name,
                            lambda: _load_bundle(
                                idle_name, timeout=LLM_RESTORE_TIMEOUT
                            ),
                            desired_state=BundleDesiredState.RUNNING,
                            reason=f"restore-llm:{reason}",
                        )
                        if load_ok:
                            self.logger.info(
                                f"VRAM: Idle bundle '{idle_name}' restored and healthy"
                            )
                            if idle_gpu:
                                restored_gpus.add(idle_gpu)
                            restored_names.append(idle_name)
                            if idle_svc_name:
                                restored_member_names.append(idle_svc_name)
                                if not await _rejoin_combined_gemma_member_after_start(
                                    idle_svc_name, timeout=LLM_RESTORE_TIMEOUT
                                ):
                                    healthy = False
                        else:
                            healthy = False
                            self.logger.warning(
                                f"VRAM: Idle bundle '{idle_name}' not healthy "
                                f"after {LLM_RESTORE_TIMEOUT}s"
                            )
                    else:
                        # Fallback: no idle bundle declared for this entry;
                        # start its unit directly and probe its configured port.
                        unit = idle_entry.get("systemd_unit") or ""
                        port = idle_entry.get("port") or 0
                        if not unit:
                            healthy = False
                            self.logger.warning(
                                f"VRAM: No idle bundle/unit for GPU {idle_gpu} "
                                f"service {idle_svc_name}"
                            )
                            continue
                        try:
                            result = await asyncio.create_subprocess_exec(
                                "systemctl", "start", unit,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            await result.wait()
                            self.logger.info(f"VRAM: Started {unit}")
                            probe_ok = await _probe_health_inline(
                                port, health_path=(svc_map.get(idle_svc_name, {}) or {}).get("health_path") or "/health",
                                timeout=2.0,
                            )
                            if probe_ok:
                                if idle_gpu:
                                    restored_gpus.add(idle_gpu)
                                restored_names.append(idle_svc_name or unit)
                                if idle_svc_name:
                                    restored_member_names.append(idle_svc_name)
                                    if not await _rejoin_combined_gemma_member_after_start(
                                        idle_svc_name, timeout=LLM_RESTORE_TIMEOUT
                                    ):
                                        healthy = False
                            else:
                                healthy = False
                        except Exception as e:
                            healthy = False
                            self.logger.warning(f"VRAM: Failed to start {unit}: {e}")
            elif "idle_services" in (config.get("scheduling") or {}):
                # B4: an explicitly empty map means no GPU idle tenant; the
                # legacy discovery below would `systemctl start` cloud members'
                # router unit.
                self.logger.info(f"VRAM: idle_services is empty, nothing to restore ({reason})")
            else:
                # Last-resort legacy fallback if scheduling.idle_services is
                # missing: preserve the previous dynamic service discovery.
                gpu_services = []
                for svc_key, svc_cfg in svc_map.items():
                    if not svc_cfg.get("enabled", True):
                        continue
                    if svc_cfg.get("priority", 99) >= 10:
                        continue
                    if svc_key == "gpu-manager" or svc_cfg.get("cpu_only"):
                        continue
                    unit = svc_cfg.get("systemd_unit", f"{svc_key}.service")
                    port = svc_cfg.get("port", 0)
                    gpu_services.append((unit, port, svc_key))
                self.logger.info(
                    f"VRAM: No idle_services map, starting {len(gpu_services)} legacy service(s) directly"
                )
                for svc, svc_port, svc_key in gpu_services:
                    if _combined_gemma_broker_owns_service(
                        svc_key, svc_map.get(svc_key, {}) or {}
                    ):
                        restored_member_names.append(svc_key)
                    try:
                        result = await asyncio.create_subprocess_exec(
                            "systemctl", "start", svc,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await result.wait()
                        self.logger.info(f"VRAM: Started {svc}")
                    except Exception as e:
                        healthy = False
                        self.logger.warning(f"VRAM: Failed to start {svc}: {e}")
                healthy = healthy and await self._wait_for_llm_healthy(timeout=LLM_RESTORE_TIMEOUT)
                if healthy:
                    for member_name in restored_member_names:
                        if not await _rejoin_combined_gemma_member_after_start(
                            member_name, timeout=LLM_RESTORE_TIMEOUT
                        ):
                            healthy = False

            # B4: an unhealthy restore stays evicted so a later call retries it.
            self.mark_llm_evicted(not healthy, source="restore_llm" if healthy else "restore_llm_unhealthy")
            if healthy:
                self._restore_failures, self._restore_retry_at = 0, 0.0
            else:
                self._restore_failures += 1
                delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (self._restore_failures - 1))
                self._restore_retry_at = time.monotonic() + delay
                self.logger.warning(
                    f"VRAM: restore unhealthy ({reason}); retry in {delay:.0f}s "
                    f"(failure {self._restore_failures})"
                )
            # Clear every per-GPU eviction flag we restored.  Older code only
            # cleared the scalar _evicted_gpu_id, which left the other GPU in a
            # dual-GPU generation path looking evicted/stale until a later
            # watchdog pass.
            for restored_gpu in restored_gpus:
                self._gpu_evicted.pop(restored_gpu, None)

            # Phase 2/Phase 0: Update per-GPU state after restore. _load_bundle
            # already stamps most of this, but doing it explicitly here keeps
            # vram.restore_llm() as a complete state transition and repairs
            # stale state from reservation-only bundles like krea2-encoder.
            for restored_gpu in restored_gpus:
                idle_bundle = _get_idle_bundle(restored_gpu)
                idle_name = idle_bundle.get("bundle_name", "idle") if idle_bundle else "idle"
                idle_priority = idle_bundle.get("priority", 1) if idle_bundle else 1
                bundle_state = _get_gpu_state(restored_gpu)
                bundle_state.loaded_bundle = idle_name
                bundle_state.bundle_priority = idle_priority
                bundle_state.state = "loaded"
                bundle_state.set_evicting(False, source="restore_llm")
                bundle_state.set_draining(False, source="restore_llm")
                bundle_state.set_loaded_bundles([idle_name], source="restore_llm")
                bundle_state.set_current_priority(idle_priority, source="restore_llm")
                self.logger.info(
                    f"VRAM: GPU {restored_gpu} restored with idle bundle "
                    f"'{idle_name}' (priority={idle_priority})"
                )
            self.logger.info(
                f"VRAM: Restored idle bundle set: {restored_names}"
            )
            return healthy

    async def _comfyui_systemd_state(self) -> dict:
        """Return the authoritative systemd state per ComfyUI systemd unit.

        Slice B: previously this returned a single string naming only the
        generic ``comfyui.service``. That allowed the scheduler to
        conclude "ComfyUI is stopped" while ``comfyui-qwen.service``
        was still active — the Krea/Qwen serialization bug. The helper
        now reports BOTH the Krea and Qwen units so reconciliation and
        the running-state commit can distinguish "both off" from
        "qwen active, krea inactive".

        Returns a dict like::

            {
                "comfyui.service": "inactive" | "active" | ...,
                "comfyui-qwen.service": "inactive" | "active" | ...,
            }
        """
        config = _load_services_config() or {}
        services = config.get("services") or {}
        krea_unit = str(
            (services.get("comfyui") or {}).get("systemd_unit")
            or "comfyui.service"
        )
        qwen_unit = str(
            (services.get("comfyui-qwen") or {}).get("systemd_unit")
            or "comfyui-qwen.service"
        )
        units = tuple(dict.fromkeys((krea_unit, qwen_unit)))
        states: dict[str, str] = {}
        for unit in units:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "systemctl", "is-active", unit,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                state = stdout.decode(errors="replace").strip().lower()
                states[unit] = state or "unknown"
            except Exception as exc:
                self.logger.warning(
                    f"VRAM: unable to read ComfyUI systemd state for {unit}: {exc}"
                )
                states[unit] = "unknown"
        # Keep canonical legacy keys as read-only aliases for older status
        # consumers; lifecycle authority remains the resolved registry unit.
        states.setdefault("comfyui.service", states.get(krea_unit, "unknown"))
        states.setdefault("comfyui-qwen.service", states.get(qwen_unit, "unknown"))
        return states

    @staticmethod
    def _checked_comfyui_runtime_configured() -> bool:
        """Return whether ComfyUI is bound to a checked host runtime."""
        config = _load_services_config() or {}
        bundles = config.get("bundles") or {}
        for name in ("comfyui", "qwen-edit-dualgpu"):
            bundle = bundles.get(name) if isinstance(bundles, dict) else None
            if isinstance(bundle, dict) and bundle.get("runtime_instance") and bundle.get(
                "runtime_profile"
            ):
                return True
        return False

    @staticmethod
    def _configured_comfyui_unit(service_name: str) -> str:
        config = _load_services_config() or {}
        services = config.get("services") or {}
        fallback = {
            "comfyui": "comfyui.service",
            "comfyui-qwen": "comfyui-qwen.service",
        }.get(service_name, f"{service_name}.service")
        return str((services.get(service_name) or {}).get("systemd_unit") or fallback)

    @staticmethod
    def _comfyui_unit_state(states: dict[str, str], unit: str) -> str:
        """Return one exact ComfyUI unit state from an authoritative snapshot.

        A string is accepted only as legacy Krea input for older internal test
        doubles; live probes always return the per-unit mapping.
        """
        if isinstance(states, str):
            return states.lower() if unit == "comfyui.service" else "unknown"
        if not isinstance(states, dict):
            return "unknown"
        return str(states.get(unit, "unknown") or "unknown").lower()

    @classmethod
    def _all_comfyui_units_stopped(cls, states: dict[str, str]) -> bool:
        """Return True only when both Krea and Qwen are authoritatively off."""
        stopped = {"inactive", "failed"}
        return all(
            cls._comfyui_unit_state(states, unit) in stopped
            for unit in ("comfyui.service", "comfyui-qwen.service")
        )

    @classmethod
    def _any_comfyui_unit_active(cls, states: dict[str, str]) -> bool:
        """Return True when either exact ComfyUI unit is active."""
        return any(
            cls._comfyui_unit_state(states, unit) == "active"
            for unit in ("comfyui.service", "comfyui-qwen.service")
        )

    async def mark_comfyui_running_after_commit(self, unit: str = "comfyui.service") -> bool:
        """Clear stopped caches only after systemd confirms committed ownership.

        Slice B: takes the unit name explicitly so callers can commit the
        running state of either Krea (``comfyui.service``) or Qwen
        (``comfyui-qwen.service``). The previous single-unit implementation
        would silently accept a Qwen-only active state as a Krea-active
        commit, leading to false "Krea running" claims.
        """
        states = await self._comfyui_systemd_state()
        state = self._comfyui_unit_state(states, unit)
        if state != "active":
            self.logger.error(
                "VRAM: refusing running-state commit; %s systemd state=%r (all=%r)",
                unit, state, states,
            )
            return False
        self._comfyui_stopped = False
        self._comfyui_auto_stopped = False
        self._comfyui_idle_since = time.time()
        self.logger.info(
            "VRAM: ComfyUI running-state committed after authoritative bundle commit (unit=%s)",
            unit,
        )
        return True

    async def _unload_comfyui_models_for_restore(self) -> bool:
        """Use ComfyUI's bounded /free API without changing backend ownership."""
        result = await _free_comfyui_models_for_oom("comfyui", timeout_s=10.0)
        if not result.get("ok"):
            self.logger.error("ComfyUI in-process model unload failed: %s", result)
            return False
        self.logger.info("ComfyUI models unloaded in-process for VRAM restoration")
        return True

    async def _restart_acestep_for_restore(self):
        """Compatibility hook: ACE is retired; checked unload owns release."""
        self.logger.debug("Retired ACE restore hook skipped")

    # ── ComfyUI lifecycle (for pipeline LLM speed boost) ──────────

    async def stop_comfyui(self, reason: str = "pipeline_llm") -> bool:
        """Stop Krea ComfyUI and update the shared cache from both exact units."""
        if self._checked_comfyui_runtime_configured():
            try:
                unit_states = await self._comfyui_systemd_state()
                self._comfyui_stopped = self._all_comfyui_units_stopped(unit_states)
                self._comfyui_auto_stopped = self._comfyui_stopped
                if self._comfyui_stopped:
                    self._comfyui_idle_since = time.time()
            except Exception as exc:
                self.logger.debug(
                    "VRAM: checked ComfyUI state sync during stop skipped: %s",
                    exc,
                )
            self.logger.info(
                "VRAM: checked ComfyUI lifecycle is controller-owned; "
                "legacy stop helper deferred (%s)", reason
            )
            return True
        unit_states = await self._comfyui_systemd_state()
        krea_state = self._comfyui_unit_state(
            unit_states, "comfyui.service"
        )
        if krea_state in {"inactive", "failed"}:
            self._comfyui_stopped = not self._any_comfyui_unit_active(unit_states)
            self.logger.info(
                "VRAM: Krea ComfyUI already stopped "
                f"({reason}, state={krea_state}, all={unit_states!r})"
            )
            return True

        if self._comfyui_stopped:
            self.logger.warning(
                "VRAM: cached ComfyUI stopped state disagrees with Krea systemd "
                f"state={krea_state!r} (all={unit_states!r}); issuing "
                "authoritative stop"
            )
            self._comfyui_stopped = False

        self.logger.info(f"VRAM: Stopping ComfyUI for LLM speed ({reason})")
        await self._free_comfyui()
        await asyncio.sleep(2)

        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "stop", "comfyui.service",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                self.logger.error(f"VRAM: ComfyUI stop failed: {stderr.decode().strip()}")
                return False
            if event_bus:
                event_bus.emit("comfyui", "unhealthy")
        except Exception as e:
            self.logger.error(f"VRAM: ComfyUI stop error: {e}")
            return False

        freed = await self._wait_for_comfyui_port_free(timeout=30)
        if not freed:
            self.logger.warning("VRAM: ComfyUI port still in use after 30s")

        unit_states = await self._comfyui_systemd_state()
        krea_state = self._comfyui_unit_state(
            unit_states, "comfyui.service"
        )
        if krea_state not in {"inactive", "failed"}:
            self.logger.error(
                "VRAM: Krea ComfyUI stop command returned but authoritative "
                f"unit state is {krea_state!r} (all={unit_states!r}); refusing "
                "stopped-state commit"
            )
            self._comfyui_stopped = False
            return False

        self._comfyui_stopped = not self._any_comfyui_unit_active(unit_states)
        self.logger.info(
            "VRAM: Krea ComfyUI service authoritatively stopped, ~9.8GB VRAM "
            "freed for LLM"
        )
        return True

    async def start_comfyui(self, reason: str = "pipeline_gpu") -> bool:
        """Compatibility start entry point backed by the managed controller."""
        self.logger.info("VRAM: requesting managed ComfyUI start (%s)", reason)
        result = await _start_comfyui_via_controller()
        return bool(isinstance(result, dict) and result.get("success") is True)

    async def _ensure_comfyui_running(self):
        """Report whether ComfyUI may receive dispatched jobs; never starts it.

        Called when the scheduler needs ComfyUI for work but it may have
        been auto-stopped or stopped after a failure. A stopped backend is
        started only by the explicit authenticated controller route, so this
        returns False immediately. The scheduler retries about once a second,
        so the notice is rate-limited rather than repeated on every tick.
        """
        if not self._comfyui_stopped and not self._comfyui_auto_stopped:
            self._comfyui_stopped_notice_at = None
            return True  # Already running
        now = time.monotonic()
        last = self._comfyui_stopped_notice_at
        if last is None or now - last >= COMFYUI_STOPPED_NOTICE_INTERVAL_S:
            self._comfyui_stopped_notice_at = now
            self.logger.warning(
                "ComfyUI is stopped; queued generation waits for the explicit "
                "authenticated /vram/start-comfyui controller route "
                "(no automatic start)"
            )
        return False

    # ── Internal helpers ───────────────────────────────────────────

    async def _check_service_active(self, service_name: str = None) -> bool:
        name = service_name or _get_idle_service_name()
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", "--quiet", name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception as e:
            self.logger.error(f"VRAM: systemctl check failed: {e}")
            return False

    async def _systemctl(self, action: str, service_name: str = None) -> bool:
        name = service_name or _get_idle_service_name()
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", action, name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                self.logger.error(
                    f"VRAM: systemctl {action} failed (rc={proc.returncode}): "
                    f"{stderr.decode().strip()}"
                )
                return False
            self.logger.info(f"VRAM: systemctl {action} {name} succeeded")
            return True
        except asyncio.TimeoutError:
            self.logger.error(f"VRAM: systemctl {action} timed out")
            return False
        except Exception as e:
            self.logger.error(f"VRAM: systemctl {action} error: {e}")
            return False

    async def _wait_for_port_free(self, timeout: float = 30, port: int = None) -> bool:
        if port is None:
            port = _get_idle_service_port()
        url = f"http://localhost:{port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with self.session.get(
                    url,
                    timeout=ClientTimeout(total=2, connect=1),
                ) as resp:
                    pass
            except Exception:
                return True
            await asyncio.sleep(2)
        return False

    async def _wait_for_vram_below(self, target_mib: int = 4096, timeout: float = 30,
                                  *, gpu_id: str | None = None) -> bool:
        """Wait on the evicted GPU; other GPUs may still serve live work."""
        start = time.time()
        while time.time() - start < timeout:
            vram_data = await _poll_vram()
            if gpu_id:
                entry = _get_vram_entry_for_gpu(vram_data, gpu_id) or {}
                used = entry.get("vram_used_mib", entry.get("used_mib", 999999))
            else:
                used = vram_data.get("used_vram_mib", 999999)
            if not isinstance(used, (int, float)) or used < 0:
                used = 999999
            if used < target_mib:
                self.logger.info(f"VRAM: Freed — {used} MiB < {target_mib} MiB threshold")
                return True
            self.logger.info(f"VRAM: Still used {used} MiB, waiting... (target <{target_mib})")
            await asyncio.sleep(3)
        return False

    async def wait_for_free_mib(self, gpu_id: str, target_mib: int,
                                timeout_s: float = 5.0,
                                poll_interval_s: float = 0.25) -> bool:
        """Phase 5.2.19: block until `gpu_id` has `target_mib` free VRAM.

        Used after `systemctl start` of a new bundle, before HTTP probe,
        to close the TOCTOU window between process-exit and PyTorch CUDA
        allocator release (verified ~1s in llama.cpp + PyTorch per
        RESEARCH_PHASE_5_2_19_EVIDENCE.md).

        Returns True if target reached within timeout, False otherwise.
        Caller should log on False but NOT fail the load — the OOM
        watchdog will catch real failure (5.2.11 bug pattern).

        Backend dispatch via _detect_backend():
        - rocm:    read _vram_cache["gpus"] by matching card/pci
        - cuda:    nvidia-smi --query-gpu=memory.free (not implemented
                   yet — falls back to True)
        - vulkan:  GEM-fd scan (not implemented — falls back to True)
        - cpu:     always True (no VRAM)

        Per-GPU lookup key (P0-2 from PHASE5.2.19_ADVERSARIAL): services
        gpu_ids are logical names ("gpu_primary", "gpu_secondary"). _poll_vram()
        returns cards by "card" name ("card0", "card1") or pci_address.
        We resolve gpu_id → pci via services.json gpu_devices, then
        match on pci_address.
        """
        deadline = time.monotonic() + timeout_s
        backend = self._detect_backend_for_gpu(gpu_id)
        # CPU-only services (gemma-vision, no gpu_id): always free.
        if backend == "cpu" or not gpu_id:
            return True
        # Resolve gpu_id → pci for matching
        pci = self._pci_for_gpu(gpu_id)
        last_free = 0
        while time.monotonic() < deadline:
            free_mib = await self._free_mib_for_async(gpu_id, pci, backend)
            last_free = free_mib
            if free_mib >= target_mib:
                return True
            await asyncio.sleep(poll_interval_s)
        self.logger.warning(
            f"wait_for_free_mib({gpu_id}): target {target_mib} MiB not "
            f"reached in {timeout_s:.1f}s (last seen: {last_free} MiB)"
        )
        return False

    def _detect_backend_for_gpu(self, gpu_id: str | None) -> str:
        """One of 'rocm' | 'cuda' | 'vulkan' | 'cpu'."""
        if gpu_id in (None, "", "cpu", "cpu_only"):
            return "cpu"
        config = _load_services_config()
        gpu_devices = config.get("gpu_devices", {}) or {}
        dev = gpu_devices.get(gpu_id, {})
        if dev.get("cpu_only"):
            return "cpu"
        # Heuristic: try rocm-smi, then nvidia-smi, then vulkaninfo
        import shutil
        if shutil.which("rocm-smi"):
            return "rocm"
        if shutil.which("nvidia-smi"):
            return "cuda"
        if shutil.which("vulkaninfo"):
            return "vulkan"
        return "cpu"

    def _pci_for_gpu(self, gpu_id: str) -> str | None:
        config = _load_services_config()
        gpu_devices = config.get("gpu_devices", {}) or {}
        return gpu_devices.get(gpu_id, {}).get("pci")

    async def _free_mib_for_async(self, gpu_id: str, pci: str | None,
                                   backend: str) -> int:
        if backend == "rocm":
            vram_data = await _poll_vram()
            gpus = vram_data.get("gpus", []) or []
            target_pci = _normalize_pci_address(pci)
            for gpu in gpus:
                if (
                    target_pci
                    and _normalize_pci_address(gpu.get("pci_address")) == target_pci
                ):
                    used = gpu.get("vram_used_mib")
                    total = gpu.get("vram_total_mib") or 16368
                    if used is None:
                        return 0
                    return max(0, total - used)
            if target_pci:
                return 0
            mapped_card = (
                (_load_services_config().get("gpu_devices", {}) or {})
                .get(gpu_id, {})
                .get("card")
            )
            for gpu in gpus:
                card = gpu.get("card", "")
                if mapped_card and card == mapped_card:
                    used = gpu.get("vram_used_mib")
                    total = gpu.get("vram_total_mib") or 16368
                    return max(0, total - (used or 0))
            return 0
        if backend == "cuda":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
                return int(stdout.decode().strip().splitlines()[0])
            except Exception:
                return 0
        # Vulkan + cpu: best-effort return True semantics upstream.
        return 1 << 30

    async def _wait_for_llm_healthy(self, timeout: float = LLM_RESTORE_TIMEOUT) -> bool:
        port = _get_idle_service_port()
        url = f"http://localhost:{port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with self.session.get(
                    url,
                    timeout=ClientTimeout(total=5, connect=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "ok":
                            llm.update_health(True)
                            return True
            except Exception:
                pass
            await asyncio.sleep(3)
        return False

    async def _free_comfyui(self):
        try:
            async with self.session.post(
                f"{COMFYUI_URL}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=ClientTimeout(total=30),
            ) as resp:
                self.logger.info(f"VRAM: ComfyUI free → status {resp.status}")
        except Exception as e:
            self.logger.error(f"VRAM: ComfyUI free failed: {e}")

    async def _wait_for_comfyui_port_free(self, timeout: float = 30) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with self.session.get(
                    f"{COMFYUI_URL}/system_stats",
                    timeout=ClientTimeout(total=2, connect=1),
                ) as resp:
                    pass
            except Exception:
                return True
            await asyncio.sleep(2)
        return False

    async def _wait_for_comfyui_ready(self, timeout: float = 120) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with self.session.get(
                    f"{COMFYUI_URL}/system_stats",
                    timeout=ClientTimeout(total=5, connect=3),
                ) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(3)
        return False

    def get_gpu_services(self) -> list:
        """Return all GPU-using services sorted by eviction priority (lowest priority = most evictable).
        Skips cpu_only services (they don't use GPU VRAM)."""
        global _services_config
        if _services_config is None:
            _services_config = _load_services_config()
        services = _services_config.get("services", {})
        gpu_svcs = []
        for name, svc in services.items():
            if svc.get("cpu_only", False):
                continue  # CPU-only services don't need VRAM management
            if svc.get("enabled") and svc.get("systemd_unit"):
                gpu_svcs.append((name, svc))
        gpu_svcs.sort(key=lambda x: x[1].get("priority", 0))
        return gpu_svcs

    async def evict_for_vram(self, needed_mib: int) -> list:
        """Evict lowest-priority non-protected services until enough VRAM is free.

        Returns list of evicted service names.
        """
        global _services_config
        if _services_config is None:
            _services_config = _load_services_config()

        evicted = []
        bundles = _resolve_bundles()
        candidates = self.get_gpu_services()
        bundle_candidates = []
        for candidate_name, candidate_svc in candidates:
            if candidate_svc.get("protected", False):
                continue
            if not candidate_svc.get("systemd_unit"):
                continue
            resolved_bundle = _get_bundle_for_service(
                candidate_name, bundles=bundles
            )
            if resolved_bundle is not None:
                bundle_candidates.append((candidate_name, resolved_bundle))

        # A bundle-backed candidate may never be downgraded to the legacy
        # per-service systemd path when the durable owner is unavailable.
        # Preflight the whole candidate set so an unbundled service cannot be
        # stopped first and leave a partial eviction behind a later refusal.
        if bundle_candidates and bundle_lifecycle_controller is None:
            self.logger.error(
                "VRAM: refusing evict_for_vram without lifecycle controller "
                "for bundle-backed candidates: %s",
                [name for name, _ in bundle_candidates],
            )
            return []

        handled_bundle_roots: set[str] = set()
        for name, svc in candidates:
            if svc.get("protected", False):
                continue
            unit = svc.get("systemd_unit")
            if not unit:
                continue

            resolved_bundle = _get_bundle_for_service(name, bundles=bundles)
            if resolved_bundle is not None:
                bundle_name = resolved_bundle["bundle_name"]
                if bundle_name in handled_bundle_roots:
                    continue
                handled_bundle_roots.add(bundle_name)
                self.logger.info(
                    "VRAM: evict_for_vram(%sMiB) — evicting bundle %s "
                    "for service %s (priority=%s)",
                    needed_mib,
                    bundle_name,
                    name,
                    svc.get("priority"),
                )
                try:
                    bundle_ok = await _run_bundle_eviction_for_vram(
                        bundle_name,
                        reason=f"evict-for-vram:{name}:{needed_mib}",
                        bundles=bundles,
                    )
                except Exception as exc:
                    bundle_ok = False
                    self.logger.error(
                        "VRAM: exception evicting bundle %s for %s: %s",
                        bundle_name,
                        name,
                        exc,
                    )
                if bundle_ok:
                    evicted.append(name)
                else:
                    self.logger.warning(
                        "VRAM: controller refused/failed bundle eviction %s",
                        bundle_name,
                    )
                continue

            # This is the genuinely unbundled compatibility path. It remains
            # allowed, but it is deliberately unreachable for any service that
            # resolves through the bundle registry above.
            self.logger.info(f"VRAM: evict_for_vram({needed_mib}MiB) — evicting {name} (priority={svc.get('priority')})")
            try:
                # Phase 5.1.5: stamp this stop as planned (we initiated
                # it to free VRAM). The Supervisor will defer pool
                # shutdown for the planned window so a transient
                # unhealthy probe during the systemd stop cascade does
                # not get misclassified as a crash.
                if event_bus is not None:
                    event_bus.mark_planned_stop(name)
                proc = await asyncio.create_subprocess_exec(
                    "sudo", "systemctl", "stop", unit,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                if event_bus:
                    event_bus.emit(name, "unhealthy")
                if proc.returncode == 0:
                    evicted.append(name)
                    self.logger.info(f"VRAM: evicted {name} for {needed_mib}MiB")
                else:
                    self.logger.warning(f"VRAM: failed to evict {name}: {stderr.decode().strip()}")
            except Exception as e:
                self.logger.error(f"VRAM: exception evicting {name}: {e}")
            finally:
                # Phase 5.2.6: caller-side clear. The stamp's only
                # purpose was to absorb the systemd stop cascade; once
                # the stop returns (success or failure), release the
                # supervisor's deferral so a real crash is not silently
                # swallowed.
                if event_bus is not None:
                    event_bus.clear_planned_stop(name)
        return evicted


def _rebuild(function, facade_globals, class_cell=None):
    closure = function.__closure__
    if closure is not None and class_cell is not None:
        closure = tuple(
            class_cell if name == "__class__" else cell
            for name, cell in zip(function.__code__.co_freevars, closure)
        )
    rebuilt = FunctionType(
        function.__code__,
        facade_globals,
        function.__name__,
        function.__defaults__,
        closure,
    )
    rebuilt.__kwdefaults__ = (
        function.__kwdefaults__.copy() if function.__kwdefaults__ else None
    )
    rebuilt.__annotations__ = function.__annotations__.copy()
    rebuilt.__doc__ = function.__doc__
    rebuilt.__qualname__ = function.__qualname__
    rebuilt.__module__ = facade_globals.get("__name__", function.__module__)
    rebuilt.__dict__.update(function.__dict__)
    return rebuilt


def _rebuild_class(template, facade_globals):
    """Return a fresh class equivalent to ``template`` bound to ``facade_globals``.

    Same mechanism as ``gpu_manager.scheduler_state._rebuild_class``: every
    function in the class dictionary is rebuilt against the facade globals and
    a ``__class__`` cell (zero-argument ``super()``) is re-pointed at the new
    class.  Defaults are carried over from the template unchanged.
    """
    class_cell = CellType()

    def rebuild(function):
        return _rebuild(function, facade_globals, class_cell)

    namespace = {}
    for key, value in vars(template).items():
        if key in ("__dict__", "__weakref__"):
            continue
        if isinstance(value, FunctionType):
            value = rebuild(value)
        elif isinstance(value, (staticmethod, classmethod)):
            value = type(value)(rebuild(value.__func__))
        elif isinstance(value, property):
            value = property(
                rebuild(value.fget) if value.fget else None,
                rebuild(value.fset) if value.fset else None,
                rebuild(value.fdel) if value.fdel else None,
                value.__doc__,
            )
        namespace[key] = value
    namespace["__module__"] = facade_globals.get("__name__", template.__module__)
    bound = type(template)(template.__name__, template.__bases__, namespace)
    class_cell.cell_contents = bound
    return bound


def bind(facade_globals):
    """Return ``VRAMLifecycleManager`` bound to ``facade_globals``.

    The class is new per call, so the facade must call ``bind`` once and keep
    the result.
    """
    if not isinstance(facade_globals, dict):
        raise TypeError("facade must be its module globals dictionary")
    return _rebuild_class(VRAMLifecycleManager, facade_globals)
