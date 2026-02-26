from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from tileflow.config import RuntimeConfig, TuneResult

try:
    import numpy as np
    import torch
except ImportError:  # pragma: no cover
    np = None
    torch = None


@dataclass(slots=True)
class TileTask:
    token_idx: int
    tile_id: str
    predicted_next_tiles: List[str]
    predicted_next_scores: Optional[Dict[str, float]] = None


class ExpertLRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[str, None] = OrderedDict()

    def touch(self, expert_id: str) -> None:
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
            return
        self.cache[expert_id] = None
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def has(self, expert_id: str) -> bool:
        return expert_id in self.cache


class PrefetchScheduler:
    def __init__(
        self,
        cfg: RuntimeConfig,
        tune: TuneResult,
        *,
        real_transfers: bool = False,
        tile_path_resolver: Optional[Callable[[str], Optional[Path]]] = None,
        transfer_cap_mb: int = 16,
    ) -> None:
        self.cfg = cfg
        self.tune = tune
        self.hots = ExpertLRU(capacity=cfg.hot_expert_slots)
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.prefetch_queue_len)
        self.current_prefetch_depth = max(cfg.prefetch_depth_min, min(cfg.prefetch_depth_max, tune.prefetch_depth))
        self._window_seen = 0
        self._window_miss = 0
        self.last_memory_pressure = 0.0
        self.real_transfers = bool(real_transfers)
        self.tile_path_resolver = tile_path_resolver
        self.transfer_cap_bytes = max(1, int(transfer_cap_mb)) * 1024 * 1024
        self.transfer_stats: dict[str, float | int] = {
            "real_mode_enabled": int(self.real_transfers),
            "tiles_fetched": 0,
            "tiles_h2d": 0,
            "bytes_read": 0,
            "read_ms_total": 0.0,
            "h2d_ms_total": 0.0,
            "fallback_simulated": 0,
        }

    async def predict_and_queue(self, predicted_tiles: Iterable[str], depth: int) -> None:
        queued = 0
        queue_budget = min(self.cfg.prefetch_queue_len, depth)
        for tile in predicted_tiles:
            if self.hots.has(tile):
                continue
            if self.queue.qsize() >= queue_budget:
                break
            if self.queue.full():
                break
            await self.queue.put(tile)
            queued += 1
            if queued >= depth:
                break

    def _resolve_tile_path(self, tile: str) -> Optional[Path]:
        if not self.tile_path_resolver:
            return None
        maybe_path = self.tile_path_resolver(tile)
        if not maybe_path:
            return None
        path = Path(maybe_path)
        if not path.exists() or not path.is_file():
            return None
        return path

    def _read_tile_bytes(self, path: Path) -> bytes:
        with path.open("rb") as handle:
            return handle.read(self.transfer_cap_bytes)

    def _copy_payload_h2d(self, payload: bytes) -> None:
        if torch is None or np is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA transfer path unavailable")
        arr = np.frombuffer(payload, dtype=np.uint8).copy()
        cpu = torch.from_numpy(arr).pin_memory()
        gpu = cpu.to("cuda:0", non_blocking=False)
        torch.cuda.synchronize()
        del gpu
        del cpu

    async def fetch_tile_nvme_to_ram(self, tile: str) -> tuple[str, Optional[bytes]]:
        if not self.real_transfers:
            self.transfer_stats["fallback_simulated"] += 1
            await asyncio.sleep(0.0025)
            return tile, None

        path = self._resolve_tile_path(tile)
        if not path:
            self.transfer_stats["fallback_simulated"] += 1
            await asyncio.sleep(0.0025)
            return tile, None

        start = time.perf_counter()
        payload = await asyncio.to_thread(self._read_tile_bytes, path)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.transfer_stats["tiles_fetched"] += 1
        self.transfer_stats["bytes_read"] += len(payload)
        self.transfer_stats["read_ms_total"] += elapsed_ms
        return tile, payload

    async def copy_h2d(self, ram_tile: tuple[str, Optional[bytes]]) -> str:
        tile, payload = ram_tile
        if payload is None:
            self.transfer_stats["fallback_simulated"] += 1
            await asyncio.sleep(0.0015)
            return f"gpu:{tile}"

        start = time.perf_counter()
        try:
            await asyncio.to_thread(self._copy_payload_h2d, payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.transfer_stats["tiles_h2d"] += 1
            self.transfer_stats["h2d_ms_total"] += elapsed_ms
            return f"gpu:{tile}"
        except Exception:
            self.transfer_stats["fallback_simulated"] += 1
            await asyncio.sleep(0.0015)
            return f"gpu:{tile}"

    def get_transfer_stats(self) -> dict[str, float | int]:
        fetched = int(self.transfer_stats["tiles_fetched"])
        moved = int(self.transfer_stats["tiles_h2d"])
        stats = dict(self.transfer_stats)
        stats["avg_read_ms"] = (float(stats["read_ms_total"]) / fetched) if fetched else 0.0
        stats["avg_h2d_ms"] = (float(stats["h2d_ms_total"]) / moved) if moved else 0.0
        return stats

    async def prefetch_worker(self) -> None:
        while True:
            tile = await self.queue.get()
            ram_tile = await self.fetch_tile_nvme_to_ram(tile)
            _ = await self.copy_h2d(ram_tile)
            self.hots.touch(tile)
            self.queue.task_done()

    def _rank_predictions(self, task: TileTask) -> list[str]:
        if not task.predicted_next_scores:
            return task.predicted_next_tiles
        ordered = sorted(task.predicted_next_tiles, key=lambda t: task.predicted_next_scores.get(t, 0.0), reverse=True)
        return ordered

    def _memory_pressure(self) -> float:
        if torch is None or not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(0).total_memory
        if total <= 0:
            return 0.0
        return float(torch.cuda.memory_reserved(0)) / float(total)

    def _effective_depth(self) -> int:
        pressure = self._memory_pressure()
        self.last_memory_pressure = pressure
        if pressure >= self.cfg.memory_pressure_high:
            return max(self.cfg.prefetch_depth_min, self.current_prefetch_depth - 1)
        return self.current_prefetch_depth

    def _feedback_step(self) -> None:
        if self._window_seen <= 0:
            return
        miss_rate = self._window_miss / self._window_seen
        if miss_rate > self.cfg.stall_high_watermark and self.current_prefetch_depth < self.cfg.prefetch_depth_max:
            self.current_prefetch_depth += 1
        elif miss_rate < self.cfg.stall_low_watermark and self.current_prefetch_depth > self.cfg.prefetch_depth_min:
            self.current_prefetch_depth -= 1
        if self.last_memory_pressure >= self.cfg.memory_pressure_high:
            self.current_prefetch_depth = max(self.cfg.prefetch_depth_min, self.current_prefetch_depth - 1)
        self._window_seen = 0
        self._window_miss = 0

    async def run_stream(self, tasks: Iterable[TileTask]) -> None:
        worker = asyncio.create_task(self.prefetch_worker())
        try:
            for task in tasks:
                self._window_seen += 1
                if not self.hots.has(task.tile_id):
                    self._window_miss += 1
                depth = self._effective_depth()
                ranked = self._rank_predictions(task)
                await self.predict_and_queue(ranked, depth=depth)
                await asyncio.sleep(0.003)
                self.hots.touch(task.tile_id)
                if self._window_seen >= self.cfg.feedback_interval:
                    self._feedback_step()
            self._feedback_step()
            await self.queue.join()
        finally:
            worker.cancel()
