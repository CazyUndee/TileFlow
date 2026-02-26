from __future__ import annotations

import statistics
import time
from dataclasses import replace
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple

from tileflow.config import RuntimeConfig, TuneResult

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# ---------------------------------------------------------------------------
# Fallback constants (measured on A100-80 GB; tune for your hardware)
# ---------------------------------------------------------------------------
_FALLBACK_IO_BW_GBS: float = 22.0
_FALLBACK_COMPUTE_BASE_S: float = 0.0045
_FALLBACK_COMPUTE_SLOPE_S_PER_GB: float = 0.0032

TileSizeMB = int  # semantic alias for clarity


@dataclass(slots=True)
class BenchmarkResults:
    io_bandwidth_gbs: float
    compute_times_s: Dict[TileSizeMB, float]


@dataclass(slots=True)
class TileCandidate:
    tile_mb: TileSizeMB
    prefetch_depth: int
    utilization: float
    compute_time_ms: float
    io_time_ms: float
    effective_io_time_ms: float
    stall_time_ms: float
    # Derived latency/throughput scores (lower stall = lower latency;
    # higher utilization = higher throughput)
    latency_score: float   # = stall_time_ms (want to minimise)
    throughput_score: float  # = utilization (want to maximise)


class _RoundResult(NamedTuple):
    io_bw_gbs: float
    compute_times_s: Dict[TileSizeMB, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_without_cuda(sizes_mb: Iterable[TileSizeMB]) -> BenchmarkResults:
    sizes_mb = list(sizes_mb)
    compute_times: Dict[TileSizeMB, float] = {
        mb: _FALLBACK_COMPUTE_BASE_S + (mb / 1024.0) * _FALLBACK_COMPUTE_SLOPE_S_PER_GB
        for mb in sizes_mb
    }
    return BenchmarkResults(
        io_bandwidth_gbs=_FALLBACK_IO_BW_GBS,
        compute_times_s=compute_times,
    )


def _aligned(mb: TileSizeMB, align_mb: int) -> TileSizeMB:
    return max(align_mb, int(round(mb / align_mb)) * align_mb)


def _sample_sizes(
    low_mb: int, high_mb: int, count: int, align_mb: int
) -> List[TileSizeMB]:
    if count <= 1 or low_mb >= high_mb:
        return [max(align_mb, _aligned(low_mb, align_mb))]
    span = high_mb - low_mb
    values: set[TileSizeMB] = set()
    for i in range(count):
        frac = i / (count - 1)
        values.add(_aligned(low_mb + int(span * frac), align_mb))
    return sorted(values)


def _initial_sizes(cfg: RuntimeConfig) -> List[TileSizeMB]:
    if not cfg.tile_sizes_mb:
        raise ValueError("RuntimeConfig.tile_sizes_mb must not be empty")
    if cfg.dynamic_tile_search:
        return _sample_sizes(
            cfg.min_tile_mb, cfg.max_tile_mb,
            cfg.tile_samples_per_round, cfg.align_mb,
        )
    return sorted(set(cfg.tile_sizes_mb))


# ---------------------------------------------------------------------------
# IO benchmark  (takes an explicit size list — no hidden cfg dependency)
# ---------------------------------------------------------------------------

def _benchmark_io_bandwidth(sizes_mb: List[TileSizeMB], cfg: RuntimeConfig) -> float:
    """Measure host→device bandwidth for each requested tile size."""
    assert torch is not None
    io_gbs: List[float] = []

    for size_mb in sizes_mb:
        num_elements = (size_mb * 1024 * 1024) // 2
        cpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16, pin_memory=True)

        # Warmup — synchronise properly so device is truly warm before timing
        for _ in range(cfg.num_warmup_runs):
            _ = cpu_tensor.to("cuda", non_blocking=True)
            torch.cuda.synchronize()

        timings: List[float] = []
        for _ in range(cfg.num_benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = cpu_tensor.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            timings.append(max(1e-9, time.perf_counter() - start))

        io_gbs.append(size_mb / 1024.0 / statistics.median(timings))

    return statistics.median(io_gbs) * cfg.safety_margin


# ---------------------------------------------------------------------------
# Compute benchmark
# ---------------------------------------------------------------------------

def _benchmark_compute(
    sizes_mb: List[TileSizeMB], cfg: RuntimeConfig
) -> Dict[TileSizeMB, float]:
    """Measure matmul compute time for each tile size."""
    assert torch is not None
    compute_times: Dict[TileSizeMB, float] = {}
    activation = torch.randn(1, cfg.hidden_dim, dtype=torch.bfloat16, device="cuda")

    for size_mb in sizes_mb:
        bytes_per_expert = cfg.hidden_dim * cfg.expert_dim * 2
        experts_in_tile = max(1, int((size_mb * 1e6) / bytes_per_expert))
        expert_weights = torch.randn(
            cfg.hidden_dim,
            experts_in_tile * cfg.expert_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

        # Warmup — synchronise after so the device is fully warm before timing
        for _ in range(cfg.num_warmup_runs):
            _ = torch.matmul(activation, expert_weights)
        torch.cuda.synchronize()

        timings: List[float] = []
        for _ in range(cfg.num_benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(cfg.matmul_ops_per_tile):
                _ = torch.matmul(activation, expert_weights)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

        compute_times[size_mb] = statistics.median(timings) / cfg.matmul_ops_per_tile

    return compute_times


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _candidate_metrics(
    tile_mb: TileSizeMB,
    compute_time_s: float,
    b_io_gbs: float,
    cfg: RuntimeConfig,
    prefetch_depth: int,
) -> TileCandidate:
    depth = max(1, prefetch_depth)
    compute_ms = compute_time_s * 1000.0
    io_ms = ((tile_mb / 1024.0) / b_io_gbs) * 1000.0

    # Effective IO time accounts for prefetch pipeline depth
    effective_io_ms = io_ms / depth + cfg.fixed_overhead_s * 1000.0

    # Stall = time the GPU waits for data (pure latency penalty)
    stall_ms = max(0.0, effective_io_ms - compute_ms)

    total_ms = compute_ms + stall_ms + cfg.routing_tax_s * 1000.0
    utilization = compute_ms / total_ms if total_ms > 0 else 0.0

    return TileCandidate(
        tile_mb=tile_mb,
        prefetch_depth=depth,
        utilization=utilization,
        compute_time_ms=compute_ms,
        io_time_ms=io_ms,
        effective_io_time_ms=effective_io_ms,
        stall_time_ms=stall_ms,
        latency_score=stall_ms,        # lower  = better latency
        throughput_score=utilization,  # higher = better throughput
    )


def _select_candidate(
    candidates: List[TileCandidate], cfg: RuntimeConfig
) -> TileCandidate:
    """
    Selection policy (low-latency + high-throughput):

    1. Restrict to candidates that meet the minimum utilization floor so we
       never return a GPU-idle tile.
    2. Among those, prefer candidates whose stall time is zero or minimal
       (compute-bound = low latency).  Break ties by preferring *higher*
       utilization (more throughput), then *smaller* tile size (lower memory
       pressure / faster context switch on the next token).
    3. If every candidate is above the target utilization ceiling (GPU may
       thermal-throttle), pick the one that is closest to the target with the
       smallest tile — keeping latency low while backing off throughput just
       enough.
    """
    if not candidates:
        raise ValueError("No candidates to evaluate")

    # Step 1 — enforce minimum utilization floor
    viable = [c for c in candidates if c.utilization >= cfg.min_util] or candidates

    # Step 2 — identify compute-bound candidates (stall ≈ 0, i.e. IO hidden)
    compute_bound = [c for c in viable if c.stall_time_ms == 0.0]
    pool = compute_bound if compute_bound else viable

    # Step 3 — from the pool, pick below the utilization ceiling first
    below_target = [c for c in pool if c.utilization <= cfg.target_util]
    if below_target:
        # Maximise throughput; break ties by minimising tile size (latency)
        return max(below_target, key=lambda c: (c.throughput_score, -c.tile_mb))

    # Everything exceeds the target — get as close as possible with smallest tile
    return min(pool, key=lambda c: (c.utilization - cfg.target_util, c.tile_mb))


# ---------------------------------------------------------------------------
# Top-level benchmark orchestration
# ---------------------------------------------------------------------------

def benchmark_io_and_compute(cfg: RuntimeConfig) -> BenchmarkResults:
    if torch is None or not torch.cuda.is_available():
        return _estimate_without_cuda(_initial_sizes(cfg))

    torch.cuda.set_device(0)

    if not cfg.dynamic_tile_search:
        sizes = list(cfg.tile_sizes_mb)
        if not sizes:
            raise ValueError("RuntimeConfig.tile_sizes_mb must not be empty")
        io_bw = _benchmark_io_bandwidth(sizes, cfg)
        compute_times = _benchmark_compute(sizes, cfg)
        return BenchmarkResults(io_bandwidth_gbs=io_bw, compute_times_s=compute_times)

    # --- Dynamic search: iteratively narrow toward the optimal tile size ----
    searched: Dict[TileSizeMB, float] = {}
    round_results: List[_RoundResult] = []
    low, high = cfg.min_tile_mb, cfg.max_tile_mb

    for _ in range(max(1, cfg.tile_search_rounds)):
        round_sizes = [
            s for s in _sample_sizes(low, high, cfg.tile_samples_per_round, cfg.align_mb)
            if s not in searched
        ]
        if not round_sizes:
            break

        # Benchmark this round's sizes
        io_bw = _benchmark_io_bandwidth(round_sizes, cfg)
        compute_times = _benchmark_compute(round_sizes, cfg)
        round_results.append(_RoundResult(io_bw_gbs=io_bw, compute_times_s=compute_times))
        searched.update(compute_times)

        # Evaluate all gathered data with the *best* IO estimate so far so
        # candidate scoring improves as more rounds complete
        best_io = statistics.median(r.io_bw_gbs for r in round_results)
        candidates = [
            _candidate_metrics(mb, t_s, best_io, cfg, cfg.prefetch_depth)
            for mb, t_s in searched.items()
        ]
        best = _select_candidate(candidates, cfg)

        # Narrow the search window around the current best tile
        half_window = max(cfg.align_mb, (high - low) // 4)
        low = max(cfg.min_tile_mb, best.tile_mb - half_window)
        high = min(cfg.max_tile_mb, best.tile_mb + half_window)

    final_io = statistics.median(r.io_bw_gbs for r in round_results) if round_results else _benchmark_io_bandwidth(list(cfg.tile_sizes_mb), cfg)
    return BenchmarkResults(io_bandwidth_gbs=final_io, compute_times_s=searched)


# ---------------------------------------------------------------------------
# Tune result
# ---------------------------------------------------------------------------

def get_s_opt(
    b_io: float, t_comps: Dict[TileSizeMB, float], cfg: RuntimeConfig
) -> TuneResult:
    candidates = [
        _candidate_metrics(mb, t_s, b_io, cfg, depth)
        for mb, t_s in t_comps.items()
        for depth in range(cfg.prefetch_depth_min, cfg.prefetch_depth_max + 1)
    ]
    best = _select_candidate(candidates, cfg)

    base_tile = best.tile_mb
    s_opt = int((base_tile * (1.0 + cfg.headroom)) // cfg.align_mb * cfg.align_mb)
    s_opt = max(cfg.align_mb, s_opt)
    s_opt = min(s_opt, cfg.max_tile_mb)  # clamp — never exceed configured ceiling

    return TuneResult(
        s_opt_mb=s_opt,
        prefetch_depth=best.prefetch_depth,
        predicted_util_pct=best.utilization * 100.0,
        b_io_gbs=b_io,
    )


def startup_tune(cfg: RuntimeConfig) -> TuneResult:
    bench = benchmark_io_and_compute(cfg)
    return get_s_opt(b_io=bench.io_bandwidth_gbs, t_comps=bench.compute_times_s, cfg=cfg)
