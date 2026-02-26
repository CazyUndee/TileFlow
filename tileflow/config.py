from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    tile_sizes_mb: tuple[int, ...] = (256, 320, 384, 448, 512, 576, 640, 768)
    dynamic_tile_search: bool = True
    min_tile_mb: int = 128
    max_tile_mb: int = 2048
    tile_search_rounds: int = 3
    tile_samples_per_round: int = 6
    hidden_dim: int = 7168
    expert_dim: int = 2048
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10
    matmul_ops_per_tile: int = 14
    safety_margin: float = 0.92
    prefetch_depth: int = 4
    headroom: float = 0.45
    target_util: float = 0.90
    min_util: float = 0.90
    fixed_overhead_s: float = 0.001
    routing_tax_s: float = 0.002
    align_mb: int = 64
    hot_expert_slots: int = 16
    prefetch_queue_len: int = 3
    prefetch_depth_min: int = 1
    prefetch_depth_max: int = 8
    feedback_interval: int = 8
    stall_high_watermark: float = 0.30
    stall_low_watermark: float = 0.05
    memory_pressure_high: float = 0.90
    memory_pressure_low: float = 0.75


@dataclass(slots=True)
class TuneResult:
    s_opt_mb: int
    prefetch_depth: int
    predicted_util_pct: float
    b_io_gbs: float
