import asyncio

from tileflow.config import RuntimeConfig, TuneResult
from tileflow.scheduler import PrefetchScheduler, TileTask


def test_scheduler_runs() -> None:
    cfg = RuntimeConfig(prefetch_depth=3, hot_expert_slots=8, prefetch_queue_len=2)
    tune = TuneResult(s_opt_mb=512, prefetch_depth=3, predicted_util_pct=96.0, b_io_gbs=18.0)
    sched = PrefetchScheduler(cfg=cfg, tune=tune)
    tasks = [
        TileTask(token_idx=0, tile_id="e1", predicted_next_tiles=["e2", "e3", "e4"]),
        TileTask(token_idx=1, tile_id="e2", predicted_next_tiles=["e3", "e4", "e5"]),
    ]
    asyncio.run(sched.run_stream(tasks))


def test_scheduler_prioritizes_routing_scores() -> None:
    cfg = RuntimeConfig(prefetch_depth=3, hot_expert_slots=8, prefetch_queue_len=3)
    tune = TuneResult(s_opt_mb=512, prefetch_depth=3, predicted_util_pct=96.0, b_io_gbs=18.0)
    sched = PrefetchScheduler(cfg=cfg, tune=tune)
    task = TileTask(
        token_idx=0,
        tile_id="e1",
        predicted_next_tiles=["e2", "e3", "e4"],
        predicted_next_scores={"e2": 0.1, "e3": 0.9, "e4": 0.4},
    )
    assert sched._rank_predictions(task) == ["e3", "e4", "e2"]


def test_scheduler_feedback_increases_depth_on_misses() -> None:
    cfg = RuntimeConfig(
        prefetch_depth=2,
        prefetch_depth_min=1,
        prefetch_depth_max=4,
        feedback_interval=2,
        stall_high_watermark=0.25,
        stall_low_watermark=0.01,
    )
    tune = TuneResult(s_opt_mb=512, prefetch_depth=2, predicted_util_pct=96.0, b_io_gbs=18.0)
    sched = PrefetchScheduler(cfg=cfg, tune=tune)
    sched._window_seen = 2
    sched._window_miss = 2
    sched._feedback_step()
    assert sched.current_prefetch_depth == 3


def test_scheduler_exposes_transfer_stats() -> None:
    cfg = RuntimeConfig(prefetch_depth=2, hot_expert_slots=8, prefetch_queue_len=2)
    tune = TuneResult(s_opt_mb=512, prefetch_depth=2, predicted_util_pct=96.0, b_io_gbs=18.0)
    sched = PrefetchScheduler(cfg=cfg, tune=tune)
    tasks = [TileTask(token_idx=0, tile_id="e1", predicted_next_tiles=["e2", "e3"])]
    asyncio.run(sched.run_stream(tasks))
    stats = sched.get_transfer_stats()
    assert "fallback_simulated" in stats
    assert "avg_read_ms" in stats
    assert "avg_h2d_ms" in stats
