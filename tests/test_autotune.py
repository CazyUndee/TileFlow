from tileflow.autotune import get_s_opt
from tileflow.config import RuntimeConfig


def test_get_s_opt_returns_aligned_size() -> None:
    cfg = RuntimeConfig(prefetch_depth=4, headroom=0.45, target_util=0.90, align_mb=64)
    t_comps = {256: 0.006, 320: 0.007, 384: 0.008, 448: 0.009}
    tune = get_s_opt(b_io=20.0, t_comps=t_comps, cfg=cfg)
    assert tune.s_opt_mb % 64 == 0
    assert cfg.prefetch_depth_min <= tune.prefetch_depth <= cfg.prefetch_depth_max


def test_get_s_opt_uses_target_util() -> None:
    t_comps = {256: 0.006, 320: 0.007, 384: 0.008, 448: 0.009, 512: 0.010}
    low = RuntimeConfig(prefetch_depth=4, headroom=0.0, target_util=0.78, min_util=0.7, align_mb=64)
    high = RuntimeConfig(prefetch_depth=4, headroom=0.0, target_util=0.83, min_util=0.7, align_mb=64)
    low_tune = get_s_opt(b_io=20.0, t_comps=t_comps, cfg=low)
    high_tune = get_s_opt(b_io=20.0, t_comps=t_comps, cfg=high)
    assert low_tune.predicted_util_pct < high_tune.predicted_util_pct


def test_get_s_opt_increases_prefetch_depth_when_io_is_slower() -> None:
    cfg = RuntimeConfig(
        prefetch_depth=2,
        prefetch_depth_min=1,
        prefetch_depth_max=8,
        headroom=0.0,
        target_util=0.90,
        min_util=0.70,
    )
    t_comps = {256: 0.008, 384: 0.010, 512: 0.012}
    fast_io = get_s_opt(b_io=80.0, t_comps=t_comps, cfg=cfg)
    slow_io = get_s_opt(b_io=6.0, t_comps=t_comps, cfg=cfg)
    assert slow_io.prefetch_depth >= fast_io.prefetch_depth
