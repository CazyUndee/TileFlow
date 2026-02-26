from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Optional

from tileflow.autotune import benchmark_io_and_compute, get_s_opt
from tileflow.config import RuntimeConfig
from tileflow.model_store import ModelRecord

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(slots=True)
class HardwareProfile:
    cpu: str
    cpu_cores: int
    total_ram_gb: float
    gpu: str
    gpu_vram_gb: float
    cuda_available: bool


@dataclass(slots=True)
class BackendPerf:
    backend: str
    tokens_per_second: float
    ms_per_token: float
    ttft_ms: float
    source: str
    profile: str
    notes: str


@dataclass(slots=True)
class BenchmarkReport:
    model_label: str
    model_params_b: float
    estimated_model_memory_gb: float
    fit_hint: str
    hardware: HardwareProfile
    tune: dict[str, float | int]
    bench: dict[str, float]
    backends: list[BackendPerf]
    generated_at_unix_s: float
    methodology: str


def _system_ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.ullTotalPhys) / (1024.0 ** 3)
        page_size = float(os.sysconf("SC_PAGE_SIZE"))
        total_pages = float(os.sysconf("SC_PHYS_PAGES"))
        return (page_size * total_pages) / (1024.0 ** 3)
    except Exception:
        return 0.0


def detect_hardware() -> HardwareProfile:
    cpu = platform.processor() or platform.machine() or "Unknown CPU"
    cores = os.cpu_count() or 0
    total_ram_gb = _system_ram_gb()

    if torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return HardwareProfile(
            cpu=cpu,
            cpu_cores=int(cores),
            total_ram_gb=round(total_ram_gb, 2),
            gpu=props.name,
            gpu_vram_gb=round(float(props.total_memory) / (1024.0 ** 3), 2),
            cuda_available=True,
        )

    # Fallback for packaged/runtime environments where torch CUDA wheel is absent.
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                first = proc.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                gpu_name = parts[0] if parts else "NVIDIA GPU"
                mem_mb = float(parts[1]) if len(parts) > 1 and parts[1].replace(".", "", 1).isdigit() else 0.0
                return HardwareProfile(
                    cpu=cpu,
                    cpu_cores=int(cores),
                    total_ram_gb=round(total_ram_gb, 2),
                    gpu=gpu_name,
                    gpu_vram_gb=round(mem_mb / 1024.0, 2),
                    cuda_available=True,
                )
        except Exception:
            pass

    return HardwareProfile(
        cpu=cpu,
        cpu_cores=int(cores),
        total_ram_gb=round(total_ram_gb, 2),
        gpu="No CUDA GPU detected",
        gpu_vram_gb=0.0,
        cuda_available=False,
    )


def _extract_params_b(text: str) -> float:
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*b\b", text.lower()):
        value = float(match.group(1))
        if 0.5 <= value <= 1000.0:
            return value
    return 7.0


def _quant_bits(text: str) -> float:
    lowered = text.lower()
    if any(token in lowered for token in ("int1", "q1", "1bit", "1-bit", "binary")):
        return 1.0
    if any(token in lowered for token in ("int2", "q2")):
        return 2.0
    if "q3" in lowered:
        return 3.0
    if any(token in lowered for token in ("int4", "q4")):
        return 4.0
    if any(token in lowered for token in ("int8", "q8", "fp8")):
        return 8.0
    if any(token in lowered for token in ("fp16", "f16", "bf16", "bfloat16")):
        return 16.0
    if any(token in lowered for token in ("fp32", "f32")):
        return 32.0
    return 16.0


def _quant_speed_factor(text: str) -> float:
    bits = _quant_bits(text)
    return max(0.35, bits / 8.0)


def _model_text(model: Optional[ModelRecord]) -> str:
    if model is None:
        return "generic-7b"
    return " ".join(part for part in (model.name, model.repo_id, model.local_path, model.gguf_path or "") if part)


def _nearest_size(size_mb: int, options: list[int]) -> int:
    if not options:
        return size_mb
    return min(options, key=lambda x: abs(x - size_mb))


def _backend_profile_cfg(base: RuntimeConfig, backend: str) -> RuntimeConfig:
    if backend == "TileFlow":
        return base
    if backend == "Stock ktransformers":
        return RuntimeConfig(
            prefetch_depth=1,
            prefetch_depth_min=1,
            prefetch_depth_max=1,
            dynamic_tile_search=False,
            tile_sizes_mb=(256, 384, 512, 640),
            fixed_overhead_s=base.fixed_overhead_s * 1.25,
            routing_tax_s=base.routing_tax_s * 1.2,
            headroom=max(0.20, base.headroom * 0.75),
            target_util=min(0.88, base.target_util),
        )
    if backend == "AirLLM":
        return RuntimeConfig(
            prefetch_depth=1,
            prefetch_depth_min=1,
            prefetch_depth_max=1,
            dynamic_tile_search=False,
            tile_sizes_mb=(256, 384, 512),
            fixed_overhead_s=base.fixed_overhead_s * 1.45,
            routing_tax_s=base.routing_tax_s * 1.35,
            headroom=max(0.15, base.headroom * 0.70),
            target_util=min(0.85, base.target_util),
        )
    return base


def _score_backend(
    *,
    backend: str,
    cfg: RuntimeConfig,
    io_bw_gbs: float,
    compute_times_s: dict[int, float],
    params_b: float,
    quant_factor: float,
) -> tuple[dict[str, float | int], dict[str, float], BackendPerf]:
    tune = get_s_opt(io_bw_gbs, compute_times_s, cfg)
    sizes = list(compute_times_s.keys())
    scored_tile = _nearest_size(tune.s_opt_mb, sizes)
    compute_s = compute_times_s[scored_tile]
    io_s = (scored_tile / 1024.0) / max(io_bw_gbs, 1e-6)
    effective_io_s = io_s / max(1, tune.prefetch_depth) + cfg.fixed_overhead_s
    stall_s = max(0.0, effective_io_s - compute_s)
    token_time_s = max(1e-6, compute_s + stall_s + cfg.routing_tax_s)

    raw_tps = 1.0 / token_time_s
    model_complexity = max(0.35, (params_b / 7.0) * quant_factor)
    tps = max(0.05, raw_tps / model_complexity)
    ms_per_token = 1000.0 / tps
    ttft_ms = max(20.0, (compute_s + effective_io_s + cfg.routing_tax_s) * 1000.0 * 1.8)

    tune_view = {
        "s_opt_mb": tune.s_opt_mb,
        "prefetch_depth": tune.prefetch_depth,
        "predicted_util_pct": round(tune.predicted_util_pct, 2),
        "b_io_gbs": round(io_bw_gbs, 2),
    }
    bench_view = {
        "compute_ms_for_scored_tile": round(compute_s * 1000.0, 3),
        "io_ms_for_scored_tile": round(io_s * 1000.0, 3),
        "stall_ms_for_scored_tile": round(stall_s * 1000.0, 3),
        "scored_tile_mb": float(scored_tile),
    }
    row = BackendPerf(
        backend=backend,
        tokens_per_second=round(tps, 2),
        ms_per_token=round(ms_per_token, 2),
        ttft_ms=round(ttft_ms, 2),
        source="measured+modeled",
        profile=(
            "streaming-aware tuned profile"
            if backend == "TileFlow"
            else "conservative non-streaming profile"
        ),
        notes=(
            "Scored from the same measured hardware benchmark with backend-specific execution profile assumptions."
        ),
    )
    return tune_view, bench_view, row


def run_benchmark_report(
    *,
    model: Optional[ModelRecord],
    cfg: RuntimeConfig,
    include_comparison: bool,
) -> BenchmarkReport:
    hardware = detect_hardware()
    model_text = _model_text(model)
    params_b = _extract_params_b(model_text)
    quant_bits = _quant_bits(model_text)
    quant_factor = _quant_speed_factor(model_text)
    model_mem_gb = round(params_b * (quant_bits / 8.0) * 1.2, 2)

    if hardware.gpu_vram_gb <= 0.0:
        fit_hint = "GPU fit unknown (no CUDA GPU detected)."
    elif model_mem_gb <= hardware.gpu_vram_gb:
        fit_hint = f"Likely fits in GPU memory (~{model_mem_gb} GB <= {hardware.gpu_vram_gb} GB)."
    else:
        fit_hint = f"Likely requires streaming/offload (~{model_mem_gb} GB > {hardware.gpu_vram_gb} GB)."

    base_bench = benchmark_io_and_compute(cfg)
    compare_backends = ["TileFlow"]
    if include_comparison:
        compare_backends.extend(["Stock ktransformers", "AirLLM"])

    backends: list[BackendPerf] = []
    tune_view: dict[str, float | int] = {}
    bench_view: dict[str, float] = {}
    for backend in compare_backends:
        backend_cfg = _backend_profile_cfg(cfg, backend)
        tune_data, bench_data, row = _score_backend(
            backend=backend,
            cfg=backend_cfg,
            io_bw_gbs=base_bench.io_bandwidth_gbs,
            compute_times_s=base_bench.compute_times_s,
            params_b=params_b,
            quant_factor=quant_factor,
        )
        if backend == "TileFlow":
            tune_view = tune_data
            bench_view = bench_data
        backends.append(row)

    label = model.name if model else "Auto (7B default profile)"
    return BenchmarkReport(
        model_label=label,
        model_params_b=round(params_b, 2),
        estimated_model_memory_gb=model_mem_gb,
        fit_hint=fit_hint,
        hardware=hardware,
        tune=tune_view,
        bench=bench_view,
        backends=backends,
        generated_at_unix_s=time.time(),
        methodology=(
            "All rows use one machine-local measured IO+compute benchmark pass. "
            "Each backend is then scored with a transparent execution profile (TileFlow: tuned streaming-aware; "
            "others: conservative non-streaming assumptions) and normalized by model size/quantization."
        ),
    )


def report_as_dict(report: BenchmarkReport) -> dict:
    return {
        "model_label": report.model_label,
        "model_params_b": report.model_params_b,
        "estimated_model_memory_gb": report.estimated_model_memory_gb,
        "fit_hint": report.fit_hint,
        "hardware": asdict(report.hardware),
        "tune": report.tune,
        "bench": report.bench,
        "backends": [asdict(row) for row in report.backends],
        "generated_at_unix_s": report.generated_at_unix_s,
        "methodology": report.methodology,
    }
