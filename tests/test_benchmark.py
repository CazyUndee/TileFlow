from tileflow.benchmark import _backend_profile_cfg, _extract_params_b, _quant_bits, _quant_speed_factor
from tileflow.config import RuntimeConfig


def test_extract_params_defaults_and_parses() -> None:
    assert _extract_params_b("meta-llama-3-8b-instruct") == 8.0
    assert _extract_params_b("model-70B-Q4") == 70.0
    assert _extract_params_b("unknown-model-name") == 7.0


def test_quant_bits_detection() -> None:
    assert _quant_bits("model-Q4_K_M.gguf") == 4.0
    assert _quant_bits("model-int8.gguf") == 8.0
    assert _quant_bits("model-fp16.safetensors") == 16.0
    assert _quant_bits("model-fp32.safetensors") == 32.0


def test_quant_speed_factor_monotonicity() -> None:
    q4 = _quant_speed_factor("model-q4")
    fp16 = _quant_speed_factor("model-fp16")
    fp32 = _quant_speed_factor("model-fp32")
    assert q4 < fp16 < fp32


def test_backend_profiles_disable_prefetch_for_non_tileflow() -> None:
    base = RuntimeConfig(prefetch_depth=4, prefetch_depth_min=1, prefetch_depth_max=8)
    stock = _backend_profile_cfg(base, "Stock ktransformers")
    airllm = _backend_profile_cfg(base, "AirLLM")
    assert stock.prefetch_depth == 1
    assert stock.prefetch_depth_min == 1
    assert stock.prefetch_depth_max == 1
    assert airllm.prefetch_depth == 1
    assert airllm.prefetch_depth_min == 1
    assert airllm.prefetch_depth_max == 1


def test_detect_hardware_uses_nvidia_smi_fallback(monkeypatch) -> None:
    import types
    import tileflow.benchmark as bm

    monkeypatch.setattr(bm, "torch", None)
    monkeypatch.setattr(bm.shutil, "which", lambda _: "nvidia-smi")

    def _fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="NVIDIA RTX 4090, 24564\n", stderr="")

    monkeypatch.setattr(bm.subprocess, "run", _fake_run)
    hw = bm.detect_hardware()
    assert hw.cuda_available is True
    assert "4090" in hw.gpu
    assert hw.gpu_vram_gb > 20
