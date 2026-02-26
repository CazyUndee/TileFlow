from tileflow.cli import _quant_hint


def test_quant_hint_full_precision_and_integer_levels() -> None:
    assert "fp32" in _quant_hint("model-FP32.gguf").lower()
    assert "fp16" in _quant_hint("model-fp16.gguf").lower()
    assert "bf16" in _quant_hint("model-bf16.gguf").lower()
    assert "fp8/int8" in _quant_hint("model-int8.gguf").lower()
    assert "q4/int4" in _quant_hint("model-Q4_K_M.gguf").lower()
    assert "q3" in _quant_hint("model-q3_k_l.gguf").lower()
    assert "int2/q2" in _quant_hint("model-int2.gguf").lower()
    assert "q1/int1/binary" in _quant_hint("model-binary-int1.gguf").lower()


def test_quant_hint_mixed_and_scaling_and_activation_patterns() -> None:
    mixed = _quant_hint("model-w8a4-perchannel-int8-fp16-act.gguf").lower()
    assert "mixed quantization" in mixed
    assert "per-channel" in mixed
    assert "fp16 activations" in mixed

    tensor = _quant_hint("model-q4-pertensor.gguf").lower()
    assert "per-tensor" in tensor
