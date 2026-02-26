from tileflow.cli import _dense_recommended_label, _dense_variant_options


def test_dense_variant_options_groups_sharded_files() -> None:
    files = [
        "bf16/model-00001-of-00002.safetensors",
        "bf16/model-00002-of-00002.safetensors",
        "fp16/model.safetensors",
    ]
    variants = _dense_variant_options(files)
    labels = [v[0] for v in variants]
    assert "bf16/model (2 shards)" in labels
    assert "fp16/model.safetensors" in labels

    bf16 = [v for v in variants if v[0] == "bf16/model (2 shards)"][0]
    assert "bf16/model-*.safetensors" in bf16[1]
    assert "bf16/model.safetensors.index.json" in bf16[1]


def test_dense_recommended_label_prefers_bf16_then_fp16() -> None:
    variants = [
        ("fp16/model.safetensors", ["fp16/model.safetensors"], "fp16/model.safetensors"),
        ("bf16/model.safetensors", ["bf16/model.safetensors"], "bf16/model.safetensors"),
    ]
    assert _dense_recommended_label(variants) == "bf16/model.safetensors"
