from tileflow.model_store import ModelStore


def test_build_allow_patterns_prefers_single_gguf() -> None:
    patterns = ModelStore._build_allow_patterns(
        repo_id="x/y",
        revision="main",
        include=["special.bin"],
        gguf_pattern="*Q4_K_M.gguf",
        include_gguf=True,
        prefer_smallest_gguf=False,
    )
    assert "*.gguf" not in patterns
    assert "*Q4_K_M.gguf" in patterns
    assert "special.bin" in patterns


def test_build_allow_patterns_can_skip_gguf() -> None:
    patterns = ModelStore._build_allow_patterns(
        repo_id="x/y",
        revision="main",
        include=None,
        gguf_pattern=None,
        include_gguf=False,
        prefer_smallest_gguf=False,
    )
    assert "*.gguf" not in patterns


def test_preferred_q4_or_int4_gguf_prefers_smallest(monkeypatch) -> None:
    class _Sibling:
        def __init__(self, name: str, size: int) -> None:
            self.rfilename = name
            self.size = size

    class _Info:
        siblings = [
            _Sibling("model-Q8_0.gguf", 800),
            _Sibling("model-Q4_K_M.gguf", 420),
            _Sibling("model-INT4.gguf", 400),
        ]

    class _Api:
        def model_info(self, **kwargs):
            return _Info()

    monkeypatch.setattr("tileflow.model_store.HfApi", lambda: _Api())
    preferred = ModelStore._preferred_q4_or_int4_gguf("x/y", "main")
    assert preferred == "model-INT4.gguf"


def test_list_gguf_files(monkeypatch) -> None:
    class _Sibling:
        def __init__(self, name: str) -> None:
            self.rfilename = name

    class _Info:
        siblings = [
            _Sibling("a-Q4_K_M.gguf"),
            _Sibling("README.md"),
            _Sibling("b-Q8_0.gguf"),
        ]

    class _Api:
        def model_info(self, **kwargs):
            return _Info()

    monkeypatch.setattr("tileflow.model_store.HfApi", lambda: _Api())
    files = ModelStore.list_gguf_files("x/y", "main")
    assert files == ["a-Q4_K_M.gguf", "b-Q8_0.gguf"]


def test_list_safetensor_files(monkeypatch) -> None:
    class _Sibling:
        def __init__(self, name: str) -> None:
            self.rfilename = name

    class _Info:
        siblings = [
            _Sibling("model-00001-of-00002.safetensors"),
            _Sibling("model-00002-of-00002.safetensors"),
            _Sibling("model.safetensors.index.json"),
            _Sibling("README.md"),
        ]

    class _Api:
        def model_info(self, **kwargs):
            return _Info()

    monkeypatch.setattr("tileflow.model_store.HfApi", lambda: _Api())
    files = ModelStore.list_safetensor_files("x/y", "main")
    assert files == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_delete_and_rename_model(tmp_path) -> None:
    store = ModelStore(home=tmp_path)
    original = {
        "alpha": {
            "name": "alpha",
            "repo_id": "x/y",
            "local_path": str(tmp_path / "models" / "alpha"),
            "revision": "main",
            "gguf_path": None,
        }
    }
    store._save(original)

    renamed = store.rename("alpha", "beta")
    assert renamed.name == "beta"
    assert store.get("alpha") is None
    assert store.get("beta") is not None

    deleted = store.delete("beta", delete_files=False)
    assert deleted is True
    assert store.get("beta") is None


def test_load_recovers_from_corrupt_registry(tmp_path) -> None:
    store = ModelStore(home=tmp_path)
    store.registry_path.write_text("{bad json", encoding="utf-8")
    loaded = store._load()
    assert loaded == {}
    assert store.registry_path.exists()


def test_matching_repo_files_filters_patterns(monkeypatch) -> None:
    class _Sibling:
        def __init__(self, name: str, size: int) -> None:
            self.rfilename = name
            self.size = size

    class _Info:
        siblings = [
            _Sibling("model-q4.gguf", 100),
            _Sibling("README.md", 10),
            _Sibling("tokenizer.json", 20),
        ]

    class _Api:
        def model_info(self, **kwargs):
            return _Info()

    monkeypatch.setattr("tileflow.model_store.HfApi", lambda: _Api())
    files = ModelStore._matching_repo_files("x/y", "main", ["*.gguf", "tokenizer*"])
    assert files == [("model-q4.gguf", 100), ("tokenizer.json", 20)]


def test_pull_streaming_reports_progress(monkeypatch, tmp_path) -> None:
    class _Sibling:
        def __init__(self, name: str, size: int) -> None:
            self.rfilename = name
            self.size = size

    class _Info:
        siblings = [_Sibling("model-q4.gguf", 4)]

    class _Api:
        def model_info(self, **kwargs):
            return _Info()

    class _Resp:
        def __enter__(self):
            self._chunks = [b"ab", b"cd", b""]
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    monkeypatch.setattr("tileflow.model_store.HfApi", lambda: _Api())
    monkeypatch.setattr("tileflow.model_store.hf_hub_url", lambda **kwargs: "https://example.test/file")
    monkeypatch.setattr("tileflow.model_store.ModelStore._hf_token", staticmethod(lambda: None))
    monkeypatch.setattr("tileflow.model_store.urllib.request.urlopen", lambda *args, **kwargs: _Resp())

    store = ModelStore(home=tmp_path)
    progress_events = []
    rec = store.pull_streaming(
        repo_id="x/y",
        name="demo",
        include=[],
        gguf_pattern="*.gguf",
        include_gguf=True,
        prefer_smallest_gguf=False,
        progress_callback=lambda payload: progress_events.append(payload),
        chunk_size=2,
    )
    assert rec.name == "demo"
    assert progress_events
    assert any((evt.get("downloaded_bytes") or 0) >= 4 for evt in progress_events)
