import pytest

from tileflow.backend_ktransformers import KTransformersBackend


class _Resp:
    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body.encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_chat_completion_parses_valid_payload(monkeypatch) -> None:
    def _fake_urlopen(_req, timeout=0):
        return _Resp('{"choices":[{"message":{"content":"hello"}}]}')

    monkeypatch.setattr("tileflow.backend_ktransformers.urllib.request.urlopen", _fake_urlopen)
    out = KTransformersBackend._chat_completion("127.0.0.1", 11434, {"model": "x"})
    assert out == "hello"


def test_chat_completion_rejects_invalid_payload(monkeypatch) -> None:
    def _fake_urlopen(_req, timeout=0):
        return _Resp('{"bad":"shape"}')

    monkeypatch.setattr("tileflow.backend_ktransformers.urllib.request.urlopen", _fake_urlopen)
    with pytest.raises(RuntimeError):
        KTransformersBackend._chat_completion("127.0.0.1", 11434, {"model": "x"})

