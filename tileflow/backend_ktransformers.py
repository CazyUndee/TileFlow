
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tileflow.autotune import startup_tune
from tileflow.config import RuntimeConfig
from tileflow.model_store import ModelRecord
from tileflow.runtime_bootstrap import ensure_runtime_ready


@dataclass(slots=True)
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 11434
    gpu_split: Optional[str] = None
    max_new_tokens: int = 512
    model_id: Optional[str] = None
    extra_args: tuple[str, ...] = ()


class KTransformersBackend:
    def __init__(self, python_exe: Optional[str] = None, ktransformers_path: Optional[str] = None) -> None:
        self.python_exe = python_exe or sys.executable
        self.ktransformers_path = ktransformers_path or os.getenv("TILEFLOW_KTRANSFORMERS_PATH")

    def ensure_installed(self) -> None:
        auto_install = os.getenv("TILEFLOW_AUTO_INSTALL_RUNTIME", "1") != "0"
        ok, msg = ensure_runtime_ready(auto_install=auto_install)
        if not ok and msg:
            print(f"[tileflow] Runtime bootstrap incomplete (soft-fail):\n{msg}", file=sys.stderr)
        if self.ktransformers_path:
            pkg = Path(self.ktransformers_path) / "ktransformers"
            if pkg.exists():
                return
        if importlib.util.find_spec("ktransformers") is None:
            raise RuntimeError(
                "ktransformers is not installed. Install it first, then retry.\n"
                "Example: pip install ktransformers\n"
                "Or use a local fork with --ktransformers-path /path/to/ktransformers"
            )

    def _build_env(self, tune_s_opt_mb: int, tune_prefetch_depth: int) -> dict[str, str]:
        env = dict(os.environ)
        env["TILEFLOW_S_OPT_MB"] = str(tune_s_opt_mb)
        env["TILEFLOW_PREFETCH_DEPTH"] = str(tune_prefetch_depth)
        if self.ktransformers_path:
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                self.ktransformers_path if not existing else f"{self.ktransformers_path}{os.pathsep}{existing}"
            )
        return env

    def serve(self, model: ModelRecord, cfg: ServeConfig, runtime_cfg: RuntimeConfig) -> int:
        self.ensure_installed()
        tune = startup_tune(runtime_cfg)
        env = self._build_env(tune.s_opt_mb, tune.prefetch_depth)

        cmd = self._server_cmd(model=model, cfg=cfg)

        return subprocess.call(cmd, env=env)

    def run_interactive(
        self,
        model: ModelRecord,
        runtime_cfg: RuntimeConfig,
        model_id: Optional[str] = None,
        extra_args: tuple[str, ...] = (),
    ) -> int:
        self.ensure_installed()
        tune = startup_tune(runtime_cfg)
        env = self._build_env(tune.s_opt_mb, tune.prefetch_depth)
        cmd = [self.python_exe, "-m", "ktransformers.local_chat"]
        if model.local_path:
            cmd.extend(["--model_path", model.local_path])
        if model_id or model.repo_id:
            cmd.extend(["--model_id", model_id or model.repo_id])
        if model.gguf_path:
            cmd.extend(["--gguf_path", model.gguf_path])
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.call(cmd, env=env)

    def run_single_prompt(
        self,
        model: ModelRecord,
        prompt: str,
        runtime_cfg: RuntimeConfig,
        serve_cfg: Optional[ServeConfig] = None,
    ) -> dict[str, Any]:
        serve_cfg = serve_cfg or ServeConfig()
        self.ensure_installed()
        tune = startup_tune(runtime_cfg)
        env = self._build_env(tune.s_opt_mb, tune.prefetch_depth)
        proc = self._spawn_server(model, serve_cfg, env=env)
        try:
            self._wait_for_server(serve_cfg.host, serve_cfg.port, timeout_s=90.0, proc=proc)
            resolved_model = self._resolve_chat_model_id(serve_cfg.host, serve_cfg.port, model, serve_cfg)
            payload = {
                "model": resolved_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": serve_cfg.max_new_tokens,
            }
            output = self._chat_completion(serve_cfg.host, serve_cfg.port, payload)
            return {
                "output": output,
                "runtime_tune": asdict(tune),
                "engine": "ktransformers.server.main",
                "host": serve_cfg.host,
                "port": serve_cfg.port,
            }
        finally:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    @staticmethod
    def _popen_kwargs() -> dict:
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = startupinfo
        return kwargs

    @staticmethod
    def _logs_dir() -> Path:
        base = Path(os.path.expanduser("~/.tileflow")) / "logs"
        base.mkdir(parents=True, exist_ok=True)
        return base

    @classmethod
    def _rotate_log(cls, name: str, max_bytes: int = 1_000_000, backups: int = 4) -> Path:
        path = cls._logs_dir() / name
        if path.exists() and path.stat().st_size >= max_bytes:
            oldest = path.with_name(f"{name}.{backups}")
            if oldest.exists():
                oldest.unlink()
            for idx in range(backups - 1, 0, -1):
                src = path.with_name(f"{name}.{idx}")
                dst = path.with_name(f"{name}.{idx + 1}")
                if src.exists():
                    src.replace(dst)
            path.replace(path.with_name(f"{name}.1"))
        return path

    @staticmethod
    def _tail_log(path: Path, max_bytes: int = 4000) -> str:
        if not path.exists():
            return ""
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")

    def _spawn_server(self, model: ModelRecord, cfg: ServeConfig, env: dict[str, str]) -> subprocess.Popen[bytes]:
        cmd = self._server_cmd(model=model, cfg=cfg)
        log_path = self._rotate_log("backend.log")
        log_file = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            **self._popen_kwargs(),
        )
        log_file.close()
        return proc

    def _server_cmd(self, model: ModelRecord, cfg: ServeConfig) -> list[str]:
        cmd = [self.python_exe, "-m", "ktransformers.server.main"]
        if model.local_path:
            cmd.extend(["--model_path", model.local_path])
        if cfg.model_id or model.repo_id:
            cmd.extend(["--model_id", cfg.model_id or model.repo_id])
        cmd.extend(
            [
                "--port",
                str(cfg.port),
                "--host",
                cfg.host,
                "--max_new_tokens",
                str(cfg.max_new_tokens),
            ]
        )
        if model.gguf_path:
            cmd.extend(["--gguf_path", model.gguf_path])
        if cfg.gpu_split:
            cmd.extend(["--gpu_split", cfg.gpu_split])
        if cfg.extra_args:
            cmd.extend(cfg.extra_args)
        return cmd

    @staticmethod
    def _wait_for_server(host: str, port: int, timeout_s: float, proc: Optional[subprocess.Popen[bytes]] = None) -> None:
        deadline = time.time() + timeout_s
        url = f"http://{host}:{port}/v1/models"
        log_path = KTransformersBackend._logs_dir() / "backend.log"
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                tail = KTransformersBackend._tail_log(log_path)
                detail = f"\n\nRecent backend log tail:\n{tail}" if tail else ""
                raise RuntimeError(
                    f"ktransformers server exited before becoming ready. See log: {log_path}{detail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2.5) as resp:
                    if resp.status < 500:
                        return
            except Exception:
                time.sleep(1.0)
        tail = KTransformersBackend._tail_log(log_path)
        detail = f"\n\nRecent backend log tail:\n{tail}" if tail else ""
        raise TimeoutError(
            f"Timed out waiting for ktransformers server to become ready. See log: {log_path}{detail}"
        )

    @staticmethod
    def _resolve_chat_model_id(host: str, port: int, model: ModelRecord, serve_cfg: ServeConfig) -> str:
        preferred = serve_cfg.model_id or model.repo_id or model.name
        url = f"http://{host}:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=10.0) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
            data = parsed.get("data") if isinstance(parsed, dict) else None
            if isinstance(data, list) and data:
                ids = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
                if preferred in ids:
                    return preferred
                if ids:
                    return str(ids[0])
        except Exception:
            pass
        return preferred

    @staticmethod
    def _chat_completion(host: str, port: int, payload: dict) -> str:
        url = f"http://{host}:{port}/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ktransformers inference request failed: {detail}") from exc
        try:
            parsed = json.loads(raw)
            choices = parsed.get("choices") if isinstance(parsed, dict) else None
            if not isinstance(choices, list) or not choices:
                raise ValueError("Missing choices list in response.")
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else None
            if not isinstance(message, dict):
                raise ValueError("Missing message payload in response.")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("Missing assistant content in response.")
            return content
        except Exception as exc:
            tail = raw[-500:] if isinstance(raw, str) else ""
            raise RuntimeError(
                "Invalid response payload from ktransformers server."
                + (f" Payload tail: {tail}" if tail else "")
            ) from exc
