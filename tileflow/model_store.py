
from __future__ import annotations

import json
import re
import shutil
from fnmatch import fnmatch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import time
import urllib.request

from huggingface_hub import HfApi, hf_hub_url, snapshot_download


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


@dataclass(slots=True)
class ModelRecord:
    name: str
    repo_id: str
    local_path: str
    revision: str = "main"
    gguf_path: Optional[str] = None


class ModelStore:
    @staticmethod
    def _hf_token() -> Optional[str]:
        # huggingface_hub token APIs have changed across versions.
        try:
            from huggingface_hub import get_token  # type: ignore

            token = get_token()
            if token:
                return token
        except Exception:
            pass
        try:
            from huggingface_hub import HfFolder  # type: ignore

            token = HfFolder.get_token()
            if token:
                return token
        except Exception:
            pass
        try:
            from huggingface_hub.utils import get_token as utils_get_token  # type: ignore

            token = utils_get_token()
            if token:
                return token
        except Exception:
            pass
        return None

    def __init__(self, home: Optional[Path] = None) -> None:
        self.home = home or (Path.home() / ".tileflow")
        self.models_dir = self.home / "models"
        self.registry_path = self.home / "models.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.home.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save({})

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = self.registry_path.with_suffix(self.registry_path.suffix + ".corrupt")
            try:
                shutil.copyfile(self.registry_path, backup)
            except Exception:
                pass
            self._save({})
            return {}
        if not isinstance(raw, dict):
            self._save({})
            return {}
        return raw

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self.registry_path.with_suffix(self.registry_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.registry_path)

    def list_models(self) -> list[ModelRecord]:
        data = self._load()
        return [ModelRecord(**payload) for payload in data.values()]

    def get(self, name: str) -> Optional[ModelRecord]:
        data = self._load()
        payload = data.get(name)
        return ModelRecord(**payload) if payload else None

    def delete(self, name: str, delete_files: bool = False) -> bool:
        data = self._load()
        payload = data.pop(name, None)
        if payload is None:
            return False
        self._save(data)
        if delete_files:
            local_path = payload.get("local_path")
            if isinstance(local_path, str) and local_path:
                path = Path(local_path)
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
        return True

    def rename(self, old_name: str, new_name: str) -> ModelRecord:
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name or not new_name:
            raise ValueError("Both old and new model names are required.")
        if old_name == new_name:
            rec = self.get(old_name)
            if not rec:
                raise KeyError(f"Model not found: {old_name}")
            return rec

        data = self._load()
        payload = data.get(old_name)
        if payload is None:
            raise KeyError(f"Model not found: {old_name}")
        if new_name in data:
            raise ValueError(f"Model already exists: {new_name}")
        rec = ModelRecord(**payload)
        rec.name = new_name
        data.pop(old_name, None)
        data[new_name] = asdict(rec)
        self._save(data)
        return rec

    def pull(
        self,
        repo_id: str,
        name: Optional[str] = None,
        revision: str = "main",
        include: Optional[list[str]] = None,
        gguf_pattern: Optional[str] = None,
        include_gguf: bool = True,
        prefer_smallest_gguf: bool = True,
    ) -> ModelRecord:
        model_name = name or _slug(repo_id)
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        allow_patterns = self._build_allow_patterns(
            repo_id=repo_id,
            revision=revision,
            include=include,
            gguf_pattern=gguf_pattern,
            include_gguf=include_gguf,
            prefer_smallest_gguf=prefer_smallest_gguf,
        )

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(model_dir),
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        rec = ModelRecord(
            name=model_name,
            repo_id=repo_id,
            local_path=str(model_dir),
            revision=revision,
            gguf_path=self._guess_gguf_path(model_dir),
        )
        data = self._load()
        data[model_name] = asdict(rec)
        self._save(data)
        return rec

    def pull_streaming(
        self,
        repo_id: str,
        name: Optional[str] = None,
        revision: str = "main",
        include: Optional[list[str]] = None,
        gguf_pattern: Optional[str] = None,
        include_gguf: bool = True,
        prefer_smallest_gguf: bool = True,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        chunk_size: int = 1024 * 1024,
    ) -> ModelRecord:
        model_name = name or _slug(repo_id)
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        allow_patterns = self._build_allow_patterns(
            repo_id=repo_id,
            revision=revision,
            include=include,
            gguf_pattern=gguf_pattern,
            include_gguf=include_gguf,
            prefer_smallest_gguf=prefer_smallest_gguf,
        )
        files = self._matching_repo_files(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        total_bytes = sum(size for _, size in files if size is not None)
        files_total = len(files)
        done = 0
        downloaded_total = 0
        last_ts = time.time()
        last_bytes = 0

        def _emit(current_file: str, current_downloaded: int, current_size: Optional[int]) -> None:
            nonlocal last_ts, last_bytes
            if not progress_callback:
                return
            now = time.time()
            elapsed = max(1e-6, now - last_ts)
            delta = downloaded_total - last_bytes
            speed = float(delta) / elapsed
            last_ts = now
            last_bytes = downloaded_total
            progress_callback(
                {
                    "status": "running",
                    "repo_id": repo_id,
                    "downloaded_bytes": downloaded_total,
                    "total_bytes": total_bytes if total_bytes > 0 else None,
                    "bytes_per_sec": speed,
                    "files_done": done,
                    "files_total": files_total,
                    "current_file": current_file,
                    "current_file_downloaded": current_downloaded,
                    "current_file_size": current_size,
                }
            )

        for filename, size in files:
            destination = model_dir / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_suffix(destination.suffix + ".part")
            token = self._hf_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
            req = urllib.request.Request(url, headers=headers)
            current_downloaded = 0
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                while True:
                    chunk = resp.read(max(1, int(chunk_size)))
                    if not chunk:
                        break
                    out.write(chunk)
                    n = len(chunk)
                    current_downloaded += n
                    downloaded_total += n
                    _emit(filename, current_downloaded, size)
            tmp.replace(destination)
            done += 1
            _emit(filename, current_downloaded, size)

        rec = ModelRecord(
            name=model_name,
            repo_id=repo_id,
            local_path=str(model_dir),
            revision=revision,
            gguf_path=self._guess_gguf_path(model_dir),
        )
        data = self._load()
        data[model_name] = asdict(rec)
        self._save(data)
        return rec

    @staticmethod
    def _matching_repo_files(repo_id: str, revision: str, allow_patterns: list[str]) -> list[tuple[str, Optional[int]]]:
        info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        matched: list[tuple[str, Optional[int]]] = []
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            if not filename:
                continue
            if any(fnmatch(filename, pattern) for pattern in allow_patterns):
                matched.append((filename, getattr(sibling, "size", None)))
        return sorted(matched, key=lambda x: x[0])

    @staticmethod
    def estimate_download_size_bytes(
        repo_id: str,
        revision: str,
        include: Optional[list[str]],
        gguf_pattern: Optional[str],
        include_gguf: bool,
        prefer_smallest_gguf: bool,
    ) -> Optional[int]:
        allow_patterns = ModelStore._build_allow_patterns(
            repo_id=repo_id,
            revision=revision,
            include=include,
            gguf_pattern=gguf_pattern,
            include_gguf=include_gguf,
            prefer_smallest_gguf=prefer_smallest_gguf,
        )
        try:
            info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        except Exception:
            return None
        total = 0
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
            if not filename or size is None:
                continue
            if any(fnmatch(filename, pattern) for pattern in allow_patterns):
                total += int(size)
        return total

    @staticmethod
    def _guess_gguf_path(model_dir: Path) -> Optional[str]:
        ggufs = sorted(model_dir.rglob("*.gguf"))
        return str(ggufs[0]) if ggufs else None

    @staticmethod
    def _build_allow_patterns(
        repo_id: str,
        revision: str,
        include: Optional[list[str]],
        gguf_pattern: Optional[str],
        include_gguf: bool,
        prefer_smallest_gguf: bool,
    ) -> list[str]:
        patterns: list[str] = [
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "*.model",
            "*.json",
            "*.safetensors",
            "*.safetensors.index.json",
        ]
        if include:
            patterns.extend(include)

        if include_gguf:
            if gguf_pattern:
                patterns.append(gguf_pattern)
            elif prefer_smallest_gguf:
                preferred = ModelStore._preferred_q4_or_int4_gguf(repo_id=repo_id, revision=revision)
                if preferred:
                    patterns.append(preferred)
                else:
                    patterns.append("*.gguf")
            else:
                patterns.append("*.gguf")
        return sorted(set(patterns))

    @staticmethod
    def list_gguf_files(repo_id: str, revision: str) -> list[str]:
        try:
            info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        except Exception:
            return []
        files: list[str] = []
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            if filename and filename.lower().endswith(".gguf"):
                files.append(filename)
        return sorted(files)

    @staticmethod
    def list_safetensor_files(repo_id: str, revision: str) -> list[str]:
        try:
            info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        except Exception:
            return []
        files: list[str] = []
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            if filename and filename.lower().endswith(".safetensors"):
                files.append(filename)
        return sorted(files)

    @staticmethod
    def list_all_model_files(repo_id: str, revision: str) -> list[str]:
        """List all model files (safetensors and GGUF) from a repository."""
        safetensors = ModelStore.list_safetensor_files(repo_id, revision)
        gguf = ModelStore.list_gguf_files(repo_id, revision)
        return sorted(set(safetensors + gguf))

    @staticmethod
    def _preferred_q4_or_int4_gguf(repo_id: str, revision: str) -> Optional[str]:
        try:
            info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        except Exception:
            return None

        best_name: Optional[str] = None
        best_size: Optional[int] = None
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            if not filename or not filename.lower().endswith(".gguf"):
                continue
            lowered = filename.lower()
            if "q4" not in lowered and "int4" not in lowered:
                continue
            size = getattr(sibling, "size", None)
            if size is None:
                if best_name is None:
                    best_name = filename
                continue
            if best_size is None or size < best_size:
                best_size = size
                best_name = filename
        return best_name

    @staticmethod
    def _smallest_gguf(repo_id: str, revision: str) -> Optional[str]:
        try:
            info = HfApi().model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        except Exception:
            return None
        smallest_name: Optional[str] = None
        smallest_size: Optional[int] = None
        for sibling in info.siblings or []:
            filename = getattr(sibling, "rfilename", None)
            if not filename or not filename.lower().endswith(".gguf"):
                continue
            size = getattr(sibling, "size", None)
            if size is None:
                continue
            if smallest_size is None or size < smallest_size:
                smallest_size = size
                smallest_name = filename
        return smallest_name
