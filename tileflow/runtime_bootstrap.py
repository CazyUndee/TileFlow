from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _run(cmd: list[str], timeout_s: int = 1800) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode, proc.stdout or ""
    except Exception as exc:
        return 1, str(exc)


def _has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    code, out = _run([nvidia_smi, "-L"], timeout_s=15)
    return code == 0 and "GPU" in out


def _pick_python_for_pip() -> str:
    # In frozen mode, sys.executable is tileflow.exe and cannot run `-m pip`.
    # Prefer a real python from PATH.
    path_python = shutil.which("python")
    if path_python:
        return path_python
    return sys.executable


def _install_torch(cuda: bool) -> tuple[bool, str]:
    py = _pick_python_for_pip()
    index = "https://download.pytorch.org/whl/cu121" if cuda else "https://download.pytorch.org/whl/cpu"
    code, out = _run([py, "-m", "pip", "install", "--upgrade", "torch", "--index-url", index], timeout_s=2400)
    return code == 0, out


def _local_kt_kernel_path() -> Path | None:
    # Dev layout from current working directory.
    dev = Path.cwd() / "ktransformers" / "kt-kernel"
    if (dev / "pyproject.toml").exists():
        return dev
    # Dev layout resolved from package location.
    package_root = Path(__file__).resolve().parents[1]
    local = package_root / "ktransformers" / "kt-kernel"
    if (local / "pyproject.toml").exists():
        return local
    # Frozen onedir layout from build script add-data target "kt-kernel-src"
    if hasattr(sys, "_MEIPASS"):
        frozen = Path(getattr(sys, "_MEIPASS")) / "kt-kernel-src"
        if (frozen / "pyproject.toml").exists():
            return frozen
    return None


def _install_kt_kernel() -> tuple[bool, str]:
    src = _local_kt_kernel_path()
    if src is None:
        return False, "Local kt-kernel source not found."
    py = _pick_python_for_pip()
    code, out = _run([py, "-m", "pip", "install", str(src)], timeout_s=3600)
    return code == 0, out


def ensure_runtime_ready(auto_install: bool = True) -> tuple[bool, str]:
    """
    Best-effort runtime bootstrap:
    - Ensure torch exists (CUDA wheel if NVIDIA GPU detected, else CPU wheel).
    - Ensure kt_kernel exists (install from bundled/local source if missing).
    """
    logs: list[str] = []
    success = True
    if not auto_install:
        return True, "Auto-install disabled."

    # torch bootstrap
    has_torch = _has_module("torch")
    if not has_torch:
        want_cuda = _has_nvidia_gpu()
        ok, out = _install_torch(cuda=want_cuda)
        success = success and ok
        logs.append(("Installed torch." if ok else "Failed to install torch.") + ("\n" + out[-1500:] if out else ""))
    else:
        logs.append("torch already available.")

    # kt-kernel bootstrap
    has_kernel = _has_module("kt_kernel")
    if not has_kernel:
        ok, out = _install_kt_kernel()
        success = success and ok
        logs.append(("Installed kt-kernel." if ok else "Failed to install kt-kernel.") + ("\n" + out[-1500:] if out else ""))
    else:
        logs.append("kt-kernel already available.")

    # Soft-fail: report unsuccessful bootstrap but do not raise here.
    return success, "\n".join(logs)
