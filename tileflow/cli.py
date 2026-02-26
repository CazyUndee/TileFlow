from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tileflow.autotune import startup_tune
from tileflow.backend_ktransformers import KTransformersBackend, ServeConfig
from tileflow.config import RuntimeConfig
from tileflow.model_store import ModelRecord, ModelStore
from tileflow.scheduler import PrefetchScheduler, TileTask
from tileflow.settings import SettingsStore

try:
    from clui.clui import ui as clui_ui
except Exception:  # pragma: no cover - optional dependency
    clui_ui = None


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(n in text for n in needles)


def _quant_hint(filename: str) -> str:
    lowered = filename.lower()

    primary = "quant level varies; check model card for exact quality vs speed tradeoff"

    if _has_any(lowered, ("fp32", "f32")):
        primary = "fp32 - full precision, highest quality, extremely heavy to run (usually not worth it)"
    elif _has_any(lowered, ("bf16", "bfloat16")):
        primary = "bf16 - nearly same quality as fp32, around 2x faster, still relatively heavy"
    elif _has_any(lowered, ("fp16", "f16", "half")):
        primary = "fp16 - similar to bf16, slightly faster, minor numeric precision differences"
    elif _has_any(lowered, ("fp8", "int8", "q8")):
        primary = "fp8/int8 - very good quality, much faster than fp16, widely used for efficiency"
    elif _has_any(lowered, ("q4", "int4")):
        primary = "q4/int4 - good quality, much lighter to run, common sweet spot for consumer GPUs (recommended)"
    elif "q3" in lowered:
        primary = "q3 - moderate to low quality, somewhat easier to run than q4/int4"
    elif _has_any(lowered, ("q2", "int2")):
        primary = "int2/q2 - low precision, mainly for very low-end hardware"
    elif _has_any(lowered, ("q1", "int1", "binary", "1bit", "1-bit")):
        primary = "q1/int1/binary - lowest precision, significant quality loss, only for extremely limited hardware"

    details: list[str] = []

    # Mixed 4/8-bit formats like w8a4, 8x4, int8-int4.
    if re.search(r"(w8a4|8x4|int8[-_ ]?(int4|a4)|q8[-_ ]?q4)", lowered):
        details.append("4/8-bit mixed quantization")

    # Per-channel / per-tensor scaling hints.
    if _has_any(lowered, ("perchannel", "per_channel", "per-channel", "channelwise", "chwise")):
        details.append("per-channel scaling (usually more accurate)")
    elif _has_any(lowered, ("pertensor", "per_tensor", "per-tensor", "tensorwise")):
        details.append("per-tensor scaling (simpler, sometimes less accurate)")

    # INT weights with FP16 activations.
    if re.search(r"(w(8|4)a16|int(8|4)[-_ ]?fp16|fp16[-_ ]?act)", lowered):
        details.append("quantized weights + fp16 activations")

    if details:
        return f"{primary}; {', '.join(details)}"
    return primary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TileFlow Ollama-like CLI with ktransformers backend.")
    sub = p.add_subparsers(dest="cmd", required=True)

    tune = sub.add_parser("tune", help="Run startup auto-tune and print JSON.")
    _add_runtime_args(tune)
    tune.add_argument("--simulate", action="store_true", help="Run async scheduler simulation.")
    tune.add_argument(
        "--real-transfers",
        action="store_true",
        help="With --simulate, use real file reads and real H2D copy when possible.",
    )
    tune.add_argument(
        "--tile-dir",
        help="Directory with tile payloads named <tile_id>.bin used by --real-transfers.",
    )
    tune.add_argument(
        "--tile-manifest",
        help="JSON file mapping tile IDs to file paths. Supports {'e14': 'path/to/file.bin'} or {'tiles': {...}}.",
    )
    tune.add_argument(
        "--tile-template",
        default="{tile_id}.bin",
        help="Filename template fallback for tile IDs not present in manifest, e.g. '{tile_id}.bin' or 'expert_{tile_id}.dat'.",
    )

    pull = sub.add_parser("pull", help="Download a model snapshot from Hugging Face.")
    pull.add_argument("repo_id", help="Hugging Face repo id, e.g. bartowski/DeepSeek-R1-GGUF")
    pull.add_argument("--name", help="Optional local alias override.")
    pull.add_argument("--revision", default="main")
    pull.add_argument("--gguf-pattern", help="GGUF glob, e.g. *Q4_K_M.gguf")
    pull.add_argument(
        "--all-gguf",
        action="store_true",
        help="Download all GGUF files (can consume significant storage).",
    )
    pull.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF files. Useful for HF safetensors models.",
    )
    pull.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra include pattern(s). Can be repeated.",
    )

    sub.add_parser("list", help="List locally pulled models.")

    rm = sub.add_parser("rm", help="Remove a model from local registry.")
    rm.add_argument("name", help="Local model alias to remove.")
    rm.add_argument(
        "--delete-files",
        action="store_true",
        help="Also delete the model directory from disk.",
    )

    rename = sub.add_parser("rename", help="Rename a local model alias.")
    rename.add_argument("old_name")
    rename.add_argument("new_name")

    run = sub.add_parser("run", help="Run model (interactive) or single prompt.")
    run.add_argument("model", nargs="?", help="Local alias from `tileflow list`, path, or HF repo id.")
    run.add_argument("--prompt", help="If set, run one prompt via server API and exit.")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=11434)
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--model-path", help="Explicit local model path.")
    run.add_argument("--hf-model-id", help="Explicit Hugging Face model id.")
    run.add_argument("--gguf-path", help="Explicit GGUF file path.")
    run.add_argument("--ktransformers-path", help="Path to local ktransformers fork root.")
    run.add_argument("--backend-arg", action="append", default=[], help="Pass-through backend arg.")
    _add_runtime_args(run)

    serve = sub.add_parser("serve", help="Start ktransformers HTTP server.")
    serve.add_argument("model", nargs="?", help="Local alias from `tileflow list`, path, or HF repo id.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11434)
    serve.add_argument("--max-new-tokens", type=int, default=512)
    serve.add_argument("--gpu-split", help="Pass-through ktransformers gpu split, e.g. 80")
    serve.add_argument("--model-path", help="Explicit local model path.")
    serve.add_argument("--hf-model-id", help="Explicit Hugging Face model id.")
    serve.add_argument("--gguf-path", help="Explicit GGUF file path.")
    serve.add_argument("--ktransformers-path", help="Path to local ktransformers fork root.")
    serve.add_argument("--backend-arg", action="append", default=[], help="Pass-through backend arg.")
    _add_runtime_args(serve)

    backend = sub.add_parser("backend", help="Configure local ktransformers fork path.")
    backend_sub = backend.add_subparsers(dest="backend_cmd", required=True)
    backend_show = backend_sub.add_parser("show", help="Show current backend settings.")
    backend_show.set_defaults(backend_cmd="show")
    backend_set = backend_sub.add_parser("set-path", help="Set local ktransformers fork path.")
    backend_set.add_argument("path")
    backend_clear = backend_sub.add_parser("clear-path", help="Clear saved fork path.")
    backend_clear.set_defaults(backend_cmd="clear-path")
    return p


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefetch-depth", type=int, default=4)
    parser.add_argument("--headroom", type=float, default=0.45)
    parser.add_argument("--target-util", type=float, default=0.90)


def _runtime_cfg(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        prefetch_depth=args.prefetch_depth,
        headroom=args.headroom,
        target_util=args.target_util,
    )


def _load_tile_manifest(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("tiles"), dict):
        data = data["tiles"]
    if not isinstance(data, dict):
        raise ValueError("Tile manifest must be a JSON object mapping tile_id -> path.")

    mapping: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            mapping[key] = value
    return mapping


def _build_tile_path_resolver(args: argparse.Namespace):
    tile_dir = Path(args.tile_dir).expanduser().resolve() if args.tile_dir else None
    tile_manifest: dict[str, str] = {}
    if args.tile_manifest:
        tile_manifest = _load_tile_manifest(Path(args.tile_manifest).expanduser().resolve())
    tile_template = args.tile_template
    if tile_template:
        try:
            _ = tile_template.format(tile_id="sample")
        except KeyError as exc:
            raise ValueError(f"Invalid --tile-template placeholder: {exc}") from exc

    def _resolve_tile_path(tile_id: str) -> Optional[Path]:
        if tile_id in tile_manifest:
            candidate = Path(tile_manifest[tile_id]).expanduser()
            if not candidate.is_absolute() and tile_dir:
                candidate = tile_dir / candidate
            return candidate.resolve()
        if tile_template:
            rendered = tile_template.format(tile_id=tile_id)
            candidate = Path(rendered).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            base = tile_dir or Path.cwd()
            return (base / candidate).resolve()
        return None

    return _resolve_tile_path


def _cmd_tune(args: argparse.Namespace) -> int:
    cfg = _runtime_cfg(args)
    tune = startup_tune(cfg)
    if clui_ui and sys.stdout.isatty():
        headers = ["s_opt_mb", "prefetch_depth", "predicted_util_pct", "b_io_gbs"]
        rows = [[tune.s_opt_mb, tune.prefetch_depth, f"{tune.predicted_util_pct:.2f}", f"{tune.b_io_gbs:.2f}"]]
        print(clui_ui.table(headers, rows))
    else:
        print(json.dumps(asdict(tune), indent=2))
    if not args.simulate:
        return 0

    tile_resolver = _build_tile_path_resolver(args) if args.real_transfers else None

    scheduler = PrefetchScheduler(
        cfg=cfg,
        tune=tune,
        real_transfers=bool(args.real_transfers),
        tile_path_resolver=tile_resolver,
    )
    tasks = [
        TileTask(token_idx=0, tile_id="e10", predicted_next_tiles=["e14", "e18", "e22"]),
        TileTask(token_idx=1, tile_id="e14", predicted_next_tiles=["e18", "e22", "e02"]),
        TileTask(token_idx=2, tile_id="e18", predicted_next_tiles=["e22", "e02", "e06"]),
    ]
    asyncio.run(scheduler.run_stream(tasks))
    print(json.dumps({"transfer_stats": scheduler.get_transfer_stats()}, indent=2))
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    store = ModelStore()
    
    # Get all available model files (safetensors + GGUF)
    all_files = store.list_all_model_files(repo_id=args.repo_id, revision=args.revision)
    
    if not all_files:
        print(f"No model files found in {args.repo_id}", file=sys.stderr)
        return 1
    
    # Create labels with helpful information for each file
    labels = []
    for filename in all_files:
        lowered = filename.lower()
        file_type = "GGUF" if lowered.endswith(".gguf") else "Safetensors"
        hint = _quant_hint(filename)
        labels.append(f"{filename} ({file_type}) - {hint}")
    
    selected_pattern = None
    
    # Interactive mode - present all options to user
    if sys.stdin.isatty():
        if clui_ui:
            selected_idx = clui_ui.select(
                labels,
                title="Choose a model file to download:",
                search=True,
            ).run()
            if selected_idx == -1:
                print("Selection cancelled.", file=sys.stderr)
                return 2
            selected_pattern = all_files[selected_idx]
        else:
            print("Choose a model file to download:")
            for idx, label in enumerate(labels, start=1):
                print(f"{idx}. {label}")
            while True:
                raw = input(f"Selection (1-{len(labels)}): ").strip()
                if not raw.isdigit():
                    print("Enter a number.")
                    continue
                choice = int(raw)
                if 1 <= choice <= len(labels):
                    selected_pattern = all_files[choice - 1]
                    break
                print("Selection out of range.")
    else:
        # Non-interactive mode - list all available files
        print("Available model files:", file=sys.stderr)
        for f in all_files:
            print(f"  - {f}", file=sys.stderr)
        print("\nPlease run interactively or specify a pattern with --gguf-pattern or --include", file=sys.stderr)
        return 1
    
    # Determine if it's a GGUF or safetensors file
    is_gguf = selected_pattern.lower().endswith(".gguf")
    
    rec = store.pull(
        repo_id=args.repo_id,
        name=args.name,
        revision=args.revision,
        include=args.include,
        gguf_pattern=selected_pattern if is_gguf else None,
        include_gguf=is_gguf,
        prefer_smallest_gguf=False,
    )
    if clui_ui and sys.stdout.isatty():
        headers = ["name", "repo_id", "local_path", "revision", "gguf_path"]
        rows = [[rec.name, rec.repo_id, rec.local_path, rec.revision, rec.gguf_path or ""]]
        print(clui_ui.table(headers, rows))
    else:
        print(json.dumps(asdict(rec), indent=2))
    return 0


def _cmd_list() -> int:
    store = ModelStore()
    models = store.list_models()
    if clui_ui and sys.stdout.isatty():
        headers = ["name", "repo_id", "local_path", "revision", "gguf_path"]
        rows = [[m.name, m.repo_id, m.local_path, m.revision, m.gguf_path or ""] for m in models]
        print(clui_ui.table(headers, rows))
    else:
        print(json.dumps([asdict(m) for m in models], indent=2))
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    store = ModelStore()
    deleted = store.delete(name=args.name, delete_files=bool(args.delete_files))
    if not deleted:
        print(f"Model not found: {args.name}", file=sys.stderr)
        return 2
    print(json.dumps({"deleted": True, "name": args.name, "delete_files": bool(args.delete_files)}, indent=2))
    return 0


def _cmd_rename(args: argparse.Namespace) -> int:
    store = ModelStore()
    try:
        rec = store.rename(old_name=args.old_name, new_name=args.new_name)
    except KeyError:
        print(f"Model not found: {args.old_name}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(asdict(rec), indent=2))
    return 0


def _resolve_model(args: argparse.Namespace) -> Optional[ModelRecord]:
    if args.model_path or args.hf_model_id:
        source = args.model_path or args.hf_model_id
        return ModelRecord(
            name=Path(source).name if source else "model",
            repo_id=args.hf_model_id or "",
            local_path=args.model_path or "",
            gguf_path=args.gguf_path,
        )

    if not args.model:
        return None

    store = ModelStore()
    rec = store.get(args.model)
    if rec:
        if args.gguf_path:
            rec.gguf_path = args.gguf_path
        return rec

    candidate = args.model
    is_hf_id = "/" in candidate and not Path(candidate).exists()
    if is_hf_id:
        return ModelRecord(name=candidate.split("/")[-1], repo_id=candidate, local_path="", gguf_path=args.gguf_path)
    return ModelRecord(name=Path(candidate).name, repo_id="", local_path=candidate, gguf_path=args.gguf_path)


def _resolve_ktransformers_path(args: argparse.Namespace) -> Optional[str]:
    if getattr(args, "ktransformers_path", None):
        return args.ktransformers_path
    return SettingsStore().load().ktransformers_path


def _cmd_run(args: argparse.Namespace) -> int:
    rec = _resolve_model(args)
    if not rec:
        print("No model provided. Pass an alias/path/repo id or set --model-path/--hf-model-id.", file=sys.stderr)
        return 2

    backend = KTransformersBackend(ktransformers_path=_resolve_ktransformers_path(args))
    cfg = _runtime_cfg(args)
    if not args.prompt:
        return backend.run_interactive(
            rec,
            cfg,
            model_id=args.hf_model_id,
            extra_args=tuple(args.backend_arg),
        )

    output = backend.run_single_prompt(
        model=rec,
        prompt=args.prompt,
        runtime_cfg=cfg,
        serve_cfg=ServeConfig(
            host=args.host,
            port=args.port,
            max_new_tokens=args.max_new_tokens,
            model_id=args.hf_model_id,
            extra_args=tuple(args.backend_arg),
        ),
    )
    print(output)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    rec = _resolve_model(args)
    if not rec:
        print("No model provided. Pass an alias/path/repo id or set --model-path/--hf-model-id.", file=sys.stderr)
        return 2

    backend = KTransformersBackend(ktransformers_path=_resolve_ktransformers_path(args))
    cfg = _runtime_cfg(args)
    return backend.serve(
        rec,
        ServeConfig(
            host=args.host,
            port=args.port,
            gpu_split=args.gpu_split,
            max_new_tokens=args.max_new_tokens,
            model_id=args.hf_model_id,
            extra_args=tuple(args.backend_arg),
        ),
        runtime_cfg=cfg,
    )


def _cmd_backend(args: argparse.Namespace) -> int:
    store = SettingsStore()
    settings = store.load()
    if args.backend_cmd == "show":
        print(json.dumps(asdict(settings), indent=2))
        return 0
    if args.backend_cmd == "set-path":
        root = Path(args.path).expanduser().resolve()
        if not (root / "ktransformers").exists():
            print(f"Invalid fork path: missing package at {root / 'ktransformers'}", file=sys.stderr)
            return 2
        settings.ktransformers_path = str(root)
        store.save(settings)
        print(json.dumps(asdict(settings), indent=2))
        return 0
    if args.backend_cmd == "clear-path":
        settings.ktransformers_path = None
        store.save(settings)
        print(json.dumps(asdict(settings), indent=2))
        return 0
    return 2


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "tune":
        raise SystemExit(_cmd_tune(args))
    if args.cmd == "pull":
        raise SystemExit(_cmd_pull(args))
    if args.cmd == "list":
        raise SystemExit(_cmd_list())
    if args.cmd == "rm":
        raise SystemExit(_cmd_rm(args))
    if args.cmd == "rename":
        raise SystemExit(_cmd_rename(args))
    if args.cmd == "run":
        raise SystemExit(_cmd_run(args))
    if args.cmd == "serve":
        raise SystemExit(_cmd_serve(args))
    if args.cmd == "backend":
        raise SystemExit(_cmd_backend(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()

