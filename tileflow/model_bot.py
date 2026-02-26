from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from huggingface_hub import HfApi, hf_hub_download

from tileflow.benchmark import HardwareProfile


@dataclass(slots=True)
class SuggestedModel:
    repo_id: str
    display_name: str
    est_params_b: float
    reason: str
    recommended_quant: str


_CANDIDATES: list[SuggestedModel] = [
    SuggestedModel(
        repo_id="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen2.5 3B Instruct",
        est_params_b=3.0,
        reason="Fast and responsive on modest hardware.",
        recommended_quant="Q4_K_M",
    ),
    SuggestedModel(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5 7B Instruct",
        est_params_b=7.0,
        reason="Strong quality/speed balance for most desktops.",
        recommended_quant="Q4_K_M",
    ),
    SuggestedModel(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B Instruct",
        est_params_b=8.0,
        reason="General-purpose instruct model with good output quality.",
        recommended_quant="Q4_K_M",
    ),
    SuggestedModel(
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        display_name="Qwen2.5 14B Instruct",
        est_params_b=14.0,
        reason="Higher quality if your GPU has enough memory or offload bandwidth.",
        recommended_quant="Q4_K_M",
    ),
    SuggestedModel(
        repo_id="mistralai/Mistral-Small-24B-Instruct-2501",
        display_name="Mistral Small 24B Instruct",
        est_params_b=24.0,
        reason="Large-model quality target for strong consumer/workstation GPUs.",
        recommended_quant="Q4_K_M",
    ),
    SuggestedModel(
        repo_id="Qwen/Qwen2.5-32B-Instruct",
        display_name="Qwen2.5 32B Instruct",
        est_params_b=32.0,
        reason="Use when memory and streaming bandwidth are both strong.",
        recommended_quant="Q4_K_M",
    ),
    SuggestedModel(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama 3.3 70B Instruct",
        est_params_b=70.0,
        reason="Flagship-tier output on high-end multi-GPU or aggressive offload setups.",
        recommended_quant="Q4_K_M",
    ),
]

_LLM_PIPELINE_TAGS = {
    "text-generation",
    "text2text-generation",
    "conversational",
    "fill-mask",
}
_LLM_HINT_TOKENS = (
    "instruct",
    "chat",
    "llm",
    "causal-lm",
    "decoder-only",
    "gpt",
    "llama",
    "qwen",
    "mistral",
    "gemma",
    "phi",
)
_NON_LLM_HINT_TOKENS = (
    "diffusion",
    "image-generation",
    "text-to-image",
    "image-text-to-text",
    "vision-language",
    "automatic-speech-recognition",
    "speech",
    "audio",
    "vision",
    "-vl",
    " vl ",
    "multimodal",
    "embedding",
    "sentence-transformers",
    "reward-model",
    "classification",
    "segmentation",
    "detector",
)


def _max_params_for_vram(vram_gb: float) -> float:
    if vram_gb < 6:
        return 3.0
    if vram_gb < 10:
        return 8.0
    if vram_gb < 16:
        return 14.0
    if vram_gb < 24:
        return 24.0
    if vram_gb < 40:
        return 32.0
    return 70.0


def _max_params_for_system(hw: HardwareProfile) -> float:
    if hw.cuda_available and hw.gpu_vram_gb > 0:
        return _max_params_for_vram(hw.gpu_vram_gb)
    ram = hw.total_ram_gb
    if ram < 16:
        return 3.0
    if ram < 32:
        return 7.0
    if ram < 64:
        return 14.0
    return 32.0


def _parse_params_b(text_parts: Iterable[str]) -> Optional[float]:
    pool = " ".join(p for p in text_parts if p).lower()
    candidates: list[float] = []

    for m in re.finditer(r"(\d+(?:[._]\d+)?)\s*b\b", pool):
        raw = m.group(1).replace("_", ".")
        try:
            val = float(raw)
        except Exception:
            continue
        if 0.5 <= val <= 200.0:
            candidates.append(val)

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)b\b", pool):
        try:
            val = float(m.group(1))
        except Exception:
            continue
        if 0.5 <= val <= 200.0:
            candidates.append(val)

    if not candidates:
        return None
    return max(candidates)


def _is_llm_model(repo_id: str, tags: list[str], pipeline_tag: Optional[str]) -> bool:
    tag = (pipeline_tag or "").strip().lower()
    haystack = " ".join([repo_id, *tags]).lower()
    if any(token in haystack for token in _NON_LLM_HINT_TOKENS):
        return False
    if tag and tag not in _LLM_PIPELINE_TAGS:
        return False
    if tag in _LLM_PIPELINE_TAGS:
        return True
    return any(token in haystack for token in _LLM_HINT_TOKENS)


def _recommended_quant(params_b: Optional[float], hw: HardwareProfile) -> str:
    budget = _max_params_for_system(hw)
    size = params_b if params_b is not None else budget
    if size <= 8.0:
        return "Q4_K_M"
    if size <= max(14.0, budget):
        return "Q4_K_M"
    return "Q3_K_M"


def _suggestion_reason(
    *,
    params_b: Optional[float],
    hw_budget: float,
    downloads: int,
    has_gguf_tag: bool,
) -> str:
    if params_b is None:
        fit = "Model size not published; ranking by popularity and compatibility tags."
    elif params_b <= hw_budget:
        fit = f"Fits your hardware budget (~{hw_budget:.0f}B target)."
    else:
        fit = f"Larger than your ideal budget (~{hw_budget:.0f}B), may rely on offload."
    distro = "High community adoption." if downloads >= 100_000 else "Emerging model with moderate adoption."
    fmt = "GGUF/quantized variants likely available." if has_gguf_tag else "Check quantized variants in model files."
    return f"{fit} {distro} {fmt}"


def _score_for_hardware(
    *,
    params_b: Optional[float],
    budget_b: float,
    downloads: int,
    likes: int,
    pipeline_tag: Optional[str],
    repo_id: str,
    tags: list[str],
) -> float:
    if params_b is None:
        fit_score = 0.58
    elif params_b <= budget_b:
        fit_score = 1.0 - min(0.45, (budget_b - params_b) / max(1.0, budget_b) * 0.45)
    else:
        fit_score = max(0.0, 1.0 - ((params_b - budget_b) / max(1.0, budget_b)) * 0.9)
    popularity = min(1.0, (downloads / 400_000.0)) * 0.8 + min(0.2, likes / 25_000.0)
    instruct_bonus = 0.12 if "instruct" in repo_id.lower() else 0.0
    gguf_bonus = 0.08 if any("gguf" in t.lower() for t in tags) else 0.0
    pipeline_bonus = 0.08 if (pipeline_tag or "").lower() == "text-generation" else 0.0
    return fit_score * 0.68 + popularity * 0.24 + instruct_bonus + gguf_bonus + pipeline_bonus


def _display_name(repo_id: str) -> str:
    tail = repo_id.split("/")[-1]
    return tail.replace("_", " ").replace("-", " ")


def suggested_models_for_hardware(hw: HardwareProfile, limit: int = 6) -> list[SuggestedModel]:
    requested = max(1, int(limit))
    budget = _max_params_for_system(hw)
    api = HfApi()
    try:
        # Pull a larger pool then rank by hardware fit + adoption signal.
        fetched = api.list_models(
            search="instruct",
            sort="downloads",
            direction=-1,
            limit=max(60, requested * 20),
            full=True,
        )
        scored: list[tuple[float, dict[str, Any]]] = []
        seen: set[str] = set()
        for model in fetched:
            repo_id = getattr(model, "id", "") or ""
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            tags = list(getattr(model, "tags", None) or [])
            pipeline_tag = getattr(model, "pipeline_tag", None)
            if not _is_llm_model(repo_id, tags, pipeline_tag):
                continue
            params_b = _parse_params_b([repo_id, *tags])
            downloads = int(getattr(model, "downloads", 0) or 0)
            likes = int(getattr(model, "likes", 0) or 0)
            score = _score_for_hardware(
                params_b=params_b,
                budget_b=budget,
                downloads=downloads,
                likes=likes,
                pipeline_tag=pipeline_tag,
                repo_id=repo_id,
                tags=tags,
            )
            scored.append(
                (
                    score,
                    {
                        "repo_id": repo_id,
                        "params_b": params_b,
                        "downloads": downloads,
                        "tags": tags,
                    },
                )
            )
        scored.sort(key=lambda x: (x[0], x[1]["downloads"]), reverse=True)
        suggestions: list[SuggestedModel] = []
        for _, item in scored[:requested]:
            params_b = item["params_b"]
            tags = item["tags"]
            suggestions.append(
                SuggestedModel(
                    repo_id=item["repo_id"],
                    display_name=_display_name(item["repo_id"]),
                    est_params_b=float(params_b) if params_b is not None else budget,
                    reason=_suggestion_reason(
                        params_b=params_b,
                        hw_budget=budget,
                        downloads=int(item["downloads"]),
                        has_gguf_tag=any("gguf" in t.lower() for t in tags),
                    ),
                    recommended_quant=_recommended_quant(params_b, hw),
                )
            )
        if suggestions:
            return suggestions
    except Exception:
        pass

    # Fallback when HF API is unavailable.
    primary = [m for m in _CANDIDATES if m.est_params_b <= budget]
    fallback = [m for m in _CANDIDATES if m.est_params_b > budget]
    ranked = primary + fallback
    return ranked[:requested]


def search_hf_models(query: str, limit: int = 12) -> list[dict[str, str | int | float | None]]:
    query = (query or "").strip()
    effective_query = query or "instruct"
    requested = max(1, int(limit))
    api = HfApi()
    results = api.list_models(
        search=effective_query,
        sort="downloads",
        direction=-1,
        limit=max(40, requested * 12),
        full=True,
    )
    items: list[dict[str, str | int | float | None]] = []
    seen: set[str] = set()
    for model in results:
        repo_id = getattr(model, "id", "") or ""
        if not repo_id or repo_id in seen:
            continue
        seen.add(repo_id)
        tags = list(getattr(model, "tags", None) or [])
        pipeline_tag = getattr(model, "pipeline_tag", None)
        if not _is_llm_model(repo_id, tags, pipeline_tag):
            continue
        params_b = _parse_params_b([repo_id, *tags])
        items.append(
            {
                "repo_id": repo_id,
                "downloads": model.downloads,
                "likes": model.likes,
                "last_modified": str(model.last_modified) if model.last_modified else None,
                "pipeline_tag": pipeline_tag,
                "params_b": params_b,
            }
        )
        if len(items) >= requested:
            break
    return items


def _quant_hint(filename: str) -> str:
    lowered = filename.lower()
    if "q4" in lowered or "int4" in lowered:
        return "Balanced quality/speed; common recommendation."
    if "q8" in lowered or "int8" in lowered or "fp8" in lowered:
        return "Higher quality, heavier VRAM/runtime."
    if "q3" in lowered:
        return "Smaller/faster, lower quality."
    if "q2" in lowered or "int2" in lowered:
        return "Very small footprint, significant quality loss."
    if "fp16" in lowered or "bf16" in lowered:
        return "Highest quality path, heavier memory use."
    return "Quantization level inferred from filename."


def _extract_description_and_images(readme_text: str) -> tuple[str, list[str]]:
    text = (readme_text or "").lstrip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]

    image_urls: list[str] = []
    for match in re.finditer(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", text):
        image_urls.append(match.group(1).strip())
    for match in re.finditer(r"<img[^>]+src=[\"'](https?://[^\"']+)[\"']", text, flags=re.IGNORECASE):
        image_urls.append(match.group(1).strip())
    deduped_images = list(dict.fromkeys(image_urls))

    lines = [line.strip() for line in text.splitlines()]
    paragraph_lines: list[str] = []
    for line in lines:
        if not line:
            if paragraph_lines:
                break
            continue
        if line.startswith("#"):
            continue
        if line.startswith("!["):
            continue
        if line.startswith("[!["):
            continue
        paragraph_lines.append(line)
    description = " ".join(paragraph_lines).strip()
    return description, deduped_images


def get_model_details(repo_id: str, revision: str = "main") -> dict:
    api = HfApi()
    info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
    siblings = info.siblings or []
    gguf_files = sorted(
        getattr(s, "rfilename", "")
        for s in siblings
        if getattr(s, "rfilename", "").lower().endswith(".gguf")
    )
    safetensors = sorted(
        getattr(s, "rfilename", "")
        for s in siblings
        if getattr(s, "rfilename", "").lower().endswith(".safetensors")
    )
    card_data = getattr(info, "cardData", None) or {}
    description = (
        card_data.get("description")
        or card_data.get("model_description")
        or card_data.get("summary")
        or ""
    )
    image_urls: list[str] = []
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", revision=revision)
        readme_text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
        md_description, md_images = _extract_description_and_images(readme_text)
        if not description and md_description:
            description = md_description
        image_urls = md_images[:6]
    except Exception:
        image_urls = []
    tags = list(getattr(info, "tags", None) or [])
    quant_options = [{"filename": name, "hint": _quant_hint(name)} for name in gguf_files]
    return {
        "repo_id": repo_id,
        "revision": revision,
        "description": description.strip(),
        "tags": tags[:20],
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "quant_options": quant_options,
        "has_gguf": bool(gguf_files),
        "safetensors_count": len(safetensors),
        "image_urls": image_urls,
    }
