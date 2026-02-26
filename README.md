# TileFlow

Run massive MoE (Mixture of Experts) models on limited VRAM using rolling weight streaming.

## What It Is

TileFlow is a CLI tool that manages local LLM inference with ktransformers. It provides:

- **Auto-tuning** - Automatically optimizes tile size and prefetch depth for your hardware
- **Rolling weight streaming** - Loads expert weights on-demand from NVMe, keeping only needed tiles in VRAM
- **Model management** - Pull and manage models from Hugging Face

## How It Works

### The Tiling Concept

Traditional LLM loading requires the entire model in VRAM. TileFlow enables running massive MoE models on tiny VRAM by:

1. **Splitting experts into tiles** - MoE models have many "experts" (specialized neural network parts). TileFlow splits these across NVMe storage.

2. **On-demand loading** - Only the currently-needed experts are loaded to GPU memory.

3. **Prefetching** - While GPU processes, the next predicted experts are loaded in the background from NVMe→RAM→GPU.

4. **Auto-tuning** - Before each run, TileFlow benchmarks your NVMe→GPU bandwidth to find the optimal tile size and prefetch depth.

The tuning results are exported to ktransformers as environment variables:
- `TILEFLOW_S_OPT_MB` - optimal tile size in MB
- `TILEFLOW_PREFETCH_DEPTH` - how many tiles ahead to prefetch

### Architecture

```
tileflow/
├── cli.py              # CLI commands
├── autotune.py         # Hardware benchmarking & tuning
├── scheduler.py        # NVMe→GPU tile prefetch scheduler
├── model_store.py      # Hugging Face model management
├── backend_ktransformers.py  # ktransformers integration
└── ...
```

## Install

### Option 1: Standalone EXE (no Python needed)

Download the latest release from GitHub releases. The EXE includes Python and all dependencies.

### Option 2: Python Package

```bash
pip install -e .
pip install ktransformers
```

## Commands

### Pull a model

```bash
tileflow pull bartowski/Llama-3.3-70B-Instruct-Q4_K_M-GGUF
```

This shows all available model files - choose one to download.

### List local models

```bash
tileflow list
```

### Run interactive chat

```bash
tileflow run <model>
```

Auto-tuning runs automatically on startup.

### Run one-shot prompt

```bash
tileflow run <model> --prompt "Your question here"
```

### Start API server

```bash
tileflow serve <model>
```

### Tune runtime

```bash
tileflow tune
```

### Configure ktransformers path

```bash
tileflow backend set-path C:/path/to/ktransformers
```

## Requirements

- Windows (EXE) or Python 3.10+ (pip install)
- ktransformers (for actual inference)
- CUDA-capable GPU (for GPU inference)
- NVMe SSD recommended (for tile streaming)

## Building the EXE

```bash
pip install pyinstaller
pyinstaller --onefile --name tileflow tileflow/cli.py
```
