# TileFlow v0.1.0 - CLI-Only Release

## Overview

TileFlow is now a **CLI-only** application. The frontend/desktop GUI has been removed in favor of a simpler, more maintainable command-line interface.

## What's New

### CLI-Only Architecture
- Removed React/Vite frontend
- Removed pywebview desktop wrapper
- Pure command-line interface using the `clui` TUI library

### Simplified Dependencies
- Only includes essential packages: `numpy`, `huggingface_hub`, `clui`
- Build size reduced to ~68MB standalone executable
- No more bundling unnecessary ML libraries

### Commands
```
tileflow --help
tileflow pull <model>     # Download models from Hugging Face
tileflow list             # List locally pulled models
tileflow run <model>     # Run inference
tileflow serve           # Start HTTP server
tileflow tune            # Run auto-tuning
```

## Installation

### From PyPI (coming soon)
```bash
pip install tileflow
```

### From Source
```bash
pip install -e .
```

### Standalone Executable
Download `tileflow.exe` from the releases page, or build it yourself:
```bash
python build_exe.py
```

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA support (for inference)
- torch (installed separately as needed)

## Features
- Rolling weight streaming for MoE models
- NVMe→GPU tile prefetch scheduling
- Auto-tuning for optimal performance
- Ollama-like CLI experience
- ktransformers backend integration

## License
MIT
