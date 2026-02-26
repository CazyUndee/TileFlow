# TileFlow
The first inference pipeline that achieves near zero performance loss that allows even mid range gpus to run frontier LLMs.

### Why we need TileFLow
Right now, AI inference apps (like Ollama and LMStudio) load the entire model into VRAM before running anything. For high-end datacentres this is fine because there is lots of VRAM in datacentre GPUS. For consumer hardware, this is not possible for most modern LLMs, which can require up to terabytes.

### How it works
Here is an explanation of how it works (It is oversimplified, but the main points are correct):

TileFLow uses a completely different way to run LLMs, instead of loading the entire model into VRAM and only grabbbing a certain chunk of it from the VRAM, we send the chunks directly from storage intelligently, so that it appears as if its doing the same thing as before, basically, we are using storage as a huge version of VRAM, meaning that we can finally run the model!


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

## How use

- Get .exe from releases
- Go on [Huggingface](https://huggingface.co) and find the model you want to use
- Double click exe and type in ```tileflow run <model``` (so if you want to run GPT 2, you would do ```tileflow run openai-community/gpt2>```

