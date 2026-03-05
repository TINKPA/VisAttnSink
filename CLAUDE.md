# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VisAttnSink — "See What You Are Told: Visual Attention Sink in Large Multimodal Models" (ICLR 2025). Built on top of LLaVA, it mitigates object hallucination in large multimodal models by identifying and redistributing attention away from "sink" tokens back to visual tokens.

## Commands

```bash
# Run inference (primary entry point)
bash B_scripts/launch.sh "A_exps/lv1.5_7b.yml" <gpu_id>
bash B_scripts/launch.sh "A_exps/lv1.5_7b.yml" 0 debug   # with debug flag

# Direct inference
python src/inference.py --exp_config A_exps/lv1.5_7b.yml --device 0

# Multi-GPU chunked inference
python src/inference.py --exp_config A_exps/lv1.5_7b.yml --device 0 --num_chunks 4 --chunk_idx 0

# POPE hallucination evaluation
python src/eval/eval_and_save_pope.py
```

## Environment Setup

```bash
# Create conda env, then install pip deps
conda create -n visattnsink python=3.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r env_pip.txt   # includes flash-attn, transformers from git, segment_anything from git
```

Key dependency: `flash-attn==2.6.3` (requires CUDA compilation).

## Architecture

### Inference Flow

`src/inference.py:eval_model()` is the single entry point. It loads a YAML config, initializes the model + logic engine, iterates over a question JSONL file, and writes answer JSONL to `E_answers/`.

### Core Logic — The Attention Sink Mechanism (`src/logic/logic.py`)

Three class-based singletons that intercept during the forward pass of each decoder layer:

1. **DimProspector** — Runs inside `TunedLlamaModel.forward()`. After specific layers, it applies RMSNorm to hidden states, checks pre-identified "sink dimensions" (hardcoded per model in `src/logic/constants.py`), and records token positions where variance exceeds threshold `tau`.

2. **HeadFork** — Runs inside `TunedLlamaAttention.forward()`. For each attention head, checks if image attention portion ≤ `rho` AND attention summation ≥ `summ`. If both conditions met, the head is flagged as "hallucinating".

3. **VARProcessor** (Visual Attention Redistribution) — Also in `TunedLlamaAttention.forward()`. For flagged heads, reduces attention to sink tokens by factor `p` and redistributes that weight proportionally to visual tokens.

**Processing order per layer:** DimProspector (pre-attention) → HeadFork (in attention) → VARProcessor (in attention)

### Model Hierarchy (`src/model/`)

```
TunedLlavaLlamaForCausalLM        # Top-level: generation + multimodal
  └─ TunedLlavaLlamaModel          # LLaVA wrapper: vision tower + projector + LLM
       ├─ CLIPVisionTower           # Vision encoder (CLIP)
       ├─ Vision Projector          # mm_hidden_size → llm hidden_size (MLP or linear)
       └─ TunedLlamaModel           # Modified LLM with DimProspector hooks
            └─ TunedLlamaDecoderLayer[]
                 └─ TunedLlamaAttention  # HeadFork + VARProcessor hooks
```

### Global State (`src/stash.py`)

`StashEngine`, `MetadataStation`, and `ValueMonitor` are class-method singletons that track model config, prompt segmentation (system/role/image/question/response boundaries), and generation state. They are activated per-sample and cleared between samples.

### Sink Dimensions (`src/logic/constants.py`)

Pre-identified per LLM backbone:
- `llama-v2-7b`: dimensions `[2533, 1415]`
- `llama-v2-13b`: dimensions `[2100, 4743]`

Model name → LLM backbone mapping is in `MODEL_LLM` dict.

## Config Parameters (`A_exps/*.yml`)

| Parameter | Meaning |
|---|---|
| `logic` | 1=enable attention sink mechanism, 0=vanilla LLaVA |
| `tau` | Threshold for DimProspector sink token identification |
| `rho` | HeadFork: max image attention portion to flag a head |
| `summ` | HeadFork: min attention summation to flag a head |
| `p` | VARProcessor: factor to scale down sink attention (0-1) |
| `except_last_layer` | 1=skip logic on the last decoder layer |
| `conv_mode` | Conversation template (vicuna_v1, llama_2, etc.) |

## Directory Conventions

- `A_exps/` — YAML experiment configs
- `B_scripts/` — Shell launch scripts
- `C_datasets/` — Dataset root (Images/ + Questions/*.jsonl)
- `D_answers/` — Placeholder for answer outputs
- `E_answers/` — Generated answer JSONL files, organized by model name
- `src/eval/` — Evaluation scripts (POPE, ScienceQA, MMBench, TextVQA, GPT-review)

## Supported Models

LLaVA-v1.5 (7B, 13B) and LLaVA-v1.6-vicuna-13b. Adding a new model requires defining its sink dimensions in `src/logic/constants.py` and mapping it in `MODEL_LLM`.
