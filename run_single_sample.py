"""Run a single ScienceQA sample with detailed output for debugging/understanding."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image

from src.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from src.utils import disable_torch_init
from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation

# --- Config ---
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "vicuna_v1"
DEVICE = "cuda:0"
TAU, RHO, P, SUMM = 20, 0.5, 0.6, 0.2

# Load the second question (a simple geography question)
DATA_DIR = "/xuanwu-tank/east/antarachugh/projects/VisAttnSink_data/ScienceQA"
with open(f"{DATA_DIR}/Questions/test-questions.jsonl") as f:
    lines = [json.loads(l) for l in f]
sample = lines[1]  # New Hampshire colony question

print("=" * 70)
print("SAMPLE INFO")
print("=" * 70)
print(f"QID:    {sample['qid']}")
print(f"Image:  {sample['image']}")
print(f"Label:  {sample['label']}")
print(f"Question:\n{sample['text']}")
print()

# --- Load model ---
print("Loading model...")
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, attn_implementation="eager", device_map=DEVICE
)

# --- Activate logic ---
MetadataStation.activate()
MetadataStation.export_model_config(model.config)
LogicEngine.activate(tau=TAU, rho=RHO, summ=SUMM, p=P, except_last_layer=1)

print(f"\nModel: {model_name}")
print(f"Logic: ON (tau={TAU}, rho={RHO}, p={P}, summ={SUMM})")
print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}")
print(f"Sink dims: {LogicEngine.dim_sink}")
print()

# --- Prepare input ---
qs = sample['text']
if model.config.mm_use_im_start_end:
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
else:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

conv = conv_templates[CONV_MODE].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, conv=conv, return_tensors="pt").unsqueeze(0).to(DEVICE)
image = Image.open(os.path.join(DATA_DIR, "Images", sample['image'])).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)[0]

print("=" * 70)
print("INPUT DETAILS")
print("=" * 70)
print(f"Image size: {image.size}")
print(f"Image tensor shape: {image_tensor.shape}")
print(f"Input IDs shape: {input_ids.shape} ({input_ids.shape[1]} tokens)")
print(f"Prompt (first 200 chars):\n{prompt[:200]}...")
print()

# --- Generate ---
print("=" * 70)
print("RUNNING INFERENCE (with VisAttnSink logic)")
print("=" * 70)

setattr(model, "tokenizer", tokenizer)
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().to(DEVICE),
        image_sizes=[image.size],
        return_dict_in_generate=True,
        output_attentions=True,
        output_hidden_states=True,
        do_sample=False,
        max_new_tokens=128,
        use_cache=True,
    )

generated = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

# --- Report ---
print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Generated response: {generated}")
print(f"Ground truth label: {sample['label']}")

# Extract just the answer letter
answer = generated.strip()
if len(answer) >= 1 and answer[0] in "ABCDE":
    pred = answer[0]
else:
    pred = answer
print(f"Predicted: {pred}")
print(f"Correct:   {'YES' if pred == sample['label'] else 'NO'}")

# --- Logic engine stats ---
print()
print("=" * 70)
print("VISATTNSINK LOGIC STATS")
print("=" * 70)

# DimProspector: which layers found sink tokens
sink_layers = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in LogicEngine.indices.items() if len(v) > 0}
print(f"DimProspector found sink tokens in {len(sink_layers)} layers")
if sink_layers:
    first_layer = min(sink_layers.keys())
    print(f"  Layer {first_layer}: {len(sink_layers[first_layer])} sink token positions")

# HeadFork: which layers had forked heads
forked_layers = {k: v for k, v in LogicEngine.forked_head.items() if isinstance(v, torch.Tensor) and len(v) > 0}
print(f"HeadFork flagged heads in {len(forked_layers)} layers")
if forked_layers:
    first_layer = min(forked_layers.keys())
    coords = forked_layers[first_layer]
    unique_heads = sorted(set(coords[:, 1].tolist()))
    print(f"  Layer {first_layer}: {len(coords)} (batch,head,query) combos across heads {unique_heads[:10]}{'...' if len(unique_heads) > 10 else ''}")

# Per-token stats
forked_per_token = LogicEngine.forked_head_per_token
total_forked = sum(
    len(v) for token_dict in forked_per_token.values()
    for v in token_dict.values()
    if isinstance(v, torch.Tensor)
)
print(f"Total forked (batch,head,query) combos across all tokens and layers: {total_forked}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)

# Cleanup
StashEngine.clear()
LogicEngine.clear()
