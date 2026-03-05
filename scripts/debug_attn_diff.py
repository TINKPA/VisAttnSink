"""Quick diagnostic: check if VAS and Vanilla attention are actually different."""
import sys, os
sys.path.insert(0, '/home/antarachugh/idountang/VisAttnSink')
os.chdir('/home/antarachugh/idountang/VisAttnSink')

import json, torch
from PIL import Image
from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from src.utils import disable_torch_init
from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation, ValueMonitor

MODEL_PATH = "liuhaotian/llava-v1.5-7b"
CONV_MODE = "vicuna_v1"
DEVICE = "cuda:0"
TAU, RHO, P, SUMM = 20, 0.5, 0.6, 0.2

DATA_DIR = "/xuanwu-tank/east/antarachugh/projects/VisAttnSink_data/ScienceQA"
with open(f"{DATA_DIR}/Questions/test-questions.jsonl") as f:
    sample = [json.loads(l) for l in f][2]

# Load model
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, attn_implementation="eager", device_map=DEVICE
)

# Prepare input
qs = DEFAULT_IMAGE_TOKEN + "\n" + sample['text']
conv = conv_templates[CONV_MODE].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, conv=conv, return_tensors="pt").unsqueeze(0).to(DEVICE)
img = Image.open(os.path.join(DATA_DIR, "Images", sample['image'])).convert("RGB")
image_tensor = process_images([img], image_processor, model.config)[0]
image_input = image_tensor.unsqueeze(0).half().to(DEVICE)
setattr(model, 'tokenizer', tokenizer)

# --- Run 1: VAS ---
print("=" * 60)
print("RUN 1: VisAttnSink")
print("=" * 60)
StashEngine.clear()
LogicEngine.clear()
MetadataStation.activate()
MetadataStation.export_model_config(model.config)
LogicEngine.activate(tau=TAU, rho=RHO, summ=SUMM, p=P, except_last_layer=1)

with torch.inference_mode():
    out_vas = model.generate(
        input_ids, images=image_input, image_sizes=[img.size],
        return_dict_in_generate=True, output_attentions=True,
        do_sample=False, max_new_tokens=5, use_cache=True,
    )

begin_pos = MetadataStation.segments['begin_pos']
vis_len = MetadataStation.metadata['vis_len']
im_start = begin_pos['image']
im_end = im_start + vis_len

n_steps = len(out_vas.attentions)
print(f"Number of attention steps: {n_steps}")
for step in range(min(n_steps, 3)):
    a = out_vas.attentions[step][15][0]  # layer 15
    print(f"  Step {step}: shape={a.shape}, mean={a.mean():.8f}")

# Save step 0 and step 1 for VAS
vas_step0 = {li: out_vas.attentions[0][li][0].cpu().float() for li in range(32)}
if n_steps > 1:
    vas_step1 = {li: out_vas.attentions[1][li][0].cpu().float() for li in range(32)}

# Check VARProcessor was active
print(f"\nLogicEngine.logic_flag: {LogicEngine.logic_flag}")
print(f"VARProcessor.logic_flag: {VARProcessor.logic_flag}")
print(f"HeadFork.logic_flag: {HeadFork.logic_flag}")

# Check forked heads
n_forked = sum(len(v) for v in LogicEngine.forked_head.values() if isinstance(v, torch.Tensor))
print(f"Total forked head entries: {n_forked}")

saved_forked = {}
for token_idx, layer_dict in LogicEngine.forked_head_per_token.items():
    for layer, coords in layer_dict.items():
        if isinstance(coords, torch.Tensor) and len(coords) > 0:
            saved_forked.setdefault(layer, {})[token_idx] = coords.clone()
print(f"Layers with forked heads: {len(saved_forked)}")

del out_vas
torch.cuda.empty_cache()

# --- Run 2: Vanilla ---
print("\n" + "=" * 60)
print("RUN 2: Vanilla")
print("=" * 60)
StashEngine.clear()
LogicEngine.clear()
LogicEngine.logic_flag = False
DimProspector.logic_flag = False
HeadFork.logic_flag = False
VARProcessor.logic_flag = False
MetadataStation.activate()
MetadataStation.export_model_config(model.config)

with torch.inference_mode():
    out_van = model.generate(
        input_ids, images=image_input, image_sizes=[img.size],
        return_dict_in_generate=True, output_attentions=True,
        do_sample=False, max_new_tokens=5, use_cache=True,
    )

van_step0 = {li: out_van.attentions[0][li][0].cpu().float() for li in range(32)}
n_steps_van = len(out_van.attentions)
if n_steps_van > 1:
    van_step1 = {li: out_van.attentions[1][li][0].cpu().float() for li in range(32)}

del out_van
torch.cuda.empty_cache()

# --- Compare ---
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("\n--- Step 0 (prefill) ---")
for li in [0, 15, 30, 31]:
    diff = (vas_step0[li] - van_step0[li]).abs()
    print(f"  Layer {li:2d}: max_diff={diff.max():.10f}, mean_diff={diff.mean():.10f}")

if n_steps > 1 and n_steps_van > 1:
    print("\n--- Step 1 (first generated token) ---")
    for li in [0, 15, 30, 31]:
        diff = (vas_step1[li] - van_step1[li]).abs()
        print(f"  Layer {li:2d}: max_diff={diff.max():.10f}, mean_diff={diff.mean():.10f}")

    # Check specific head that was flagged
    for li in sorted(saved_forked.keys())[:3]:
        for token_idx, coords in saved_forked[li].items():
            heads = sorted(set(coords[:, 1].tolist()))
            for h in heads[:2]:
                a_van = van_step1[li][h].squeeze(0)
                a_vas = vas_step1[li][h].squeeze(0)
                diff_img = (a_vas[im_start:im_end] - a_van[im_start:im_end]).abs()
                print(f"  Layer {li}, Head {h}, Token {token_idx}: img_diff max={diff_img.max():.10f}, mean={diff_img.mean():.10f}")
            break  # just first token
        break  # just first layer

# Also check: are prefill attentions the same? They should be for step 0
# since VARProcessor only modifies the attention weights (not KV cache)
print("\n--- Are VAS step0 and Vanilla step0 identical? ---")
all_same = True
for li in range(32):
    if not torch.allclose(vas_step0[li], van_step0[li], atol=1e-6):
        diff = (vas_step0[li] - van_step0[li]).abs()
        print(f"  Layer {li}: DIFFERENT! max_diff={diff.max():.8f}")
        all_same = False
if all_same:
    print("  YES - all layers identical at step 0")

if n_steps > 1:
    print("\n--- Are VAS step1 and Vanilla step1 identical? ---")
    all_same = True
    for li in range(32):
        if not torch.allclose(vas_step1[li], van_step1[li], atol=1e-6):
            diff = (vas_step1[li] - van_step1[li]).abs()
            print(f"  Layer {li}: DIFFERENT! max_diff={diff.max():.8f}")
            all_same = False
    if all_same:
        print("  YES - all layers identical at step 1 (THIS IS THE BUG)")
