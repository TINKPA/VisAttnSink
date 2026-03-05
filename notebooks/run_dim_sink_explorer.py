"""
Dimension-wise Attention Sink Explorer — non-interactive runner.
Generates static plots for all layers and saves them.
"""
import sys, os
project_root = '/home/antarachugh/idountang/VisAttnSink'
os.chdir(project_root)
sys.path.insert(0, project_root)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from src.utils import disable_torch_init
from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation, ValueMonitor
from src.logic.constants import DIM_SINK

# === Config ===
IMAGE_PATH = 'notebooks/IMG_0226 Small.jpeg'
PROMPT = 'is there a water bottle in this image?'
MODEL_PATH = 'liuhaotian/llava-v1.5-7b'
CONV_MODE = 'vicuna_v1'
DEVICE = 'cuda:0'
TAU, RHO, P, SUMM = 20, 0.5, 0.6, 0.2
SINK_DIMS = DIM_SINK['llama-v2-7b'].tolist()
OUT_DIR = 'notebooks/dim_sink_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

print(f'Sink dimensions: {SINK_DIMS}')

# === Load model ===
print('Loading model...')
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, attn_implementation='eager', device_map=DEVICE
)
n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f'Model: {model_name}, Layers: {n_layers}, Hidden: {hidden_dim}')

# === Prepare input ===
img = Image.open(IMAGE_PATH).convert('RGB')
qs = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT
conv = conv_templates[CONV_MODE].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, conv=conv, return_tensors='pt'
).unsqueeze(0).to(DEVICE)
image_tensor = process_images([img], image_processor, model.config)[0]
image_input = image_tensor.unsqueeze(0).half().to(DEVICE)
print(f'Input tokens: {input_ids.shape[1]}')

# === Run 1: Vanilla ===
StashEngine.clear(); LogicEngine.clear()
LogicEngine.logic_flag = False
DimProspector.logic_flag = False
HeadFork.logic_flag = False
VARProcessor.logic_flag = False
MetadataStation.activate()
MetadataStation.export_model_config(model.config)
setattr(model, 'tokenizer', tokenizer)

print('Running Vanilla inference...')
with torch.inference_mode():
    out_van = model.generate(
        input_ids, images=image_input, image_sizes=[img.size],
        return_dict_in_generate=True, output_hidden_states=True,
        do_sample=False, max_new_tokens=128, use_cache=True,
    )

gen_van = tokenizer.batch_decode(out_van.sequences, skip_special_tokens=True)[0].strip()
hs_vanilla = [out_van.hidden_states[0][i][0].cpu().float() for i in range(n_layers + 1)]
print(f'Vanilla response: {gen_van}')
del out_van; torch.cuda.empty_cache()

# === Run 2: VisAttnSink ===
StashEngine.clear(); LogicEngine.clear()
for subcls in [DimProspector, HeadFork, VARProcessor]:
    if 'logic_flag' in subcls.__dict__:
        delattr(subcls, 'logic_flag')

MetadataStation.activate()
MetadataStation.export_model_config(model.config)
LogicEngine.activate(tau=TAU, rho=RHO, summ=SUMM, p=P, except_last_layer=1)

print('Running VisAttnSink inference...')
with torch.inference_mode():
    out_vas = model.generate(
        input_ids, images=image_input, image_sizes=[img.size],
        return_dict_in_generate=True, output_hidden_states=True,
        do_sample=False, max_new_tokens=128, use_cache=True,
    )

gen_vas = tokenizer.batch_decode(out_vas.sequences, skip_special_tokens=True)[0].strip()
begin_pos = dict(MetadataStation.segments['begin_pos'])
vis_len = MetadataStation.metadata['vis_len']
im_start = begin_pos['image']
im_end = im_start + vis_len
hs_vas = [out_vas.hidden_states[0][i][0].cpu().float() for i in range(n_layers + 1)]

print(f'VAS response: {gen_vas}')
print(f'Segments: {begin_pos}')
print(f'Visual tokens: {vis_len} (pos {im_start}:{im_end})')

del out_vas; torch.cuda.empty_cache()
StashEngine.clear(); LogicEngine.clear()

# === Free model GPU memory ===
del model, image_input, image_tensor, input_ids
torch.cuda.empty_cache()
print('Model freed from GPU.')

# === Compute phi ===
def compute_phi(hidden_states, eps=1e-6):
    h = hidden_states.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    return (h * torch.rsqrt(variance + eps)).abs()

print('Computing phi() for all layers...')
phi_van = {}
phi_vas = {}
for i in range(n_layers):
    phi_van[i] = compute_phi(hs_vanilla[i]).numpy()
    phi_vas[i] = compute_phi(hs_vas[i]).numpy()

del hs_vanilla, hs_vas

# === Helper ===
bos_pos = 0

def get_representative_tokens(phi_layer, im_start, im_end, sink_dims):
    img_phi = phi_layer[im_start:im_end]
    sink_vals = np.stack([img_phi[:, d] for d in sink_dims], axis=-1)
    max_sink = sink_vals.max(axis=-1)
    hi_idx = im_start + int(np.argmax(max_sink))
    lo_idx = im_start + int(np.argmin(max_sink))
    return hi_idx, lo_idx

# === Plot per-layer figures ===
print(f'Generating per-layer plots...')
SAMPLE_LAYERS = [0, 2, 5, 10, 15, 20, 25, 31]

for layer in SAMPLE_LAYERS:
    pv = phi_van[layer]
    ps = phi_vas[layer]
    hi_van, lo_van = get_representative_tokens(pv, im_start, im_end, SINK_DIMS)
    hi_vas, lo_vas = get_representative_tokens(ps, im_start, im_end, SINK_DIMS)

    tokens_van = [
        (bos_pos, 'BOS', '#4CAF50'),
        (hi_van, 'img (high sink)', '#F44336'),
        (lo_van, 'img (low sink)', '#2196F3'),
    ]
    tokens_vas = [
        (bos_pos, 'BOS', '#4CAF50'),
        (hi_vas, 'img (high sink)', '#F44336'),
        (lo_vas, 'img (low sink)', '#2196F3'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharey='row')
    fig.suptitle(
        f'Layer {layer}/{n_layers-1} — phi(h) = |RMSNorm(h)| across {hidden_dim} dimensions\n'
        f'Vanilla: "{gen_van}"  |  VAS: "{gen_vas}"',
        fontsize=13, fontweight='bold'
    )

    for col, (phi_data, tokens, title) in enumerate([
        (pv, tokens_van, 'Vanilla (Baseline)'),
        (ps, tokens_vas, 'VisAttnSink'),
    ]):
        for row, (pos, label, color) in enumerate(tokens):
            ax = axes[row, col]
            vals = phi_data[pos]
            max_val = vals.max()
            max_sink_val = max(vals[d] for d in SINK_DIMS)

            # Bin every 16 dimensions together for cleaner look
            bin_size = 16
            n_bins = hidden_dim // bin_size
            binned = vals[:n_bins * bin_size].reshape(n_bins, bin_size).max(axis=1)
            bin_x = np.arange(n_bins) * bin_size + bin_size / 2
            ax.bar(bin_x, binned, width=bin_size * 0.9, color=color, alpha=0.85)
            for d in SINK_DIMS:
                ax.axvline(x=d, color='black', linestyle='--', linewidth=3, alpha=0.8)
                ax.text(d, max_val * 0.95, f'{d}', ha='center', va='top',
                        fontsize=12, fontweight='bold')

            tag = f'phi({label}) = {max_sink_val:.1f}'
            ax.set_title(f'{title}:  {tag}' if row == 0 else tag, fontsize=14)
            ax.set_xlim(0, hidden_dim)
            if row == 2:
                ax.set_xlabel('Dimension')
            if col == 0:
                ax.set_ylabel('|phi(h)|')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'layer_{layer:02d}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

# === Summary: sink dim activation across all layers ===
print('Generating summary plot...')
layers_range = list(range(n_layers))

fig, axes = plt.subplots(1, len(SINK_DIMS), figsize=(7 * len(SINK_DIMS), 5))
if len(SINK_DIMS) == 1:
    axes = [axes]

for di, dim in enumerate(SINK_DIMS):
    ax = axes[di]
    bos_van = [phi_van[l][bos_pos, dim] for l in layers_range]
    bos_vas = [phi_vas[l][bos_pos, dim] for l in layers_range]
    img_mean_van = [phi_van[l][im_start:im_end, dim].mean() for l in layers_range]
    img_mean_vas = [phi_vas[l][im_start:im_end, dim].mean() for l in layers_range]

    ax.plot(layers_range, bos_van, 'g-o', markersize=6, linewidth=3, label='BOS (vanilla)')
    ax.plot(layers_range, bos_vas, 'g--s', markersize=6, linewidth=3, label='BOS (VAS)')
    ax.plot(layers_range, img_mean_van, 'r-o', markersize=6, linewidth=3, label='Img mean (vanilla)')
    ax.plot(layers_range, img_mean_vas, 'r--s', markersize=6, linewidth=3, label='Img mean (VAS)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('phi() value')
    ax.set_title(f'Dimension {dim}')
    ax.legend(fontsize=8)

fig.suptitle(
    f'Sink Dimension Activation Across Layers\n'
    f'Prompt: "{PROMPT}"',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
path = os.path.join(OUT_DIR, 'summary_across_layers.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {path}')

print(f'\nDone! All plots saved to {OUT_DIR}/')
