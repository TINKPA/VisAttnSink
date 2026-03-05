"""
Architectural diagram: token sequence → MHA → visual attention map
Replicates the paper's Figure 1 style with user's own image + prompt.
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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from src.utils import disable_torch_init
from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation, ValueMonitor
from src.logic.constants import DIM_SINK

# ============================================================
# Config
# ============================================================
IMAGE_PATH = 'notebooks/IMG_0226 Small.jpeg'
PROMPT = 'is there a water bottle in this image?'
MODEL_PATH = 'liuhaotian/llava-v1.5-7b'
CONV_MODE = 'vicuna_v1'
DEVICE = 'cuda:0'
TAU, RHO, P, SUMM = 20, 0.5, 0.6, 0.2
SINK_DIMS = DIM_SINK['llama-v2-7b'].tolist()
OUT_DIR = 'notebooks/dim_sink_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load model
# ============================================================
print('Loading model...')
disable_torch_init()
model_name = get_model_name_from_path(MODEL_PATH)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH, None, model_name, attn_implementation='eager', device_map=DEVICE
)
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
print(f'Model: {model_name}, Layers: {n_layers}, Heads: {n_heads}')

# ============================================================
# Prepare input
# ============================================================
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

# ============================================================
# Run 1: VisAttnSink (to get sink indices + forked heads + attention)
# ============================================================
StashEngine.clear(); LogicEngine.clear()
for subcls in [DimProspector, HeadFork, VARProcessor]:
    if 'logic_flag' in subcls.__dict__:
        delattr(subcls, 'logic_flag')
MetadataStation.activate()
MetadataStation.export_model_config(model.config)
LogicEngine.activate(tau=TAU, rho=RHO, summ=SUMM, p=P, except_last_layer=1)
setattr(model, 'tokenizer', tokenizer)

print('Running VisAttnSink inference...')
with torch.inference_mode():
    out_vas = model.generate(
        input_ids, images=image_input, image_sizes=[img.size],
        return_dict_in_generate=True, output_attentions=True,
        do_sample=False, max_new_tokens=128, use_cache=True,
    )

gen_vas = tokenizer.batch_decode(out_vas.sequences, skip_special_tokens=True)[0].strip()
begin_pos = dict(MetadataStation.segments['begin_pos'])
vis_len = MetadataStation.metadata['vis_len']
im_start = begin_pos['image']
im_end = im_start + vis_len

# Save attention from first generated token for a representative layer
# Pick a middle layer with strong sink behavior
TARGET_LAYER = n_layers // 2  # layer 16

# Attention shape: [bsz, n_heads, q_len, kv_len]
# For generation step 1 (first new token), q_len=1
step = 1 if len(out_vas.attentions) > 1 else 0
attn_vas = out_vas.attentions[step][TARGET_LAYER][0].cpu().float()  # [n_heads, 1, kv_len]

# Save sink indices for this layer
saved_indices = {}
for layer, indices in LogicEngine.indices.items():
    if hasattr(indices, '__len__') and len(indices) > 0:
        saved_indices[layer] = indices.clone()

# Save forked heads
saved_forked = {}
for layer in LogicEngine.forked_head:
    fh = LogicEngine.forked_head[layer]
    if isinstance(fh, torch.Tensor) and len(fh) > 0:
        saved_forked[layer] = fh.clone()

print(f'VAS response: {gen_vas}')
print(f'Segments: {begin_pos}, vis_len={vis_len}')
del out_vas; torch.cuda.empty_cache()

# ============================================================
# Run 2: Vanilla (attention only)
# ============================================================
StashEngine.clear(); LogicEngine.clear()
LogicEngine.logic_flag = False
DimProspector.logic_flag = False
HeadFork.logic_flag = False
VARProcessor.logic_flag = False
MetadataStation.activate()
MetadataStation.export_model_config(model.config)

print('Running Vanilla inference...')
with torch.inference_mode():
    out_van = model.generate(
        input_ids, images=image_input, image_sizes=[img.size],
        return_dict_in_generate=True, output_attentions=True,
        do_sample=False, max_new_tokens=128, use_cache=True,
    )

gen_van = tokenizer.batch_decode(out_van.sequences, skip_special_tokens=True)[0].strip()
attn_van = out_van.attentions[step][TARGET_LAYER][0].cpu().float()

print(f'Vanilla response: {gen_van}')
del out_van; torch.cuda.empty_cache()

# Free GPU
del model, image_input, image_tensor, input_ids
torch.cuda.empty_cache()
print('Model freed.')

# ============================================================
# Prepare data for the diagram
# ============================================================
patches_per_side = 24  # sqrt(576)

# Average attention across all heads, squeeze query dim
avg_attn_van = attn_van.mean(dim=0).squeeze(0).numpy()  # [kv_len]
avg_attn_vas = attn_vas.mean(dim=0).squeeze(0).numpy()

# Pick a single flagged head for more dramatic visualization
# Find the head with max difference in image attention
img_attn_diff = []
for h in range(n_heads):
    van_img = attn_van[h, 0, im_start:im_end].sum().item()
    vas_img = attn_vas[h, 0, im_start:im_end].sum().item()
    img_attn_diff.append(vas_img - van_img)
best_head = int(np.argmax(img_attn_diff))
print(f'Best head (max image attn increase): {best_head}, delta={img_attn_diff[best_head]:.4f}')

head_attn_van = attn_van[best_head, 0].numpy()  # [kv_len]
head_attn_vas = attn_vas[best_head, 0].numpy()

# Get sink positions for target layer
sink_positions = set()
if TARGET_LAYER in saved_indices:
    sink_positions = set(saved_indices[TARGET_LAYER].tolist())
img_sink_positions = sorted([p for p in sink_positions if im_start <= p < im_end])

# Reshape image attention to 24x24 grid
img_attn_grid_van = head_attn_van[im_start:im_end].reshape(patches_per_side, patches_per_side)
img_attn_grid_vas = head_attn_vas[im_start:im_end].reshape(patches_per_side, patches_per_side)

# Find sink and relevant (high non-sink attention) patches
sink_mask = np.zeros((patches_per_side, patches_per_side), dtype=bool)
for pos in img_sink_positions:
    rel = pos - im_start
    r, c = rel // patches_per_side, rel % patches_per_side
    sink_mask[r, c] = True

# Find the highest-attention non-sink patch (= "relevant" patch)
nonsink_attn = img_attn_grid_vas.copy()
nonsink_attn[sink_mask] = 0
rel_r, rel_c = np.unravel_index(nonsink_attn.argmax(), nonsink_attn.shape)

# Find the highest-attention sink patch
sink_attn = img_attn_grid_vas.copy()
sink_attn[~sink_mask] = 0
if sink_attn.max() > 0:
    sink_r, sink_c = np.unravel_index(sink_attn.argmax(), sink_attn.shape)
else:
    sink_r, sink_c = 0, 0

print(f'Sink patches: {int(sink_mask.sum())}, Representative sink patch: ({sink_r},{sink_c})')
print(f'Relevant (high attn non-sink) patch: ({rel_r},{rel_c})')

# ============================================================
# DRAW THE DIAGRAM
# ============================================================
img_np = np.array(img)
h_img, w_img = img_np.shape[:2]

fig = plt.figure(figsize=(22, 11))

# Color palette
C_BOS = '#4CAF50'
C_SINK = '#F44336'
C_RELEVANT = '#2196F3'
C_SYSTEM = '#E0E0E0'
C_TEXT = '#E0E0E0'
C_IMG = '#FFF3E0'
C_ATTN_HIGH = '#4A148C'  # deep purple
C_ATTN_LOW = '#E1BEE7'   # light purple
C_MHA = '#F5F5F5'
C_ARROW = '#757575'

# ============================================================
# (1) LEFT IMAGE with bounding boxes
# ============================================================
ax_img_l = fig.add_axes([0.01, 0.12, 0.18, 0.72])
ax_img_l.imshow(img_np)
ax_img_l.axis('off')

# Draw grid-aligned bounding boxes
pw = w_img / patches_per_side
ph = h_img / patches_per_side

# Sink patch box (red)
ax_img_l.add_patch(plt.Rectangle(
    (sink_c * pw, sink_r * ph), pw, ph,
    fill=False, edgecolor=C_SINK, linewidth=6
))
# Relevant patch box (blue)
ax_img_l.add_patch(plt.Rectangle(
    (rel_c * pw, rel_r * ph), pw, ph,
    fill=False, edgecolor=C_RELEVANT, linewidth=6
))

# ============================================================
# (2) PRE-ATTENTION TOKEN COLUMN
# ============================================================
ax_tokens_pre = fig.add_axes([0.22, 0.05, 0.12, 0.88])
ax_tokens_pre.set_xlim(0, 1)
ax_tokens_pre.set_ylim(0, 1)
ax_tokens_pre.axis('off')

def draw_token_box(ax, x, y, w, h, text, facecolor='white', edgecolor='black',
                   textcolor='black', fontsize=9, fontweight='normal', star=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=4)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=textcolor)
    if star:
        ax.text(x - 0.02, y + h, '\u2606', ha='center', va='center',
                fontsize=16, color='#FF8F00', fontweight='bold')

# Token layout (bottom to top: text, image, system/BOS)
bw, bh = 0.8, 0.038  # box width, height
gap = 0.006
bx = 0.1  # left margin

# --- System tokens ---
y = 0.92
draw_token_box(ax_tokens_pre, bx, y, bw, bh, '<BOS>', facecolor=C_BOS,
               edgecolor='#2E7D32', textcolor='white', fontweight='bold', star=True)

y -= (bh + gap)
draw_token_box(ax_tokens_pre, bx, y, bw, bh, 'A', facecolor=C_SYSTEM)
y -= (bh + gap)
ax_tokens_pre.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')

# Label: System
ax_tokens_pre.text(bx + bw + 0.06, 0.92, 'System', ha='left', va='center', fontsize=10, fontweight='bold')

# --- Image tokens ---
y_img_top = 0.72
n_img_show = 6  # show this many img token boxes
y = y_img_top

# First few img tokens (normal)
draw_token_box(ax_tokens_pre, bx, y, bw, bh, '<img>', facecolor=C_IMG)
y -= (bh + gap)
# Sink img token (red)
draw_token_box(ax_tokens_pre, bx, y, bw, bh, '<img>', facecolor=C_SINK,
               textcolor='white', fontweight='bold', star=True)
y -= (bh + gap)
draw_token_box(ax_tokens_pre, bx, y, bw, bh, '<img>', facecolor=C_IMG)
y -= (bh + gap)
ax_tokens_pre.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')
y -= (bh + gap)
# Relevant img token (blue)
draw_token_box(ax_tokens_pre, bx, y, bw, bh, '<img>', facecolor=C_RELEVANT,
               textcolor='white', fontweight='bold')
y -= (bh + gap)
draw_token_box(ax_tokens_pre, bx, y, bw, bh, '<img>', facecolor=C_IMG)
y -= (bh + gap)
ax_tokens_pre.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')

# Label: Image
ax_tokens_pre.text(bx + bw + 0.06, y_img_top, 'Image', ha='left', va='center', fontsize=10, fontweight='bold')

# Arrow from left image to token column
fig.patches.append(FancyArrowPatch(
    (0.19, 0.5), (0.22, 0.5),
    arrowstyle='->', mutation_scale=30, color=C_ARROW, linewidth=5,
    transform=fig.transFigure, figure=fig
))

# --- Text tokens ---
y_text_top = 0.34
y = y_text_top

# Get actual text tokens from prompt
prompt_words = PROMPT.split()
for i, word in enumerate(prompt_words[:4]):
    draw_token_box(ax_tokens_pre, bx, y, bw, bh, word, facecolor=C_TEXT)
    y -= (bh + gap)
if len(prompt_words) > 4:
    ax_tokens_pre.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')

# Label: Text
ax_tokens_pre.text(bx + bw + 0.06, y_text_top, 'Text', ha='left', va='center', fontsize=10, fontweight='bold')

# ============================================================
# (3) MHA BLOCK
# ============================================================
ax_mha = fig.add_axes([0.38, 0.08, 0.08, 0.82])
ax_mha.set_xlim(0, 1)
ax_mha.set_ylim(0, 1)
ax_mha.axis('off')

mha_box = FancyBboxPatch((0.1, 0.02), 0.8, 0.96, boxstyle="round,pad=0.02",
                          facecolor=C_MHA, edgecolor='#424242', linewidth=4)
ax_mha.add_patch(mha_box)
ax_mha.text(0.5, 0.5, 'Multi-Head\nAttention', ha='center', va='center',
            fontsize=12, fontweight='bold', rotation=90, color='#424242')

# Arrow pre-tokens → MHA
fig.patches.append(FancyArrowPatch(
    (0.35, 0.5), (0.38, 0.5),
    arrowstyle='->', mutation_scale=30, color=C_ARROW, linewidth=5,
    transform=fig.transFigure, figure=fig
))

# Arrow MHA → post-tokens
fig.patches.append(FancyArrowPatch(
    (0.46, 0.5), (0.49, 0.5),
    arrowstyle='->', mutation_scale=30, color=C_ARROW, linewidth=5,
    transform=fig.transFigure, figure=fig
))

# ============================================================
# (4) POST-ATTENTION TOKEN COLUMN (with purple attention shading)
# ============================================================
ax_tokens_post = fig.add_axes([0.49, 0.05, 0.12, 0.88])
ax_tokens_post.set_xlim(0, 1)
ax_tokens_post.set_ylim(0, 1)
ax_tokens_post.axis('off')

# Compute attention weights for coloring
# Normalize attention for color mapping (use full range including BOS)
attn_max = head_attn_vas.max()
attn_min = head_attn_vas.min()

def attn_to_purple(attn_val, attn_min, attn_max):
    """Map attention value to purple shade."""
    if attn_max - attn_min < 1e-10:
        t = 0.5
    else:
        t = (attn_val - attn_min) / (attn_max - attn_min)
    t = max(0.0, min(1.0, t))  # clamp to [0, 1]
    # Interpolate between light purple and deep purple
    r = int(225 * (1-t) + 74 * t)
    g = int(190 * (1-t) + 20 * t)
    b = int(231 * (1-t) + 140 * t)
    return f'#{r:02x}{g:02x}{b:02x}'

# BOS attention
bos_attn = head_attn_vas[0]
bos_color = attn_to_purple(bos_attn, attn_min, attn_max)

# --- System tokens (post) ---
y = 0.92
draw_token_box(ax_tokens_post, bx, y, bw, bh, '<BOS>', facecolor=bos_color,
               edgecolor='#2E7D32', textcolor='white', fontweight='bold', star=True)
y -= (bh + gap)
draw_token_box(ax_tokens_post, bx, y, bw, bh, 'A', facecolor=C_SYSTEM)
y -= (bh + gap)
ax_tokens_post.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')

# --- Image tokens (post, with attention coloring) ---
y = y_img_top

# Sample representative img token positions for coloring
# Normal img
p0 = im_start + 10
c0 = attn_to_purple(head_attn_vas[p0], attn_min, attn_max)
draw_token_box(ax_tokens_post, bx, y, bw, bh, '<img>', facecolor=c0, textcolor='white')
y -= (bh + gap)

# Sink img (high attention in vanilla, should be reduced in VAS)
sink_example_pos = img_sink_positions[0] if img_sink_positions else im_start + 50
c_sink = attn_to_purple(head_attn_vas[sink_example_pos], attn_min, attn_max)
draw_token_box(ax_tokens_post, bx, y, bw, bh, '<img>', facecolor=c_sink,
               edgecolor=C_SINK, textcolor='white', fontweight='bold', star=True)
y -= (bh + gap)

p2 = im_start + 100
c2 = attn_to_purple(head_attn_vas[p2], attn_min, attn_max)
draw_token_box(ax_tokens_post, bx, y, bw, bh, '<img>', facecolor=c2, textcolor='white')
y -= (bh + gap)
ax_tokens_post.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')
y -= (bh + gap)

# Relevant img (should have gained attention)
rel_pos = im_start + rel_r * patches_per_side + rel_c
c_rel = attn_to_purple(head_attn_vas[rel_pos], attn_min, attn_max)
draw_token_box(ax_tokens_post, bx, y, bw, bh, '<img>', facecolor=c_rel,
               edgecolor=C_RELEVANT, textcolor='white', fontweight='bold')
y -= (bh + gap)

p4 = im_start + 400
c4 = attn_to_purple(head_attn_vas[p4], attn_min, attn_max)
draw_token_box(ax_tokens_post, bx, y, bw, bh, '<img>', facecolor=c4, textcolor='white')
y -= (bh + gap)
ax_tokens_post.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')

# --- Text tokens (post) ---
y = y_text_top
for i, word in enumerate(prompt_words[:4]):
    tp = begin_pos['inst_q'] + i
    ct = attn_to_purple(head_attn_vas[min(tp, len(head_attn_vas)-1)], attn_min, attn_max)
    draw_token_box(ax_tokens_post, bx, y, bw, bh, word, facecolor=ct, textcolor='white')
    y -= (bh + gap)
if len(prompt_words) > 4:
    ax_tokens_post.text(bx + bw/2, y + bh/2, '...', ha='center', va='center', fontsize=10, color='gray')

# ============================================================
# "Reshape" label + arrow
# ============================================================
fig.text(0.64, 0.52, 'Reshape\n24x24', ha='center', va='center',
         fontsize=11, fontweight='bold', color='#424242')
fig.patches.append(FancyArrowPatch(
    (0.62, 0.5), (0.67, 0.5),
    arrowstyle='->', mutation_scale=30, color=C_ARROW, linewidth=5,
    transform=fig.transFigure, figure=fig
))

# ============================================================
# (5) RIGHT IMAGE: Visual Attention Map
# ============================================================
ax_img_r = fig.add_axes([0.68, 0.12, 0.30, 0.72])
ax_img_r.imshow(img_np)

# Overlay attention heatmap
ax_img_r.imshow(
    img_attn_grid_vas, cmap='Purples', alpha=0.55,
    interpolation='nearest', extent=[0, w_img, h_img, 0],
    vmin=0, vmax=img_attn_grid_vas.max()
)

# Mark sink patch (red box)
ax_img_r.add_patch(plt.Rectangle(
    (sink_c * pw, sink_r * ph), pw, ph,
    fill=False, edgecolor=C_SINK, linewidth=4, linestyle='--'
))
# Mark relevant patch (blue box)
ax_img_r.add_patch(plt.Rectangle(
    (rel_c * pw, rel_r * ph), pw, ph,
    fill=False, edgecolor=C_RELEVANT, linewidth=6
))
ax_img_r.axis('off')
ax_img_r.set_title('Visual Attention Map', fontsize=13, fontweight='bold', pad=10)

# ============================================================
# Dashed arrows from left image patches to right image patches
# ============================================================
# Sink patch arrow (red dashed)
fig.patches.append(FancyArrowPatch(
    (0.19, 0.65), (0.68, 0.65),
    arrowstyle='->', mutation_scale=30, color=C_SINK,
    linewidth=5, linestyle='--',
    transform=fig.transFigure, figure=fig,
    connectionstyle='arc3,rad=-0.15'
))

# Relevant patch arrow (blue dashed)
fig.patches.append(FancyArrowPatch(
    (0.19, 0.35), (0.68, 0.35),
    arrowstyle='->', mutation_scale=30, color=C_RELEVANT,
    linewidth=5, linestyle='--',
    transform=fig.transFigure, figure=fig,
    connectionstyle='arc3,rad=0.15'
))

# ============================================================
# (6) LEGEND
# ============================================================
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_SINK,
           markeredgecolor=C_SINK, markersize=12, label='irrelevant visual token (sink)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_RELEVANT,
           markeredgecolor=C_RELEVANT, markersize=12, label='relevant visual token'),
    Line2D([0], [0], marker='*', color='#FF8F00', markersize=14,
           linestyle='None', label='sink token'),
    mpatches.Patch(facecolor=C_ATTN_HIGH, edgecolor='gray', label='high attention weight'),
    mpatches.Patch(facecolor=C_ATTN_LOW, edgecolor='gray', label='low attention weight'),
]
fig.legend(handles=legend_elements, loc='upper left', fontsize=9,
           frameon=True, fancybox=True, framealpha=0.9,
           bbox_to_anchor=(0.01, 0.99), ncol=3)

# Prompt label
fig.text(0.09, 0.04, f'Prompt: "{PROMPT}"', ha='center', va='center',
         fontsize=15, fontweight='bold', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

# Model + layer info
fig.text(0.50, 0.01,
         f'Model: {model_name}  |  Layer {TARGET_LAYER}, Head {best_head}  |  '
         f'Vanilla: "{gen_van[:50]}..."  |  VAS: "{gen_vas[:50]}..."',
         ha='center', va='center', fontsize=10, color='gray')

# ============================================================
# Save
# ============================================================
out_path = os.path.join(OUT_DIR, 'arch_diagram.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'\nSaved: {out_path}')

# ============================================================
# Also generate vanilla vs VAS side-by-side attention maps
# ============================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

vmax = max(img_attn_grid_van.max(), img_attn_grid_vas.max())

ax1.imshow(img_np)
im1 = ax1.imshow(img_attn_grid_van, cmap='Purples', alpha=0.55,
                  interpolation='nearest', extent=[0, w_img, h_img, 0], vmin=0, vmax=vmax)
for r in range(patches_per_side):
    for c in range(patches_per_side):
        if sink_mask[r, c]:
            ax1.add_patch(plt.Rectangle((c*pw, r*ph), pw, ph,
                          fill=False, edgecolor=C_SINK, linewidth=2.5, linestyle='--'))
ax1.set_title(f'Vanilla — Layer {TARGET_LAYER}, Head {best_head}\n"{gen_van[:60]}"', fontsize=11)
ax1.axis('off')

ax2.imshow(img_np)
im2 = ax2.imshow(img_attn_grid_vas, cmap='Purples', alpha=0.55,
                  interpolation='nearest', extent=[0, w_img, h_img, 0], vmin=0, vmax=vmax)
for r in range(patches_per_side):
    for c in range(patches_per_side):
        if sink_mask[r, c]:
            ax2.add_patch(plt.Rectangle((c*pw, r*ph), pw, ph,
                          fill=False, edgecolor=C_SINK, linewidth=2.5, linestyle='--'))
ax2.set_title(f'VisAttnSink — Layer {TARGET_LAYER}, Head {best_head}\n"{gen_vas[:60]}"', fontsize=11)
ax2.axis('off')

fig2.colorbar(im2, ax=[ax1, ax2], label='Attention weight', shrink=0.7)
fig2.suptitle(f'Visual Attention Map Comparison (red dashed = sink patches)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
out_path2 = os.path.join(OUT_DIR, 'attn_map_comparison.png')
fig2.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'Saved: {out_path2}')

print('\nDone!')
