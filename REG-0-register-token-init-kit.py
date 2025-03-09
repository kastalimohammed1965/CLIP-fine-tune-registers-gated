import os
import torch
import clip
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model = model.float()

"""
The purpose of this code is to initialize Register Tokens with
'naturally occuring' self-emergent register tokens.

As we can recognize them by their norms, we just collect them 
for pre-trained CLIP for a given image dataset.

For more information / background:

Vision Transformers Need Registers
arxiv.org/abs/2309.16588v2
"""

# Doesn't need to be your train dataset. Should be a diverse dataset with some-thousand images.
# That way, we get 'natural' inits for training explicit registers.
# In my example / models, I am using ImageNet val with 50,000 images (~15 min on RTX 4090):
root_img_dir = "F:/AI_DATASET/ILSVRC2012/val" # or any large image folder (we don't need text)

# Normal tokens (in ViT-L/14) seem to have norm: ~50 to ~80, register: ~100 to ~250
# Example result (from the above):
# L2 Norm Statistics of Top Register Tokens:
# Top-1 Register Token - Mean: 221.79, Max: 250.58, Min: 159.71
# Top-2 Register Token - Mean: 173.07, Max: 216.73, Min: 90.04
# Top-3 Register Token - Mean: 133.57, Max: 194.41, Min: 90.01
# Top-4 Register Token - Mean: 116.72, Max: 178.09, Min: 90.02

save_dir = "regtokens"
os.makedirs(save_dir, exist_ok=True)

# Recursively collect all image paths from all subfolders
def get_all_images(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

img_files = get_all_images(root_img_dir)

# Store the top 4 register token (to then use for separate inits -> 4 [REG] tokens)
register_accumulators = {i: [] for i in range(4)}  # Top-1, Top-2, Top-3, Top-4
l2_stats = {i: [] for i in range(4)}  # Store norms for min/max/mean calc

# Function to extract register tokens from an image
def extract_register_tokens(model, image_path, norm_threshold=90): # should avoid normal / local info patch tokens
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Vision Transformer Processing
        x = model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # CLS + Patch Tokens
        cls_token = model.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)

        # Transformer
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)

        # Extract Patch Norms
        patch_embeddings = x[:, 1:, :]  # Exclude CLS
        patch_norms = torch.norm(patch_embeddings, dim=-1).squeeze(0)

        # Find High-Norm Register Tokens
        high_norm_indices = torch.where(patch_norms > norm_threshold)[0]
        high_norm_values = patch_norms[high_norm_indices]

        if len(high_norm_indices) == 0:
            return None  # No register tokens in this image

        # Get top-4 highest register tokens per image
        sorted_indices = torch.argsort(high_norm_values, descending=True)[:4]
        sorted_high_norms = high_norm_values[sorted_indices]
        sorted_register_tokens = patch_embeddings[:, high_norm_indices[sorted_indices], :]

        return sorted_register_tokens.squeeze(0), sorted_high_norms.cpu().numpy()

# Process images and accumulate register tokens for each
for img_path in tqdm(img_files, desc="Processing images"):
    registers = extract_register_tokens(model, img_path)

    if registers is not None:
        sorted_registers, sorted_norms = registers
        for i in range(len(sorted_registers)):  # Top-1 to Top-4
            register_accumulators[i].append(sorted_registers[i].cpu().numpy())
            l2_stats[i].append(sorted_norms[i])
    else:
        print(f"Warning: No register tokens found for {img_path}")

# Compute mean register tokens per rank, but skip if no tokens exist
mean_registers = {}
for i in range(4):
    if len(register_accumulators[i]) > 0:
        mean_registers[i] = np.mean(np.array(register_accumulators[i]), axis=0)
    else:
        mean_registers[i] = None  # No registers found for this rank

# Save found register as '.pt' for use in CLIP fine-tuning code (will be loaded automatically, if left at default dir)
for i in range(4):
    if mean_registers[i] is not None:
        torch.save(torch.tensor(mean_registers[i], dtype=torch.float32), f"{save_dir}/top{i+1}_mean.pt")
    else:
        print(f"Warning: No valid register tokens for Top-{i+1}, skipping save.")

# Compute L2 norm statistics safely (skip empty lists)
l2_norm_stats = {}
for i in range(4):
    if len(l2_stats[i]) > 0:
        l2_norm_stats[i] = {
            "mean": np.mean(l2_stats[i]),
            "max": np.max(l2_stats[i]),
            "min": np.min(l2_stats[i])
        }
    else:
        l2_norm_stats[i] = {"mean": None, "max": None, "min": None}  # Avoid errors

# Print L2 statistics for comparison
print("\nL2 Norm Statistics of Top Register Tokens:")
for i in range(4):
    if l2_norm_stats[i]["mean"] is not None:
        print(f"Top-{i+1} Register Token - Mean: {l2_norm_stats[i]['mean']:.2f}, Max: {l2_norm_stats[i]['max']:.2f}, Min: {l2_norm_stats[i]['min']:.2f}")
    else:
        print(f"No register tokens found for Top-{i+1}")