import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import load_file
import argparse
import torch.nn as nn

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longINFERclipregXGATED as clip
from longINFERclipregXGATED.model import CLIP

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize CLIP ViT Patch Norms')
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-REG-GATED-full-model.safetensors", help="Path to a ViT-L/14 model, pickle (.pt) or .safetensors")
    parser.add_argument('--image_folder', default="EX-image-vis", help="Folder with images to get norms for")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

model, preprocess = clip.load(model_name_or_path, device=device)

model = model.float() # full precision


out_dir = "plots/longREG-XGATED/patch-norm"
os.makedirs(out_dir, exist_ok=True)

# Image & token directories
img_dir = args.image_folder
tok_dir = "EX-tokens-vis" # not implemented / not needed for just patch norms



# ---------- MAIN CODE ----------

# Image transformation (ensures compatibility with CLIP)
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

# Function to encode text tokens independently (for future use)
def clip_encode_text(model, token):
    with torch.no_grad():
        text_tokens = clip.tokenize([token]).to(device)
        text_features = model.encode_text(text_tokens)
        return text_features

# Sanitize text tokens for safe filenames
def sanitize_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '_', text)  # Replaces invalid characters with '_'


def clip_encode_image(model, image):
    """
    Compute L2 norms for all tokens:
      - 256 patch tokens from the transformer output.
      - CLS token (post ln_post)
      - REG tokens (post ln_post)
      - FUSED token computed via the gating mechanism.
    Returns a numpy array of shape (256 + 6,) where the first 256 entries correspond 
    to patch token norms and the last 6 entries are [CLS, REG1, REG2, REG3, REG4, FUSED].
    """
    with torch.no_grad():
        # Assume 'image' is already preprocessed appropriately
        x = image  # starting with the input image
        
        # ---- Convolution and reshape (same as forward) ----
        x = model.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        # Create CLS and REG tokens
        cls_token = model.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        register_tokens = model.visual.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        reg_count = register_tokens.shape[1]  # e.g., 4 register tokens
        
        # Concatenate tokens: CLS, REG, and patch tokens
        x = torch.cat([cls_token, register_tokens, x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        
        # ---- Transformer processing ----
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer input
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD back to batch-first
        
        # ---- Post-processing of extra tokens ----
        # Process CLS and REG tokens with ln_post
        cls_token_post = model.visual.ln_post(x[:, 0, :])  # Shape: [batch, width]
        reg_tokens_post = model.visual.ln_post(x[:, 1:1+reg_count, :])  # Shape: [batch, reg_count, width]
        # Patch tokens remain unprocessed here (as in the original patch norms code)
        patch_tokens = x[:, 1+reg_count:, :]  # Expecting 256 patch tokens
        
        # ---- Compute L2 norms for each token ----
        # Assuming batch size 1, so we squeeze the batch dimension
        norm_patch = torch.norm(patch_tokens, dim=-1).squeeze(0)  # Shape: [256]
        norm_cls = torch.norm(cls_token_post, dim=-1).squeeze(0)      # Scalar
        norm_regs = torch.norm(reg_tokens_post, dim=-1).squeeze(0)    # Shape: [reg_count]
        
        # ---- Compute FUSED token ----
        # Average the REG tokens to get a summary
        reg_summary = reg_tokens_post.mean(dim=1)  # Shape: [batch, width]
        fusion_input = torch.cat([cls_token_post, reg_summary], dim=-1)  # Shape: [batch, 2*width]
        gate = torch.sigmoid(model.visual.fusion_mlp(fusion_input))  # Shape: [batch, 1]
        fused = gate * cls_token_post + (1 - gate) * reg_summary  # Fused representation
        #if model.visual.proj is not None:
        #    fused = fused @ model.visual.proj
        norm_fused = torch.norm(fused, dim=-1).squeeze(0)  # Scalar
        
        # ---- Combine norms ----
        # Order: first 256 patch tokens, then extra tokens: [CLS, REG tokens..., FUSED]
        extra_norms = torch.cat([norm_cls.unsqueeze(0), norm_regs, norm_fused.unsqueeze(0)], dim=0)
        combined_norms = torch.cat([norm_patch, extra_norms], dim=0)
        
        return combined_norms.cpu().detach().numpy()


# ----------------------------------------------------------------------
# Updated heatmap function to handle the additional "FUSED" token.
def save_heatmap(norms, image_name, text_token):
    """
    Saves a heatmap where the first 256 norms correspond to the 16x16 patch grid,
    and the extra norms correspond to [CLS, REG1, REG2, REG3, REG4, FUSED].
    """
    # ---- Extract norms for patch tokens and extra tokens ----
    patch_norms = norms[:256].reshape(16, 16)  # 16x16 grid for image patches
    extra_norms = norms[256:]  # Expected order: [CLS, REG1, REG2, REG3, REG4, FUSED]
    
    # ---- Determine color scale ----
    vmin, vmax = norms.min(), norms.max()
    
    # ---- Create main heatmap for patch tokens ----
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(patch_norms, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("L2 Norm")
    ax.set_title(f"Patch Norm Heatmap: {image_name}")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ---- Append extra tokens (CLS, REG, FUSED) below the main heatmap ----
    fig.subplots_adjust(bottom=0.2)  # add space at the bottom
    ax_extra = fig.add_axes([0.1, 0.05, 0.8, 0.1])  # adjust as needed for your layout
    
    # Reshape extra norms to a row
    extra_norms = np.array(extra_norms).reshape(1, -1)
    im_extra = ax_extra.imshow(extra_norms, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
    
    # ---- Set token labels (adjusted to include FUSED) ----
    labels = ["CLS"] + [f"REG{i+1}" for i in range(4)] + ["FUSED"]
    ax_extra.set_xticks(range(len(labels)))
    ax_extra.set_xticklabels(labels, rotation=0, fontsize=10, fontweight="bold")
    ax_extra.set_yticks([])
    ax_extra.set_title("CLS, REG & FUSED Token Norms", fontsize=12, pad=10)
    
    # ---- Save the figure ----
    sanitized_token = sanitize_filename(text_token)
    filename = f"{out_dir}/heatmap_{image_name}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# Function to generate L2 norm bar plot
def save_barplot(norms, image_name, text_token):
    sanitized_token = sanitize_filename(text_token)
    filename = f"{out_dir}/l2norm_{image_name}.png"

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(norms)), norms)
    plt.xlabel("Patch Token ID")
    plt.ylabel("L2 Norm")
    plt.title(f"L2 Norms of ViT Image Tokens: {image_name}")
    plt.savefig(filename)
    plt.close()

# Main processing loop
csv_records = []

for img_file in tqdm(os.listdir(img_dir)):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_name = os.path.splitext(img_file)[0]
        image_path = os.path.join(img_dir, img_file)
        #text_path = os.path.join(tok_dir, f"tokens_{image_name}.txt")

        #if not os.path.exists(text_path):
        #    print(f"Warning: No token file for {image_name}. Skipping.")
        #    continue

        # Load and encode image
        image = load_image(image_path)
        image_norms = clip_encode_image(model, image)

        # Process tokens
        token = "dummy"
        #with open(text_path, "r", encoding="utf-8") as f:
        #    tokens = f.read().strip().split()

        #token = tokens[0] # don't need text for just patch norm
        #text_features = clip_encode_text(model, token) # TODO: unified code

        # Save heatmap and bar plot
        save_heatmap(image_norms, image_name, token)
        save_barplot(image_norms, image_name, token)

        # Save norms with flagging
        for idx, norm in enumerate(image_norms):
            flag = "CLS" if idx == 0 else "!" if norm > 100 else ""
            csv_records.append([idx, norm, flag, image_name, token])


# Save results to CSV
df = pd.DataFrame(csv_records, columns=["TokenID", "Norm", "Flag", "Image", "TextToken"])
df.to_csv(f"{out_dir}/patch_norms.csv", index=False)

print(f"Processing complete. Check '{out_dir}' for visualizations and CSV output.")
