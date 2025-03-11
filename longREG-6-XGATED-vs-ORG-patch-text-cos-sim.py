import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from safetensors.torch import load_file
import argparse

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longINFERclipregXGATED as clip
from longmodel import longclip as orgclip



device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------- MANUALLY DEFINE IMAGE AND TEXTS HERE -------------------------- 

filename = "catshoe.png"
text_prompts = ["cat", "meow", "purr", "necklace", "speaker", "shoe", "shoes", "eye", "nose", "ear", "book", "metal object", "grumpy", "shoelace", "pink t", "horrified", "angry"]

#filename = "rifle.png"
#text_prompts = ["plunger", "couch", "stick", "toilet tool", "polearm", "improvised spear", "rifle", "vacuum attachment", "toy weapon", "meme material"]

#filename = "wfh.png"
#text_prompts = ["home office", "printer", "desk", "bookshelf", "leather chair", "laptop", "clock", "cable", "pictureframe", "brown object", "red object", "green object", "blue object"]

# ------------------------------------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize Long-CLIP patch-text cosine similarity')
    parser.add_argument('--base_model', default="models/LongCLIP-L.safetensors", help="Path to original ViT-L/14 model, pickle (.pt) or .safetensors")
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-REG-GATED-full-model.safetensors", help="Path to fine-tuned ViT-L/14 model, pickle (.pt) or .safetensors")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

model, preprocess = clip.load(model_name_or_path, device=device)
model = model.float()

modelorg, preprocess = orgclip.load(args.base_model, device=device)
modelorg = modelorg.float()



# -------- MAIN CODE --------

image = Image.open(f"EX-image-vis/{filename}")
image_input = preprocess(image).unsqueeze(0).to(device)

# Function to encode image features for CLIP model
def clip_encode_image(model, image_input):
    with torch.no_grad():
        x = model.visual.conv1(image_input)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  

        cls_token = model.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        register_tokens = model.visual.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)

        x = torch.cat([cls_token, register_tokens, x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        return x

# Function to encode image features for original CLIP
def orgclip_encode_image(modelorg, image_input):
    with torch.no_grad():
        x = modelorg.visual.conv1(image_input)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)  

        cls_token = modelorg.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + modelorg.visual.positional_embedding.to(x.dtype)
        x = modelorg.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  
        x = modelorg.visual.transformer(x)
        x = x.permute(1, 0, 2)
        return x

# Loop through each text prompt
for obj in text_prompts:
    text_prompt = [f"a {obj}"]  # Format prompt

    # Encode text features for both models
    text_features_myclip = model.encode_text(clip.tokenize(text_prompt).to(device)).float()
    text_features_myclip /= text_features_myclip.norm(dim=-1, keepdim=True)

    text_features_orgclip = modelorg.encode_text(orgclip.tokenize(text_prompt).to(device)).float()
    text_features_orgclip /= text_features_orgclip.norm(dim=-1, keepdim=True)

    # Encode image for both models
    x_myclip = clip_encode_image(model, image_input)
    x_orgclip = orgclip_encode_image(modelorg, image_input)

    for clip_model, x, text_features, model_name in zip(
        [model, modelorg], 
        [x_myclip, x_orgclip], 
        [text_features_myclip, text_features_orgclip], 
        ["regclip", "orgclip"]
    ):
        # Compute number of patches
        N_patches = x.shape[1] - 1  # Exclude CLS token
        h = w = int(np.sqrt(N_patches))

        # Apply the correct patch skipping rule
        if model_name == "regclip":
            x = x[:, 5:, :]  # Skip CLS and register tokens
        else:
            x = x[:, 1:, :]  # Skip only CLS token

        # Compute per-patch cosine similarity
        cos_sim_values = []

        for i in range(x.shape[1]):  
            patch = x[:, i, :]
            patch = clip_model.visual.ln_post(patch)  
            patch = patch @ clip_model.visual.proj  
            patch /= patch.norm(dim=-1, keepdim=True)

            cos_sim = torch.nn.functional.cosine_similarity(patch.view(1, -1), text_features.view(1, -1))
            cos_sim_values.append(cos_sim)

        cos_sim = torch.stack(cos_sim_values).cpu().detach().numpy()
        cos_sim = cos_sim.reshape(h, w)

        # Normalize heatmap
        cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())

        # Ensure image is in correct format
        image_np = np.array(image)

        # Resize heatmap to match image size
        heatmap = np.uint8(255 * cos_sim)
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Plot results
        plt.figure(figsize=(8, 8))
        plt.imshow(image_np)  
        plt.imshow(heatmap, cmap="jet_r", alpha=0.5)  
        plt.axis("off")
        plt.title(f"Model: {model_name} -- Text: '{text_prompt[0]}'", fontsize=25)
        plt.tight_layout()

        # Save the image
        os.makedirs(f"plots/longREG-XGATED/patch-cos", exist_ok=True) 
        save_path = f"plots/longREG-XGATED/patch-cos/{filename}_{obj}_{model_name}_seg.png"
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Saved: {save_path}")

print("Batch processing complete!")
