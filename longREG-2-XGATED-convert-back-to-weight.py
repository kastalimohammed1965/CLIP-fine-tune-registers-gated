import os
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longINFERclipregXGATED.model
from longINFERclipregXGATED.model import CLIP

which_finetune_number = "20" # 20 for e.g. longclip_ft_20.pt

def convert_back_to_original(state_dict):
    """Convert Geometric Parameterization back to standard weights."""
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.endswith(".theta"):
            base_key = key.replace(".theta", "")
            r_key = base_key + ".r"
            new_weight = state_dict[r_key] * F.normalize(value, p=2, dim=1)
            new_state_dict[base_key + ".weight"] = new_weight
        elif key.endswith(".r") or key.endswith(".theta"):
            continue  # Skip the .r and .theta keys
        else:
            new_state_dict[key] = value

    return new_state_dict

# Load the fine-tuned model directly
model_path = f"CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints/longclip_ft_{which_finetune_number}.pt"
modelft = torch.load(model_path)

# Convert the fine-tuned model's state_dict back to standard format (undo Geometric Parametrization)
fine_tuned_state_dict = modelft.state_dict()
original_state_dict = convert_back_to_original(fine_tuned_state_dict)

# Explicitly retain new parameters
original_state_dict["visual.register_tokens"] = modelft.visual.register_tokens.detach().cpu()
original_state_dict["visual.fusion_mlp"] = {name: param.detach().cpu() for name, param in modelft.visual.fusion_mlp.named_parameters()}
original_state_dict["visual.transformer.intermediate_fusion_mlps"] = {name: param.detach().cpu() for name, param in modelft.visual.transformer.intermediate_fusion_mlps.named_parameters()}


# Save the corrected state_dict only
torch.save(original_state_dict, f"CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints/longclip_ft_{which_finetune_number}_state_dict.pt")

print(f"\n--------> Model state_dict successfully converted and saved as 'longclip_ft_{which_finetune_number}_state_dict.pt'")

# Load state_dict (converted version)
state_dict_path = f"CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints/longclip_ft_{which_finetune_number}_state_dict.pt"
state_dict = torch.load(state_dict_path)

# Extract required model parameters
embed_dim = state_dict["text_projection"].shape[1]
vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
vision_width = state_dict["visual.conv1.weight"].shape[0]
vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
context_length = state_dict["positional_embedding"].shape[0]
vocab_size = state_dict["token_embedding.weight"].shape[0]
transformer_width = state_dict["ln_final.weight"].shape[0]
transformer_heads = transformer_width // 64
transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

# Compute Image Resolution from Positional Embedding
grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # Ignore CLS token
input_resolution = grid_size * vision_patch_size  # Dynamically computed
pos_embeds=state_dict["visual.positional_embedding"].shape[0]

print(f"\nDEBUG: vision_layers {vision_layers}, vision_width {vision_width}, vision_patch_size {vision_patch_size}")
print(f"DEBUG: grid_size {grid_size}, input_resolution {input_resolution}, pos_embeds {pos_embeds}\n")


# Recreate fusion_mlp with the correct architecture
fusion_mlp = nn.Sequential(
    nn.Linear(2 * vision_width, vision_width),
    nn.ReLU(),
    nn.Linear(vision_width, 1)
)

# Recreate model and load state_dict
model = CLIP(
    embed_dim=embed_dim,
    image_resolution=input_resolution,
    vision_layers=vision_layers,
    vision_width=vision_width,
    vision_patch_size=vision_patch_size,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers,
)

# Load the saved state_dict for fusion_mlp
fusion_mlp.load_state_dict(state_dict["visual.fusion_mlp"])
# Intermediate MLP are already part of vision.transformer as visual.transformer.intermediate_fusion_mlps

# Assign it back to the model
model.visual.fusion_mlp = fusion_mlp

# Load state_dict into model
model.load_state_dict(state_dict, strict=False)

# Save the full model as a pickled '.pt' file
torch.save(model, f"CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints/longclip_ft_{which_finetune_number}_backtoweight.pt")

print(f"--------> Full pickled model saved as 'longclip_ft_{which_finetune_number}_backtoweight.pt'")


# ------------------- ABLATION CODE, IMPLEMENTATION -------------------

# Get the number of register tokens from the fine-tuned model.
num_registers = modelft.visual.register_tokens.shape[0]

# --- Compute the new positional embedding by slicing out the register token positions ---
# The original "visual.positional_embedding" is of shape: (1 + num_registers + num_patches, width),
# where index 0 is CLS, indices 1:1+num_registers are REG, and the rest are patch tokens.
original_pos_emb = original_state_dict["visual.positional_embedding"]
# Keep CLS at index 0 and patch tokens from index (1 + num_registers) onward.
new_pos_emb = torch.cat([original_pos_emb[:1], original_pos_emb[1+num_registers:]], dim=0)

# --- Create two variants of the state_dict ---
# 1. REG Ablation: Remove only the register tokens.
ablated_state_dict_REG = original_state_dict.copy()
ablated_state_dict_REG.pop("visual.register_tokens", None)  # Remove register tokens.
# Update positional embedding to the sliced one.
ablated_state_dict_REG["visual.positional_embedding"] = nn.Parameter(new_pos_emb)

# 2. ALL Ablation: Remove register tokens, fusion_mlp, and intermediate gating MLPs.
ablated_state_dict_ALL = original_state_dict.copy()
ablated_state_dict_ALL.pop("visual.register_tokens", None)  # Remove register tokens.
ablated_state_dict_ALL.pop("visual.fusion_mlp", None)         # Remove the final fusion MLP.
# Remove all keys corresponding to intermediate gating MLPs.
keys_to_remove = [k for k in ablated_state_dict_ALL.keys() if k.startswith("visual.transformer.intermediate_fusion_mlps")]
for k in keys_to_remove:
    ablated_state_dict_ALL.pop(k, None)
# Update positional embedding to the sliced one.
ablated_state_dict_ALL["visual.positional_embedding"] = nn.Parameter(new_pos_emb)

# --- Extract required model parameters from the REG ablated state dict ---
embed_dim = ablated_state_dict_REG["text_projection"].shape[1]
vision_layers = len([k for k in ablated_state_dict_REG.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
vision_width = ablated_state_dict_REG["visual.conv1.weight"].shape[0]
vision_patch_size = ablated_state_dict_REG["visual.conv1.weight"].shape[-1]
# For text, context_length is taken from the text positional embedding (key: "positional_embedding")
context_length = ablated_state_dict_REG["positional_embedding"].shape[0]
vocab_size = ablated_state_dict_REG["token_embedding.weight"].shape[0]
transformer_width = ablated_state_dict_REG["ln_final.weight"].shape[0]
transformer_heads = transformer_width // 64
transformer_layers = len(set(k.split(".")[2] for k in ablated_state_dict_REG if k.startswith("transformer.resblocks")))

# Compute the new grid size and input resolution from the updated visual positional embedding.
grid_size = round((ablated_state_dict_REG["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # (257-1=256) => 16 for 256 patch tokens
input_resolution = grid_size * vision_patch_size

print(f"\nDEBUG (ablation): vision_layers {vision_layers}, vision_width {vision_width}, vision_patch_size {vision_patch_size}")
print(f"DEBUG (ablation): grid_size {grid_size}, input_resolution {input_resolution}\n")

# --- Create and save the REG Ablation model ---
from clip.model import CLIP

ablated_model_REG = CLIP(
    embed_dim=embed_dim,
    image_resolution=input_resolution,
    vision_layers=vision_layers,
    vision_width=vision_width,
    vision_patch_size=vision_patch_size,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers,
)

# Load the REG-ablated state dict (note: keys for register_tokens are missing, as desired)
ablated_model_REG.load_state_dict(ablated_state_dict_REG, strict=False)

# Pointless. Skipping this to just all-ablations model below.
#torch.save(ablated_model_REG, f"CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints/longclip_ft_{which_finetune_number}_backtoweight_ablated_REG.pt")
#print(f"--------> Full pickled ablated REG model saved as 'longclip_ft_{which_finetune_number}_backtoweight_ablated_REG.pt'")

# --- Create and save the ALL Ablation model ---
ablated_model_ALL = CLIP(
    embed_dim=embed_dim,
    image_resolution=input_resolution,
    vision_layers=vision_layers,
    vision_width=vision_width,
    vision_patch_size=vision_patch_size,
    context_length=context_length,
    vocab_size=vocab_size,
    transformer_width=transformer_width,
    transformer_heads=transformer_heads,
    transformer_layers=transformer_layers,
)

# Load the ALL-ablated state dict (keys for register_tokens, fusion_mlp, and intermediate fusion MLPs are missing)
ablated_model_ALL.load_state_dict(ablated_state_dict_ALL, strict=False)
torch.save(ablated_model_ALL, f"CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints/longclip_ft_{which_finetune_number}_backtoweight_ablated.pt")
print(f"--------> Full pickled ablated ALL model saved as 'longclip_ft_{which_finetune_number}_backtoweight_ablated.pt'")
