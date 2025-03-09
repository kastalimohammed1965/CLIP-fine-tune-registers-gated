import torch
import INFERclipregXGATED as clip
from collections import OrderedDict
from safetensors.torch import load_file
import argparse

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# If you want to see some numbers.
# Don't even need to run this, here they are: 

# REG-GATED CLIP:       Total Parameters: 452,815,117
# OpenAI/CLIP:          Total Parameters: 427,616,513

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_submodule_parameters(model, submodule_name):
    submodule = dict(model.named_modules()).get(submodule_name, None)
    if submodule is None:
        return 0
    return sum(p.numel() for p in submodule.parameters())

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize CLIP Attention heatmaps')
    parser.add_argument('--use_model', 
                        default="CLIPneedsREGISTERS/REG-XGATED/ft-checkpoints/clip_ft_12_backtoweight.pt", 
                        help="Path to a ViT-L/14 model, pickle (.pt) or .safetensors")
    parser.add_argument('--token_folder', default="EX-tokens-vis", help="Folder with gradient ascent .txt files of CLIP's opinions (or yours)")
    parser.add_argument('--image_folder', default="EX-image-vis", help="Folder with images, matching for .txt files: 'image.png' -> 'tokens_image.txt'")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model


if model_name_or_path.endswith(".safetensors"):
    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
    
    # Load ViT-L/14 explicitly
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    # Load the safetensors state_dict and apply
    state_dict = load_file(model_name_or_path)
    model.load_state_dict(state_dict)

else:
    print("Detected non-.safetensors file. Attempting to load as a pickle...")
    
    # Load normally as per the existing logic
    model, preprocess = clip.load(model_name_or_path, device=device, jit=False)

model = model.float() # full precision

# Total parameters
total_params = count_parameters(model)

# Parameters in visual.transformer vs. text transformer
visual_transformer_params = count_submodule_parameters(model.visual, "transformer")
text_transformer_params = count_submodule_parameters(model, "transformer")

# Detailed breakdown
visual_resblocks_params = count_submodule_parameters(model.visual, "transformer.resblocks")
visual_intermediate_fusion_mlps_params = count_submodule_parameters(model.visual, "transformer.intermediate_fusion_mlps")
visual_fusion_mlp_params = count_submodule_parameters(model.visual, "fusion_mlp")

# Print results
print("CLIP ViT-L/14 Model Parameter Stats")
print("-" * 40)
print(f"Total Parameters: {total_params:,}")
print(f"Visual Transformer Parameters: {visual_transformer_params:,}")
print(f"Text Transformer Parameters: {text_transformer_params:,}")
print("\nDetailed Visual Transformer Breakdown:")
print(f"- Resblocks: {visual_resblocks_params:,}")
print(f"- Intermediate Fusion MLPs: {visual_intermediate_fusion_mlps_params:,}")
print(f"- Final Fusion MLP: {visual_fusion_mlp_params:,}")
