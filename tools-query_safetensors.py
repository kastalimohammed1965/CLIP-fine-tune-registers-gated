import torch
from safetensors.torch import load_file

# Just prints the keys in a model.

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Load the model state dictionary from the .safetensors file
state_dict = load_file("CLIPneedsREGISTERS/REG-XGATED/ft-checkpoints/clip_ft_12_backtoweight.safetensors")

def print_model_structure(state_dict):
    """Print the structure of the model state dictionary."""
    print(f"{'Key':<50} {'Shape':<30} {'Dtype'}")
    print("-" * 90)
    for key, tensor in state_dict.items():
        print(f"{key:<50} {str(tensor.shape):<30} {str(tensor.dtype)}")

# Print the structure of the model state dictionary
print_model_structure(state_dict)
