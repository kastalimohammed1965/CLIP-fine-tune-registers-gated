import torch
from safetensors.torch import save_file
import argparse
import os

import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# IMPORTANT: Run 'REG-2-XGATED-convert-back-to-weight.py' first. 
# Input needs to be a "_backtoweight.pt" model without GmP.

# NOTE: This script does **NOT** convert anything to HuggingFace / transformers format.
# It is merely for converting an (unsafe) full model object / pickle into a state_dict safetensors as-is.
# You can only use it to load a .safetensors state_dict into an OpenAI "import clip" model.

def flatten_nested_tensors(state_dict):
    """ Flattens nested dictionaries of tensors before saving to safetensors """
    flat_dict = {}

    for key, value in state_dict.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                flat_dict[f"{key}.{subkey}"] = subval
        else:
            flat_dict[key] = value

    return flat_dict

def unflatten_nested_tensors(state_dict, original_keys):
    """ Reconstructs nested dictionaries of tensors after loading from safetensors """
    nested_dict = {}

    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) > 1 and parts[0] in original_keys:
            if parts[0] not in nested_dict:
                nested_dict[parts[0]] = {}
            nested_dict[parts[0]][parts[1]] = value
        else:
            nested_dict[key] = value

    return nested_dict

def convert_pickle_to_safetensors(input_path, output_path=None):
    """ Converts a PyTorch .pt/.bin/.pth checkpoint (either state_dict or full model) to .safetensors """

    print(f"Loading PyTorch checkpoint from: {input_path}")
    loaded_obj = torch.load(input_path, map_location="cpu")

    # Extract state_dict
    if isinstance(loaded_obj, dict):
        state_dict = loaded_obj
        print("Detected state_dict.")
    else:
        print("Detected full model object. Extracting state_dict...")
        if hasattr(loaded_obj, "state_dict"):
            state_dict = loaded_obj.state_dict()
        else:
            raise ValueError("The loaded object does not have a state_dict method.")

    # Flatten nested structures before saving
    original_keys = [key for key, value in state_dict.items() if isinstance(value, dict)]
    state_dict = flatten_nested_tensors(state_dict)

    # Default output path if not specified
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".safetensors"

    # Save as safetensors
    print(f"Saving safetensors checkpoint to: {output_path}")
    save_file(state_dict, output_path)

    print("[---! SUCCESS !---] Conversion complete. ALL keys, including `visual.register_tokens`, are retained!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt/.bin/.pth to .safetensors")
    parser.add_argument("input_path", help="Path to the input .pt/.bin/.pth file")
    parser.add_argument("--output_path", default=None, help="Path to save the output .safetensors file (optional)")
    
    args = parser.parse_args()
    
    convert_pickle_to_safetensors(args.input_path, args.output_path)

