"""
Builds on the basic demo that showcases the key capabilities of the Direct Ascent Synthesis (DAS) method by Stanislav Fort and Jonathan Withaker,
described in the paper [Direct Ascent Synthesis: Revealing Hidden Generative Capabilities in Discriminative Models
https://arxiv.org/abs/2502.07753 | https://github.com/stanislavfort/Direct_Ascent_Synthesis

Uses a heavily modified version of Original CLIP Gradient Ascent Script: by Twitter / X: @advadnoun

Uses a heavily modified version of Original Feature Visualization by Hamid Kazemi: https://github.com/hamidkazemi22/vit-visualization

THIS: by zer0int https://github.com/zer0int
"""

import warnings
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os
import sys
import gc
import math
import copy
import datetime
import argparse
import tqdm
import random
from colorama import Fore, Style
import longINFERclipregXGATED as clip
from longINFERclipregXGATED.model import QuickGELU
import open_clip
import numpy as np
import torch
from torch.optim import SGD
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import collections
from safetensors.torch import load_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.utils
import torchvision
from PIL import Image
import kornia.augmentation as kaugs
import kornia
from scipy import stats
from scipy.ndimage import gaussian_filter
from safetensors.torch import load_file
# Custom imports
from cliptools import TotalVariation, ActivationNorm, LossArray
from cliptools import ViTFeatHook, ViTEnsFeatHook, ClipGeLUHook
from cliptools import Clip, Tile, Jitter, RepeatBatch, ColorJitter, GaussianNoise
from cliptools import new_init, save_intermediate_step, save_image
from cliptools import get_clip_vit_dimensions, normalize_for_clip, Normalization, fix_random_seed
from cliptools import save_model_dtypes, convert_model_to_full_precision, restore_model_dtypes
from cliptools import ClipNeuronCaptureHook, ClipViTWrapper
from cliptools import raw_to_real_image, real_to_raw_image

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = 'cuda'

try:
    from torch.amp import autocast, GradScaler
except:
    from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def parse_model_name(values):
    model = values[0]
    extra = values[1] if len(values) > 1 else None

    if model.startswith("OpenAI"):
        # OpenAI models: second argument must be None or a .pt file
        if isinstance(extra, str) and extra.lower() == "none":
            extra = None  # Convert "None" string to actual None
        elif extra is not None and not extra.endswith((".pt", ".safetensors")):
            raise argparse.ArgumentTypeError(
                f"Invalid file format for '{model}': Expected a .pt or .safetensors checkpoint or 'None'."
            )
    else:
        # open_clip models: second argument must exist and must NOT be a .pt file
        if extra is None:
            raise argparse.ArgumentTypeError(f"Model '{model}' requires a dataset name (e.g., 'laion2b_s34b_b79k').")
        if extra.endswith(".pt"):
            raise argparse.ArgumentTypeError(f"Invalid dataset name '{extra}' for '{model}': Expected a dataset name, not a .pt file.")

    return model, extra # tuple

parser = argparse.ArgumentParser()


# ----------------------------------------------------------
#  So many args + help, I'll just call this: THE MANUAL. :)
# ----------------------------------------------------------
parser = argparse.ArgumentParser(description="CLI-based CLIP Direct Ascent Image Generation")
# Less important stuff (for you to tweak)
parser.add_argument("--results_dir", type=str, default="results", help="Base directory to save all outputs to; defaults to subfolder 'results'")
parser.add_argument('--total_steps', type=int, default=100, help="How many steps to optimize for; defaults to 100")
# Stats for nerds
parser.add_argument("--make_plots", action='store_true', help="Generate additional informative plots with stats for nerds")
parser.add_argument("--make_lossplots", action='store_true', help="Visualize loss landscape for gradient ascent text embeddings")
# IMPORTANT --- Make faster: Change augs_cp to 32. Increase augs_cp for quality.
parser.add_argument('--augs_cp', type=int, default=64, help="How many augmentations to get a gradient from at once; defaults to 64")
parser.add_argument('--multi', type=int, default=6, help="How many images to generate for multi-generation; defaults to 6")
parser.add_argument('--ga_batch_size', default=8, type=int, help="Gradient Ascent batch_size, if '--use_image'. Defaults to 8, minimum: 2")
# Change this if CUDA OOM, especially when using ViT-L (try 16 or 8):
parser.add_argument('--batch_size', default=16, type=int, help="Batch Size for Direct Ascent. Defaults to 16.")
# Images to load (use up to 5):
parser.add_argument('--img0', type=str, default="EX-image-vis/catshoe.png", help="Main image path. Defaults to images/cat.png")
parser.add_argument('--img1', type=str, default=None, help="Optional 2nd image path to pass")
parser.add_argument('--img2', type=str, default=None, help="Optional 3rd image path to pass")
parser.add_argument('--img3', type=str, default=None, help="Optional 4th image path to pass")
parser.add_argument('--img4', type=str, default=None, help="Optional 5th image path to pass")
# IMPORTANT --- --use_image = using image instead of text prompt - this will also replace --img0 with the --use_image
parser.add_argument('--use_image', type=str, default=None, help="Path to a single image to use as main input AND as input instead of text (gradient ascent)")
parser.add_argument("--make_anti", action='store_true', help="Makes an additional anti-text image for *minimum* cosine similarity if --use_image, else image of anti-text prompt (gradient ascent)")
# Prompt, if using text prompt (IMPORTANT --- will be discarded if --use_image):
parser.add_argument("--txt1", type=str, default="a beautiful photo of a cat sits on shoes resting., detailed", help="Main prompt, weight (1.0). Default: 'a beautiful photo of a cat sits on shoes resting., detailed'")
parser.add_argument("--negtxt1", type=str, default="optical character recognition", help="Negative prompt 1 (-0.3). Default (tries to prevent text in image): 'optical character recognition'")
parser.add_argument("--txt2", type=str, default="octane render, unreal engine, ray tracing, volumetric lighting", help="Secondary prompt, weight (0.3). Default: 'octane render, unreal engine, ray tracing, volumetric lighting'")
parser.add_argument("--negtxt2", type=str, default="multiple exposure", help="Negative prompt 2 (-0.3). Default: 'multiple exposure'")
# IMPORTANT --- Neuron (Feature Visualization) arguments - adds an image (or images) of CLIP's MLP features to the batch.
parser.add_argument('--use_neuron', action='store_true', help="Makes an image of a CLIP ViT Feature ('Neuron') with max activation value resulting from --use_image (if None, from img0).")
parser.add_argument('--all_neurons', action='store_true', help="Makes images of the top activating feature of ALL layers of CLIP (12 or 24), and appends them all to the batch of images")
# Big CLIP ViT-L's features are a noisy, chaotic confusion at the final layer. Alas use earlier for ViT-L, use final layer for 12-layer smaller CLIP, by default:
parser.add_argument('--vit_l_neuron', default=4, type=int, help="Target Layer -(int) for 'neuron', from end of transformer; 1 results in '-1' -> final layer. Default: 4")
parser.add_argument('--vit_neuron', default=1, type=int, help="Target Layer -(int) for 'neuron', from end of transformer small / 12-layer models. Default: 1")
# IMPORTANT --- custom open_clip or (fine-tuned, pre-trained) OpenAI models; to use your custom fine-tune: --model_name 'OpenAI-ViT-L/14' 'path/to/custom_model.pt'
parser.add_argument("--use_model", nargs="+", type=str, default=["OpenAI-models/Long-ViT-L-14-REG-GATED-full-model.safetensors", 'models/Long-ViT-L-14-REG-GATED-full-model.safetensors'], help="Primary model name. Usage: --model_name OpenAI-ViT-L/14 path/to/custom_model.pt (.safetensors)")
parser.add_argument("--custom_model2", nargs="+", type=str, default=None, help="Second model name (optional)")
parser.add_argument("--custom_model3", nargs="+", type=str, default=None, help="Third model name (optional)")
# Mostly deterministic, at least for Gradient Ascent Text Embeddings + Neurons (Features):
parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends, torch, numpy)")
# Skip any of the tasks by supplying these:
parser.add_argument('--no1', action='store_true', help="Skips task #1")
parser.add_argument('--no2', action='store_true', help="Skips task #2")
parser.add_argument('--no3', action='store_true', help="Skips task #3")
parser.add_argument('--no4', action='store_true', help="Skips task #4")
# Manual layer selection for Vision and Text Encoder. First model (--model_name) OpenAI only.
parser.add_argument("--manu_vit", action='store_true', help="Use manual final layer (resblock) for Vision Encoder (ViT)")
parser.add_argument("--manu_txt", action='store_true', help="Use manual final layer (resblock) for Text Encoder (TxT)")
# SDXL uses penultimate (second-to-last) -2 layer instead of final for CLIP Text Encoder.
# -1 means final layer (changes nothing), -2 means penultimate, and so on. Counting from back of transformer.
parser.add_argument('--set_vit', default=1, type=int, help="Manual Final Layer -(int) to use for ViT. Default: 1")
parser.add_argument('--set_txt', default=1, type=int, help="Manual Final Layer -(int) to use for TxT. Default: 1")
# Skip the final layer normalization before projection by using these args:
parser.add_argument("--skip_ln_vit", action='store_true', help="Skip final ViT layer_norm before projection")
parser.add_argument("--skip_ln_txt", action='store_true', help="Skip final TxT layer_norm before projection")



args = parser.parse_args()
args.model_name = args.use_model

# Validate and convert model_name
args.model_name = tuple(parse_model_name(args.model_name))
if args.custom_model2:
    args.custom_model2 = tuple(parse_model_name(args.custom_model2))
if args.custom_model3:
    args.custom_model3 = tuple(parse_model_name(args.custom_model3))

if args.use_image:
    image_name = img_name = os.path.splitext(os.path.basename(args.use_image))[0]
    print(Fore.CYAN + Style.BRIGHT + "Using input image to create text embedding. Text prompt inputs will be ignored." + Fore.RESET)

if args.deterministic:
    fix_random_seed()

image_name = "dummy"
ga_batch_size = args.ga_batch_size
multi_many = args.multi
results_dir = args.results_dir
steps = args.total_steps
augmentation_copies = args.augs_cp
batch_size = args.batch_size
vit_l_neuron = args.vit_l_neuron
vit_neuron = args.vit_neuron

clipname = args.model_name[0].replace("/", "-").replace("@", "-")

print_once_v=False
print_once_t=False
last_txt = None
last_vit = None
# -------
#  Utils
# -------

def get_model_type(model_name):
    # Normalizes model name for comparison
    normalized_name = model_name.replace("OpenAI-", "").replace("/", "-").strip("'\"")
    return normalized_name

def manu_clip_encode_text(model, text, last=-args.set_txt):
    global print_once_t
    global last_txt
    n_ctx = text.shape[-1]
    x = model.token_embedding(text)
    x = x + model.positional_embedding[:n_ctx]
    x = x.permute(1, 0, 2)

    # Transformer pass with manual layer selection
    total_layers = len(model.transformer.resblocks)
    if last < 0:
        last = total_layers + last
        if last < 0: # If last would be out of range, reset to use layer 0.
            print(Fore.RED + Style.BRIGHT + f"Final layer to use out of range with {last}. Setting to layer 0 instead." + Fore.RESET)
            print(f"Total Layers in the current model: {total_layers}")
            last = 0
    last_txt=last
    if not print_once_t:
        print(Fore.CYAN + Style.BRIGHT + f"\nSelected as last layer (TxT): {last}" + Fore.RESET)
        print_once_t=True

    for layer in model.transformer.resblocks[:last + 1]:
        x = layer(x)

    x = x.permute(1, 0, 2)  # LND -> NLD

    if not args.skip_ln_txt:
        x = model.ln_final(x)

    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection
    return x

def manu_clip_encode_image(model, image, last=-1):
    global print_once_v
    global last_vit
    # Initial convolutional (patch) embedding
    x = model.visual.conv1(image)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    cls_token = model.visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    register_tokens = model.visual.register_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)

    x = torch.cat([cls_token, register_tokens, x], dim=1)
    #x = torch.cat([class_emb, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND

    # Transformer pass with manual layer selection
    total_layers = len(model.visual.transformer.resblocks)
    if last < 0:
        last = total_layers + last
        if last < 0: # If last would be out of range, reset to use layer 0.
            print(Fore.RED + Style.BRIGHT + f"Final layer to use out of range with {last}. Setting to layer 0 instead." + Fore.RESET)
            print(f"Total Layers in the current model: {total_layers}")
            last = 0
    last_vit=last
    if not print_once_v:
        print(Fore.MAGENTA + Style.BRIGHT + f"\nSelected as last layer (ViT): {last}" + Fore.RESET)
        print_once_v=True

    for layer in model.visual.transformer.resblocks[:last + 1]:
        x = layer(x)

    x = x.permute(1, 0, 2)  # LND -> NLD

    if args.skip_ln_vit:
        x = x[:, 0, :] # Do not apply final layer normalization
    else:
        x = model.visual.ln_post(x[:, 0, :])

    if model.visual.proj is not None:
        x = x @ model.visual.proj

    return x

def get_many_text_features(model, tokenizer, texts):
    # If texts is a list of strings, tokenize and encode.
    if isinstance(texts, (list, tuple)) and isinstance(texts[0], str):
        tokenized_text = tokenizer(texts).to("cuda")
        if args.manu_txt:
            return manu_clip_encode_text(model, tokenized_text)
        else:
            return model.encode_text(tokenized_text)
    # Otherwise, assume texts is already a tensor of embeddings.
    elif isinstance(texts, torch.Tensor):
        return texts
    else:
        raise ValueError(
            "Unexpected type for texts in get_many_text_features.")

def get_many_image_features(model, batch_of_images):
    image_features = manu_clip_encode_image(model, batch_of_images, last=-args.set_vit)
    return image_features

def loss_between_images_and_text(model, images, text_features, target_values=None):
    text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)
    if args.manu_vit:
        image_features = manu_clip_encode_image(model, images)
    else:
        image_features = model.encode_image(images)
    image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
    scores = image_features_normed @ text_features_normed.T
    if target_values is None:
        return torch.mean(scores, axis=1)
    else:
        return torch.mean(scores * torch.Tensor(target_values).to("cuda").reshape([1,-1]), axis=1)

def add_jitter(x,size=3,res=224):
    in_res = x.shape[2]
    if size > 0:
        x_shift = np.random.choice(range(-size,size+1))
        y_shift = np.random.choice(range(-size,size+1))
    elif size == 0:
        x_shift = 0
        y_shift = 0
    x = torch.roll(x,shifts=(x_shift, y_shift),dims=(-2,-1))
    x = x[:,:,(in_res-res)//2:(in_res-res)//2+res,(in_res-res)//2:(in_res-res)//2+res]
    return x

def add_noise(x, scale=0.1):
    return x + torch.rand_like(x) * scale

def get_optimal_grid(n, max_cols=5):
    cols = min(max_cols, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    return rows, cols

def make_image_augmentations(image_in, count=1, clip_output=True, jitter_scale=3, noise_scale=0.1):
    images_collected = []
    for _ in range(count):
        image_resized_now = image_in
        image_resized_now = add_jitter(image_resized_now,size=jitter_scale) if jitter_scale is not None else add_jitter(image_resized_now,size=0)
        image_resized_now = add_noise(image_resized_now,scale=noise_scale) if noise_scale is not None else image_resized_now
        images_collected.append(image_resized_now)

    return torch.clip((torch.cat(images_collected,axis=0)),0,1) if clip_output else (torch.cat(images_collected,axis=0))

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(image)

def load_images(image_paths, batch_size=batch_size):
    valid_image_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_image_paths:
        print("None of the provided image paths exist.")
        return None
    dataset = ImageDataset(valid_image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return dataloader

def load_clip_models(models_to_load):
    global preprocess
    global tokenizer
    models_and_tokenizers = []

    for model_str, data_str in models_to_load:
        print(f"Loading {model_str} on data {data_str}")

        if not model_str.startswith('OpenAI'): # Open_CLIP models
            model_str = model_str.strip("'\"")
            data_str = data_str.strip("'\"")  # Fix: Clean data_str properly
            model, _, preprocess = open_clip.create_model_and_transforms(model_str, pretrained=data_str)    
            model.to("cuda").float()
            tokenizer = open_clip.get_tokenizer(model_str)

            mean = preprocess.transforms[-1].mean
            std = preprocess.transforms[-1].std

            models_and_tokenizers.append((model, tokenizer, mean, std))

        else: # OpenAI CLIP models
            print(f"Loading {model_str} on data {data_str}")
            model, preprocess = clip.load(model_str.split("OpenAI-")[1])
            model.to("cuda").eval().float()
            tokenizer = clip.tokenize

            custom_path = data_str
            if custom_path == "obsolete_lets_make_sure_we_dont_go_here":
                ext = os.path.splitext(custom_path)[-1]

                if ext == ".safetensors":
                    state_dict = load_file(custom_path)
                    print("Detected .safetensors file. Loading ViT-L/14 and applying file as state_dict...")
                else:
                    checkpoint = torch.load(custom_path, map_location="cuda")

                    if isinstance(checkpoint, torch.jit.ScriptModule):  
                        # It's a full JIT model
                        state_dict = checkpoint.state_dict()
                    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        # It's a model checkpoint containing `state_dict`
                        state_dict = checkpoint["state_dict"]
                    elif isinstance(checkpoint, dict):
                        # It's already a raw `state_dict`
                        state_dict = checkpoint
                    elif type(checkpoint).__name__.endswith("CLIP"):  
                        # General check: Model class name ends in "CLIP"
                        state_dict = checkpoint.state_dict()
                    else:
                        raise ValueError(f"Unexpected model format in {custom_path}, type: {type(checkpoint)}")

                model.load_state_dict(state_dict)
                print(f"Loaded fine-tuned weights from {custom_path}")

            mean = preprocess.transforms[-1].mean
            std = preprocess.transforms[-1].std
            models_and_tokenizers.append((model, tokenizer, mean, std))

    return models_and_tokenizers


def save_images(collected_images, results_dir, next_num, task_type, large_resolution=224, original_resolution=224,
                guiding_images_batch=None, start_images_batch=None,
                show_guiding_image=False, show_starting_image=False):

    os.makedirs(results_dir, exist_ok=True)

    # Get existing filenames and determine next available number
    existing = [int(f.split("_")[1].split(".")[0])
                for f in os.listdir(results_dir)
                if f.startswith(f"{task_type}_") and f.endswith(".png") and f.split("_")[1].split(".")[0].isdigit()]

    next_num = max(existing, default=-1) + 1 # Ensure no overwriting
    offset = (large_resolution - original_resolution) // 2
    versions = collected_images[-1].shape[0]

    extra = 0
    if show_guiding_image:
        extra += 1
    if show_starting_image:
        extra += 1
    total = versions + extra

    suffix_txt = f"_txt{last_txt}" if last_txt is not None else ""
    suffix_vit = f"_vit{last_vit}" if last_vit is not None else ""

    # Save each individual image separately (no borders, unique filenames)
    for v in range(versions):
        img = collected_images[-1][v][:, offset:offset + original_resolution, offset:offset + original_resolution]
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

        individual_filename = f"{task_type}_{next_num + v}{suffix_txt}{suffix_vit}.png"
        individual_path = os.path.join(results_dir, individual_filename)

        plt.imsave(individual_path, np.clip((img * 255).astype(np.uint8), 0, 255))
        print(f"Saved individual image: {individual_path}")

    # Compute optimal layout for the multi-image canvas
    rows, cols = get_optimal_grid(total, max_cols=5)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=112)

    if rows == 1:
        axes = np.array(axes).reshape(1, -1) # Ensure it's always iterable
    elif cols == 1:
        axes = np.array(axes).reshape(-1, 1)

    axes = axes.flatten()

    for v, ax in enumerate(axes):
        if v < total:
            if v < versions:
                img = collected_images[-1][v][:, offset:offset + original_resolution, offset:offset + original_resolution]
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
            elif show_guiding_image and not (show_guiding_image and show_starting_image):
                img = guiding_images_batch[0].detach().cpu().numpy().transpose(1, 2, 0)
            elif show_guiding_image and show_starting_image:
                if v == total - 2:
                    img = guiding_images_batch[0].detach().cpu().numpy().transpose(1, 2, 0)
                else:
                    img = start_images_batch[0].detach().cpu().numpy().transpose(1, 2, 0)[offset:-offset, offset:-offset]

            ax.imshow(np.clip((img * 255).astype(np.uint8), 0, 255))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
        else:
            ax.remove()

    combined_filename = f"all_{task_type}_{next_num}{suffix_txt}{suffix_vit}.png"
    combined_path = os.path.join(results_dir, combined_filename)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(combined_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Saved combined image: {combined_path}")

# ----------------
#  Visualizations
# ----------------

def smooth_loss_landscape(loss_landscape, sigma=2.0):
    return gaussian_filter(loss_landscape, sigma=sigma)

def visualize_loss_landscape(image, model, lats, many_tokens, prompt, nom, augment, 
                             best_embeddings, worst_embeddings, results_dir="results"):
    """
    Approximates the loss landscape using finite differences in multiple random directions.
    """

    grid_size = 75  # May wanna turn this down to 50 if it takes too long
    alpha_range = np.linspace(-1.5, 1.5, grid_size)
    beta_range = np.linspace(-1.5, 1.5, grid_size)

    loss_landscape_best = np.zeros((grid_size, grid_size))
    loss_landscape_worst = np.zeros((grid_size, grid_size))

    num_directions = 5  # More averaging for better smoothing
    direction_1 = sum([torch.randn_like(best_embeddings).to(best_embeddings.device) for _ in range(num_directions)]) / num_directions
    direction_2 = sum([torch.randn_like(best_embeddings).to(best_embeddings.device) for _ in range(num_directions)]) / num_directions

    direction_1 /= direction_1.norm()
    direction_2 /= direction_2.norm()

    print("Computing loss landscape... (this will take up to a few minutes)")

    expected_dtype = next(model.parameters()).dtype

    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            perturbed_best = best_embeddings + alpha * direction_1 + beta * direction_2
            perturbed_worst = worst_embeddings + alpha * direction_1 + beta * direction_2

            perturbed_best = perturbed_best.to(dtype=expected_dtype)
            perturbed_worst = perturbed_worst.to(dtype=expected_dtype)

            with torch.no_grad():
                loss_best, _, _ = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
                loss_worst, _, _ = ascend_txt_inverse(image, model, lats, many_tokens, prompt, nom, augment, best_embeddings)

            loss_landscape_best[i, j] = loss_best.mean().item()
            loss_landscape_worst[i, j] = loss_worst.mean().item()

        if i % 10 == 0:
            print(f"Progress: {i}/{grid_size}")

    print("Loss landscape computation complete.")

    # Gaussian smoothing, so looks like landscape (and not like spike galore)
    loss_landscape_best = smooth_loss_landscape(loss_landscape_best, sigma=2.5)
    loss_landscape_worst = smooth_loss_landscape(loss_landscape_worst, sigma=2.5)

    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=200, constrained_layout=True)

    # Best embeddings landscape (Max Cosine)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(alpha_grid, beta_grid, loss_landscape_best, cmap='viridis', edgecolor='none', antialiased=True)
    ax1.set_title("Loss Landscape (Text -> Max Cosine -> Image)", fontsize=16)
    ax1.set_xlabel("Direction 1")
    ax1.set_ylabel("Direction 2")
    ax1.set_zlabel("Loss")

    # Worst embeddings landscape (Min Cosine)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(alpha_grid, beta_grid, loss_landscape_worst, cmap='inferno', edgecolor='none', antialiased=True)
    ax2.set_title("Loss Landscape (Text -> Min Cosine -> Text[Image])", fontsize=16)
    ax2.set_xlabel("Direction 1")
    ax2.set_ylabel("Direction 2")
    ax2.set_zlabel("Loss")

    # Save high-resolution plot
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    save_path = f"{results_dir}/plots/{image_name}_ga_loss_landscape.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Saved high-resolution loss landscape plot to {save_path}.")

def visualize_individual_resolutions(
    all_image_perturbations,
    version_i = 0,
    selection_resolutions = [1,2,4,8,16,32,64],
    large_resolution = 224,
    ):

    # visualizing the image perturbations in all_image_perturbations
    plt.figure(figsize=(3*len(selection_resolutions),3*2))
    for i,p in enumerate(all_image_perturbations):
        if all_image_perturbations[i].shape[2] in selection_resolutions:
            print(f"{i} {all_image_perturbations[i].shape}")

            plt.subplot(2,len(selection_resolutions),selection_resolutions.index(all_image_perturbations[i].shape[2])+1)

            data = all_image_perturbations[i][version_i]
            data = raw_to_real_image(data).detach().cpu().numpy().transpose([1,2,0])
            data = data - np.min(data)
            data = data / np.max(data)

            plt.xticks([],[])
            plt.yticks([],[])
            plt.gca().set_axis_off()  # Turn off the axis
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove x-axis ticks
            plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove y-axis ticks

            plt.subplot(2,len(selection_resolutions),selection_resolutions.index(all_image_perturbations[i].shape[2])+1+len(selection_resolutions))

            # use the interpolation from the attack function on the data
            data_interpolated = F.interpolate(all_image_perturbations[i], size=(large_resolution, large_resolution), mode='bicubic')

            data = raw_to_real_image(data_interpolated[version_i]).detach().cpu().numpy().transpose([1,2,0])
            data = data - np.min(data)
            data = data / np.max(data)

            plt.imshow(data)

            plt.xticks([],[])
            plt.yticks([],[])
            plt.gca().set_axis_off()  # Turn off the axis
            plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove x-axis ticks
            plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Remove y-axis ticks

            os.makedirs(f"{results_dir}/plots", exist_ok=True)
            plt.savefig(f"{results_dir}/plots/{image_name}_individuals_{i}.png")

def analyze_perturbations(all_image_perturbations):
    """
    Analyze image perturbations across different resolutions and fit a power law.

    Parameters:
    all_image_perturbations: list of torch tensors containing perturbations at different resolutions

    Returns:
    slope: float, the power law exponent
    intercept: float, the y-intercept of the fitted line
    r_value: float, the correlation coefficient
    """
    # Calculate variances and get resolutions
    rs = []
    vars = []
    for i, p in enumerate(all_image_perturbations):
        vars.append(torch.var(all_image_perturbations[i]).detach().cpu().numpy())
        rs.append(all_image_perturbations[i].shape[2])

    # Convert to numpy arrays and take logs
    log_rs = np.log(rs)
    log_vars = np.log(vars)

    # Fit line to log-log data
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_rs, log_vars)

    # Create plot
    plt.figure(figsize=(5, 5))

    # Plot data points
    plt.loglog(rs, vars, 'o', color='skyblue', label='Data')

    # Plot fitted line
    x_fit = np.array([min(rs), max(rs)])
    y_fit = np.exp(intercept + slope * np.log(x_fit))
    plt.loglog(x_fit, y_fit, '--', color='red',
               label=f'Fitted line (slope = {slope:.3f})')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Resolution',fontsize=14)
    plt.ylabel('Variance',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Perturbation Variance vs. Resolution',fontsize=14)
    plt.legend()
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    plt.savefig(f"{results_dir}/plots/{image_name}_perturbations.png")
    print(f"Power law exponent: {slope:.3f}")
    print(f"R-squared value: {r_value**2:.3f}")

    return slope, intercept, r_value


# ---------------------------------------------
#  Feature Visualization / args: --use_neuron
# ---------------------------------------------

def register_hooks(model, num_layers):
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, QuickGELU):
            hook = ClipNeuronCaptureHook(module, layer_idx)
            hooks.append(hook)
            layer_idx += 1
            if layer_idx >= num_layers:
                break
    return hooks

def get_all_top_neurons(hooks, k=10):
        all_top_neurons = []

        for hook in hooks:
            layer_idx, top_value, top_index = hook.get_top_neuron()
            if top_value is not None:
                all_top_neurons.append((layer_idx, top_value, top_index))

        # Sort by activation value (highest first)
        all_top_neurons.sort(key=lambda x: x[1], reverse=True)

        # Return top neuron values
        return all_top_neurons

def wrap_clip_model(model, device: str = 'cuda'):
    return ClipViTWrapper(model).to(device)

class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200, saver: bool = True,
                 print_every: int = 55, **_):
        self.loss = loss_array
        self.saver = saver
        self.pre_aug = pre_aug
        self.post_aug = post_aug
        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.Tensor = None, optimizer: optim.Optimizer = None, layer: int = None, feature: int = None, clipname: str = None):
        if not img.is_cuda or img.device != torch.device('cuda:0'):
            img = img.to('cuda:0')
        if not img.requires_grad:
            img.requires_grad_()

        optimizer = optimizer if optimizer is not None else optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver:
                save_intermediate_step(img, i, layer, feature, clipname)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()

        optimizer.state = collections.defaultdict(dict)
        return img

def generate_single(model, clipname, layer, feature, image_size, lr, steps, print_every, save_every, saver, coefficient, tv):
    loss = LossArray()
    loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient * tv)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
    image = new_init(image_size, 1)

    visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient)
    image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

    os.makedirs(f"{results_dir}/neurons", exist_ok=True)
    save_path = f'{results_dir}/neurons/{clipname}_L{layer}_F{feature}.png'
    save_image(image, save_path)

    return image, save_path

def get_clip_feature(model, preprocess, primary_image, vit_l_neuron, vit_neuron):
    input_dims, num_layers, num_features = get_clip_vit_dimensions(model, preprocess)
    model = wrap_clip_model(model)

    print(f"Selected input dimension: {input_dims}")
    print(f"Number of Layers: {num_layers} with {num_features} Features/Layer\n")

    if primary_image is not None:
        hooks = register_hooks(model, num_layers)
        _ = model(primary_image)  # Forward pass to capture activations
        all_top_neurons = get_all_top_neurons(hooks)

        # Print all top neurons found per layer
        # TODO: Make user select layer manually based on this list?
        print(Fore.GREEN + Style.BRIGHT + "---------------- TOP NEURONS -----------------" + Fore.RESET)
        for layer_idx, activation_value, feature_idx in all_top_neurons:
            print(f"Layer {layer_idx}, Feature {feature_idx}, Activation Value: {activation_value}")

        if all_top_neurons:
            layer, activation_value, feature = max(all_top_neurons, key=lambda x: x[1])  # Get highest activation overall
            print(f"Highest Activation - Layer {layer}, Feature {feature}, Value: {activation_value}")
        else:
            layer, feature = num_layers - 1, torch.randint(0, num_features, (1,)).item()
            activation_value = None
    else:
        layer, feature = num_layers - 4 if num_layers > 12 else num_layers - 1, torch.randint(0, num_features, (1,)).item()
        activation_value = None

    # Layer Picker
    if "L/14" in args.model_name[0] or "L-14" in args.model_name[0]:
        layer = num_layers - vit_l_neuron
    else:
        layer = num_layers - vit_neuron
    print(Fore.CYAN + Style.BRIGHT + f"\nGenerating with Layer {layer}, Feature {feature}.\n" + Fore.RESET)
    return generate_single(model, clipname, layer, feature, input_dims, 1.0, 400, 10, 10, False, 0.00005, 1.0)


# ------------------------------------------------------
#  Gradient Ascent Text Embeddings / args: --use_image
# ------------------------------------------------------

def load_ga_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

def augment(into, augs):
    return augs(into)

# Gradient Ascent manual encode_text
def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('<|startoftext|>', '').replace('<|endoftext|>', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('$', '')
        j = j.replace('%', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        tokens = j.split()
        unique_tokens.update(tokens)

    os.makedirs("TOK", exist_ok=True)
    with open(f"TOK/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, ga_batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        self.ga_batch_size = ga_batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt

        # Initialize parameters for softmax
        st = torch.zeros(ga_batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(ga_batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(
            ga_batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1

        self.update_padding()

    def update_padding(self):
        pad_length = 77 - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.ga_batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        self.soft = F.gumbel_softmax(
            self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat(
            [self.start, self.prompt_embeddings, self.soft, self.pad], 1)
        return fin


# Gradient Ascent for maximizing cosine similarity (image-text)
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment, inverse=False):
    iii = nom(augment(image[:, :3, :, :].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    loss = -100 * torch.cosine_similarity(
        tx.unsqueeze(0), iii.unsqueeze(1), -1
    ).view(-1, lats.normu.shape[0]).T.mean(1)
    if inverse:
        loss = -loss  # Invert loss so optimizer pushes embeddings apart (i.e. minimizes similarity)
    return loss, tx, lll

# Modified Function for text-text optimization (minimizing similarity)
def ascend_txt_inverse(image, model, lats, many_tokens, prompt, nom, augment, best_text_embeddings, inverse=False):
    assert best_text_embeddings is not None, "best_text_embeddings must be provided for inverse training."
    # Instead of computing image embeddings, we use the best text embeddings from step 1.
    iii = best_text_embeddings.detach()  # Detach to prevent gradients flowing into the saved embedding
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    loss = -100 * torch.cosine_similarity(
        tx.unsqueeze(0), iii.unsqueeze(1), -1
    ).view(-1, lats.normu.shape[0]).T.mean(1)
    if inverse: # Invert loss so optimizer pushes embeddings apart (i.e. minimizes similarity)
        loss = -loss # It's still gonna oscillate though, due to the nature of the task
    return loss, tx, lll

# Standard training for best embeddings remains unchanged.
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment, inverse=False):
    with autocast("cuda"):
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment, inverse=inverse)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll

# Modified training function for inverse optimization
def train_inverse(image, model, lats, many_tokens, prompt, optimizer, nom, augment, inverse=False, best_text_embeddings=None):
    with autocast("cuda"):
        loss1, tx, lll = ascend_txt_inverse(image, model, lats, many_tokens, prompt, nom, augment, best_text_embeddings, inverse=inverse)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll

def initialize_training_objects(args):
    """Initializes tokenizer, latent variables, augmentations, optimizer, and scheduler."""
    tok = clip.simple_tokenizer.SimpleTokenizer()
    nom = Normalization(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    ).cuda()
    bests = {i: 'None' for i in range(1000, 1006)}
    prompt = clip.tokenize('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i not in (0, 49406, 49407)]

    lats = Pars(args.ga_batch_size, 4, prompt).cuda()
    augs = torch.nn.Sequential(kaugs.RandomAffine(degrees=10, translate=0.1, p=0.8).cuda()).cuda()

    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.8)

    return tok, nom, bests, prompt, lats, augs, optimizer, scheduler

def train_text_embeddings(img, model, lats, many_tokens, prompt, optimizer, scheduler, nom, augment, training_iterations, checkin_step, tok, bests, img_name, inverse=False, best_text_embeddings=None):

    if not inverse:
        best_loss = float('inf')
    else:
        best_loss = float('-inf')

    for j in range(training_iterations):
        if not inverse:
            loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment, inverse=inverse)
        else:
            loss, tx, lll = train_inverse(img, model, lats, many_tokens, prompt, optimizer, nom, augment, inverse=inverse, best_text_embeddings=best_text_embeddings)

        current_loss = loss.mean().item()

        if (not inverse and current_loss < best_loss) or (inverse and current_loss > best_loss):
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New top loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)

        scheduler.step()

        if j % checkin_step == 0:
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
            trusted_worst_text_embeddings = copy.deepcopy(tx.detach())

    os.makedirs("txtembeds", exist_ok=True)
    if not inverse:
        torch.save(best_text_embeddings, f"txtembeds/{img_name}_text_embedding.pt")
        print(Fore.MAGENTA + Style.BRIGHT + "Best text embedding saved to 'txtembeds'.\n" + Fore.RESET)
        return best_text_embeddings
    else:
        torch.save(trusted_worst_text_embeddings, f"txtembeds/{img_name}_text_embedding_inverse.pt")
        print(Fore.MAGENTA + Style.BRIGHT + "Worst text embedding saved to 'txtembeds'.\n" + Fore.RESET)
        return trusted_worst_text_embeddings


def generate_target_text_embeddings(img_path, model, args, training_iterations=340, checkin_step=10, many_tokens=4):
    """Generates both best and worst text embeddings."""
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_ga_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

    tok, nom, bests, prompt, lats, augs, optimizer, scheduler = initialize_training_objects(args)

    # Generate best embeddings (cosine similarity maximization: image-text)
    best_text_embeddings = train_text_embeddings(
        img, model, lats, many_tokens, prompt, optimizer, scheduler, nom, augs, 
        training_iterations=training_iterations, checkin_step=checkin_step, 
        tok=tok, bests=bests, img_name=img_name, inverse=False, best_text_embeddings=None
    )

    del optimizer, scheduler, lats, augs, bests, prompt
    torch.cuda.empty_cache()

    # Reinitialize for worst embeddings (cosine similarity minimization: text-text)
    tok, nom, bests, prompt, lats, augs, optimizer, scheduler = initialize_training_objects(args)

    worst_text_embeddings = train_text_embeddings(
        img, model, lats, many_tokens, prompt, optimizer, scheduler, nom, augs, 
        training_iterations=training_iterations, checkin_step=checkin_step,
        tok=tok, bests=bests, img_name=img_name, inverse=True, best_text_embeddings=best_text_embeddings
    )

    if args.make_lossplots:
        tok, nom, bests, prompt, lats, augs, optimizer, scheduler = initialize_training_objects(args)
        # Store model's original dtypes
        original_dtypes = save_model_dtypes(model)
        # Convert model to full precision
        convert_model_to_full_precision(model)
        visualize_loss_landscape(img, model, lats, many_tokens, prompt, nom, augs, best_text_embeddings, worst_text_embeddings)
        # Restore original dtypes
        restore_model_dtypes(model, original_dtypes)

    del optimizer, scheduler, lats, augs, bests, prompt
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    return img, best_text_embeddings, worst_text_embeddings, img_path

def generate_anti_text_embeddings(target_texts_and_values, model, preprocess, tokenizer, args, training_iterations=340, checkin_step=10, many_tokens=4):
    input_dims, num_layers, num_features = get_clip_vit_dimensions(model, preprocess)
    img_name = image_name
    # Generate a dummy image tensor so I can avoid editing code designed to work with an image (here: not used, just passed around)
    img = torch.tensor(torch.randn(1, 3, input_dims, input_dims)).cuda().permute(0, 3, 1, 2) / 255
    img = F.interpolate(img, (input_dims, input_dims))

    if target_texts_and_values:
        target_values = [val for (val, _) in target_texts_and_values]

        if isinstance(target_texts_and_values[0][1], str):
            target_texts = [txt for (_, txt) in target_texts_and_values]
            text_features_list = get_many_text_features(model, tokenizer, target_texts).detach().to("cuda")
        else:
            print("That didn't work.")
            return

    # Ensure best_text_embeddings is a tensor with batch size args.ga_batch_size
    batch_size = ga_batch_size
    num_texts = text_features_list.shape[0]  # Original number of encoded texts

    if num_texts < batch_size:
        repeat_factor = (batch_size // num_texts) + 1
        best_text_embeddings = text_features_list.repeat((repeat_factor, 1))[:batch_size]
    else:
        best_text_embeddings = text_features_list[:batch_size]

    best_text_embeddings = best_text_embeddings.to("cuda")
    os.makedirs("txtembeds", exist_ok=True)
    torch.save(best_text_embeddings, f"txtembeds/{img_name}_text_embedding.pt")
    tok, nom, bests, prompt, lats, augs, optimizer, scheduler = initialize_training_objects(args)

    worst_text_embeddings = train_text_embeddings(
        img, model, lats, many_tokens, prompt, optimizer, scheduler, nom, augs, 
        training_iterations=training_iterations, checkin_step=checkin_step,
        tok=tok, bests=bests, img_name=img_name, inverse=True, best_text_embeddings=best_text_embeddings
    )

    del optimizer, scheduler, lats, augs, bests, prompt
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    return img, best_text_embeddings, worst_text_embeddings, "generated_image"



# ---------------------------
#  Direct Ascent Synthesis
# ---------------------------

def generate_image(
    models_and_tokenizers,
    target_texts_and_values,
    starting_image=None,
    guiding_images_tensor=None,
    original_resolution=224,
    large_resolution=224+2*56,  # adding the buffer on the side
    resolutions=range(1, 336+1),
    batch_size=batch_size,
    lr = 2e-1,
    steps=steps,
    jitter_scale=56,  # x-y pixel jitter
    noise_scale=0.2,
    augmentation_copies=augmentation_copies,# how many augmentations to get a gradient from at once
    gradient_sign_only=False,  # to get only the gradient signs, needs 1e4 smaller LR or so
    attack_size_factor=None,  # for limiting the attack size
    step_to_show=10,  # not implemented in cli
    multiple_generations_at_once=multi_many,
    separate_image_weights=None,
    inpainting_mask=None,
    greyscale_only=False,
):
    global all_image_perturbations
    global collected_images
    target_texts=None

    gradient_sign_only=True
    lr = 1e-4
    steps=steps*4
    print(Fore.RED + Style.BRIGHT + f"Setting gradient_sign_only=True for ViT-L.\nSetting LR={lr}. Setting steps*4={steps}." + Fore.RESET)

    # TODO / temp FIX: If a starting image is provided, override large_resolution with its resolution.
    if starting_image is not None:
        provided_res = starting_image.shape[2]  # shape (N, 3, H, W)
        if provided_res != large_resolution:
            print(f"Overriding large_resolution ({large_resolution}) with starting_image resolution ({provided_res})")
            large_resolution = provided_res

    # --- Prepare target text features and target values ---
    if target_texts_and_values:
        target_values = [val for (val, _) in target_texts_and_values]
        if isinstance(target_texts_and_values[0][1], str):
            target_texts = [txt for (_, txt) in target_texts_and_values]
            text_features_list = [
                get_many_text_features(model, tokenizer, target_texts).detach().to("cuda")
                for (model, tokenizer, _, _) in models_and_tokenizers
            ]
        else:
            text_features_list = [
                torch.stack([txt for (_, txt) in target_texts_and_values], dim=0).to("cuda")
                for _ in models_and_tokenizers
            ]
    else:
        target_values = []
        text_features_list = None

    resolutions = sorted(set(resolutions))

    normalize_fns_list = [
        lambda x, mean=mean, std=std: normalize_for_clip(x, mean, std)
        for (_, _, mean, std) in models_and_tokenizers
    ]

    # --- Process guiding images if provided ---
    if guiding_images_tensor is not None:
        # Compute guiding image features for each model once.
        guiding_images_features_list = [
            get_many_image_features(model, normalize_fns_list[j](guiding_images_tensor.to("cuda")))
            .detach()
            .to("cuda")
            for j, (model, tokenizer, _, _) in enumerate(models_and_tokenizers)
        ]
        if text_features_list is not None:
            text_features_list = [
                torch.cat([text_features_list[i], guiding_images_features_list[i]], dim=0)
                for i in range(len(text_features_list))
            ]
        else:
            text_features_list = guiding_images_features_list

        if separate_image_weights is None:
            target_values = target_values + ([1.0] * guiding_images_tensor.shape[0])
        else:
            target_values = target_values + separate_image_weights


        if text_features_list is not None:
            text_features_list = [
                torch.cat([text_features_list[i], guiding_images_features_list[i]], dim=0)
                for i in range(len(text_features_list))
            ]
            if len(target_values) > 0:
                target_values += [1.0] * guiding_images_tensor.shape[0]
        else:
            text_features_list = guiding_images_features_list
            target_values = [1.0] * guiding_images_tensor.shape[0]

    image_count = augmentation_copies
    batch_count = int(np.ceil(image_count / batch_size))


    # --- Prepare starting image ---
    if starting_image is None:
        np_image_now = np.ones((multiple_generations_at_once, 3, large_resolution, large_resolution)) * 0.5
    else:
        # Use the provided starting image as is and update the batch size.
        np_image_now = starting_image  
        multiple_generations_at_once = np_image_now.shape[0]

    torch_image_raw = real_to_raw_image(torch.Tensor(np_image_now).to("cuda"))
    original_image = torch.Tensor(np_image_now).to("cuda")

    very_start_image = np.array(np_image_now)  # legacy variable

    # Initialize total_perturbation as a tensor instead of a float
    total_perturbation = torch.zeros_like(torch_image_raw).to("cuda")

    all_image_perturbations = [torch.zeros((multiple_generations_at_once, 3, res, res), device="cuda", requires_grad=True) for res in resolutions]

    optimizer = SGD(all_image_perturbations, lr=lr)
    scheduler = LambdaLR(optimizer, lambda step: 1.0)

    images_to_start_raw = torch_image_raw.to("cuda")

    collected_images = []
    tqdm_range = tqdm.tqdm(range(steps))
    for step in tqdm_range:
        # Reinitialize total_perturbation as a tensor each iteration
        total_perturbation = torch.zeros_like(images_to_start_raw).to("cuda")

        for i, p in enumerate(all_image_perturbations):
            upscaled = F.interpolate(p, size=(large_resolution, large_resolution), mode='bicubic' if resolutions[i] > 1 else 'nearest')
            total_perturbation = total_perturbation + upscaled

        image_perturbation = total_perturbation

        if greyscale_only:
            image_perturbation = torch.mean(image_perturbation, axis=1, keepdims=True)

        if inpainting_mask is not None:
            image_perturbation.register_hook(lambda grad: grad * inpainting_mask)

        if attack_size_factor is None:
            collected_images.append(raw_to_real_image((images_to_start_raw + image_perturbation)).detach().cpu().numpy())
        else:
            collected_images.append(np.clip((original_image + attack_size_factor*(raw_to_real_image(images_to_start_raw + image_perturbation) - original_image)).detach().cpu().numpy(),0,1))

        # --- Compute loss over augmentation batches ---
        losses_split = []
        losses = []
        for it in range(batch_count):
            i1 = it * batch_size
            i2 = min((it + 1) * batch_size, image_count)

            for i_model,(model,tokenizer,mean,std) in enumerate(models_and_tokenizers):
                model.eval()

                if attack_size_factor is None:
                    image_to_aug = raw_to_real_image(images_to_start_raw + image_perturbation)
                else:
                    image_to_aug = original_image.to("cuda") + attack_size_factor * (raw_to_real_image(images_to_start_raw + image_perturbation) - original_image.to("cuda"))

                aug_variations = make_image_augmentations(
                    image_to_aug.to("cuda"),
                    count=(i2 - i1),
                    jitter_scale=jitter_scale,
                    noise_scale=noise_scale
                )

                with autocast("cuda"):
                    loss = -1.0 * loss_between_images_and_text(
                        model,
                        normalize_fns_list[i_model](torch.clip(aug_variations[: i2 - i1], 0, 1)),
                        text_features_list[i_model],
                        target_values=target_values
                    )
                loss = torch.mean(loss) * multiple_generations_at_once
                losses.append(loss.item())
                loss.backward(retain_graph=True)

        # Fixes gradients going DEFCON1 in large models / ViT-L
        if gradient_sign_only:
            for p in all_image_perturbations:
                p.grad.data.sign_()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            ell_infty = np.max(np.abs(collected_images[-1][0] - collected_images[0][0])) * 255
            tqdm_range.set_description(f"Step {step}, Loss: {np.mean(losses):.3f}, ell_inf: {ell_infty:.1f}/255")

    return collected_images, all_image_perturbations

# -----
# Main
# -----

def main():
    args = parser.parse_args()
    args.model_name = args.use_model
    save_path=None
    date_str = datetime.datetime.now().strftime("%m-%d-%Y")
    base_dir = os.path.join(args.results_dir, date_str)
    os.makedirs(base_dir, exist_ok=True)
    next_number = max([int(folder) for folder in os.listdir(base_dir) if folder.isdigit()] or [-1]) + 1
    results_dir = os.path.join(base_dir, str(next_number))
    os.makedirs(results_dir, exist_ok=True)
    next_num = next_number  # For naming files
    if args.use_image or args.use_neuron or args.all_neurons:
        assert args.model_name[0].startswith("OpenAI"), "Main --model_name must be 'OpenAI-*' for --use_image gradient ascent and --use_neuron. For example: --model_name 'OpenAI-ViT-B/32'"

    # Get models to load
    models_to_load = []
    if len(args.model_name) == 1:
        args.model_name.append(None)  # Ensure there's always a second element
    models_to_load.append(tuple(args.model_name))  # default is 'OpenAI-ViT-B/32'

    if args.custom_model2 is not None:
        if len(args.custom_model2) == 1:
            args.custom_model2.append(None)
        models_to_load.append(tuple(args.custom_model2))

    if args.custom_model3 is not None:
        if len(args.custom_model3) == 1:
            args.custom_model3.append(None) 
        models_to_load.append(tuple(args.custom_model3))

    print(f"\nFinal models_to_load: {models_to_load}")

    # Ensure all models are of the same type
    model_types = {get_model_type(model[0]) for model in models_to_load}
    print(Fore.GREEN + Style.BRIGHT + f"Model types: {model_types}" + Fore.RESET)
    if len(model_types) > 1:
        print(f"ERROR: Incompatible model types detected: {model_types}.")
        print("All models must be of the same base architecture to avoid dimension mismatches.")
        sys.exit(1)

    # Load models
    models_and_tokenizers = load_clip_models(models_to_load)

    # For gradient ascent, visualization: Use first model (must be OpenAI).
    default_model = models_and_tokenizers[0][0]

    # Use CLIP opinion (gradient ascent) about image instead of text prompt
    if args.use_image is not None:
        img, best_text_embeddings, worst_text_embeddings, img_path = generate_target_text_embeddings(
            args.use_image, default_model, args, training_iterations=340, checkin_step=10, many_tokens=4
        )
        print(f"Done processing image: {img_path}")

        img_name = os.path.splitext(os.path.basename(args.use_image))[0]
        best_embed = torch.load(f"txtembeds/{img_name}_text_embedding.pt")
        worst_embed = torch.load(f"txtembeds/{img_name}_text_embedding_inverse.pt")

        primary_image = args.use_image

        target_texts_and_values = [
            (1.0, best_embed[0]),
            (-0.3, worst_embed[0]),
            (0.3, best_embed[1]),
            (-0.3, worst_embed[1]),
        ]
    else:
        primary_image = args.img0

        target_texts_and_values = [
            (1.0, args.txt1),
            (-0.3, args.negtxt1),
            (0.3, args.txt2),
            (-0.3, args.negtxt2),
        ]

    # Get images
    image_paths = [primary_image, args.img1, args.img2, args.img3, args.img4, save_path]    
    # Sanitize the ones that are None:
    image_paths = list(filter(None, [primary_image, args.img1, args.img2, save_path]))

    # Obtain maximum activating neurons (feature activation visualization) images and append
    if args.use_neuron or args.all_neurons:
        if isinstance(primary_image, str):
            primary_neuron_image = copy.deepcopy(preprocess(Image.open(primary_image)).unsqueeze(0).to(device))

        if not args.all_neurons:
            neuron_image, save_path = get_clip_feature(default_model, preprocess, primary_neuron_image, args.vit_l_neuron, args.vit_neuron)
            image_paths.append(save_path)

        if args.all_neurons:
            input_dims, num_layers, num_features = get_clip_vit_dimensions(default_model, preprocess)

            total_layers = num_layers - 1
            save_paths = {}

            for i in range(2, total_layers + 1):  # Start from 2nd layer (0 = input, 1 = second) -> up to final layer
                if "L/14" in args.model_name[0] or "L-14" in args.model_name[0]:
                    vit_l_neuron = i
                    vit_neuron = None
                else:
                    vit_neuron = i
                    vit_l_neuron = None

                neuron_image, save_paths[i] = get_clip_feature(default_model, preprocess, primary_neuron_image, vit_l_neuron, vit_neuron)

            image_paths.extend(save_paths.values())

    # Load images
    loader = load_images(image_paths)
    start_images_batch = next(iter(loader))


    # ------------
    # MAIN TASKS
    # ------------

    if not args.no1:
        # --- Task 1: Text-based Image Generation ---
        print(Fore.MAGENTA + f"\n----------------------------------------------")
        print("-- " + Fore.MAGENTA + Style.BRIGHT + "Task 1: Text-based Image Generation" + " --")
        print(Fore.MAGENTA + f"----------------------------------------------\n" + Fore.RESET)

        task_type = "textbased"
        collected_images, perturbations = generate_image(models_and_tokenizers, target_texts_and_values)
        save_images(collected_images, results_dir, next_num, task_type)

        if args.make_plots:
            slope, intercept, r_value = analyze_perturbations(perturbations)
            visualize_individual_resolutions(all_image_perturbations)

        # INTERMISSION - make an ANTI-TEXT about the text prompt (or image) (min cos similarity)
        if args.make_anti:
            if args.use_image is not None:

                img_name = os.path.splitext(os.path.basename(args.use_image))[0]
                best_embed_anti = torch.load(f"txtembeds/{img_name}_text_embedding.pt")
                worst_embed_anti = torch.load(f"txtembeds/{img_name}_text_embedding_inverse.pt")

                target_texts_and_values = [
                    (1.0, worst_embed_anti[0]),
                    (-0.3, best_embed_anti[0]),
                    (0.3, worst_embed_anti[1]),
                    (-0.3, best_embed_anti[1]),
                ]
            else:
                target_texts_and_values = [
                    (1.0, args.txt1),
                    (-0.3, args.negtxt1),
                    (0.3, args.txt2),
                    (-0.3, args.negtxt2),
                ]
                img, best_text_embeddings, worst_text_embeddings, img_path = generate_anti_text_embeddings(
                    target_texts_and_values, default_model, preprocess, tokenizer, args, training_iterations=340, checkin_step=10, many_tokens=4
                )
                img_name = image_name
                best_embed_anti = torch.load(f"txtembeds/{img_name}_text_embedding.pt")
                worst_embed_anti = torch.load(f"txtembeds/{img_name}_text_embedding_inverse.pt")

                target_texts_and_values = [
                    (1.0, worst_embed_anti[0]),
                    (-0.3, best_embed_anti[0]),
                    (0.3, worst_embed_anti[1]),
                    (-0.3, best_embed_anti[1]),
                ]

            # --- Task 1b: Anti-Text-based Image Generation ---
            print(Fore.MAGENTA + f"\n----------------------------------------------")
            print("-- " + Fore.MAGENTA + Style.BRIGHT + "Task 1b: Text-based Image (Anti)" + " --")
            print(Fore.MAGENTA + f"----------------------------------------------\n" + Fore.RESET)

            task_type = "textbased_anti"
            collected_images, perturbations = generate_image(models_and_tokenizers, target_texts_and_values)
            save_images(collected_images, results_dir, next_num, task_type)

            # Reset to the NON-ANTI / normal texts
            if args.use_image is not None:
                img_name = os.path.splitext(os.path.basename(args.use_image))[0]
                best_embed = torch.load(f"txtembeds/{img_name}_text_embedding.pt")
                worst_embed = torch.load(f"txtembeds/{img_name}_text_embedding_inverse.pt")

                target_texts_and_values = [
                    (1.0, best_embed[0]),
                    (-0.3, worst_embed[0]),
                    (0.3, best_embed[1]),
                    (-0.3, worst_embed[1]),
                ]
            else:
                target_texts_and_values = [
                    (1.0, args.txt1),
                    (-0.3, args.negtxt1),
                    (0.3, args.txt2),
                    (-0.3, args.negtxt2),
                ]

        # END INTERMISSION --------------

    if not args.no2:
        # --- Task 2: Generation Stability (Multiple) ---
        print(Fore.MAGENTA + f"\n----------------------------------------------")
        print("-- " + Fore.MAGENTA + Style.BRIGHT + "Task 2: Generation Stability (Multiple)" + " --")
        print(Fore.MAGENTA + f"----------------------------------------------\n" + Fore.RESET)

        task_type = "genstab"
        print(f"Generating {multi_many} images.")
        stability_images, _ = generate_image(models_and_tokenizers, target_texts_and_values, multiple_generations_at_once=multi_many)
        save_images(stability_images, results_dir, next_num, task_type)

    if not args.no3:
        # --- Task 3: Style Transfer ---
        print(Fore.MAGENTA + f"\n----------------------------------------------")
        print("-- " + Fore.MAGENTA + Style.BRIGHT + "Task 3: Style Transfer (Image-Image)" + " --")
        print(Fore.MAGENTA + f"----------------------------------------------\n" + Fore.RESET)

        task_type = "styletrans"
        large_resolution = 224
        eps = 0.1
        start_images_batch = start_images_batch * (1 - 2 * eps) + eps
        if start_images_batch.dim() == 3:
            start_images_batch = start_images_batch.unsqueeze(0)

        pad_h = (large_resolution - start_images_batch.shape[2]) // 2
        pad_w = (large_resolution - start_images_batch.shape[3]) // 2
        start_images_batch = F.pad(start_images_batch, (pad_w, pad_w, pad_h, pad_h), "constant", 0.5)

        guiding_images_batch = start_images_batch.clone()
        style_images, _ = generate_image(
            models_and_tokenizers,
            target_texts_and_values=[],
            guiding_images_tensor=guiding_images_batch,
            starting_image=start_images_batch.detach().cpu().numpy()
        )
        try:
            save_images(
                style_images,
                results_dir,
                next_num,
                task_type,
                guiding_images_batch=guiding_images_batch,
                start_images_batch=start_images_batch,
                show_guiding_image=True,
                show_starting_image=True
            )
        except ValueError:
            pass  # TODO: fix this if needed

    if not args.no4:
        # --- Task 4: Reconstruction ---
        print(Fore.MAGENTA + f"\n----------------------------------------------")
        print("-- " + Fore.MAGENTA + Style.BRIGHT + "Task 4: Reconstruction From Image" + " --")
        print(Fore.MAGENTA + f"----------------------------------------------\n" + Fore.RESET)

        task_type = "reconstr"
        recon_texts_and_values = [(0.6, "cohesive single subject"), (-0.6, "multiple exposure")]
        guiding_images_batch = start_images_batch.clone()
        recon_images, _ = generate_image(
            models_and_tokenizers,
            target_texts_and_values=recon_texts_and_values,
            guiding_images_tensor=guiding_images_batch
        )
        save_images(recon_images, results_dir, next_num, task_type, guiding_images_batch=guiding_images_batch, show_guiding_image=True)

if __name__ == '__main__':
    main()
