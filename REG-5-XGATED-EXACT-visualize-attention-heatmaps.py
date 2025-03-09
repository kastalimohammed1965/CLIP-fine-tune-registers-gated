import os
import sys
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from safetensors.torch import load_file
import argparse

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import ATTNclipXGATED as clip
from ATTNclipXGATED.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

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

print('Creating CLIP heatmaps...')

# Control context expansion
# Number of (skip) layers for image Transformer (from the back, i.e. -1 = final layer, -2 = penultimate, and so on)
start_layer =  -1 # Try setting this to -2, -3, -4, -5
start_layer_text =  -1 # Try setting this to -2 -- it's SDXL-style penultimate layer use! :)

image_folder = args.image_folder
token_folder = args.token_folder
heatmap_folder = "plots/REG-XGATED/attn-heatmap-exact"
os.makedirs(heatmap_folder, exist_ok=True)
overlay_token_in_image=True # set to false to get the heatmaps without the text token info overlay
font_size = 50

"""
Uses torch.nn.functional.interpolate with mode='nearest-exact'.
Much slower. Large image dimensions. See also: pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
"""

# ---------- MAIN CODE ----------

"""
Attention visualization code below, original author:
github.com/hila-chefer/Transformer-MM-Explainability
"""

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)

    # -------------------------------- SLICE THE CLS and REG TOKENS! --------------------------------
    #image_relevance = R[:, 0, 1:] # Original CLIP
    image_relevance = R[:, 0, 5:]  # Skip CLS and REG tokens, keep only patches
    # -------------------------------- SLICE THE CLS and REG TOKENS! --------------------------------

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    if start_layer_text == -1:
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

def show_heatmap_on_text(text, text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text_slice = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text_slice / R_text_slice.sum()
    text_scores = text_scores.flatten()
    text_tokens = _tokenizer.encode(text)
    text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
    # Visualization for text relevance is not displayed here.

# --- NEW FUNCTION: overlay_heatmap_on_image ---
def overlay_heatmap_on_image(image_relevance, orig_image):
    """
    Resizes the heatmap to match the original image dimensions and overlays it using plt.imshow.
    """
    # Convert the original image to an RGB numpy array
    orig_rgb = np.array(orig_image.convert('RGB'))
    width, height = orig_image.size  # PIL returns (width, height)

    # Process image_relevance: reshape and interpolate to match original image size
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=(height, width), mode='nearest-exact')
    image_relevance = image_relevance.reshape(height, width).cpu().numpy()
    # Normalize the heatmap
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min() + 1e-8)
    # Convert to 8-bit
    heatmap_uint8 = np.uint8(255 * image_relevance)

    # Create overlay using plt.imshow with the jet_r colormap
    plt.figure(figsize=(8, 8))
    plt.imshow(orig_rgb)
    plt.imshow(heatmap_uint8, cmap="jet", alpha=0.5)
    plt.axis('off')
    plt.tight_layout(pad=0)

    fig = plt.gcf()
    return fig

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Convert images to PNG format
all_files = os.listdir(image_folder)
for file in all_files:
    file_path = os.path.join(image_folder, file)
    try:
        img = Image.open(file_path)
        img = img.convert('RGBA')  # Convert to RGBA for PNG
        new_file_path = os.path.join(image_folder, os.path.splitext(file)[0] + '.png')
        img.save(new_file_path, "PNG")
        if not file_path.endswith('.png'):
            os.remove(file_path)
    except IOError:
        print(f"{file} is not a valid image.")

# List of all image files in the image_folder
image_files = glob.glob(f"{image_folder}/*.png")

for img_file in image_files:
    img_name = os.path.basename(os.path.splitext(img_file)[0])
    token_file = f"{token_folder}/tokens_{img_name}.txt"
    with open(token_file, 'r') as f:
        tokens = f.read().split()

    img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
    print(f"Processing {img_file} tokens...")

    for token in tokens:
        texts = [token]
        text = clip.tokenize(texts).to(device)

        R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
        batch_size = text.shape[0]
        for i in range(batch_size):
            show_heatmap_on_text(texts[i], text[i], R_text[i])
            # Generate the overlay plot using the new function
            fig = overlay_heatmap_on_image(R_image[i], orig_image=Image.open(img_file))
            heatmap_filename = f"{heatmap_folder}/{img_name}_{token}_ViT{start_layer}_TxT{start_layer_text}.png"
            fig.savefig(heatmap_filename, bbox_inches='tight', pad_inches=0)
            plt.close(fig)


if overlay_token_in_image:
    # Specify the directory that contains the heatmap images
    directory = heatmap_folder

    # Get a list of all image files in the directory
    image_files = glob.glob(os.path.join(directory, '*.png'))

    def get_font_in_order(font_names, font_size):
        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, font_size)
            except IOError:
                continue
        raise ValueError(f"None of the fonts {font_names} are available.")

    def extract_text(filename):
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        parts = name_without_ext.split('_', 1)
        if len(parts) > 1:
            return parts[1]
        return name_without_ext


    # Please edit this if it's an AI-hallucination and your font is in some other place. :P
    primary_font_names = [
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows
        "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS/iOS
        "/System/Library/Fonts/Core/AppleColorEmoji@2x.ttc",  # macOS/iOS
        "/System/Library/Fonts/Core/AppleColorEmoji-160px.ttc",  # macOS/iOS
        "/usr/share/fonts/NotoColorEmoji.ttf",  # Linux/Android
        "arialn.ttf",
        "DejaVuSansCondensed.ttf",
        "segoeui.ttf",
        "NotoSans-Regular.ttf",
        "symbola.ttf",
        "arial.ttf"
    ]
    fallback_font_names = [
        "NotoColorEmoji.ttf",  # Common fallback for emojis
        "Apple Color Emoji.ttc",  # Another fallback for macOS/iOS
        "symbola.ttf"  # Fallback for symbols
    ]


    primary_font = get_font_in_order(primary_font_names, font_size)


    def draw_text_with_fallback(draw, text, position, font, fallback_fonts, fill='white'):
        try:
            draw.text(position, text, font=font, fill=fill)
        except UnicodeEncodeError:
            fallback_font = get_font_in_order(fallback_fonts, font.size)
            draw.text(position, text, font=fallback_font, fill=fill)

    for image_file in image_files:
        img = Image.open(image_file).convert('RGBA')
        draw = ImageDraw.Draw(img)

        # Extract the relevant part of the filename
        text_to_write = extract_text(image_file)

        # Write the extracted text into the image with fallback support for emojis
        draw_text_with_fallback(draw, text_to_write, (10, 10), primary_font, fallback_font_names, 'white')

        # Save the image, overwriting the original file
        img.save(image_file)

    print('Done writing filename overlay into images.')
