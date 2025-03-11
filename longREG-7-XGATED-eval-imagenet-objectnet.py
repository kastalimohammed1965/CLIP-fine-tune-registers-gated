import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from safetensors.torch import load_file
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import longINFERclipregXGATED as clip
from longINFERclipregXGATED.model import CLIP


# Get this dataset here (no sign-up etc. required): objectnet.dev/mvt/
csv_file = 'E:/AI_DATASET/dataset-difficulty-CLIP/data_release_2023/human_responses.csv'
image_folder = 'E:/AI_DATASET/dataset-difficulty-CLIP/data_release_2023/all/'

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate zero-shot accuracy on ImageNet/ObjectNet')
    parser.add_argument('--use_model', default="models/Long-ViT-L-14-REG-GATED-full-model.safetensors", help="Path to a ViT-L/14 model, pickle (.pt) or .safetensors")
    return parser.parse_args()

args = parse_arguments()
model_name_or_path = args.use_model

model, preprocess = clip.load(model_name_or_path, device=device)

model = model.float() # full precision


def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=BICUBIC, max_size=None, antialias=True),
        transforms.Lambda(_convert_image_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    return transform(image)


class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx]['label']

        return image, label

def evaluate_model(model, dataloader):
    correct = 0
    total = 0

    for batch_images, batch_labels in tqdm(dataloader):
        batch_images = batch_images.to(device)
        batch_texts = clip.tokenize(batch_labels).to(device)

        with torch.no_grad():
            image_embeddings = model.encode_image(batch_images)
            text_embeddings = model.encode_text(batch_texts)
            logits_per_image = (image_embeddings @ text_embeddings.T).softmax(dim=-1)

            _, top_indices = logits_per_image.topk(1, dim=-1)
            
            for i, label in enumerate(batch_labels):
                if label == batch_labels[top_indices[i, 0].item()]:
                    correct += 1
                total += 1
    
    accuracy = correct / total
    return accuracy

dataset = CroppedImageCSVFileDataset(csv_file, image_folder, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)

model_accuracy = evaluate_model(model, dataloader)
print(f"GATED Long-REG-CLIP fine-tune, Accuracy on MVT ImageNet/ObjectNet: {model_accuracy:.4f}")

local_path = "plots/longREG-XGATED/imagenet"
os.makedirs(local_path, exist_ok=True)
with open(f"{local_path}/imagenet-objectnet.txt", "w", encoding='utf-8') as f:
    f.write(f"XGATED Long-REG-CLIP fine-tune, Accuracy on MVT ImageNet/ObjectNet: {model_accuracy:.4f}")

print(f"\nResults saved to {local_path}.")