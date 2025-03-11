import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

from sklearn.metrics import f1_score, accuracy_score
from adabelief_pytorch import AdaBelief

from PIL import Image
import matplotlib.pyplot as plt
from colorama import Fore, Style
from tqdm import tqdm

# Import CLIP with GmP, REG, GATES. See 'model.py' for details:
import longTRAINgmpCLIPregXGATED as clip


# Suppress warnings spam from torch, especially
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

training_losses = []
validation_losses = []
print("\n")

# Save training plots with matplotlib to:
plots_folder = 'CLIPneedsREGISTERS/longREG-XGATED/ft-plots'
os.makedirs(plots_folder, exist_ok=True)

# Save model .pt files to: 
ft_checkpoints_folder = 'CLIPneedsREGISTERS/longREG-XGATED/ft-checkpoints'
os.makedirs(ft_checkpoints_folder, exist_ok=True)

# Save verbose text / training logs to:
text_logs_folder = 'CLIPneedsREGISTERS/longREG-XGATED/ft-logs'
os.makedirs(text_logs_folder, exist_ok=True)

# If not unfreeze_all=True
def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    if epoch < adjust_after:
        return 1  # Initial slower unfreeze rate
    else:
        return increase_rate  # Increased rate after initial pass

def unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=False):
    if unfreeze_all:
        print(Fore.GREEN + "All params require gradient\n" + Style.RESET_ALL)
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
        layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= total_layers - layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False


def monitor_gradient_norms(gradient_norms, threshold=1e-5):
    alert_messages = []
    for name, norms in gradient_norms.items():
        mean_norm = sum(norms) / len(norms)
        # Check for vanishing or exploding gradients.
        if mean_norm < threshold:
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 1000:
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        # Always display for any fusion MLP parameters (final or intermediate)
        if "fusion_mlp" in name or "intermediate_fusion_mlps" in name:
            alert_messages.append(Fore.GREEN + f"Fusion MLP params: {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        # And always display for register tokens
        if "visual.register_tokens" in name:
            alert_messages.append(Fore.GREEN + f"Register tokens: {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    for message in alert_messages:
        print(message)


def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    plt.figure(figsize=(20, 10))
    
    cmap = plt.get_cmap('Spectral')   
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    colors = cmap(range(len(sorted_layers)))
    
    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.legend(loc='upper right', fontsize='small')
    
    if use_log_scale:
        plt.yscale('log')
        plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}_log.png")
    else:
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    
    plt.close()

def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epochs_x):
        plt.plot(epochs_x, training_losses, label='Training Loss')
    if len(validation_losses) == len(epochs_x):
        plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    if len(logits_images) == len(epochs_x):
        plt.plot(epochs_x, logits_images, label='Average Logits')
    if len(logits_texts) == len(epochs_x):
        plt.plot(epochs_x, logits_texts, label='Average Logits')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        if len(labels) >= 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to the first label if less than 2 are available
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label])  # Tokenize the label

        return image, text.squeeze(0)  # Remove the extra dimension

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, smoothing=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Apply label smoothing
        N = logits.size(0)
        smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate loss manually using log-softmax and smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs).sum(dim=1).mean()

        return (loss_img + loss_txt) / 2

       
# Custom hook to scale the feature activation
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_indices, scale_factor):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_indices = feature_indices
        self.scale_factor = scale_factor
        self.handle = None
        self.register_hook()

    def hook_fn(self, module, input, output):
        for feature_idx in self.feature_indices:
            output[:, :, feature_idx] *= self.scale_factor
        return output

    def register_hook(self):
        layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

def register_hooks(model, adverb_neurons_layers, scale_factors):
    hooks = []
    for layer_idx, feature_indices in adverb_neurons_layers.items():
        scale_factor = scale_factors[layer_idx]
        hook = FeatureScalerHook(model, layer_idx, feature_indices, scale_factor)
        hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
        
        
# Scale up activation value of 'adverb neurons'
# Long story... See here: github.com/zer0int/CLIP-fine-tune
adverb_neurons_layers = {
    23: [281],
    20: [168, 1297],
    22: [2432]
}
scale_factors = {
    23: 100,
    20: 100,
    22: 1000
}

contrastive_loss = ContrastiveLoss(temperature=0.07)

# Model loading
clipmodel = "models/LongCLIP-L.safetensors"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clipmodel, device=device)

# Settings
unfreeze_all = True
EPOCHS = 20
max_learning_rate = 5e-6
learning_rate = 6e-7
batch_size = 36

# Training dataset and dataloader: COCO-SPRIGHT, cropped to square. Get it here: huggingface.co/datasets/SPRIGHT-T2I/spright_coco
# Note! .json assumes dataset to be in subfolder 'data', e.g. "data/9/0.jpg" - make sure move & load the .json there!
#dataset1 = ImageTextDataset("path/to/SPRIGHT-T2I/data-square", "path/to/SPRIGHT-T2I/data-square/COCO-SPRIGHT-short-train-0_9.json", transform=preprocess)
#concatenated_dataset = ConcatDataset([dataset1]) 
#train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset. Note: .json assumes dataset to be in subfolder 'data', e.g. "data/10/0.jpg" - make sure move & load the .json there!
#val_dataset1 = ImageTextDataset("path/to/SPRIGHT-T2I/data-square", "path/to/SPRIGHT-T2I/data-square/COCO-SPRIGHT-short-val-10_11.json", transform=preprocess)
#concatenated_valdataset = ConcatDataset([val_dataset1])
#val_dataloader = DataLoader(concatenated_valdataset, batch_size=batch_size, shuffle=False)



# Training dataset and dataloader: COCO-SPRIGHT (see HuggingFace), cropped to square. See my Git zer0int/CLIP-fine-tune for truncated labels .json
dataset1 = ImageTextDataset("F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square", "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-train-0_9.json", transform=preprocess)
dataset2 = ImageTextDataset("E:/AI_Dataset/COCA-SPRIGHT", "E:/AI_Dataset/COCA-SPRIGHT/labels00000-truncated.json", transform=preprocess)
dataset3 = ImageTextDataset("E:/AI_Dataset/COCA-SPRIGHT", "E:/AI_Dataset/COCA-SPRIGHT/labels00001-truncated.json", transform=preprocess)
dataset4 = ImageTextDataset("E:/AI_Dataset/COCA-SPRIGHT", "E:/AI_Dataset/COCA-SPRIGHT/labels00002-truncated-split-train.json", transform=preprocess)

concatenated_dataset = ConcatDataset([dataset1]) 
train_dataloader = DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)

# Validation dataset and dataloader
val_dataset1 = ImageTextDataset("F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square", "F:/AI_DATASET/SPRITE-spatial-labels/COCO/data-square/short-coco-sprite-val-10_11.json", transform=preprocess)
val_dataset2 = ImageTextDataset("E:/AI_Dataset/COCA-SPRIGHT", "E:/AI_Dataset/COCA-SPRIGHT/labels00002-truncated-split-val.json", transform=preprocess)
concatenated_valdataset = ConcatDataset([val_dataset1])
val_dataloader = DataLoader(concatenated_valdataset, batch_size=batch_size, shuffle=False)




total_steps = len(train_dataloader) * EPOCHS

# Define parameter groups
visual_parameters = [p for p in model.visual.transformer.parameters() if p.requires_grad] # This includes new learnable intermediate MLP gates
transformer_parameters = [p for p in model.transformer.parameters() if p.requires_grad]

# Per-parameter learning rates
param_groups = [
    {'params': visual_parameters, 'lr': 6e-7},
    {'params': transformer_parameters, 'lr': 6e-8},
    {'params': model.token_embedding.parameters(), 'lr': 6e-8},
    {'params': [model.positional_embedding], 'lr': 6e-8},
    {'params': [model.visual.positional_embedding, model.visual.class_embedding], 'lr': 3e-7},
    {'params': [model.visual.register_tokens], 'lr': 1e-5}, # New learnable register tokens
    {'params': model.visual.fusion_mlp.parameters(), 'lr': 8e-6},  # New learnable gating mechanism (after the transformer resblocks)
    {'params': [model.visual.proj, model.text_projection], 'lr': 3e-7},
    {'params': [model.visual.ln_pre.weight, model.visual.ln_pre.bias, model.visual.ln_post.weight, model.visual.ln_post.bias], 'lr': 3e-7},
    {'params': [model.ln_final.weight, model.ln_final.bias, model.visual.conv1.weight], 'lr': 3e-7}
]

# Optimizer and Scheduler
optimizer = AdaBelief(param_groups, lr=learning_rate, eps=1e-14, betas=(0.9, 0.999), weight_decay=1e-3, weight_decouple=True, rectify=True, print_change_log=False)

scheduler = OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=total_steps, pct_start=0.2, anneal_strategy='cos')

model = model.float()

print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print("== START == \n")
print(f"\nChecking on the register tokens... If init correctly, there should be tensor:\n{model.visual.register_tokens}")
print(f"\nChecking on the text embeddings shape... If init correctly, there should be 248:\n{model.positional_embedding.shape}\n")

def trainloop():
    contrastive_loss = ContrastiveLoss(temperature=0.07).to(device)
    logits_images = []
    logits_texts = []

    accumulation_steps = 2  # Effective batch size will be batch_size * accumulation_steps
    scaler = GradScaler()
    hooks = register_hooks(model, adverb_neurons_layers, scale_factors) # Register hooks for activation value manipulation
    for epoch in range(EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
        train_dataloader_prog = train_dataloader
        train_dataloader_all = train_dataloader
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)

        optimizer.zero_grad()

        for batch_idx, (images, texts) in progress_bar:
            images, texts = images.to(device), texts.to(device)
            batch_logits_images = []
            batch_logits_texts = []

            with autocast():
                logits_per_image, logits_per_text = model(images, texts)
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                total_loss = contrastive_loss(logits_per_image, logits_per_text)
                acc, f1 = calculate_metrics(logits_per_image, ground_truth)
                train_accs.append(acc)
                train_f1s.append(f1)

            scaler.scale(total_loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            batch_logits_images.append(logits_per_image.mean().item())
            batch_logits_texts.append(logits_per_text.mean().item())

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad_norm = parameter.grad.norm().item()
                    gradient_norms.setdefault(name, []).append(grad_norm)

            monitor_gradient_norms(gradient_norms) # Comment this out if you don't want the red spam of exploding gradients

            total_train_loss += total_loss.item()

            progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}  --  Logits Image: {batch_logits_images[-1]:.3f}, Text: {batch_logits_texts[-1]:.3f}'})

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)

        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)

        plot_gradient_norms(gradient_norms, epoch)

        epoch_train_acc = sum(train_accs) / len(train_accs)
        epoch_train_f1 = sum(train_f1s) / len(train_f1s)
        with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
            f.write(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_train_loss:.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")

        model.eval()
        total_val_loss = 0.0
        print("Running Validation...")
        with torch.no_grad():
            for images, texts in val_dataloader:
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                images, texts = images.to(device), texts.to(device)
                logits_per_image, logits_per_text = model(images, texts)
                val_loss = contrastive_loss(logits_per_image, logits_per_text)
                total_val_loss += val_loss.item()
                val_acc, val_f1 = calculate_metrics(logits_per_image, ground_truth)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

        avg_val_loss = total_val_loss / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        if epoch >= 1:
            plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)

        epoch_val_acc = sum(val_accs) / len(val_accs)
        epoch_val_f1 = sum(val_f1s) / len(val_f1s)

        if epoch >= 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
            plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.savefig(f"{plots_folder}/loss_plot_epoch_{epoch + 1}.png")
            plt.close()

        print(Fore.YELLOW + "======================== STATS =============================")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)

        with open(f"{text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
            f.write("======================== STATS =============================\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            f.write("============================================================\n")

        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            model_path = f"{ft_checkpoints_folder}/longclip_ft_{epoch+1}.pt"
            remove_hooks(hooks)
            torch.save(model, model_path)
            print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)
            hooks = register_hooks(model, adverb_neurons_layers, scale_factors)

    remove_hooks(hooks)

trainloop()