#!/usr/bin/env python
# Simple DiT training script (Windows compatible)

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import time
import json
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
import random

# Add DiT to path
dit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiT")
sys.path.append(dit_path)

# Import DiT modules
try:
    from models import DiT_models
    from diffusion import create_diffusion
except ImportError:
    print(f"Error: Could not import DiT modules. Make sure DiT is at: {dit_path}")
    sys.exit(1)

# Setup diffusion model and image processing
latent_size = 32
image_size = 256

# Simple dataset class
class SimpleDataset(Dataset):
    def __init__(self, size=10000, image_size=256):
        self.size = size
        self.image_size = image_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random image (only for testing)
        img = torch.randn(3, self.image_size, self.image_size)
        # Normalize to [-1, 1]
        img = torch.clamp(img, -1, 1)
        return img

# DynamicTanh implementation
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.elementwise_affine:
            return self.weight * torch.tanh(self.alpha * x) + self.bias
        else:
            return torch.tanh(self.alpha * x)

# AdaptiveDynamicTanh implementation
class AdaptiveDynamicTanh(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5, 
                 lambda_factor=0.1, smooth_factor=0.99, eps=1e-6, alpha_min=0.1, alpha_max=2.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.lambda_factor = lambda_factor
        self.smooth_factor = smooth_factor
        self.eps = eps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Base alpha parameter (learnable)
        self.alpha_base = nn.Parameter(torch.ones(1) * alpha_init_value)
        
        # Weights and bias (if using elementwise affine)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Register a buffer for tracking the gradient norm (not a learnable parameter)
        self.register_buffer("grad_norm_smoothed", torch.tensor([1.0]))
        
        # Flag to indicate if we're in training mode
        self.adaptive_enabled = True

    def compute_alpha(self):
        if self.adaptive_enabled and self.training:
            # α(t) = α₀ * (1 + λ/(ε + G_t))
            alpha = self.alpha_base * (1 + self.lambda_factor / (self.grad_norm_smoothed + self.eps))
            # Clip alpha to prevent extreme values
            return torch.clamp(alpha, self.alpha_min, self.alpha_max)
        return self.alpha_base
    
    def update_grad_norm(self):
        if not self.training or not self.adaptive_enabled:
            return
            
        # Compute the gradient norm from ONLY this layer's parameters
        total_norm = 0.0
        params = [self.alpha_base]
        if self.elementwise_affine:
            params.extend([self.weight, self.bias])
            
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        
        # Apply exponential smoothing to avoid rapid fluctuations
        self.grad_norm_smoothed = (
            self.smooth_factor * self.grad_norm_smoothed + 
            (1 - self.smooth_factor) * torch.tensor([grad_norm], device=self.alpha_base.device)
        )

    def forward(self, x):
        # Get current adaptive alpha value
        alpha = self.compute_alpha()
        
        # Apply tanh with adaptive alpha
        x = torch.tanh(alpha * x)
        
        # Apply weights and bias if using elementwise affine
        if self.elementwise_affine:
            return self.weight * x + self.bias
        else:
            return x

# Conversion utilities
def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, module.elementwise_affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

def convert_ln_to_adyt(module, lambda_factor=0.1, smooth_factor=0.99, alpha_min=0.1, alpha_max=2.0):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = AdaptiveDynamicTanh(
            module.normalized_shape, 
            module.elementwise_affine,
            lambda_factor=lambda_factor,
            smooth_factor=smooth_factor,
            alpha_min=alpha_min,
            alpha_max=alpha_max
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_adyt(child, lambda_factor, smooth_factor, alpha_min, alpha_max))
    del module
    return module_output

def update_adyt_grad_norms(model):
    for module in model.modules():
        if isinstance(module, AdaptiveDynamicTanh):
            module.update_grad_norm()

def create_dataset(use_cifar=True, subset_size=10000):
    if use_cifar:
        print(f"Using CIFAR-100 dataset with {subset_size} samples")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Use a subset of the dataset
        if subset_size and subset_size < len(dataset):
            indices = random.sample(range(len(dataset)), subset_size)
            dataset = Subset(dataset, indices)
            
        # Extract only the images (ignore labels)
        class ImageOnlyDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                img, _ = self.dataset[idx]
                return img
                
        return ImageOnlyDataset(dataset)
    else:
        # Use a simple dataset with random noise for testing
        return SimpleDataset(size=subset_size)

def main():
    parser = argparse.ArgumentParser(description="Train a DiT model with specified normalization")
    parser.add_argument("--norm", type=str, default="ln", choices=["ln", "dyt", "adyt"], 
                        help="Normalization method to use")
    parser.add_argument("--output-dir", type=str, default="results", 
                        help="Directory to save results")
    parser.add_argument("--model", type=str, default="DiT-B/4", 
                        choices=list(DiT_models.keys()), 
                        help="DiT model size")
    parser.add_argument("--steps", type=int, default=1000, 
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--subset-size", type=int, default=10000,
                        help="Number of samples to use from dataset")
    parser.add_argument("--lambda-factor", type=float, default=0.1, 
                        help="Lambda factor for ADYT")
    parser.add_argument("--smooth-factor", type=float, default=0.99, 
                        help="Smooth factor for ADYT gradient EMA")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    if args.norm == "ln":
        method_name = "layernorm"
    elif args.norm == "dyt":
        method_name = "dynamictanh"
    else:
        method_name = "adaptivedynamictanh"
        
    output_dir = os.path.join(args.output_dir, method_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create dataset and dataloader
    dataset = create_dataset(use_cifar=True, subset_size=args.subset_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Create diffusion model
    diffusion = create_diffusion(timestep_respacing="")
    
    # Create model
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=1000  # Default for ImageNet
    )
    
    # Apply normalization method
    if args.norm == "dyt":
        print("Converting LayerNorm layers to DynamicTanh")
        model = convert_ln_to_dyt(model)
    elif args.norm == "adyt":
        print(f"Converting LayerNorm layers to AdaptiveDynamicTanh (lambda={args.lambda_factor})")
        model = convert_ln_to_adyt(model, args.lambda_factor, args.smooth_factor)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Training loop
    model.train()
    step = 0
    loss_log = []
    start_time = time.time()
    
    print(f"Starting training for {args.steps} steps...")
    
    # Create log file
    log_file = os.path.join(output_dir, "training.log")
    with open(log_file, "w") as f:
        f.write(f"step,loss,time\n")
    
    # Main training loop
    while step < args.steps:
        for batch in dataloader:
            # Get batch
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Generate random timesteps
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (x.shape[0],), device=device
            )
            
            # Encode image to latent space (simulate VAE)
            latents = torch.randn(x.shape[0], 4, latent_size, latent_size, device=device)
            
            # Add noise to latents based on timestep
            noise = torch.randn_like(latents)
            noisy_latents = diffusion.q_sample(latents, timesteps, noise=noise)
            
            # Model prediction
            model_output = model(noisy_latents, timesteps, y=None)
            
            # Calculate loss
            if True:  # diffusion.training_losses == prediction of noise
                target = noise
            else:
                target = latents
            loss = F.mse_loss(model_output, target)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            
            # Update ADYT gradient norms if using ADYT
            if args.norm == "adyt":
                update_adyt_grad_norms(model)
                
            # Clip gradients (optional)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Log progress
            loss_log.append(loss.item())
            if step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Step: {step}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
                with open(log_file, "a") as f:
                    f.write(f"{step},{loss.item()},{elapsed}\n")
            
            # Save checkpoint
            if step % 500 == 0:
                checkpoint = {
                    'step': step,
                    'model': model.state_dict(),
                    'args': vars(args),
                    'loss': loss.item()
                }
                torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_{step}.pt"))
                
            step += 1
            if step >= args.steps:
                break
    
    # Save final checkpoint
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'args': vars(args),
        'loss': loss.item()
    }
    torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_final.pt"))
    
    print(f"Training complete! Results saved to {output_dir}")
    
if __name__ == "__main__":
    main() 