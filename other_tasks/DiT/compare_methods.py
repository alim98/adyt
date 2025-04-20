#!/usr/bin/env python
# Script to train and compare LayerNorm, DynamicTanh, and AdaptiveDynamicTanh on DiT
# Run from DiT directory after applying the necessary patches

import os
import sys
import json
import argparse
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import signal
from PIL import Image
from torchvision.utils import make_grid, save_image

# Function to run a command and capture its output
def run_command(cmd, cwd=None):
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        cwd=cwd,
        universal_newlines=True
    )
    
    # Save process to allow termination on keyboard interrupt
    run_command.current_process = process
    
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Command failed with error: {error}")
        return None
    return output.strip()

# Handle keyboard interrupts gracefully
def signal_handler(sig, frame):
    if hasattr(run_command, 'current_process'):
        print("\nTerminating current process...")
        run_command.current_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Parse logs to extract metrics
def parse_training_logs(log_file):
    losses = []
    grad_norms = []
    lr_values = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "loss:" in line:
                try:
                    loss_str = line.split("loss:")[1].split()[0]
                    losses.append(float(loss_str))
                except (IndexError, ValueError):
                    pass
            
            if "grad_norm:" in line:
                try:
                    grad_norm_str = line.split("grad_norm:")[1].split()[0]
                    grad_norms.append(float(grad_norm_str))
                except (IndexError, ValueError):
                    pass
            
            if "lr:" in line:
                try:
                    lr_str = line.split("lr:")[1].split()[0]
                    lr_values.append(float(lr_str))
                except (IndexError, ValueError):
                    pass
    
    return {
        'losses': losses,
        'grad_norms': grad_norms,
        'lr_values': lr_values
    }

# Function to extract training info from DiT checkpoint
def extract_checkpoint_info(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        return {
            'step': ckpt.get('step', 0),
            'epoch': ckpt.get('epoch', 0),
            'model_args': ckpt.get('args', {})
        }
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

# Create visualizations from parsed metrics
def create_visualizations(metrics_dict, output_dir, figsize=(12, 8)):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for plots
    plt.style.use('ggplot')
    
    # List of metrics to compare
    methods = list(metrics_dict.keys())
    
    # 1. Loss comparison
    plt.figure(figsize=figsize)
    for method in methods:
        if len(metrics_dict[method]['losses']) > 0:
            # Smooth losses with moving average for better visualization
            losses = metrics_dict[method]['losses']
            if len(losses) > 10:
                smoothed_losses = np.convolve(losses, np.ones(10)/10, mode='valid')
                plt.plot(smoothed_losses, label=f"{method} (smoothed)")
            else:
                plt.plot(losses, label=method)
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Training Steps (hundreds)')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Gradient norm comparison
    plt.figure(figsize=figsize)
    for method in methods:
        if len(metrics_dict[method]['grad_norms']) > 0:
            # Smooth grad norms with moving average
            grad_norms = metrics_dict[method]['grad_norms']
            if len(grad_norms) > 10:
                smoothed_norms = np.convolve(grad_norms, np.ones(10)/10, mode='valid')
                plt.plot(smoothed_norms, label=f"{method} (smoothed)")
            else:
                plt.plot(grad_norms, label=method)
    
    plt.title('Gradient Norm Comparison')
    plt.xlabel('Training Steps (hundreds)')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grad_norm_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Final performance table
    try:
        # Create a table showing final loss and gradient norm for each method
        final_results = {}
        for method in methods:
            if len(metrics_dict[method]['losses']) > 0:
                final_results[method] = {
                    'final_loss': metrics_dict[method]['losses'][-1],
                    'min_loss': min(metrics_dict[method]['losses']),
                    'final_grad_norm': metrics_dict[method]['grad_norms'][-1] if metrics_dict[method]['grad_norms'] else None
                }
        
        # Save as JSON
        with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
            
        # Create a visualization of the table
        fig, ax = plt.figure(figsize=(8, 3)), plt.gca()
        ax.axis('tight')
        ax.axis('off')
        
        cell_text = []
        for method in methods:
            if method in final_results:
                cell_text.append([
                    method,
                    f"{final_results[method]['final_loss']:.4f}",
                    f"{final_results[method]['min_loss']:.4f}",
                    f"{final_results[method]['final_grad_norm']:.4f}" if final_results[method]['final_grad_norm'] is not None else "N/A"
                ])
        
        if cell_text:
            table = ax.table(
                cellText=cell_text,
                colLabels=['Method', 'Final Loss', 'Min Loss', 'Final Grad Norm'],
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'results_table.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error creating results table: {e}")

# Function to collect image samples from a model
def collect_samples(model_dir, samples_dir, num_samples=4):
    os.makedirs(samples_dir, exist_ok=True)
    
    # Look for latest checkpoint
    checkpoints = list(Path(model_dir).glob('*.pt'))
    if not checkpoints:
        print(f"No checkpoints found in {model_dir}")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Generate samples using the DiT generate.py script
    try:
        cmd = f"python generate.py --model DiT-B/4 --ckpt {latest_checkpoint} --num-fid-samples {num_samples}"
        dit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DiT")
        output = run_command(cmd, cwd=dit_dir)
        
        # Find generated samples (this depends on how the DiT generate.py works)
        sample_files = list(Path(dit_dir).glob('samples/*.png'))[:num_samples]
        
        # Copy samples to our samples dir
        for i, sample_file in enumerate(sample_files):
            shutil.copy(sample_file, os.path.join(samples_dir, f"sample_{i}.png"))
            
        return True
    except Exception as e:
        print(f"Error generating samples: {e}")
        return False

# Function to create a PyTorch dataset config file
def create_dataset_config(dit_dir, download_dir, use_subset=False, subset_size=50000):
    """Creates a dataset config file to use PyTorch datasets instead of local files"""
    # Make sure the directory exists
    dit_dir = os.path.abspath(dit_dir)
    if not os.path.isdir(dit_dir):
        raise FileNotFoundError(f"DiT directory not found at: {dit_dir}")
    
    # Create the full path for the config file
    config_path = os.path.join(dit_dir, "dataset_config.py")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Create the dataset module file
    with open(config_path, 'w') as f:
        f.write(f"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import random

def get_dataset(global_batch_size, root=r"{download_dir}"):
    # Create transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # Create download directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Load CIFAR-100 dataset directly
    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ]),
        download=True
    )
    
    # Use a subset if requested
    if {use_subset}:
        indices = random.sample(range(len(dataset)), min({subset_size}, len(dataset)))
        dataset = Subset(dataset, indices)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
""")
    
    return config_path

# Function to create a command for training
def create_training_command(method, args, is_windows=False):
    """Creates the appropriate training command based on OS"""
    if is_windows:
        # On Windows, use direct Python execution instead of torchrun
        cmd = (
            f"python train.py "
            f"--model {args.model} "
            f"--lr {args.lr} "
            f"{args.dataset_arg} "
            f"--results-dir {method['results_dir']} "
            f"--global-batch-size {args.batch_size} "  # Use batch_size as global_batch_size directly
            f"--epochs 1 "  # We'll control training via --max-steps
            f"--max-steps {args.steps} "
            f"--no-distributed "  # Add this flag to disable distributed training
            f"{method['args']}"
        )
    else:
        # On Linux/Mac, use torchrun as before
        cmd = (
            f"torchrun --standalone --nproc_per_node={args.num_gpus} train.py "
            f"--model {args.model} "
            f"--lr {args.lr} "
            f"{args.dataset_arg} "
            f"--results-dir {method['results_dir']} "
            f"--global-batch-size {args.batch_size * args.num_gpus} "
            f"--epochs 1 "  # We'll control training via --max-steps
            f"--max-steps {args.steps} "
            f"{method['args']}"
        )
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Train and compare different normalization methods on DiT")
    parser.add_argument("--dit-dir", type=str, default="other_tasks/DiT/DiT", help="Path to DiT repo")
    parser.add_argument("--data-path", type=str, default="", help="Path to ImageNet training data (local folder)")
    parser.add_argument("--use-torch-datasets", action="store_true", help="Use PyTorch's datasets instead of local files")
    parser.add_argument("--dataset-download-dir", type=str, default="./datasets", help="Directory to download datasets when using --use-torch-datasets")
    parser.add_argument("--use-subset", action="store_true", help="Use a smaller subset of the dataset for faster training")
    parser.add_argument("--subset-size", type=int, default=50000, help="Number of images to use when --use-subset is enabled")
    parser.add_argument("--results-dir", type=str, default="comparison_results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="DiT-B/4", choices=["DiT-B/4", "DiT-L/4", "DiT-XL/2"], help="DiT model size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--steps", type=int, default=5000, help="Number of training steps per method")
    parser.add_argument("--sample", action="store_true", help="Generate samples after training")
    parser.add_argument("--adyt-lambda", type=float, default=0.1, help="Lambda factor for ADYT")
    parser.add_argument("--adyt-smooth", type=float, default=0.99, help="Smooth factor for ADYT")
    parser.add_argument("--skip-ln", action="store_true", help="Skip LayerNorm training")
    parser.add_argument("--skip-dyt", action="store_true", help="Skip DynamicTanh training")
    parser.add_argument("--skip-adyt", action="store_true", help="Skip AdaptiveDynamicTanh training")
    parser.add_argument("--windows-mode", action="store_true", help="Run in Windows-compatible mode (no distributed training)")
    
    args = parser.parse_args()
    
    # Detect Windows OS
    is_windows = os.name == 'nt'
    if is_windows and args.num_gpus > 1:
        print("Warning: Multi-GPU training on Windows is not supported. Setting num_gpus to 1.")
        args.num_gpus = 1
    
    # Validate arguments
    if not args.use_torch_datasets and not args.data_path:
        parser.error("--data-path is required when not using --use-torch-datasets")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"dit_comparison_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    dit_dir = "other_tasks/DiT"
    # Resolve paths
    dit_dir = os.path.abspath(args.dit_dir)
    download_dir = os.path.abspath(args.dataset_download_dir)
    
    # Check if DiT directory exists
    if not os.path.isdir(dit_dir):
        print(f"Error: DiT directory not found at {dit_dir}")
        print(f"Please clone the DiT repo to this location or specify the correct path with --dit-dir")
        print(f"Current working directory is: {os.getcwd()}")
        # List directories in other_tasks/DiT to help the user
        print("\nAvailable directories in other_tasks/DiT:")
        try:
            for item in os.listdir(os.path.join("other_tasks", "DiT")):
                full_path = os.path.join("other_tasks", "DiT", item)
                if os.path.isdir(full_path):
                    print(f"  - {item}")
        except Exception as e:
            print(f"Could not list directories: {e}")
        sys.exit(1)
    
    # Set up training for each method
    methods = []
    
    if not args.skip_ln:
        methods.append({"name": "LayerNorm", "args": ""})
    
    if not args.skip_dyt:
        methods.append({"name": "DynamicTanh", "args": " --use-dyt"})
    
    if not args.skip_adyt:
        methods.append({
            "name": "AdaptiveDynamicTanh", 
            "args": f" --use-adyt --lambda-factor {args.adyt_lambda} --smooth-factor {args.adyt_smooth}"
        })
    
    # If using PyTorch datasets, create the dataset configuration
    if args.use_torch_datasets:
        print("Using PyTorch datasets instead of local files")
        try:
            dataset_config = create_dataset_config(
                dit_dir, 
                download_dir,
                args.use_subset,
                args.subset_size
            )
            print(f"Created dataset configuration at {dataset_config}")
            args.dataset_arg = "--use-torch-datasets"
        except Exception as e:
            print(f"Error creating dataset configuration: {e}")
            sys.exit(1)
    else:
        if not args.data_path:
            print("Error: You must specify --data-path when not using --use-torch-datasets")
            sys.exit(1)
        args.dataset_arg = f"--data-path {args.data_path}"

    # Create a patch to add the no-distributed flag to DiT's train.py
    if is_windows:
        no_dist_patch_path = os.path.join(os.path.dirname(__file__), "no-distributed.patch")
        with open(no_dist_patch_path, 'w') as f:
            f.write("""From 01dc036d356c11ef0cd298de550e2802f928c5f8 Mon Sep 17 00:00:00 2001
From: User <user@example.com>
Date: Mon, 17 Mar 2025 19:42:55 +0000
Subject: [PATCH] add-no-distributed-option

---
 train.py  | 22 +++++++++++++++++++++-
 1 file changed, 21 insertions(+), 1 deletion(-)

diff --git a/train.py b/train.py
index 3bc8c87..c9e5a24 100644
--- a/train.py
+++ b/train.py
@@ -66,6 +66,7 @@ def main(args):
     parser.add_argument("--log-frequency", type=int, default=100)
     parser.add_argument("--ckpt-frequency", type=int, default=50_000)
+    parser.add_argument("--no-distributed", action="store_true", help="Disable distributed training (for Windows)")
     args = parser.parse_args()
 
     assert torch.cuda.is_available(), "Training currently requires at least one GPU."
@@ -79,8 +80,17 @@ def main(args):
     
     assert os.path.exists(args.data_path), f"Data path {args.data_path} not found!"
 
-    # Setup processes.
-    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
+    # Setup processes (unless no-distributed is specified)
+    if args.no_distributed:
+        print("Running without distributed training")
+        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
+        rank = 0
+        world_size = 1
+        local_rank = 0
+    else:
+        # Normal distributed setup
+        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
+        rank = dist.get_rank()
+        world_size = dist.get_world_size()
+        local_rank = int(os.environ.get("LOCAL_RANK", 0))
+        device = torch.device(f"cuda:{local_rank}")
+
-    rank = dist.get_rank()
-    device = torch.device(f"cuda:{rank}")
-- 
2.34.1""")
        
        # Apply the patch
        try:
            print("Applying no-distributed patch for Windows compatibility...")
            run_command(f"git apply {no_dist_patch_path}", cwd=dit_dir)
        except Exception as e:
            print(f"Warning: Failed to apply no-distributed patch: {e}")
            print("You may need to manually add the --no-distributed flag support to train.py")

    # Train each method and collect metrics
    metrics = {}
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Training with {method['name']}")
        print(f"{'='*80}")
        
        # Create method-specific directory
        method_dir = os.path.join(results_dir, method["name"].lower().replace(" ", "_"))
        os.makedirs(method_dir, exist_ok=True)
        method["results_dir"] = method_dir
        
        # Set up log file
        log_file = os.path.join(method_dir, "training.log")
        
        # Create command for training
        cmd = create_training_command(method, args, is_windows)
        
        print(f"Running command: {cmd}")
        
        # Start training process
        process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=dit_dir
        )
        
        # Save process to allow termination
        run_command.current_process = process
        
        # Stream output to both console and log file
        with open(log_file, 'w') as f:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                f.write(line)
                f.flush()
                
                # Early stopping if process has terminated
                if process.poll() is not None:
                    break
        
        # Wait for process to complete
        process.wait()
        
        # Generate samples if requested
        if args.sample:
            samples_dir = os.path.join(method_dir, "samples")
            collect_samples(method_dir, samples_dir)
            
        # Parse training logs
        metrics[method["name"]] = parse_training_logs(log_file)
        
        # Copy best checkpoint to results directory with method-specific name
        try:
            checkpoints = list(Path(method_dir).glob('*.pt'))
            if checkpoints:
                best_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                shutil.copy(
                    best_checkpoint, 
                    os.path.join(results_dir, f"{method['name'].lower().replace(' ', '_')}_checkpoint.pt")
                )
        except Exception as e:
            print(f"Error copying best checkpoint: {e}")
    
    # Create visualizations from collected metrics
    create_visualizations(metrics, results_dir)
    
    # Final comparison report
    print("\nTraining complete!")
    print(f"All results saved to: {results_dir}")
    
    # If samples were generated, create a grid of samples from all methods
    if args.sample:
        try:
            all_samples = []
            for method in methods:
                method_name = method["name"].lower().replace(" ", "_")
                samples_dir = os.path.join(results_dir, method_name, "samples")
                sample_files = list(Path(samples_dir).glob('*.png'))
                if sample_files:
                    # Load and add method name as text
                    for sample_file in sample_files:
                        img = Image.open(sample_file)
                        # Convert to tensor
                        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1)) / 255.0
                        all_samples.append(img_tensor)
                        
            if all_samples:
                # Create grid
                grid = make_grid(all_samples, nrow=len(samples_dir))
                save_image(grid, os.path.join(results_dir, "all_samples_grid.png"))
                print(f"Sample grid saved to: {os.path.join(results_dir, 'all_samples_grid.png')}")
        except Exception as e:
            print(f"Error creating sample grid: {e}")
    
    print("\nExecution summary:")
    for method in methods:
        method_metrics = metrics.get(method["name"], {})
        num_steps = len(method_metrics.get("losses", []))
        final_loss = method_metrics.get("losses", [])[-1] if method_metrics.get("losses", []) else "N/A"
        print(f"- {method['name']}: {num_steps} steps, final loss: {final_loss}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 