#!/usr/bin/env python
# Script to train and compare LayerNorm, DynamicTanh, and AdaptiveDynamicTanh on DiT with ImageNet

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
import cv2

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

# Function to create a command for training
def create_training_command(method, args, is_windows=False):
    """Creates the appropriate training command based on OS"""
    # Fix paths for Windows
    results_dir_fixed = os.path.abspath(method['results_dir']).replace('\\', '/')
    imagenet_dir_fixed = os.path.abspath(args.imagenet_dir).replace('\\', '/')
    
    # Set appropriate num_workers for the platform
    num_workers = 0 if is_windows else 4
    
    if is_windows:
        # On Windows, use direct Python execution instead of torchrun
        cmd = (
            f"python train.py "
            f"--model {args.model} "
            f"--lr {args.lr} "
            f"--data-path \"{imagenet_dir_fixed}\" "
            f"--use-torch-datasets "
            f"--results-dir \"{results_dir_fixed}\" "
            f"--global-batch-size {args.batch_size} "
            f"--epochs 1 "
            f"--max-steps {args.steps} "
            f"--no-distributed "
            f"--num-workers {num_workers} "
            f"{method['args']}"
        )
    else:
        # On Linux/Mac, use torchrun as before
        cmd = (
            f"torchrun --standalone --nproc_per_node={args.num_gpus} train.py "
            f"--model {args.model} "
            f"--lr {args.lr} "
            f"--data-path \"{imagenet_dir_fixed}\" "
            f"--use-torch-datasets "
            f"--results-dir \"{results_dir_fixed}\" "
            f"--global-batch-size {args.batch_size * args.num_gpus} "
            f"--epochs 1 "
            f"--max-steps {args.steps} "
            f"--num-workers {num_workers} "
            f"{method['args']}"
        )
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="Train and compare different normalization methods on DiT with ImageNet")
    parser.add_argument("--dit-dir", type=str, default="other_tasks/DiT/DiT", help="Path to DiT repo")
    parser.add_argument("--imagenet-dir", type=str, default="C:/Users/WINGPU/Desktop/DyT_2/other_tasks/DINO/data", help="Path to ImageNet directory (containing ILSVRC folder)")
    parser.add_argument("--max-images", type=int, default=5000, help="Maximum number of images to use from dataset")
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
        
    # Force windows mode on Windows
    if is_windows:
        args.windows_mode = True
    
    # Validate arguments
    if not args.imagenet_dir:
        parser.error("--imagenet-dir is required to specify the path to the ImageNet dataset")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"dit_comparison_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Resolve paths
    dit_dir = os.path.abspath(args.dit_dir)
    imagenet_dir = os.path.abspath(args.imagenet_dir)
    
    # Ensure results_dir is absolute
    results_dir = os.path.abspath(results_dir)
    
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
    
    # Check if the ImageNet directory exists
    if not os.path.isdir(imagenet_dir):
        print(f"Error: ImageNet directory not found at {imagenet_dir}")
        sys.exit(1)
    
    # Check if ILSVRC directory exists
    ilsvrc_path = os.path.join(imagenet_dir, "ILSVRC")
    if not os.path.isdir(ilsvrc_path):
        print(f"Error: ILSVRC directory not found at {ilsvrc_path}")
        print("Please ensure your ImageNet directory contains the ILSVRC folder")
        sys.exit(1)
        
    # Copy custom dataset files to DiT directory
    print("Setting up custom dataset files...")
    
    # 1. Copy custom_dataset.py
    custom_dataset_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_dataset.py")
    if not os.path.exists(custom_dataset_src):
        print("Creating custom dataset module...")
        with open(custom_dataset_src, 'w') as f:
            f.write("""import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class KaggleImageNetDataset(Dataset):
    \"\"\"Custom dataset for loading ImageNet data directly from Kaggle's ILSVRC structure\"\"\"
    
    def __init__(self, root_dir, split='train', transform=None, max_images=None):
        \"\"\"
        Args:
            root_dir (string): Path to ImageNet directory (containing ILSVRC folder)
            split (string): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
            max_images (int, optional): Maximum number of images to use (for testing/debugging)
        \"\"\"
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.max_images = max_images
        self.image_paths = []
        self.labels = []
        
        ilsvrc_path = os.path.join(self.root_dir, "ILSVRC")
        if not os.path.isdir(ilsvrc_path):
            raise FileNotFoundError(f"ILSVRC directory not found at {ilsvrc_path}")
        
        # Different path structure for training and validation
        if split == 'train':
            data_path = os.path.join(ilsvrc_path, "Data", "CLS-LOC", "train")
            if not os.path.isdir(data_path):
                raise FileNotFoundError(f"Training data not found at {data_path}")
            
            # Get class folders (synsets)
            synsets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            
            # Create a mapping from synset to numerical label
            self.synset_to_label = {synset: i for i, synset in enumerate(synsets)}
            
            # Collect images and labels
            for synset in synsets:
                synset_path = os.path.join(data_path, synset)
                image_files = [f for f in os.listdir(synset_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.tif', '.tiff', '.bmp'))]
                
                # Add images to dataset
                for img_file in image_files:
                    self.image_paths.append(os.path.join(synset_path, img_file))
                    self.labels.append(self.synset_to_label[synset])
                    
                    # Check if we've reached the maximum number of images
                    if self.max_images is not None and len(self.image_paths) >= self.max_images:
                        break
                
                # Check again after processing each synset
                if self.max_images is not None and len(self.image_paths) >= self.max_images:
                    break
        
        elif split == 'val':
            data_path = os.path.join(ilsvrc_path, "Data", "CLS-LOC", "val")
            if not os.path.isdir(data_path):
                raise FileNotFoundError(f"Validation data not found at {data_path}")
            
            # For validation, we need to map images to their labels
            # This typically requires a label mapping file, but for our purposes
            # we'll just assign a dummy label (0) to all validation images
            image_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.tif', '.tiff', '.bmp'))]
            
            for img_file in image_files:
                self.image_paths.append(os.path.join(data_path, img_file))
                self.labels.append(0)  # Dummy label
                
                # Check if we've reached the maximum number of images
                if self.max_images is not None and len(self.image_paths) >= self.max_images:
                    break
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and process the image
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            blank = torch.zeros(3, 256, 256) if self.transform else Image.new('RGB', (256, 256), (0, 0, 0))
            return blank, label

def get_dataloader(root_dir, batch_size=32, split='train', max_images=None, num_workers=4):
    \"\"\"Create a DataLoader for the Kaggle ImageNet dataset\"\"\"
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Create the dataset
    dataset = KaggleImageNetDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        max_images=max_images
    )
    
    # Create and return the DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )""")
    
    # Copy to DiT dir
    custom_dataset_dst = os.path.join(dit_dir, "custom_dataset.py")
    shutil.copy2(custom_dataset_src, custom_dataset_dst)
    
    # 2. Create dataset_config.py in DiT dir
    dataset_config_path = os.path.join(dit_dir, "dataset_config.py")
    with open(dataset_config_path, 'w') as f:
        f.write(f"""import os
import sys
import torch
from torch.utils.data import DataLoader
import importlib

# Import the custom dataset
from custom_dataset import KaggleImageNetDataset, get_dataloader

def get_dataset(global_batch_size, root=r"{imagenet_dir}", max_images={args.max_images}, num_workers={0 if is_windows else 4}):

    # Create transformations - these match DiT's requirements
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Create dataset
    dataset = KaggleImageNetDataset(
        root_dir=root,
        split='train',
        transform=transform,
        max_images=max_images
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
""")
    
    print(f"Dataset files set up successfully in {dit_dir}")
        
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

    # Train each method and collect metrics
    metrics = {}
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Training with {method['name']}")
        print(f"{'='*80}")
        
        # Create method-specific directory
        method_dir = os.path.abspath(os.path.join(results_dir, method["name"].lower().replace(" ", "_")))
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
    
    # Clean up temp directory
    print("Cleaning up temporary dataset directory...")
    try:
        shutil.rmtree(temp_data_dir)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory: {e}")
    
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