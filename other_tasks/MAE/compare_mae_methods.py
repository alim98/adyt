import argparse
import datetime
import json
import numpy as np
import os
import time
import math
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import copy
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image

# Fix import conflicts
import sys
# Add path to make sure we get the huggingface datasets library
site_packages = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python312", "site-packages")
if os.path.exists(site_packages):
    sys.path.insert(0, site_packages)
from datasets import load_dataset
from torch.utils.data import DataLoader

import timm
import timm.optim

# Fix path to MAE modules
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mae_dir = os.path.join(base_dir, "other_tasks", "MAE", "mae")
if os.path.exists(mae_dir):
    sys.path.insert(0, mae_dir)
    print(f"Added MAE directory to path: {mae_dir}")
else:
    print(f"MAE directory not found at: {mae_dir}")
    print("Current directory:", os.getcwd())
    print("Available directories:", os.listdir(os.path.dirname(os.path.abspath(__file__))))

try:
    import util.misc as misc
    import util.lr_sched as lr_sched
    from util.misc import NativeScalerWithGradNormCount as NativeScaler
    import models_mae
    from engine_pretrain import train_one_epoch
    print("Successfully imported MAE modules")
except ImportError as e:
    print(f"Error importing MAE modules: {e}")
    print("sys.path:", sys.path)
    raise

# Import DynamicTanh implementations
try:
    from dynamic_tanh import convert_ln_to_dyt
    from dynamic_tanh_adaptive import convert_ln_to_adyt, update_adyt_grad_norms
    print("Successfully imported DynamicTanh modules")
except ImportError as e:
    print(f"Error importing DynamicTanh modules: {e}")
    raise


class ImageNetDataset(torch.utils.data.Dataset):
    """
    Custom ImageNet dataset that loads images from local directory
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Center crop 95% area as in the original code
        width, height = image.size
        left = int(0.04 * width)
        top = int(0.04 * height)
        right = int(0.96 * width)
        bottom = int(0.96 * height)
        image = image.crop((left, top, right, bottom))
        
        # Extract class ID from path
        if "train" in image_path:
            # For training images, class is the directory name
            class_id = int(image_path.split(os.sep)[-2])
        else:
            # For validation images, extract from filename
            filename = os.path.basename(image_path)
            class_id = int(filename.split('_')[0])
        
        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)
        
        return image, class_id


class HFMappedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, img_key="img", label_key="label"):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.img_key = img_key
        self.label_key = label_key

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        # Handle different possible formats from Hugging Face
        image = sample[self.img_key]
        if isinstance(image, Image.Image):
            pass  # Already a PIL Image
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            # If it's a path or other format, try this
            image = Image.open(image).convert('RGB')
        
        label = sample[self.label_key]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_args_parser():
    parser = argparse.ArgumentParser('MAE Normalization Method Comparison', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs for comparison (use smaller value for quick comparison)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--imagenet_dir', default='ILSVRC', type=str,
                        help='Path to the ImageNet dataset directory')
    parser.add_argument('--train_dir', default='Data/CLS-LOC/train', type=str,
                        help='Path to training data relative to imagenet_dir')
    parser.add_argument('--val_dir', default='Data/CLS-LOC/val', type=str,
                        help='Path to validation data relative to imagenet_dir')

    parser.add_argument('--output_dir', default='./output_comparison',
                        help='path where to save comparison results')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # AdaptiveDynamicTanh parameters
    parser.add_argument('--adyt_lambda', type=float, default=0.1,
                        help='Lambda factor for AdaptiveDynamicTanh')
    parser.add_argument('--adyt_smooth', type=float, default=0.9,
                        help='Smoothing factor for AdaptiveDynamicTanh')
    parser.add_argument('--adyt_alpha_min', type=float, default=0.1,
                        help='Minimum alpha for AdaptiveDynamicTanh')
    parser.add_argument('--adyt_alpha_max', type=float, default=2.0,
                        help='Maximum alpha for AdaptiveDynamicTanh')
    
    # Add option to limit dataset size for faster testing
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to use (for quick testing)')

    return parser


def train_model(args, model_name, model_factory_fn):
    # Set up output directory
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Record start time
    start_time_total = time.time()
    
    # Initialize device and seed
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Load local ImageNet dataset
    print(f"Loading ImageNet dataset from {args.imagenet_dir}")
    
    # Find image paths
    train_dir = os.path.join(args.imagenet_dir, args.train_dir)
    val_dir = os.path.join(args.imagenet_dir, args.val_dir)
    
    # Find training images
    train_paths = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                train_paths.append(os.path.join(root, file))
    
    # Find validation images
    val_paths = []
    for root, dirs, files in os.walk(val_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                val_paths.append(os.path.join(root, file))
    
    # Limit dataset size if needed
    if args.max_images is not None:
        print(f"Limiting dataset to {args.max_images} images for faster testing")
        if len(train_paths) > args.max_images:
            train_paths = train_paths[:args.max_images]
    
    print(f"Found {len(train_paths)} training images and {len(val_paths)} validation images")
    
    # Set up data transformations
    transform_train = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
    # Create dataset and dataloader
    dataset_train = ImageNetDataset(
        image_paths=train_paths,
        transform=transform_train
    )
    
    print(f"Dataset size: {len(dataset_train)}")

    # Set up sampler and dataloader
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # Initialize tensorboard writer
    log_writer = SummaryWriter(log_dir=log_dir)
    
    # Create and initialize model
    model = model_factory_fn()
    model.to(device)
    
    # Set up optimizer
    eff_batch_size = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print(f"Model: {model_name}")
    print(f"Learning rate: {args.lr:.6f}")
    
    # Set up optimizer with weight decay
    param_groups = timm.optim.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    # Metrics to track
    all_train_stats = []
    loss_values = []
    epoch_times = []
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train one epoch
        if model_name == "AdaptiveDynamicTanh":
            # Custom training loop for AdaptiveDynamicTanh with gradient norm updates
            model.train()
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = f'Epoch: [{epoch}]'
            print_freq = 20

            accum_iter = args.accum_iter
            optimizer.zero_grad()

            for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)):
                # Adjust learning rate per step
                if data_iter_step % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, args)

                samples = samples.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss /= accum_iter
                loss_scaler(loss, optimizer, parameters=model.parameters(),
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
                
                # Custom: Update the AdaptiveDynamicTanh gradient norms
                update_adyt_grad_norms(model)
                
                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()

                torch.cuda.synchronize()

                metric_logger.update(loss=loss_value)
                
                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

                loss_value_reduce = misc.all_reduce_mean(loss_value)
                if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                    epoch_1000x = int((data_iter_step / len(data_loader_train) + epoch) * 1000)
                    log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                    log_writer.add_scalar('lr', lr, epoch_1000x)

            # Gather all stats
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        else:
            # Standard training loop for other models
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )
        
        # Record metrics
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        train_stats_dict = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                           'epoch': epoch, 
                           'time': epoch_time}
        all_train_stats.append(train_stats_dict)
        loss_values.append(train_stats.get('loss', 0))
        
        # Save to log file
        with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(train_stats_dict) + "\n")
        
        # Save checkpoint every 20 epochs or on last epoch
        if epoch % 20 == 0 or epoch + 1 == args.epochs:
            ckpt_path = os.path.join(output_dir, f"checkpoint-epoch{epoch}.pth")
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            torch.save(save_dict, ckpt_path)
    
    # Calculate total time
    total_time = time.time() - start_time_total
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    # Save summary metrics
    summary = {
        'model': model_name,
        'total_epochs': args.epochs,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'final_loss': loss_values[-1] if loss_values else None,
        'min_loss': min(loss_values) if loss_values else None,
        'min_loss_epoch': loss_values.index(min(loss_values)) if loss_values else None,
    }
    
    with open(os.path.join(output_dir, "summary.json"), mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    
    print(f'{model_name} training completed.')
    print(f'Total time: {total_time_str}')
    print(f'Average epoch time: {avg_epoch_time:.2f} seconds')
    print(f'Final loss: {loss_values[-1] if loss_values else None}')
    print(f'Best loss: {min(loss_values) if loss_values else None} at epoch {loss_values.index(min(loss_values)) if loss_values else None}')
    
    # Save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values)
    plt.title(f"{model_name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    
    return summary, loss_values, data_loader_train


def visualize_reconstructions(model, sample_images, mask_ratio, device, output_dir, model_name):
    """
    Visualize original images and their MAE reconstructions
    
    Args:
        model: MAE model
        sample_images: Tensor of sample images [B, C, H, W]
        mask_ratio: Masking ratio to use
        device: Device to run inference on
        output_dir: Directory to save visualizations
        model_name: Name of the model for filename
    """
    # Set model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Move images to device
        images = sample_images.to(device, non_blocking=True)
        
        # Get reconstructions from model
        loss, y_pred, mask = model(images, mask_ratio=mask_ratio)
        
        # Move to CPU for visualization
        images = images.cpu()
        y_pred = y_pred.cpu()
        mask = mask.cpu()
        
        # Create visualization grid
        fig, axes = plt.subplots(nrows=len(images), ncols=3, figsize=(15, 5*len(images)))
        
        # If only one image, make axes indexable
        if len(images) == 1:
            axes = axes.reshape(1, -1)
        
        # Loop through images
        for i, (img, pred, m) in enumerate(zip(images, y_pred, mask)):
            # Original image
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Masked image (replace masked patches with gray)
            masked_img = img.clone()
            patch_size = 16  # MAE patch size
            num_patches = (img.shape[1] // patch_size) * (img.shape[2] // patch_size)
            bool_mask = mask.reshape(-1) == 1
            
            # Create a mask for visualization
            vis_mask = torch.ones_like(img)
            for p in range(num_patches):
                if bool_mask[p]:
                    patch_h = (p // int(img.shape[2] / patch_size)) * patch_size
                    patch_w = (p % int(img.shape[2] / patch_size)) * patch_size
                    vis_mask[:, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] = 0.5
            
            masked_vis = img * vis_mask
            masked_np = masked_vis.permute(1, 2, 0).numpy()
            masked_np = (masked_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            masked_np = np.clip(masked_np, 0, 1)
            
            # Reconstruction
            pred_np = pred.permute(1, 2, 0).numpy()
            pred_np = (pred_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            pred_np = np.clip(pred_np, 0, 1)
            
            # Plot images
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masked_np)
            axes[i, 1].set_title(f'Masked ({int(mask_ratio*100)}%)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_np)
            axes[i, 2].set_title('Reconstruction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_reconstructions.png'), dpi=200)
        plt.close()


def visualize_comparison(models, model_names, sample_images, mask_ratio, device, output_dir):
    """
    Create a side-by-side comparison of reconstructions from different models
    
    Args:
        models: List of MAE models
        model_names: List of model names
        sample_images: Tensor of sample images [B, C, H, W]
        mask_ratio: Masking ratio to use
        device: Device to run inference on
        output_dir: Directory to save visualizations
    """
    # Number of models
    num_models = len(models)
    
    # Number of images
    num_images = len(sample_images)
    
    # Set models to eval mode
    for model in models:
        model.eval()
    
    # Predictions from all models
    all_preds = []
    
    with torch.no_grad():
        # Move images to device
        images = sample_images.to(device, non_blocking=True)
        
        # Get reconstructions from each model
        for model in models:
            loss, y_pred, mask = model(images, mask_ratio=mask_ratio)
            all_preds.append(y_pred.cpu())
        
        # Create visualization grid
        fig, axes = plt.subplots(nrows=num_images, ncols=num_models+1, figsize=(5*(num_models+1), 5*num_images))
        
        # If only one image, make axes indexable
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        # Loop through images
        for i, img in enumerate(sample_images):
            # Original image
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Plot original image
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Plot reconstructions from each model
            for j, (model_name, preds) in enumerate(zip(model_names, all_preds)):
                pred = preds[i]
                pred_np = pred.permute(1, 2, 0).numpy()
                pred_np = (pred_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                pred_np = np.clip(pred_np, 0, 1)
                
                axes[i, j+1].imshow(pred_np)
                axes[i, j+1].set_title(model_name)
                axes[i, j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=200)
        plt.close()


def main():
    # Parse arguments
    args = get_args_parser()
    args = args.parse_args()
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize results storage
    all_summaries = []
    all_loss_values = {}
    
    # Train with LayerNorm (original MAE)
    print("\n==== Training with LayerNorm (Original MAE) ====\n")
    def create_original_model():
        return models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    
    layernorm_summary, layernorm_losses, data_loader_train = train_model(args, "LayerNorm", create_original_model)
    all_summaries.append(layernorm_summary)
    all_loss_values["LayerNorm"] = layernorm_losses
    
    # Train with DynamicTanh
    print("\n==== Training with DynamicTanh ====\n")
    def create_dyt_model():
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        return convert_ln_to_dyt(model)
    
    dyt_summary, dyt_losses, _ = train_model(args, "DynamicTanh", create_dyt_model)
    all_summaries.append(dyt_summary)
    all_loss_values["DynamicTanh"] = dyt_losses
    
    # Train with AdaptiveDynamicTanh
    print("\n==== Training with AdaptiveDynamicTanh ====\n")
    def create_adyt_model():
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        return convert_ln_to_adyt(
            model, 
            lambda_factor=args.adyt_lambda,
            smooth_factor=args.adyt_smooth,
            alpha_min=args.adyt_alpha_min,
            alpha_max=args.adyt_alpha_max
        )
    
    adyt_summary, adyt_losses, _ = train_model(args, "AdaptiveDynamicTanh", create_adyt_model)
    all_summaries.append(adyt_summary)
    all_loss_values["AdaptiveDynamicTanh"] = adyt_losses
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_summaries)
    comparison_df.to_csv(os.path.join(args.output_dir, "comparison_table.csv"), index=False)
    print("\n==== Comparison Summary ====\n")
    print(comparison_df)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    for model_name, losses in all_loss_values.items():
        plt.plot(losses, label=model_name)
    
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_comparison.png"))
    plt.close()
    
    # Create timing comparison
    times = [summary['avg_epoch_time'] for summary in all_summaries]
    models = [summary['model'] for summary in all_summaries]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, times)
    plt.title("Average Epoch Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "timing_comparison.png"))
    
    print("\nComparison complete. Results saved to", args.output_dir)
    
    # Generate visualizations with actual data samples
    print("\nGenerating reconstruction visualizations...")
    
    # Use the same data for all models for fair comparison
    test_batch = next(iter(data_loader_train))
    sample_images = test_batch[0][:4]  # Use first 4 images from batch
    
    # Create models for visualization
    ln_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss).to(args.device)
    dyt_model = convert_ln_to_dyt(models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)).to(args.device)
    adyt_model = convert_ln_to_adyt(
        models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss),
        lambda_factor=args.adyt_lambda,
        smooth_factor=args.adyt_smooth,
        alpha_min=args.adyt_alpha_min,
        alpha_max=args.adyt_alpha_max
    ).to(args.device)
    
    # Generate visualizations
    visualize_reconstructions(ln_model, sample_images, args.mask_ratio, args.device, 
                             args.output_dir, "LayerNorm")
    visualize_reconstructions(dyt_model, sample_images, args.mask_ratio, args.device, 
                             args.output_dir, "DynamicTanh")
    visualize_reconstructions(adyt_model, sample_images, args.mask_ratio, args.device, 
                             args.output_dir, "AdaptiveDynamicTanh")
    
    # Combined visualization of all methods
    print("Creating combined reconstruction visualization...")
    visualize_comparison(
        [ln_model, dyt_model, adyt_model],
        ["LayerNorm", "DynamicTanh", "AdaptiveDynamicTanh"],
        sample_images, args.mask_ratio, args.device, args.output_dir
    )
    
    print("\nAll visualizations saved to", args.output_dir)


if __name__ == '__main__':
    main() 