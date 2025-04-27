# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import importlib.util

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, is_distributed=True, rank=0):
    """
    Create a logger that writes to a log file and stdout.
    """
    if not is_distributed or rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    if not args.no_distributed:
        dist.init_process_group("nccl")
        assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        world_size = dist.get_world_size()
    else:
        # No distributed training
        rank = 0
        device = 0
        seed = args.global_seed
        world_size = 1

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, not args.no_distributed, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, not args.no_distributed, rank)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    if args.no_distributed:
        model = model.to(device)
    else:
        model = DDP(model.to(device), device_ids=[rank])
        
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    if args.use_torch_datasets:
        # Import the dataset_config module dynamically
        try:
            logger.info("Using PyTorch datasets from dataset_config.py")
            spec = importlib.util.spec_from_file_location("dataset_config", "dataset_config.py")
            dataset_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_config)
            
            # Get the dataloader from the config, passing the data_path
            logger.info(f"Using data path: {args.data_path}")
            loader = dataset_config.get_dataset(args.global_batch_size, root=args.data_path)
            dataset_size = len(loader.dataset)
            logger.info(f"Dataset from dataset_config contains {dataset_size:,} images")
        except Exception as e:
            logger.error(f"Error loading dataset from dataset_config.py: {e}")
            raise
    else:
        # Use the normal ImageFolder dataset
        assert os.path.exists(args.data_path), f"Data path {args.data_path} not found!"
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        
        # Set up the data loader with distributed sampler if needed
        if not args.no_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.global_seed
            )
        else:
            sampler = None
            
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // world_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    if args.no_distributed:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if not args.use_torch_datasets and sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        # Handle both cases: when loader returns (x, y) and when it returns just x
        for batch in loader:
            # Only continue training for the specified number of steps
            if args.max_steps is not None and train_steps >= args.max_steps:
                break
            
            # Handling for both tuple returns (image, label) and single tensor returns
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x = batch
                # Create dummy labels, not used in unconditional generation
                y = torch.zeros(x.shape[0], dtype=torch.long, device=device)
                
            x = x.to(device)
            y = y.to(device)
            
            # Update the model with the current batch
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            # Encode image to latent space
            with torch.no_grad():
                # Scale input from [0, 1] to [-1, 1] if needed
                if x.min() >= 0 and x.max() <= 1:
                    x = 2 * x - 1
                # Encode images
                encoder_posterior = vae.encode(x).latent_dist
                z = encoder_posterior.sample() * 0.18215
            
            # Compute loss
            loss = diffusion.training_losses(
                model if args.no_distributed else model.module, 
                z, 
                t, 
                model_kwargs={"y": y}
            )["loss"].mean()
            
            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # Update EMA model
            if args.no_distributed:
                update_ema(ema, model)
            else:
                update_ema(ema, model.module)
                
            # Update metrics
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            # Log metrics
            if train_steps % args.log_every == 0:
                # Compute timestamp
                torch.cuda.synchronize()
                end_time = time()
                
                # Compute throughput
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = args.global_batch_size * steps_per_sec
                
                # Compute average loss
                avg_loss = running_loss / log_steps
                
                # Log to console
                logger.info(
                    f"Step: {train_steps}/{args.max_steps} "
                    f"Loss: {avg_loss:.4f} "
                    f"Steps/Sec: {steps_per_sec:.2f} "
                    f"Samples/Sec: {samples_per_sec:.2f} "
                )
                
                # Reset metrics
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.state_dict() if args.no_distributed else model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "step": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
     
        # End training when we reach the max steps
        if args.max_steps is not None and train_steps >= args.max_steps:
            logger.info(f"Reached {args.max_steps} steps, ending training.")
            break
            
    if not args.no_distributed:
        # Destroy process group
        cleanup()

    # Save final checkpoint
    if rank == 0:
        logger.info("Saving final checkpoint...")
        
        checkpoint = {
            "model": model.state_dict() if args.no_distributed else model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args,
            "step": train_steps
        }
        
        final_checkpoint_path = f"{checkpoint_dir}/final.pt"
        torch.save(checkpoint, final_checkpoint_path)
        logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--data-path", type=str, help="Path to the training data directory.")
    # General hyperparameters
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--no-distributed", action="store_true", help="Disable distributed training (for Windows)")
    # Additional arguments for comparison tests
    parser.add_argument("--use-torch-datasets", action="store_true", help="Use PyTorch datasets from dataset_config.py")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # Arguments for the activation function variants
    parser.add_argument("--use-dyt", action="store_true", help="Use DynamicTanh instead of LayerNorm")
    parser.add_argument("--use-adyt", action="store_true", help="Use AdaptiveDynamicTanh instead of LayerNorm")
    parser.add_argument("--lambda-factor", type=float, default=0.1, help="Lambda factor for AdaptiveDynamicTanh")
    parser.add_argument("--smooth-factor", type=float, default=0.99, help="Smooth factor for AdaptiveDynamicTanh")
    args = parser.parse_args()
    
    # Ensure either data-path or use-torch-datasets is provided
    if not args.data_path and not args.use_torch_datasets:
        parser.error("Either --data-path or --use-torch-datasets is required")
        
    main(args)
