import os
import sys
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Add parent directory to path so we can import the custom dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_dataset import KaggleImageNetDataset

# Wrapper class to match DiT's expected format
class DiTDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # DiT expects ONLY images, not (image, label) tuples
        image, label = self.dataset[idx]
        return image  # Return only the image, not the label

def get_dataset(global_batch_size, root=None, max_images=5000, num_workers=0):
    """
    Create a dataset for DiT training using Kaggle's ImageNet structure
    
    Args:
        global_batch_size: Batch size to use
        root: Path to the ImageNet directory (containing ILSVRC folder)
        max_images: Maximum number of images to use (for testing/debugging)
        num_workers: Number of workers for the DataLoader
    
    Returns:
        DataLoader for the dataset
    """
    if root is None:
        # Default path - use current path from args
        root = "C:/Users/WINGPU/Desktop/ADyt/other_tasks/DINO/data"
    
    # Create transformations - these match DiT's requirements
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Create dataset
    base_dataset = KaggleImageNetDataset(
        root_dir=root,
        split='train',
        transform=transform,
        max_images=max_images
    )
    
    # Wrap the dataset to match DiT's expectations (only images, no labels)
    dataset = DiTDatasetWrapper(base_dataset)
    
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