
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import random

def get_dataset(global_batch_size, root=r"C:\Users\alim9\Documents\codes\DyT\datasets"):
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
    if False:
        indices = random.sample(range(len(dataset)), min(50000, len(dataset)))
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
