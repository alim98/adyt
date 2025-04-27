import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, DataLoader
import random
from PIL import Image
import glob

class KaggleImageNetDataset(Dataset):
    """Dataset for directly using Kaggle ImageNet format"""
    
    def __init__(self, root_dir, transform=None, max_images=None):
        self.transform = transform
        self.image_paths = []
        
        # Look for the correct structure: ILSVRC/Data/CLS-LOC/train/
        train_dir = os.path.join(root_dir, "ILSVRC", "Data", "CLS-LOC", "train")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Could not find ImageNet train directory at {train_dir}")
            
        print(f"Loading ImageNet from: {train_dir}")
        
        # Get all class folders
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        # Collect image paths from all class folders
        for class_dir in class_dirs:
            class_path = os.path.join(train_dir, class_dir)
            # Get all image files in this class folder
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG']:
                images.extend(glob.glob(os.path.join(class_path, ext)))
                
            self.image_paths.extend(images)
            
            # Check if we've reached the maximum number of images
            if max_images is not None and len(self.image_paths) >= max_images:
                self.image_paths = self.image_paths[:max_images]
                break
        
        print(f"Loaded {len(self.image_paths)} images from {len(class_dirs)} classes")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and process the image
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            # DiT expects ONLY the image, no label
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            blank = torch.zeros(3, 256, 256) if self.transform else Image.new('RGB', (256, 256), (0, 0, 0))
            return blank

def get_dataset(global_batch_size, root=None):
    """
    Create a dataset for DiT training using Kaggle's ImageNet structure
    
    Args:
        global_batch_size: Batch size to use
        root: Path to the dataset directory containing ILSVRC folder
    
    Returns:
        DataLoader for the dataset
    """
    if root is None:
        # Default path - will be overridden by the --data-path argument
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    # Create transformations - these match DiT's requirements
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Create dataset with direct access to Kaggle ImageNet data
    dataset = KaggleImageNetDataset(
        root_dir=root,
        transform=transform,
        max_images=None  # Use all available images
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=global_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
