import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class KaggleImageNetDataset(Dataset):
    """Custom dataset for loading ImageNet data directly from Kaggle's ILSVRC structure"""
    
    def __init__(self, root_dir, split='train', transform=None, max_images=None):
        """
        Args:
            root_dir (string): Path to ImageNet directory (containing ILSVRC folder)
            split (string): 'train' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
            max_images (int, optional): Maximum number of images to use (for testing/debugging)
        """
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
    """Create a DataLoader for the Kaggle ImageNet dataset"""
    
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
    ) 