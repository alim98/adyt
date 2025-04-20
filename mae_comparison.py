import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import datetime

print("Starting MAE comparison script...")

# Import normalization modules
try:
    print("Importing DynamicTanh modules...")
    from dynamic_tanh import DynamicTanh
    from dynamic_tanh_adaptive import AdaptiveDynamicTanh, update_adyt_grad_norms
    print("Successfully imported DynamicTanh modules")
except ImportError as e:
    print(f"Error importing dynamic_tanh modules: {e}")
    print("Make sure they are in your path.")
    exit(1)

# Simple ViT-based MAE architecture
class MAE(nn.Module):
    def __init__(self, norm_type='dyt', lambda_factor=0.1, smooth_factor=0.9,
                alpha_min=0.1, alpha_max=2.0,
                img_size=28, patch_size=7, in_chans=1, embed_dim=64,
                encoder_depth=4, decoder_depth=2, num_heads=4, mlp_ratio=4.0,
                mask_ratio=0.75):
        super(MAE, self).__init__()
        
        self.norm_type = norm_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        
        # Calculate parameters
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Select normalization type for creating blocks
        if norm_type == 'dyt':
            norm_layer = lambda size: DynamicTanh(size, True)
        elif norm_type == 'adyt':
            norm_layer = lambda size: AdaptiveDynamicTanh(
                size, True, 
                lambda_factor=lambda_factor, 
                smooth_factor=smooth_factor,
                alpha_min=alpha_min,
                alpha_max=alpha_max
            )
        elif norm_type == 'ln':
            norm_layer = lambda size: nn.LayerNorm(size)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer
            )
            for _ in range(encoder_depth)
        ])
        
        # Encoder normalization
        self.encoder_norm = norm_layer(embed_dim)
        
        # Decoder embedding - project from encoder dim to decoder dim (can be different)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        
        # Mask token (for replacing masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Decoder position embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        # Decoder normalization
        self.decoder_norm = norm_layer(embed_dim)
        
        # Decoder prediction head - project from embedding dimension to patch dimension
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
    
    def patchify(self, imgs):
        """Convert images to patches"""
        B, C, H, W = imgs.shape
        p = self.patch_size
        assert H == W and H % p == 0
        
        # Extract patches and reshape
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [B, H//p, W//p, p, p, C]
        patches = x.reshape(B, (H // p) * (W // p), p * p * C)  # [B, num_patches, p*p*C]
        return patches
    
    def random_masking(self, x, mask_ratio):
        """Perform random masking by per-sample shuffling"""
        B, N, D = x.shape  # batch, num_patches, dim
        len_keep = int(N * (1 - mask_ratio))
        
        # Generate noise for random masking
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep patches, drop the remaining ones
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate mask (1 is masked, 0 is kept)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Reorder to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Convert images to patches
        patches = self.patchify(x)
        
        # Convert patches to embedding dimension
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply encoder transformer blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Final encoder normalization
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore, patches
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Expand mask tokens to the entire sequence
        B, N, D = x.shape
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - N, 1)
        
        # Concatenate visible tokens and mask tokens
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # Reorder tokens to the original order
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add position embeddings
        x = x_ + self.decoder_pos_embed
        
        # Apply decoder transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final decoder normalization
        x = self.decoder_norm(x)
        
        # Project to pixel space (patch dimension)
        x = self.decoder_pred(x)
        
        return x
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Encoder: embed patches, apply transformer blocks, and random masking
        latent, mask, ids_restore, patches = self.forward_encoder(imgs, mask_ratio)
        
        # Decoder: predict pixel values for masked patches
        pred = self.forward_decoder(latent, ids_restore)
        
        return pred, mask, patches
    
    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = self.img_size // p
        B = x.shape[0]
        
        # Reshape to match original image dimensions
        x = x.reshape(B, h, w, p, p, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.in_chans, self.img_size, self.img_size)
        return x
    
    def reconstruct(self, imgs, mask_ratio=None):
        # Forward pass to get predictions
        pred, mask, patches = self.forward(imgs, mask_ratio)
        
        # Reshape mask to match image dimensions for visualization
        p = self.patch_size
        mask = mask.reshape(-1, self.img_size // p, self.img_size // p)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, p * p)
        mask = mask.reshape(-1, self.img_size // p, self.img_size // p, p, p)
        mask = mask.permute(0, 1, 3, 2, 4)
        mask = mask.reshape(-1, self.img_size, self.img_size)
        
        # Convert predicted patches to images
        pred_img = self.unpatchify(pred.reshape(-1, self.num_patches, p * p * self.in_chans))
        
        # Create reconstruction (masked patches are filled with predictions)
        recon = imgs.clone()
        mask_expanded = mask.unsqueeze(1).repeat(1, self.in_chans, 1, 1)
        recon = torch.where(mask_expanded.bool(), pred_img, imgs)
        
        return recon, pred_img, mask


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer):
        super(TransformerBlock, self).__init__()
        
        # Self-attention
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x):
        # Self-attention with skip connection
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_output
        
        # MLP with skip connection
        x = x + self.mlp(self.norm2(x))
        
        return x


# Train function with timing and stability metrics
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    
    # Track metrics
    train_loss = 0
    train_acc = 0
    batch_losses = []
    start_time = time.time()
    
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass (returns predictions, mask, and original patches)
        pred, mask, patches = model(data)
        
        # Calculate loss (MSE on the masked patches only)
        # Now pred and patches should have the same shape
        loss = F.mse_loss(pred, patches, reduction='none')
        
        # Apply mask to compute loss only on masked patches
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * patches.shape[-1] + 1e-6)
        
        # Calculate accuracy as percentage of pixels within threshold error
        # This defines accuracy as pixels within 0.1 (10%) of the true value
        with torch.no_grad():
            threshold = 0.1
            correct = (torch.abs(pred - patches) < threshold).float() * mask.unsqueeze(-1)
            accuracy = correct.sum() / (mask.sum() * patches.shape[-1] + 1e-6)
            train_acc += accuracy.item()
        
        # Backward pass
        loss.backward()
        
        # Update gradient norms if using AdaptiveDynamicTanh
        if model.norm_type == 'adyt':
            update_adyt_grad_norms(model)
            
        optimizer.step()
        
        # Update metrics
        batch_loss = loss.item()
        train_loss += batch_loss
        batch_losses.append(batch_loss)
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), batch_loss, accuracy.item() * 100))
    
    # Calculate metrics
    epoch_time = time.time() - start_time
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    loss_std = np.std(batch_losses)  # Measure of training stability
    
    return {
        'loss': train_loss,
        'acc': train_acc,
        'time': epoch_time,
        'stability': loss_std,
        'batch_losses': batch_losses
    }


# Evaluation function
def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    all_reconstructions = []
    all_originals = []
    all_masks = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Forward pass
            pred, mask, patches = model(data)
            
            # Calculate loss (MSE on the masked patches only)
            loss = F.mse_loss(pred, patches, reduction='none')
            
            # Apply mask to compute loss only on masked patches
            loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * patches.shape[-1] + 1e-6)
            
            # Calculate accuracy as percentage of pixels within threshold error
            threshold = 0.1
            correct = (torch.abs(pred - patches) < threshold).float() * mask.unsqueeze(-1)
            accuracy = correct.sum() / (mask.sum() * patches.shape[-1] + 1e-6)
            
            test_loss += loss.item()
            test_acc += accuracy.item()
            
            # Save reconstructions for visualization
            if len(all_reconstructions) < 8:  # Save only a few samples
                recon, pred_img, vis_mask = model.reconstruct(data)
                # Take only a single image/batch for visualization
                all_reconstructions.append(recon[:1].cpu())
                all_originals.append(data[:1].cpu())
                all_masks.append(vis_mask[:1].cpu())
    
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    
    # Generate visualization of reconstructions
    if all_reconstructions:
        visualize_reconstructions(all_originals, all_reconstructions, all_masks, model.norm_type)
    
    return test_loss, test_acc


# Function to visualize reconstructions
def visualize_reconstructions(originals, reconstructions, masks, norm_type):
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    for i in range(8):
        # Original
        orig = originals[i][0].squeeze().numpy()
        axes[0, i].imshow(orig, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Mask - ensure it's a 2D array for imshow
        mask = masks[i].squeeze().numpy()
        if mask.ndim > 2:
            mask = mask[0] if mask.shape[0] == 1 else mask[0, :, :]  # Take first channel/image if multiple
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Mask')
        
        # Reconstruction
        recon = reconstructions[i][0].squeeze().numpy()
        axes[2, i].imshow(recon, cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstruction')
    
    plt.tight_layout()
    plt.savefig(f'mae_reconstructions_{norm_type}.png')
    plt.close()


# Function to plot training metrics for comparison
def plot_comparison(metrics_dict):
    # Create a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot training metrics comparison
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(3, 2, 1)
    for norm_type, metrics in metrics_dict.items():
        plt.plot(metrics['train_losses'], label=f"{norm_type}")
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('MAE Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot test loss
    plt.subplot(3, 2, 2)
    for norm_type, metrics in metrics_dict.items():
        plt.plot(metrics['test_losses'], label=f"{norm_type}")
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('MAE Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training accuracy
    plt.subplot(3, 2, 3)
    for norm_type, metrics in metrics_dict.items():
        plt.plot(metrics['train_accs'], label=f"{norm_type}")
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('MAE Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot test accuracy
    plt.subplot(3, 2, 4)
    for norm_type, metrics in metrics_dict.items():
        plt.plot(metrics['test_accs'], label=f"{norm_type}")
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('MAE Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot training time
    plt.subplot(3, 2, 5)
    norm_types = list(metrics_dict.keys())
    times = [np.mean(metrics_dict[nt]['epoch_times']) for nt in norm_types]
    plt.bar(norm_types, times)
    plt.ylabel('Average Epoch Time (s)')
    plt.title('Training Time Comparison')
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    # Plot stability (std dev of batch losses in last epoch)
    plt.subplot(3, 2, 6)
    stability = [metrics_dict[nt]['stability'][-1] for nt in norm_types]
    plt.bar(norm_types, stability)
    plt.ylabel('Loss Standard Deviation')
    plt.title('Training Stability (Lower is Better)')
    for i, v in enumerate(stability):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f'mae_comparison_{timestamp}.png')
    print(f"Comparison plot saved as mae_comparison_{timestamp}.png")
    
    # Create a summary table
    summary = {
        'norm_type': [],
        'final_train_loss': [],
        'final_test_loss': [],
        'final_train_acc': [],
        'final_test_acc': [],
        'avg_epoch_time': [],
        'stability': []
    }
    
    for norm_type in norm_types:
        metrics = metrics_dict[norm_type]
        summary['norm_type'].append(norm_type)
        summary['final_train_loss'].append(metrics['train_losses'][-1])
        summary['final_test_loss'].append(metrics['test_losses'][-1])
        summary['final_train_acc'].append(metrics['train_accs'][-1] * 100)  # Convert to percentage
        summary['final_test_acc'].append(metrics['test_accs'][-1] * 100)    # Convert to percentage
        summary['avg_epoch_time'].append(np.mean(metrics['epoch_times']))
        summary['stability'].append(metrics['stability'][-1])
    
    # Print summary table
    print("\n=== MAE COMPARISON SUMMARY ===")
    print(f"{'Norm Type':<10} {'Train Loss':<12} {'Test Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Avg Time/Epoch':<15} {'Stability':<12}")
    print("="*85)
    
    for i in range(len(summary['norm_type'])):
        print(f"{summary['norm_type'][i]:<10} "
              f"{summary['final_train_loss'][i]:<12.6f} "
              f"{summary['final_test_loss'][i]:<12.6f} "
              f"{summary['final_train_acc'][i]:<12.2f}% "
              f"{summary['final_test_acc'][i]:<12.2f}% "
              f"{summary['avg_epoch_time'][i]:<15.2f}s "
              f"{summary['stability'][i]:<12.6f}")
    
    return summary


def run_experiment(args):
    # Configure device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create timestamp for saving files
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Configure data loaders
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # Load Fashion-MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    try:
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, **train_kwargs)
        test_loader = DataLoader(test_dataset, **test_kwargs)
    except Exception as e:
        print(f"Error loading Fashion-MNIST dataset: {str(e)}")
        return
    
    # Configure models and track metrics
    results = {}
    norm_types = ['dyt', 'adyt', 'ln']
    
    # Filter based on norm type if specified
    if args.norm != 'all' and args.norm in norm_types:
        norm_types = [args.norm]
    
    for norm_type in norm_types:
        print(f"\n{'='*50}")
        print(f"Training MAE with {norm_type} normalization")
        print(f"{'='*50}")
        
        # Create model based on normalization type
        model = MAE(
            norm_type=norm_type,
            lambda_factor=args.lambda_factor,
            smooth_factor=args.smooth_factor,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            mask_ratio=args.mask_ratio
        ).to(device)
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Track metrics
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        epoch_times = []
        stability = []
        
        # Training loop
        for epoch in range(1, args.epochs + 1):
            metrics = train(model, device, train_loader, optimizer, epoch, args.log_interval)
            test_loss, test_acc = evaluate(model, device, test_loader)
            
            train_losses.append(metrics['loss'])
            test_losses.append(test_loss)
            train_accs.append(metrics['acc'])
            test_accs.append(test_acc)
            epoch_times.append(metrics['time'])
            stability.append(metrics['stability'])
            
            scheduler.step()
            
            print(f'Epoch {epoch}: '
                  f'Train Loss: {metrics["loss"]:.6f}, '
                  f'Test Loss: {test_loss:.6f}, '
                  f'Train Acc: {metrics["acc"]*100:.2f}%, '
                  f'Test Acc: {test_acc*100:.2f}%, '
                  f'Time: {metrics["time"]:.2f}s, '
                  f'Stability: {metrics["stability"]:.6f}')
            
            # Save per-epoch metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': metrics['loss'],
                'test_loss': test_loss,
                'train_acc': metrics['acc'],
                'test_acc': test_acc,
                'epoch_time': metrics['time'],
                'stability': metrics['stability']
            }
            
            # Save metrics to file
            metrics_file = f'mae_metrics_{norm_type}_{timestamp}.txt'
            with open(metrics_file, 'a') as f:
                if epoch == 1:
                    f.write('Epoch\tTrain Loss\tTest Loss\tTrain Acc\tTest Acc\tTime\tStability\n')
                f.write(f"{epoch}\t{metrics['loss']:.6f}\t{test_loss:.6f}\t{metrics['acc']:.6f}\t{test_acc:.6f}\t{metrics['time']:.2f}\t{metrics['stability']:.6f}\n")
        
        # Save results
        results[norm_type] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'epoch_times': epoch_times,
            'stability': stability,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'final_train_acc': train_accs[-1],
            'final_test_acc': test_accs[-1]
        }
        
        # Save final model state
        model_save_path = f"mae_fashion_mnist_{norm_type}_{timestamp}.pt"
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_losses[-1],
            'test_loss': test_losses[-1],
            'train_acc': train_accs[-1],
            'test_acc': test_accs[-1],
            'args': args
        }, model_save_path)
        print(f"\nSaved model state to {model_save_path}")
    
    # Plot comparison results
    summary = plot_comparison(results)
    
    return results, summary


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MAE with DyT vs ADyT vs LN Comparison on Fashion-MNIST')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='testing batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of training epochs (default: 60)')
    parser.add_argument('--lr', type=float, default=1.5e-4,
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lambda-factor', type=float, default=0.1,
                        help='lambda factor for ADyT (default: 0.1)')
    parser.add_argument('--smooth-factor', type=float, default=0.9,
                        help='smooth factor for ADyT (default: 0.9)')
    parser.add_argument('--alpha-min', type=float, default=0.1,
                        help='minimum value for alpha in ADyT (default: 0.1)')
    parser.add_argument('--alpha-max', type=float, default=2.0,
                        help='maximum value for alpha in ADyT (default: 2.0)')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='mask ratio for MAE (default: 0.75)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging (default: 100)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the trained models')
    parser.add_argument('--norm', type=str, default='all', choices=['dyt', 'adyt', 'ln', 'all'],
                        help='normalization method to use (default: all)')
    
    args = parser.parse_args()
    
    # Run the experiment
    run_experiment(args)


if __name__ == '__main__':
    main() 