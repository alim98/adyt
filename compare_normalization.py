import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Import normalization modules
try:
    from dynamic_tanh import DynamicTanh
    from dynamic_tanh_adaptive import AdaptiveDynamicTanh, update_adyt_grad_norms
except ImportError:
    print("Warning: Could not import dynamic_tanh modules. Make sure they are in your path.")
    exit(1)

# Simple CNN Model with configurable normalization
class SimpleCNN(nn.Module):
    def __init__(self, norm_type='ln', lambda_factor=0.1, smooth_factor=0.9):
        super(SimpleCNN, self).__init__()
        self.norm_type = norm_type
        
        # Define the model
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        
        # Select normalization type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(128)
        elif norm_type == 'dyt':
            self.norm = DynamicTanh(128, True)
        elif norm_type == 'adyt':
            self.norm = AdaptiveDynamicTanh(128, True, lambda_factor=lambda_factor, smooth_factor=smooth_factor)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
            
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Simple Vision Transformer for MNIST
class SimpleViT(nn.Module):
    def __init__(self, norm_type='dyt', lambda_factor=0.1, smooth_factor=0.9,
                 img_size=28, patch_size=7, in_chans=1, embed_dim=64, 
                 depth=4, num_heads=4, mlp_ratio=4.0):
        super(SimpleViT, self).__init__()
        self.norm_type = norm_type
        
        # Calculate parameters
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Select normalization type for creating blocks
        if norm_type == 'dyt':
            norm_layer = lambda size: DynamicTanh(size, True)
        elif norm_type == 'adyt':
            norm_layer = lambda size: AdaptiveDynamicTanh(size, True, lambda_factor=lambda_factor, smooth_factor=smooth_factor)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
            
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # MLP head
        self.head = nn.Linear(embed_dim, 10)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Patch embedding [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.patch_embed(x)
        
        # Flatten patches to sequence [B, embed_dim, P, P] -> [B, P*P, embed_dim]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :(H * W + 1)]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Take class token as representation
        x = x[:, 0]
        
        # Apply classifier head
        x = self.head(x)
        
        return F.log_softmax(x, dim=1)

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

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    
    # Track metrics
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # Update gradient norms if using AdaptiveDynamicTanh
        if model.norm_type == 'adyt':
            update_adyt_grad_norms(model)
            
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    train_loss /= len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    
    return test_loss, test_acc

def run_experiment(args):
    # Configure device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Configure data loaders
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, **train_kwargs)
        test_loader = DataLoader(test_dataset, **test_kwargs)
    except Exception as e:
        print(f"Error loading MNIST dataset: {str(e)}")
        return
    
    # Configure ViT models with different normalization types
    results = {}
    norm_types = ['dyt', 'adyt']
    
    # Filter based on norm type if specified
    if args.norm != 'all' and args.norm in norm_types:
        norm_types = [args.norm]
    
    # Track lambda factors for ADyT
    if args.lambda_sweep:
        lambda_factors = [0.05, 0.1, 0.2, 0.5]
        # If specific lambda provided, use that instead
        if args.norm == 'adyt' and args.lambda_factor > 0:
            lambda_factors = [args.lambda_factor]
        
        for lambda_factor in lambda_factors:
            print(f"\n{'='*50}")
            print(f"Training ViT with ADyT normalization (lambda={lambda_factor})")
            print(f"{'='*50}")
            
            model = SimpleViT(
                norm_type='adyt',
                lambda_factor=lambda_factor,
                smooth_factor=args.smooth_factor
            ).to(device)
            
            # Create optimizer and scheduler
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
            
            # Track metrics
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            
            # Training loop
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train(
                    model, device, train_loader, optimizer, epoch, args.log_interval
                )
                test_loss, test_acc = test(model, device, test_loader)
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                
                scheduler.step()
            
            # Save results
            results[f'adyt_lambda_{lambda_factor}'] = {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_accs': test_accs,
                'final_test_acc': test_accs[-1]
            }
            
            # Save model if requested
            if args.save_model:
                torch.save(model.state_dict(), f"mnist_vit_adyt_lambda_{lambda_factor}.pt")
    else:
        # Standard comparison between DyT and ADyT
        for norm_type in norm_types:
            print(f"\n{'='*50}")
            print(f"Training ViT with {norm_type.upper()} normalization")
            print(f"{'='*50}")
            
            # Create model
            model = SimpleViT(
                norm_type=norm_type,
                lambda_factor=args.lambda_factor,
                smooth_factor=args.smooth_factor
            ).to(device)
            
            # Create optimizer and scheduler
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
            
            # Track metrics
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            
            # Training loop
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train(
                    model, device, train_loader, optimizer, epoch, args.log_interval
                )
                test_loss, test_acc = test(model, device, test_loader)
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                
                scheduler.step()
            
            # Save results
            results[norm_type] = {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_accs': test_accs,
                'final_test_acc': test_accs[-1]
            }
            
            # Save model if requested
            if args.save_model:
                torch.save(model.state_dict(), f"mnist_vit_{norm_type}.pt")
    
    # Plot results
    plot_results(results, 'vit')
    
    # Print summary
    print("\n\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for norm_type, metrics in results.items():
        print(f"  {norm_type.upper()}: {metrics['final_test_acc']:.2f}%")
    
    return results

def plot_results(results, model_name):
    """Plot training curves and performance comparison"""
    # Training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    for norm_type, data in results.items():
        ax1.plot(data['train_losses'], label=f"{norm_type}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{model_name.upper()} Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training accuracy
    for norm_type, data in results.items():
        ax2.plot(data['train_accs'], label=f"{norm_type}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy (%)')
    ax2.set_title(f'{model_name.upper()} Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot test loss
    for norm_type, data in results.items():
        ax3.plot(data['test_losses'], label=f"{norm_type}")
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Loss')
    ax3.set_title(f'{model_name.upper()} Test Loss')
    ax3.legend()
    ax3.grid(True)
    
    # Plot test accuracy
    for norm_type, data in results.items():
        ax4.plot(data['test_accs'], label=f"{norm_type}")
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title(f'{model_name.upper()} Test Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png')
    print(f"Plot saved to {model_name}_training_curves.png")
    
    # Bar chart of final performance
    plt.figure(figsize=(10, 6))
    norm_types = list(results.keys())
    final_test_accs = [results[nt]['final_test_acc'] for nt in norm_types]
    
    plt.bar(norm_types, final_test_accs)
    plt.ylabel('Final Test Accuracy (%)')
    plt.title(f'{model_name.upper()} Final Performance Comparison')
    plt.ylim(min(final_test_accs) - 2, 100)
    
    # Add values on top of bars
    for i, v in enumerate(final_test_accs):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    plt.savefig(f'{model_name}_final_performance.png')
    print(f"Performance comparison saved to {model_name}_final_performance.png")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DyT vs ADyT Comparison on MNIST with ViT')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='testing batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='learning rate decay factor (default: 0.7)')
    parser.add_argument('--lambda-factor', type=float, default=0.1,
                        help='lambda factor for ADyT (default: 0.1)')
    parser.add_argument('--smooth-factor', type=float, default=0.9,
                        help='smooth factor for ADyT (default: 0.9)')
    parser.add_argument('--lambda-sweep', action='store_true', default=False,
                        help='run experiments with different lambda factors')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging (default: 100)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the trained models')
    parser.add_argument('--norm', type=str, default='all', choices=['dyt', 'adyt', 'all'],
                        help='normalization method to use (default: all)')
    
    args = parser.parse_args()
    
    # Run the experiment
    run_experiment(args)

if __name__ == '__main__':
    main() 