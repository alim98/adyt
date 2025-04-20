import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from dynamic_tanh import DynamicTanh, convert_ln_to_dyt
    from dynamic_tanh_adaptive import AdaptiveDynamicTanh, convert_ln_to_adyt, update_adyt_grad_norms
except ImportError:
    print("Warning: Could not import dynamic_tanh modules")

class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking normalization methods"""
    def __init__(self, norm_layer='ln', hidden_dim=1024, depth=4):
        super().__init__()
        self.norm_type = norm_layer
        
        # First layer
        layers = [nn.Linear(784, hidden_dim)]
        
        # Middle layers with normalization
        for i in range(depth - 2):
            if norm_layer == 'ln':
                layers.append(nn.LayerNorm(hidden_dim))
            elif norm_layer == 'dyt':
                layers.append(DynamicTanh(hidden_dim, True))
            elif norm_layer == 'adyt':
                layers.append(AdaptiveDynamicTanh(hidden_dim, True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Last layer
        layers.append(nn.Linear(hidden_dim, 10))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class VisionTransformer(nn.Module):
    """Simplified Vision Transformer for benchmarking normalization methods"""
    def __init__(self, norm_layer='ln', embed_dim=256, num_layers=6, num_heads=8):
        super().__init__()
        self.norm_type = norm_layer
        
        # Embedding layer
        self.embed = nn.Linear(784, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = {}
            
            # Pre-norm
            if norm_layer == 'ln':
                block['norm1'] = nn.LayerNorm(embed_dim)
            elif norm_layer == 'dyt':
                block['norm1'] = DynamicTanh(embed_dim, True)
            elif norm_layer == 'adyt':
                block['norm1'] = AdaptiveDynamicTanh(embed_dim, True)
                
            # Attention
            block['attn'] = nn.MultiheadAttention(embed_dim, num_heads)
            
            # Post-norm
            if norm_layer == 'ln':
                block['norm2'] = nn.LayerNorm(embed_dim)
            elif norm_layer == 'dyt':
                block['norm2'] = DynamicTanh(embed_dim, True)
            elif norm_layer == 'adyt':
                block['norm2'] = AdaptiveDynamicTanh(embed_dim, True)
                
            # MLP
            block['mlp'] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            
            self.blocks.append(nn.ModuleDict(block))
        
        # Final norm and projection
        if norm_layer == 'ln':
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == 'dyt':
            self.norm = DynamicTanh(embed_dim, True)
        elif norm_layer == 'adyt':
            self.norm = AdaptiveDynamicTanh(embed_dim, True)
            
        self.head = nn.Linear(embed_dim, 10)
    
    def forward(self, x):
        # Reshape from [B, 784] to [B, 784//16, 16] for attention
        B = x.shape[0]
        seq_len = 49  # 784 / 16
        token_dim = 16
        
        # Project to embedding dimension
        x = self.embed(x)
        
        # Reshape for attention: [B, L, D] where L=seq_len
        x = x.view(B, seq_len, -1)
        
        # Apply transformer blocks
        for block in self.blocks:
            # Self-attention with pre-norm
            residual = x
            x = block['norm1'](x)
            x_q = x.transpose(0, 1)  # [L, B, D]
            x_kv = x.transpose(0, 1)  # [L, B, D]
            x_attn, _ = block['attn'](x_q, x_kv, x_kv)
            x_attn = x_attn.transpose(0, 1)  # [B, L, D]
            x = residual + x_attn
            
            # MLP with pre-norm
            residual = x
            x = block['norm2'](x)
            x = residual + block['mlp'](x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Final norm and projection
        x = self.norm(x)
        x = self.head(x)
        
        return x

def load_mnist():
    """Load MNIST dataset for benchmarking"""
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        return train_loader, test_loader
    except ImportError:
        print("Warning: torchvision is required to load MNIST")
        return None, None

def benchmark_speed(model_class, norm_type, device, input_size, warmup=10, repeat=100):
    """Benchmark forward and backward pass speed"""
    model = model_class(norm_layer=norm_type).to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(128, input_size, device=device)
    
    # Warmup
    for _ in range(warmup):
        model(x)
    
    # Measure forward pass time
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    for _ in range(repeat):
        model(x)
    torch.cuda.synchronize() if device == 'cuda' else None
    forward_time = time.time() - start_time
    
    # Measure backward pass time
    model.train()
    y = torch.randint(0, 10, (128,), device=device)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(warmup):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
    
    # Measure
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    for _ in range(repeat):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        if norm_type == 'adyt':
            update_adyt_grad_norms(model)
    torch.cuda.synchronize() if device == 'cuda' else None
    backward_time = time.time() - start_time
    
    return forward_time / repeat, backward_time / repeat

def benchmark_accuracy(model_class, norm_type, device, train_loader, test_loader, epochs=5):
    """Benchmark training and test accuracy"""
    model = model_class(norm_layer=norm_type).to(device)
    model.train()
    
    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            if norm_type == 'adyt':
                update_adyt_grad_norms(model)
                
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        train_loss = epoch_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        test_acc = 100. * correct / total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1]
    }

def plot_results(results, model_name):
    """Plot training curves and performance comparison"""
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training loss
    for norm_type, data in results.items():
        ax1.plot(data['train_losses'], label=f"{norm_type}")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{model_name} Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot test accuracy
    for norm_type, data in results.items():
        ax2.plot(data['test_accs'], label=f"{norm_type}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title(f'{model_name} Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png')
    
    # Bar chart of final performance
    plt.figure(figsize=(10, 6))
    norm_types = list(results.keys())
    final_test_accs = [results[nt]['final_test_acc'] for nt in norm_types]
    
    plt.bar(norm_types, final_test_accs)
    plt.ylabel('Final Test Accuracy (%)')
    plt.title(f'{model_name} Final Performance Comparison')
    plt.ylim(min(final_test_accs) - 2, 100)
    
    # Add values on top of bars
    for i, v in enumerate(final_test_accs):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    plt.savefig(f'{model_name}_final_performance.png')

def run_benchmarks():
    parser = argparse.ArgumentParser(description='Benchmark AdaptiveDynamicTanh (ADyT)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run benchmarks on')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'transformer', 'both'],
                        help='Model architecture to benchmark')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for accuracy benchmark')
    parser.add_argument('--speed_only', action='store_true', help='Only run speed benchmarks')
    parser.add_argument('--accuracy_only', action='store_true', help='Only run accuracy benchmarks')
    
    args = parser.parse_args()
    
    # Load MNIST dataset
    train_loader, test_loader = load_mnist()
    
    # Define models to benchmark
    models = []
    if args.model in ['mlp', 'both']:
        models.append(('MLP', SimpleMLP, 784))
    if args.model in ['transformer', 'both']:
        models.append(('Transformer', VisionTransformer, 784))
    
    # Normalization methods to benchmark
    norm_types = ['ln', 'dyt', 'adyt']
    
    # Speed benchmarks
    if not args.accuracy_only:
        print("\n=== Speed Benchmark ===")
        for model_name, model_class, input_size in models:
            print(f"\nBenchmarking {model_name}...")
            speed_results = {}
            
            for norm_type in norm_types:
                print(f"  Testing {norm_type.upper()}...")
                forward_time, backward_time = benchmark_speed(
                    model_class, norm_type, args.device, input_size)
                speed_results[norm_type] = {
                    'forward': forward_time * 1000,  # ms
                    'backward': backward_time * 1000  # ms
                }
                print(f"    Forward: {forward_time*1000:.3f} ms")
                print(f"    Backward: {backward_time*1000:.3f} ms")
                print(f"    Total: {(forward_time+backward_time)*1000:.3f} ms")
            
            # Create speed comparison table
            print("\nSpeed Comparison (ms):")
            print(f"{'Method':<8} {'Forward':<10} {'Backward':<10} {'Total':<10}")
            print("-" * 40)
            for norm_type, times in speed_results.items():
                print(f"{norm_type.upper():<8} {times['forward']:<10.3f} {times['backward']:<10.3f} {times['forward'] + times['backward']:<10.3f}")
    
    # Accuracy benchmarks
    if not args.speed_only and train_loader is not None and test_loader is not None:
        print("\n=== Accuracy Benchmark ===")
        for model_name, model_class, _ in models:
            print(f"\nBenchmarking {model_name}...")
            accuracy_results = {}
            
            for norm_type in norm_types:
                print(f"  Training with {norm_type.upper()}...")
                accuracy_results[norm_type] = benchmark_accuracy(
                    model_class, norm_type, args.device, train_loader, test_loader, args.epochs)
            
            # Plot results
            plot_results(accuracy_results, model_name)
            
            # Print final results
            print("\nFinal Test Accuracy:")
            for norm_type, data in accuracy_results.items():
                print(f"  {norm_type.upper()}: {data['final_test_acc']:.2f}%")

if __name__ == "__main__":
    run_benchmarks() 