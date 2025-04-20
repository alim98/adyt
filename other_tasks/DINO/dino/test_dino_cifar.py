import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Add the parent directory to the path to import DINO modules
sys.path.append('dino')  # Adjust this path to point to your DINO installation

try:
    from vision_transformer import vit_small, vit_tiny
    from vision_transformer import DINOHead
    from utils import MultiCropWrapper
    from dynamic_tanh import DynamicTanh, convert_ln_to_dyt
    from dynamic_tanh_adaptive import AdaptiveDynamicTanh, convert_ln_to_adyt, update_adyt_grad_norms
except ImportError:
    print("Error importing DINO modules. Please make sure 'dino' directory is in your path")
    print("and the necessary modules are available.")
    sys.exit(1)

class DINO_Loss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks
        """
        student_out = student_output / self.student_temp
        teacher_out = teacher_output / self.teacher_temp
        
        # Teacher softmax with centering and sharpening
        teacher_out = torch.softmax(teacher_out, dim=-1).detach()
        
        # Student softmax
        student_out = torch.log_softmax(student_out, dim=-1)
        
        # Calculate cross entropy loss
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        
        # Update center for teacher output (non-distributed version)
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)  # Average over batch
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
        return loss

def get_transforms():
    """Define data transforms for CIFAR-10"""
    # Basic augmentation for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return transform_train, transform_test

def get_datasets(transform_train, transform_test):
    """Load CIFAR-10 datasets"""
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    return trainset, testset

def get_dataloaders(trainset, testset, batch_size, num_workers=2):
    """Create data loaders for training and testing"""
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def train_epoch(student, teacher, trainloader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    student.train()
    teacher.eval()  # Teacher doesn't need gradients
    
    running_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(trainloader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, _) in enumerate(pbar):
        # Move data to device
        inputs = inputs.to(device)
        
        # Forward pass
        student_output = student(inputs)
        with torch.no_grad():
            teacher_output = teacher(inputs)
        
        # Compute loss
        loss = criterion(student_output, teacher_output)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Update gradient norms for ADyT if used
        if args.norm_type == 'adyt':
            update_adyt_grad_norms(student)
        
        optimizer.step()
        
        # Update running statistics
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        # Update teacher weights
        with torch.no_grad():
            m = args.momentum_teacher  # Momentum parameter for EMA
            for param_student, param_teacher in zip(student.parameters(), teacher.parameters()):
                param_teacher.data.mul_(m).add_((1 - m) * param_student.data)
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / total_samples})
    
    return running_loss / total_samples

def evaluate(student, testloader, device):
    """Evaluate model on test set"""
    student.eval()
    
    # For now, we're just computing loss on test set, not accuracy
    # since DINO is self-supervised and doesn't have labels
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            outputs = student(inputs)
            
            # For simplicity, we'll just compute L2 norm as a proxy for quality
            loss = torch.norm(outputs, dim=1).mean()
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    return running_loss / total_samples

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test DINO with different normalization layers on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
    parser.add_argument('--norm-type', type=str, default='ln', choices=['ln', 'dyt', 'adyt'], 
                        help='type of normalization layer to use')
    parser.add_argument('--lambda-factor', type=float, default=0.5, help='lambda factor for ADyT')
    parser.add_argument('--smooth-factor', type=float, default=0.99, help='smooth factor for ADyT')
    parser.add_argument('--alpha-min', type=float, default=0.1, help='minimum alpha for ADyT')
    parser.add_argument('--alpha-max', type=float, default=2.0, help='maximum alpha for ADyT')
    parser.add_argument('--output-dir', type=str, default='./results', help='output directory')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--momentum-teacher', type=float, default=0.996, help='teacher momentum for EMA')
    parser.add_argument('--weight-decay', type=float, default=0.04, help='weight decay')
    parser.add_argument('--num-workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    args = parser.parse_args()
    
    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data transforms and datasets
    transform_train, transform_test = get_transforms()
    trainset, testset = get_datasets(transform_train, transform_test)
    
    # Create data loaders
    trainloader, testloader = get_dataloaders(
        trainset, testset, args.batch_size, args.num_workers)
    
    print(f"Dataset: CIFAR-10 - {len(trainset)} training samples, {len(testset)} test samples")
    
    # Create model
    print(f"Creating DINO model with {args.norm_type} normalization...")
    
    # Create base model - using vit_tiny for faster training
    student = vit_tiny(patch_size=4, img_size=[32, 32])  # Smaller patch size for CIFAR-10
    teacher = vit_tiny(patch_size=4, img_size=[32, 32])
    
    # Apply normalization type
    if args.norm_type == 'dyt':
        print("Converting LayerNorm to DynamicTanh...")
        student = convert_ln_to_dyt(student)
        teacher = convert_ln_to_dyt(teacher)
    elif args.norm_type == 'adyt':
        print(f"Converting LayerNorm to AdaptiveDynamicTanh (λ={args.lambda_factor}, β={args.smooth_factor})...")
        student = convert_ln_to_adyt(
            student, args.lambda_factor, args.smooth_factor, args.alpha_min, args.alpha_max)
        teacher = convert_ln_to_adyt(
            teacher, args.lambda_factor, args.smooth_factor, args.alpha_min, args.alpha_max)
    
    # Initialize student and teacher weights
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Add DINO head
    embed_dim = student.embed_dim
    student = MultiCropWrapper(
        student, 
        DINOHead(embed_dim, out_dim=65536, use_bn=False, norm_last_layer=True)
    )
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, out_dim=65536, use_bn=False, norm_last_layer=True)
    )
    
    # Initialize teacher weights with student weights
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.copy_(p_s.data)
    
    # Move models to device
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Setup criterion and optimizer
    criterion = DINO_Loss(out_dim=65536, teacher_temp=0.04, student_temp=0.1)
    criterion = criterion.to(device)
    
    optimizer = optim.AdamW(
        student.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Save model architecture summary
    with open(os.path.join(args.output_dir, f"model_summary_{args.norm_type}.txt"), 'w') as f:
        f.write(f"DINO Model with {args.norm_type} normalization\n")
        f.write(str(student))
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    train_losses = []
    test_losses = []
    times = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            student, teacher, trainloader, criterion, optimizer, device, epoch, args)
        
        # Evaluate
        test_loss = evaluate(student, testloader, device)
        
        # Record time
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        
        # Save metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save metrics to file
        with open(os.path.join(args.output_dir, f"metrics_{args.norm_type}.txt"), 'a') as f:
            if epoch == 1:
                f.write("Epoch\tTrain Loss\tTest Loss\tTime(s)\n")
            f.write(f"{epoch}\t{train_loss:.6f}\t{test_loss:.6f}\t{epoch_time:.2f}\n")
    
    # Save model
    torch.save({
        'epoch': args.epochs,
        'student_state_dict': student.state_dict(),
        'teacher_state_dict': teacher.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'args': args,
    }, os.path.join(args.output_dir, f"dino_cifar10_{args.norm_type}.pt"))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, args.epochs + 1), test_losses, 'r-', label='Test Loss')
    plt.title(f'Loss Curves ({args.norm_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(1, args.epochs + 1), times)
    plt.title(f'Training Time per Epoch ({args.norm_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"training_curves_{args.norm_type}.png"))
    
    print(f"Training completed. Model and metrics saved to {args.output_dir}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'times': times,
        'avg_time': np.mean(times),
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
    }

def run_comparison():
    """Run comparison of all three normalization types"""
    print("Running DINO comparison on CIFAR-10 with LN, DyT, and ADyT...")
    
    # Common arguments
    base_args = [
        "--epochs", "5",
        "--batch-size", "16",
        "--output-dir", "./results/cifar10_dino",
        "--lr", "0.0005",
        "--momentum-teacher", "0.996",
        "--weight-decay", "0.04",
    ]
    
    # Results dictionary
    results = {}
    
    # Run with LayerNorm (baseline)
    print("\n" + "="*50)
    print("Running DINO with LayerNorm...")
    print("="*50)
    sys.argv = ["test_dino_cifar.py"] + base_args + ["--norm-type", "ln"]
    results['ln'] = main()
    
    # Run with DynamicTanh
    print("\n" + "="*50)
    print("Running DINO with DynamicTanh...")
    print("="*50)
    sys.argv = ["test_dino_cifar.py"] + base_args + ["--norm-type", "dyt"]
    results['dyt'] = main()
    
    # Run with AdaptiveDynamicTanh
    print("\n" + "="*50)
    print("Running DINO with AdaptiveDynamicTanh...")
    print("="*50)
    sys.argv = ["test_dino_cifar.py"] + base_args + [
        "--norm-type", "adyt",
        "--lambda-factor", "0.5",
        "--smooth-factor", "0.99",
        "--alpha-min", "0.1",
        "--alpha-max", "2.0",
    ]
    results['adyt'] = main()
    
    # Create comparison plots
    create_comparison_plots(results)
    
    return results

def create_comparison_plots(results):
    """Create plots comparing the different normalization types"""
    norm_types = list(results.keys())
    epochs = len(results[norm_types[0]]['train_losses'])
    
    # Create comparison figure
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for norm_type in norm_types:
        plt.plot(range(1, epochs + 1), results[norm_type]['train_losses'], label=norm_type)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot test loss
    plt.subplot(2, 2, 2)
    for norm_type in norm_types:
        plt.plot(range(1, epochs + 1), results[norm_type]['test_losses'], label=norm_type)
    plt.title('Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot average epoch time
    plt.subplot(2, 2, 3)
    avg_times = [results[nt]['avg_time'] for nt in norm_types]
    plt.bar(norm_types, avg_times)
    plt.title('Average Epoch Time')
    plt.xlabel('Normalization Type')
    plt.ylabel('Time (s)')
    for i, v in enumerate(avg_times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    plt.grid(True)
    
    # Plot final loss comparison
    plt.subplot(2, 2, 4)
    final_train_losses = [results[nt]['final_train_loss'] for nt in norm_types]
    final_test_losses = [results[nt]['final_test_loss'] for nt in norm_types]
    
    x = np.arange(len(norm_types))
    width = 0.35
    
    plt.bar(x - width/2, final_train_losses, width, label='Train')
    plt.bar(x + width/2, final_test_losses, width, label='Test')
    
    plt.title('Final Loss Comparison')
    plt.xlabel('Normalization Type')
    plt.ylabel('Loss')
    plt.xticks(x, norm_types)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("./results/cifar10_dino/comparison.png")
    
    # Print summary table
    print("\n=== DINO COMPARISON SUMMARY ===")
    print(f"{'Norm Type':<10} {'Final Train Loss':<20} {'Final Test Loss':<20} {'Avg Time/Epoch':<20}")
    print("="*70)
    
    for norm_type in norm_types:
        print(f"{norm_type:<10} {results[norm_type]['final_train_loss']:<20.6f} "
              f"{results[norm_type]['final_test_loss']:<20.6f} "
              f"{results[norm_type]['avg_time']:<20.2f}s")
    
    print(f"\nComparison plot saved as ./results/cifar10_dino/comparison.png")

if __name__ == "__main__":
    run_comparison() 