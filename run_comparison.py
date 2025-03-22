#!/usr/bin/env python
import os
import subprocess
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def run_command(cmd):
    """Run a command and return its output"""
    print(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Stream the output as it comes
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def extract_metrics(log_file):
    """Extract training and validation metrics from log file"""
    epochs, train_loss, test_loss, test_acc1, test_acc5 = [], [], [], [], []
    
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found!")
        return epochs, train_loss, test_loss, test_acc1, test_acc5
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                epochs.append(data['epoch'])
                
                # Training metrics
                if 'train_loss' in data:
                    train_loss.append(data['train_loss'])
                
                # Testing metrics
                if 'test_loss' in data:
                    test_loss.append(data['test_loss'])
                if 'test_acc1' in data:
                    test_acc1.append(data['test_acc1'])
                if 'test_acc5' in data:
                    test_acc5.append(data['test_acc5'])
            except json.JSONDecodeError:
                continue
    
    return epochs, train_loss, test_loss, test_acc1, test_acc5

def plot_comparison(output_dirs, save_dir):
    """Generate comparison plots for the three models"""
    plt.figure(figsize=(20, 15))
    
    # Define colors and labels
    colors = ['blue', 'red', 'green']
    labels = ['LayerNorm (LN)', 'DynamicTanh (DyT)', 'AdaptiveDynamicTanh (ADyT)']
    
    # Extract metrics for each model
    all_epochs, all_train_loss, all_test_loss, all_test_acc1, all_test_acc5 = [], [], [], [], []
    
    for i, output_dir in enumerate(output_dirs):
        log_file = os.path.join(output_dir, 'log.txt')
        epochs, train_loss, test_loss, test_acc1, test_acc5 = extract_metrics(log_file)
        
        all_epochs.append(epochs)
        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        all_test_acc1.append(test_acc1)
        all_test_acc5.append(test_acc5)
    
    # Plot Training Loss
    plt.subplot(2, 2, 1)
    for i in range(len(output_dirs)):
        if all_train_loss[i]:
            plt.plot(all_epochs[i], all_train_loss[i], color=colors[i], label=labels[i])
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Test Loss
    plt.subplot(2, 2, 2)
    for i in range(len(output_dirs)):
        if all_test_loss[i]:
            plt.plot(all_epochs[i], all_test_loss[i], color=colors[i], label=labels[i])
    plt.title('Test Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Test Accuracy (Top-1)
    plt.subplot(2, 2, 3)
    for i in range(len(output_dirs)):
        if all_test_acc1[i]:
            plt.plot(all_epochs[i], all_test_acc1[i], color=colors[i], label=labels[i])
    plt.title('Test Accuracy (Top-1)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Test Accuracy (Top-5)
    plt.subplot(2, 2, 4)
    for i in range(len(output_dirs)):
        if all_test_acc5[i]:
            plt.plot(all_epochs[i], all_test_acc5[i], color=colors[i], label=labels[i])
    plt.title('Test Accuracy (Top-5)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add timestamp to the plot
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.5, 0.01, f"Generated on {timestamp}", ha='center', fontsize=10)
    
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'performance_comparison.png')
    plt.savefig(plot_path)
    print(f"Performance comparison plot saved to {plot_path}")
    
    # Generate a summary table
    summary_table(output_dirs, labels, all_epochs, all_test_acc1, all_test_acc5, save_dir)

def summary_table(output_dirs, labels, all_epochs, all_test_acc1, all_test_acc5, save_dir):
    """Generate a summary table with best and final metrics"""
    summary_data = []
    
    for i in range(len(output_dirs)):
        if all_test_acc1[i]:
            best_acc1 = max(all_test_acc1[i])
            best_epoch = all_epochs[i][all_test_acc1[i].index(best_acc1)]
            
            final_acc1 = all_test_acc1[i][-1] if all_test_acc1[i] else "N/A"
            final_epoch = all_epochs[i][-1] if all_epochs[i] else "N/A"
            
            summary_data.append({
                'Model': labels[i],
                'Best Top-1 Acc': f"{best_acc1:.2f}%",
                'Best Epoch': best_epoch,
                'Final Top-1 Acc': f"{final_acc1:.2f}%" if isinstance(final_acc1, (int, float)) else final_acc1,
                'Final Epoch': final_epoch
            })
    
    # Create summary file
    summary_path = os.path.join(save_dir, 'performance_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Performance Summary\n")
        f.write("==================\n\n")
        
        # Write table header
        f.write(f"{'Model':<25} {'Best Top-1 Acc':<15} {'Best Epoch':<15} {'Final Top-1 Acc':<15} {'Final Epoch':<15}\n")
        f.write(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15} {'-'*15}\n")
        
        # Write table rows
        for data in summary_data:
            f.write(f"{data['Model']:<25} {data['Best Top-1 Acc']:<15} {data['Best Epoch']:<15} {data['Final Top-1 Acc']:<15} {data['Final Epoch']:<15}\n")
    
    print(f"Performance summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Run experiments comparing LN, DyT, and ADyT")
    parser.add_argument('--model', default='vit_base_patch16_224', help='Model to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=4e-3, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs')
    parser.add_argument('--data_set', default='CIFAR', help='Dataset to use (CIFAR or IMNET)')
    parser.add_argument('--data_path', default='./data', help='Path to dataset')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers')
    parser.add_argument('--output_dir', default='./results', help='Base output directory')
    parser.add_argument('--warmup_epochs', default=20, type=int, help='Number of warmup epochs')
    parser.add_argument('--adyt_lambda', default=0.5, type=float, help='Lambda factor for ADyT')
    parser.add_argument('--adyt_smooth', default=0.99, type=float, help='Smoothing factor for ADyT')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoints if available')
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{args.output_dir}/{args.model}_{timestamp}"
    
    # Define output directories for each model
    output_dirs = [
        f"{base_output_dir}/ln",
        f"{base_output_dir}/dyt",
        f"{base_output_dir}/adyt"
    ]
    
    # Create output directories
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set environment variable for GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Define commands for each model
    commands = [
        f"python main.py --model {args.model} --batch_size {args.batch_size} --lr {args.lr} --norm_type ln --output_dir {output_dirs[0]} --data_set {args.data_set} --data_path {args.data_path} --epochs {args.epochs} --warmup_epochs {args.warmup_epochs} --num_workers {args.num_workers}",
        f"python main.py --model {args.model} --batch_size {args.batch_size} --lr {args.lr} --norm_type dyt --output_dir {output_dirs[1]} --data_set {args.data_set} --data_path {args.data_path} --epochs {args.epochs} --warmup_epochs {args.warmup_epochs} --num_workers {args.num_workers}",
        f"python main.py --model {args.model} --batch_size {args.batch_size} --lr {args.lr} --norm_type adyt --output_dir {output_dirs[2]} --data_set {args.data_set} --data_path {args.data_path} --epochs {args.epochs} --warmup_epochs {args.warmup_epochs} --num_workers {args.num_workers} --adyt_lambda {args.adyt_lambda} --adyt_smooth {args.adyt_smooth}"
    ]
    
    # Add resume flag if needed
    if args.resume:
        commands = [cmd + " --resume" for cmd in commands]
    
    # Save configuration
    config_path = os.path.join(base_output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Experiment configuration saved to {config_path}")
    
    # Run each model training
    for i, cmd in enumerate(commands):
        model_name = ["LayerNorm", "DynamicTanh", "AdaptiveDynamicTanh"][i]
        print(f"\n\n{'='*80}")
        print(f"Starting training for {model_name}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        run_command(cmd)
        end_time = time.time()
        
        hours, remainder = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n\n{'='*80}")
        print(f"Finished training {model_name} in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"{'='*80}\n")
    
    # Generate comparison plots
    plot_comparison(output_dirs, base_output_dir)
    
    print(f"\n\nAll experiments completed successfully!")
    print(f"Results saved to {base_output_dir}")

if __name__ == "__main__":
    main() 