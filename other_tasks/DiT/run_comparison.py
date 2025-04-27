#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparison script for DiT models using LayerNorm, DynamicTanh, and AdaptiveDynamicTanh
"""

import os
import sys
import argparse
import time
import json
import shutil
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Make sure the script runs from the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Compare DiT models with different normalization methods")
    
    # Basic options
    parser.add_argument("--data-path", type=str, default="C:/Users/WINGPU/Desktop/DyT_2/other_tasks/DINO/data/ILSVRC/Data/CLS-LOC/train",
                        help="Path to ImageNet training data")
    parser.add_argument("--results-dir", type=str, default="comparison_results",
                        help="Directory to save results")
    parser.add_argument("--model", type=str, default="DiT-B/4", 
                        choices=["DiT-B/4", "DiT-L/4", "DiT-XL/2"],
                        help="DiT model size")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Maximum number of training steps")
                        
    # Method selection
    parser.add_argument("--skip-ln", action="store_true",
                        help="Skip LayerNorm training")
    parser.add_argument("--skip-dyt", action="store_true",
                        help="Skip DynamicTanh training")
    parser.add_argument("--skip-adyt", action="store_true",
                        help="Skip AdaptiveDynamicTanh training")
    
    # ADyT parameters                    
    parser.add_argument("--adyt-lambda", type=float, default=0.1,
                        help="Lambda factor for ADyT")
    parser.add_argument("--adyt-smooth", type=float, default=0.99,
                        help="Smooth factor for ADyT")
    
    # Run on Windows without distributed
    parser.add_argument("--no-distributed", action="store_true", default=True,
                        help="Disable distributed training (for Windows)")
                        
    return parser.parse_args()

def create_experiment_dir(base_dir):
    """Create a timestamped experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"comparison_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def run_training(args, experiment_dir, variant, use_dyt=False, use_adyt=False):
    """Run training for a specific variant"""
    variant_dir = os.path.join(experiment_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)
    
    # Write configuration to the variant directory
    with open(os.path.join(variant_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create train command
    cmd = [
        "python", "DiT/train.py",
        "--data-path", args.data_path,
        "--results-dir", variant_dir,
        "--model", args.model,
        "--global-batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--max-steps", str(args.max_steps),
        "--lr", str(args.lr),
        "--no-distributed"  # For Windows compatibility
    ]
    
    # Add variant-specific flags
    if use_dyt:
        cmd.append("--use-dyt")
    elif use_adyt:
        cmd.append("--use-adyt")
        cmd.extend(["--lambda-factor", str(args.adyt_lambda)])
        cmd.extend(["--smooth-factor", str(args.adyt_smooth)])
    
    # Execute training
    print(f"\n{'='*80}")
    print(f"Running {variant} training with command:")
    print(" ".join(cmd))
    print(f"{'='*80}\n")
    
    log_file = os.path.join(variant_dir, "training.log")
    with open(log_file, "w") as f:
        # Pass the command as a string to support Windows
        process = subprocess.Popen(
            " ".join(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True
        )
        
        # Stream the output to both console and log file
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
            
        process.wait()
        
    if process.returncode != 0:
        print(f"ERROR: {variant} training failed with code {process.returncode}")
        return False
    
    print(f"\n{variant} training completed successfully!")
    return True

def extract_metrics(experiment_dir):
    """Extract metrics from training logs"""
    variants = ["LayerNorm", "DynamicTanh", "AdaptiveDynamicTanh"]
    metrics = {}
    
    for variant in variants:
        variant_dir = os.path.join(experiment_dir, variant)
        if not os.path.exists(variant_dir):
            continue
            
        log_file = os.path.join(variant_dir, "training.log")
        if not os.path.exists(log_file):
            continue
            
        metrics[variant] = {"step": [], "loss": [], "samples_per_sec": []}
        
        with open(log_file, "r") as f:
            for line in f:
                if "Step:" in line and "Loss:" in line and "Samples/Sec:" in line:
                    try:
                        # Extract metrics from the line
                        step_str = line.split("Step:")[1].split("/")[0].strip()
                        loss_str = line.split("Loss:")[1].split("Steps/Sec:")[0].strip()
                        samples_str = line.split("Samples/Sec:")[1].strip()
                        
                        metrics[variant]["step"].append(int(step_str))
                        metrics[variant]["loss"].append(float(loss_str))
                        metrics[variant]["samples_per_sec"].append(float(samples_str))
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing log line: {line}")
                        print(f"Error details: {str(e)}")
    
    return metrics

def create_plots(metrics, experiment_dir):
    """Create comparison plots from training metrics"""
    colors = {
        "LayerNorm": "blue",
        "DynamicTanh": "green",
        "AdaptiveDynamicTanh": "red"
    }
    
    # Create loss comparison plot
    plt.figure(figsize=(10, 6))
    for variant, data in metrics.items():
        if "step" in data and "loss" in data and len(data["step"]) > 0:
            plt.plot(data["step"], data["loss"], label=variant, color=colors[variant])
    
    plt.title("Loss Comparison", fontsize=16)
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(experiment_dir, "loss_comparison.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    
    # Create throughput comparison plot
    plt.figure(figsize=(10, 6))
    for variant, data in metrics.items():
        if "step" in data and "samples_per_sec" in data and len(data["step"]) > 0:
            plt.plot(data["step"], data["samples_per_sec"], label=variant, color=colors[variant])
    
    plt.title("Training Throughput Comparison", fontsize=16)
    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Samples/Second", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    throughput_plot_path = os.path.join(experiment_dir, "throughput_comparison.png")
    plt.savefig(throughput_plot_path, dpi=300)
    plt.close()
    
    # Create final metrics table
    results = {}
    for variant, data in metrics.items():
        if "loss" in data and len(data["loss"]) > 0:
            # Get the final metrics
            final_loss = data["loss"][-1]
            avg_throughput = sum(data["samples_per_sec"]) / len(data["samples_per_sec"])
            
            results[variant] = {
                "final_loss": final_loss,
                "avg_throughput": avg_throughput
            }
    
    # Save results as JSON
    with open(os.path.join(experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Create a results table image
    # Simple table rendering as an image
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    
    table_data = [["Method", "Final Loss", "Avg. Throughput (samples/sec)"]]
    for variant, metrics in results.items():
        table_data.append([
            variant, 
            f"{metrics['final_loss']:.4f}", 
            f"{metrics['avg_throughput']:.1f}"
        ])
    
    table = plt.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    plt.title("DiT Training Results Comparison", fontsize=16, pad=20)
    plt.tight_layout()
    
    table_path = os.path.join(experiment_dir, "results_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return loss_plot_path, throughput_plot_path, table_path

def create_readme(experiment_dir, args, paths):
    """Create a README file with experiment details"""
    loss_plot, throughput_plot, table = paths
    readme_path = os.path.join(experiment_dir, "README.md")
    
    with open(readme_path, "w") as f:
        f.write("# DiT Normalization Methods Comparison\n\n")
        f.write(f"Experiment conducted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Learning Rate: {args.lr}\n")
        f.write(f"- Training Steps: {args.max_steps}\n")
        f.write(f"- ADyT Lambda: {args.adyt_lambda}\n")
        f.write(f"- ADyT Smooth Factor: {args.adyt_smooth}\n\n")
        
        f.write("## Results\n\n")
        f.write("### Loss Comparison\n\n")
        f.write(f"![Loss Comparison](./loss_comparison.png)\n\n")
        
        f.write("### Throughput Comparison\n\n")
        f.write(f"![Throughput Comparison](./throughput_comparison.png)\n\n")
        
        f.write("### Final Results\n\n")
        f.write(f"![Results Table](./results_table.png)\n\n")
        
        # Load and include the detailed results
        try:
            with open(os.path.join(experiment_dir, "results.json"), "r") as results_file:
                results = json.load(results_file)
                f.write("```json\n")
                f.write(json.dumps(results, indent=2))
                f.write("\n```\n")
        except:
            f.write("Detailed results not available.\n")
    
    return readme_path

def main():
    args = parse_args()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(args.results_dir)
    print(f"Created experiment directory: {experiment_dir}")
    
    # Track which variants were successfully trained
    successful_variants = []
    
    # Run LayerNorm training (baseline)
    if not args.skip_ln:
        success = run_training(args, experiment_dir, "LayerNorm")
        if success:
            successful_variants.append("LayerNorm")
    
    # Run DynamicTanh training
    if not args.skip_dyt:
        success = run_training(args, experiment_dir, "DynamicTanh", use_dyt=True)
        if success:
            successful_variants.append("DynamicTanh")
    
    # Run AdaptiveDynamicTanh training
    if not args.skip_adyt:
        success = run_training(args, experiment_dir, "AdaptiveDynamicTanh", use_adyt=True)
        if success:
            successful_variants.append("AdaptiveDynamicTanh")
    
    if not successful_variants:
        print("ERROR: No training variants completed successfully.")
        return
    
    # Extract metrics from logs
    print("Extracting metrics from training logs...")
    metrics = extract_metrics(experiment_dir)
    
    # Create comparison plots and tables
    print("Creating comparison visualizations...")
    plot_paths = create_plots(metrics, experiment_dir)
    
    # Create a README with experiment details
    readme_path = create_readme(experiment_dir, args, plot_paths)
    print(f"Created experiment summary: {readme_path}")
    
    print(f"\nComparison completed! Results saved to: {experiment_dir}")
    print(f"\nSuccessfully trained variants: {', '.join(successful_variants)}")

if __name__ == "__main__":
    main() 