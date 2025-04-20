# DINO with ADyT on CIFAR-10

This guide explains how to run and compare DINO with different normalization layers (LayerNorm, DynamicTanh, and AdaptiveDynamicTanh) on the CIFAR-10 dataset.

## Overview

The test script `test_dino_cifar.py` implements a simplified version of DINO that:

1. Uses the CIFAR-10 dataset (automatically downloaded via torchvision)
2. Uses a smaller ViT architecture (ViT-Tiny with patch size 4)
3. Runs for 10 epochs by default
4. Compares three normalization strategies:
   - Standard LayerNorm (baseline)
   - DynamicTanh (DyT)
   - AdaptiveDynamicTanh (ADyT)
5. Generates comparative plots and metrics

## Prerequisites

Before running the test, you need to:

1. Clone the DINO repository (if you haven't already):
```
git clone https://github.com/facebookresearch/dino.git
```

2. Copy the necessary files to the DINO directory:
```
cp dynamic_tanh.py dino/
cp dynamic_tanh_adaptive.py dino/
cp test_dino_cifar.py dino/
cp run_cifar_test.sh dino/
cp run_cifar_test.bat dino/
```

3. Install the required packages:
```
pip install torch torchvision tqdm matplotlib numpy
```

## Running the Test

### On Linux/macOS:

```
cd dino
chmod +x run_cifar_test.sh
./run_cifar_test.sh
```

### On Windows:

```
cd dino
run_cifar_test.bat
```

## Results

The test will:

1. Train and evaluate DINO with all three normalization types
2. Save per-epoch metrics to text files in `./results/cifar10_dino/`
3. Save model checkpoints for each normalization type
4. Generate individual training curves for each normalization type
5. Create a comparative plot showing all three methods side by side
6. Print a summary table with final metrics

The key output files are:

- `results/cifar10_dino/metrics_ln.txt` - Per-epoch metrics for LayerNorm
- `results/cifar10_dino/metrics_dyt.txt` - Per-epoch metrics for DynamicTanh
- `results/cifar10_dino/metrics_adyt.txt` - Per-epoch metrics for AdaptiveDynamicTanh
- `results/cifar10_dino/dino_cifar10_ln.pt` - LayerNorm model checkpoint
- `results/cifar10_dino/dino_cifar10_dyt.pt` - DynamicTanh model checkpoint
- `results/cifar10_dino/dino_cifar10_adyt.pt` - AdaptiveDynamicTanh model checkpoint
- `results/cifar10_dino/training_curves_*.png` - Individual training curves
- `results/cifar10_dino/comparison.png` - Comparative plots of all three methods

## Customizing the Test

To customize the test parameters, you can edit the `base_args` list in the `run_comparison()` function of `test_dino_cifar.py`. For example:

```python
base_args = [
    "--epochs", "20",            # Increase the number of epochs
    "--batch-size", "128",       # Increase batch size
    "--output-dir", "./results/cifar10_dino",
    "--lr", "0.001",             # Adjust learning rate
    "--momentum-teacher", "0.996",
    "--weight-decay", "0.04",
]
```

For AdaptiveDynamicTanh, you can also customize these parameters:

```python
"--lambda-factor", "0.5",    # Controls adaptivity strength
"--smooth-factor", "0.99",   # EMA smoothing factor
"--alpha-min", "0.1",        # Minimum alpha value
"--alpha-max", "2.0",        # Maximum alpha value
``` 