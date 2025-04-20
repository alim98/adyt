# Comparing LayerNorm, DynamicTanh (DyT), and AdaptiveDynamicTanh (ADyT) for DiT

This directory contains scripts to help you train, compare, and visualize the performance of three different normalization methods on Diffusion Transformers (DiT):

1. **LayerNorm (LN)**: The baseline normalization used in the original DiT paper
2. **DynamicTanh (DyT)**: Our proposed alternative that replaces LayerNorm with a learnable tanh-based activation
3. **AdaptiveDynamicTanh (ADyT)**: An enhanced version of DyT that adapts the alpha parameter during training based on gradient information

## Prerequisites

Before running the comparison script, make sure you have:

1. Cloned the DiT repository to `other_tasks/DiT/DiT`:
   ```
   git clone https://github.com/facebookresearch/DiT.git other_tasks/DiT/DiT
   ```

2. Applied all necessary patches as described in the main README.md:
   ```
   # Apply learning rate fix
   cp learning-rate-fix.patch DiT
   cd DiT
   git apply learning-rate-fix.patch
   
   # Apply DynamicTanh patch (needed for both DyT and ADyT)
   cp dynamic_tanh.py DiT
   cp dynamic-tanh.patch DiT
   git apply dynamic-tanh.patch
   
   # Apply DynamicTanh flag fix (adds --use-dyt CLI option)
   cp dynamic-tanh-fix.patch DiT
   git apply dynamic-tanh-fix.patch
   
   # Apply AdaptiveDynamicTanh patch
   cp adaptive_dynamic_tanh.py DiT
   cp adaptive-dynamic-tanh.patch DiT
   git apply adaptive-dynamic-tanh.patch
   
   # Apply PyTorch datasets patch (allows using torchvision datasets)
   cp torch-datasets.patch DiT
   git apply torch-datasets.patch
   ```

3. Set up the Python environment with necessary dependencies (see main README.md)

4. Prepared your ImageNet data for training OR enabled PyTorch datasets mode (see below)

## Using the Comparison Script

The `compare_methods.py` script trains DiT models with each normalization method and generates comprehensive visualizations comparing their performance.

### Basic Usage

```bash
python compare_methods.py --data-path /path/to/imagenet/train --results-dir ./comparison_results
```

This will:
1. Train DiT-B/4 with each of the three normalization methods (LN, DyT, ADyT)
2. Run 5000 training steps for each method
3. Save checkpoints for each method
4. Generate visualizations comparing their performance
5. Save all results to a timestamped directory in `./comparison_results`

### Using PyTorch Datasets

If you don't have a local copy of ImageNet, you can use PyTorch's dataset functionality:

```bash
python compare_methods.py --use-torch-datasets --dataset-download-dir ./datasets --results-dir ./comparison_results
```

This will:
1. Automatically download and use ImageNet from torchvision.datasets (or CIFAR-100 as a fallback if ImageNet download fails)
2. Store the downloaded dataset in the specified directory
3. Train the models using this dataset

For faster testing, you can use a subset of the dataset:
```bash
python compare_methods.py --use-torch-datasets --use-subset --subset-size 10000
```

### Customizing the Comparison

You can customize the comparison with various command-line arguments:

```bash
python compare_methods.py \
  --data-path /path/to/imagenet/train \
  --results-dir ./my_results \
  --model DiT-L/4 \
  --lr 2e-4 \
  --batch-size 32 \
  --num-gpus 4 \
  --steps 10000 \
  --sample \
  --adyt-lambda 0.2 \
  --adyt-smooth 0.95
```

### Command-line Arguments

#### Basic Options:
- `--dit-dir`: Path to the DiT repository (default: "DiT")
- `--results-dir`: Directory to save results (default: "comparison_results")
- `--model`: DiT model size (choices: "DiT-B/4", "DiT-L/4", "DiT-XL/2", default: "DiT-B/4")
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size per GPU (default: 64)
- `--num-gpus`: Number of GPUs to use (default: 1)
- `--steps`: Number of training steps per method (default: 5000)
- `--sample`: Generate samples after training

#### Dataset Options:
- `--data-path`: Path to local ImageNet training data
- `--use-torch-datasets`: Use PyTorch's datasets instead of local files
- `--dataset-download-dir`: Directory to download datasets when using --use-torch-datasets (default: "./datasets")
- `--use-subset`: Use a smaller subset of the dataset for faster training
- `--subset-size`: Number of images to use when --use-subset is enabled (default: 50000)

#### Method Selection:
- `--skip-ln`: Skip LayerNorm training
- `--skip-dyt`: Skip DynamicTanh training
- `--skip-adyt`: Skip AdaptiveDynamicTanh training

#### ADYT Parameters:
- `--adyt-lambda`: Lambda factor for ADYT (default: 0.1)
- `--adyt-smooth`: Smooth factor for ADYT gradient EMA (default: 0.99)

### Examples

1. **Using PyTorch datasets with a small subset for quick testing:**
   ```bash
   python compare_methods.py \
     --use-torch-datasets \
     --use-subset \
     --subset-size 5000 \
     --steps 1000 \
     --batch-size 16
   ```

2. **Only compare DyT and ADyT with PyTorch datasets:**
   ```bash
   python compare_methods.py \
     --use-torch-datasets \
     --skip-ln \
     --steps 2000
   ```

3. **Full-scale comparison with samples using local ImageNet data:**
   ```bash
   python compare_methods.py \
     --data-path /path/to/imagenet/train \
     --model DiT-L/4 \
     --steps 50000 \
     --num-gpus 8 \
     --sample
   ```

## Output

The script generates a comprehensive set of outputs:

1. **Training logs**: Detailed logs for each method
2. **Checkpoints**: Final checkpoint for each method
3. **Visualizations**:
   - Loss comparison plot
   - Gradient norm comparison plot
   - Results table (as image and JSON)
   - Sample images (if `--sample` is specified)
   - Sample grid comparing all methods (if `--sample` is specified)

## Analyzing Results

After running the comparison, you can explore the results directory to:

1. Compare the training dynamics between methods (loss curves)
2. Analyze the gradient behavior (gradient norm plots)
3. Compare the final performance (results table)
4. Visually assess the quality of generated samples (if `--sample` is specified)

The script makes it easy to conduct a fair comparison between the different normalization approaches, helping you understand the benefits of our proposed DyT and ADyT methods. 