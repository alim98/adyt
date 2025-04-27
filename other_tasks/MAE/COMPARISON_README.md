# MAE Normalization Methods Comparison

This directory contains scripts to compare three different normalization approaches for Masked Autoencoders (MAE):

1. **LayerNorm** - The original implementation used in MAE
2. **DynamicTanh** - The Dynamic Tanh activation replacement for LayerNorm
3. **AdaptiveDynamicTanh** - An enhanced version of DynamicTanh with adaptive alpha adjustment

## Setup

Follow these steps to prepare for the comparison:

1. Set up the Python environment according to the original MAE README:
```
conda create -n MAE python=3.9
conda activate MAE
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 tensorboard
```

2. Install additional dependencies:
```
pip install pandas matplotlib
```

3. Prepare your local ImageNet dataset:
   - Download the ImageNet Object Localization Challenge dataset
   - Extract it to a directory with the following structure:
   ```
   ILSVRC/
   ├── Data/
   │   └── CLS-LOC/
   │       ├── train/
   │       │   ├── n01440764/
   │       │   │   ├── n01440764_10026.JPEG
   │       │   │   └── ...
   │       │   └── ...
   │       └── val/
   │           ├── ILSVRC2012_val_00000001.JPEG
   │           └── ...
   ```

4. Clone and prepare the MAE repository:
```
git clone https://github.com/facebookresearch/mae.git
cd mae
git apply ../compatibility-fix.patch
```

5. Set the Python path:
```
cd ..
# On Windows:
$env:PYTHONPATH += ";$(pwd)/mae"

# On Linux/MacOS:
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
```

## Running the Comparison

Run the comparison script:

```
# Windows
.\run_comparison.bat --imagenet_dir "path\to\your\ILSVRC"

# Linux/macOS
./run_comparison.sh --imagenet_dir "/path/to/your/ILSVRC"
```

This will:
1. Train MAE with LayerNorm
2. Train MAE with DynamicTanh
3. Train MAE with AdaptiveDynamicTanh
4. Generate comparison charts and tables

### Command Line Arguments

You can customize the comparison with these arguments:

- `--epochs` - Number of epochs for each method (default: 100)
- `--batch_size` - Batch size (default: 64)
- `--model` - Model architecture (default: mae_vit_base_patch16)
- `--imagenet_dir` - Path to ImageNet dataset (default: ILSVRC)
- `--output_dir` - Directory to save results (default: ./output_comparison)
- `--max_images` - Maximum number of images to use (for faster testing)

For example, to run a shorter comparison with 20 epochs and limited images:

```
.\run_comparison.bat --epochs 20 --max_images 10000 --output_dir ./quick_comparison
```

### AdaptiveDynamicTanh Parameters

You can tune the AdaptiveDynamicTanh hyperparameters:

- `--adyt_lambda` - Lambda factor (adjustment strength) (default: 0.1)
- `--adyt_smooth` - Smoothing factor for gradient norm EMA (default: 0.9)
- `--adyt_alpha_min` - Minimum alpha value (default: 0.1)
- `--adyt_alpha_max` - Maximum alpha value (default: 2.0)

## Results and Analysis

After running the comparison, you'll find these results in the output directory:

1. **Loss Curves**: Individual training loss curves for each method
2. **Comparison Plot**: Combined loss curves for all three methods
3. **Timing Comparison**: Bar chart showing average epoch time for each method
4. **Summary Table**: CSV file with key metrics for all methods
5. **Reconstructions**: Side-by-side comparison of reconstructions from each method

### Example Analysis for Paper

You can use the generated visualizations directly in your paper. Here's how to interpret the results:

1. **Loss Curves**: Lower loss generally indicates better reconstruction quality. 
   - Look for which method achieves the lowest loss
   - Observe convergence speed (which method reaches low loss faster)

2. **Timing Comparison**: Shows the computational efficiency of each method.
   - AdaptiveDynamicTanh may have slightly higher computational cost than DynamicTanh
   - Both may have lower computational cost compared to LayerNorm

3. **Reconstruction Quality**: Visual comparison of reconstructed images.
   - Compare how well each method reconstructs the masked regions
   - Look for preservation of fine details and texture

4. **Statistical Analysis**:
   - `min_loss`: Lowest loss achieved during training
   - `min_loss_epoch`: Epoch where lowest loss was achieved
   - `avg_epoch_time`: Average time per epoch

## Extending the Experiment

To experiment with different configurations:

1. For longer training, increase epochs:
```
.\run_comparison.bat --epochs 800
```

2. For full ImageNet training:
```
.\run_comparison.bat --imagenet_dir "path\to\ILSVRC"
```

3. For larger model size:
```
.\run_comparison.bat --model mae_vit_large_patch16
```

4. For hyperparameter tuning of AdaptiveDynamicTanh:
```
.\run_comparison.bat --adyt_lambda 0.2 --adyt_smooth 0.95
``` 