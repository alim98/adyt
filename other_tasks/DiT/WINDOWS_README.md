# DiT Comparison on Windows: LayerNorm vs DyT vs ADyT

This guide explains how to run a comparison between three normalization methods for Diffusion Transformers (DiT) on Windows:

1. **LayerNorm (LN)** - The original normalization used in DiT
2. **DynamicTanh (DyT)** - A learnable tanh-based activation alternative
3. **AdaptiveDynamicTanh (ADyT)** - An enhanced version of DyT with gradient-based adaptation

## Prerequisites

1. Clone the DiT repository (if not already done):
   ```
   git clone https://github.com/facebookresearch/DiT.git other_tasks/DiT/DiT
   ```

2. Apply the necessary patches to add DyT and ADyT support:
   ```
   cd other_tasks/DiT
   copy dynamic_tanh.py DiT\
   copy adaptive_dynamic_tanh.py DiT\
   cd DiT
   git apply ..\dynamic-tanh.patch
   git apply ..\dynamic-tanh-fix.patch
   git apply ..\adaptive-dynamic-tanh.patch
   git apply ..\no-distributed.patch
   ```

3. Install dependencies according to the DiT project's requirements (PyTorch, diffusers, etc.)

4. The ImageNet dataset should be available at:
   ```
   C:\Users\WINGPU\Desktop\DyT_2\other_tasks\DINO\data\ILSVRC\Data\CLS-LOC\train
   ```
   - If your dataset is located elsewhere, you'll need to modify the path in the run_comparison.bat file

## Running the Comparison

### Easy Method: Using the Batch File

1. Simply run the batch file by double-clicking `run_comparison.bat`
   - This will run the comparison with default settings (batch size 32, 2000 steps)

2. To customize the comparison, you can also run with parameters:
   ```
   run_comparison.bat 16 1000
   ```
   - First parameter: batch size (default: 32)
   - Second parameter: maximum training steps (default: 2000)

### Advanced Method: Using Python Directly

For more control over the comparison, you can run the Python script directly:

```
python run_comparison.py --data-path "C:/Users/WINGPU/Desktop/DyT_2/other_tasks/DINO/data/ILSVRC/Data/CLS-LOC/train" --batch-size 16 --max-steps 1000 --adyt-lambda 0.2 --adyt-smooth 0.95
```

Available options:
- `--skip-ln`: Skip training with LayerNorm
- `--skip-dyt`: Skip training with DynamicTanh
- `--skip-adyt`: Skip training with AdaptiveDynamicTanh
- `--adyt-lambda`: Lambda factor for ADyT (default: 0.1)
- `--adyt-smooth`: Smooth factor for ADyT (default: 0.99)
- `--model`: DiT model size (choices: DiT-B/4, DiT-L/4, DiT-XL/2) (default: DiT-B/4)
- `--lr`: Learning rate (default: 1e-4)

## Understanding the Results

After running the comparison, the results will be saved in the `comparison_results` directory. A new timestamped subdirectory will be created for each run, containing:

1. **Training logs** for each method
2. **Loss comparison plots** showing how each method performed
3. **Throughput comparison** showing training speed
4. **Results table** summarizing the final metrics
5. **README.md** with details about the experiment

## Customizing the ImageNet Path

If your ImageNet dataset is located in a different directory, you have two options:

1. Edit the `run_comparison.bat` file:
   - Change the `DATA_PATH` variable to the correct path

2. Run the Python script directly with the correct path:
   ```
   python run_comparison.py --data-path "YOUR_IMAGENET_PATH"
   ```

## Troubleshooting

- **Error loading ImageNet**: Make sure the dataset is properly organized in the expected ImageNet format
- **CUDA out of memory**: Reduce the batch size (e.g., `run_comparison.bat 8 1000`)
- **Process killed**: Try reducing the model size by using `--model DiT-B/4` (the smallest model variant) 