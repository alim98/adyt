.\run_comparison.bat --imagenet_dir "path\to\your\ILSVRC" --max_images 10000 --epochs 20

# Using MAE with Hugging Face ImageNet-1K Dataset
pip install dill
pip install huggingface_hub
pip install pyarrow
This guide provides instructions for training Masked Autoencoders (MAE) using the Hugging Face ImageNet-1K dataset. This approach eliminates the need to download the full ImageNet dataset locally.

# Install other required packages
pip install timm==1.0.15 tensorboard

# Install Hugging Face packages with explicit versions
pip install datasets==2.16.0
pip install huggingface_hub==0.20.3
pip install dill==0.3.7
pip install pyarrow==15.0.0
cd other_tasks/MAE
$env:PYTHONPATH += ";$(pwd)/mae"
python hf_mae_pretrain.py
cp compatibility-fix.patch mae
cd mae
git apply compatibility-fix.patch


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

pip install huggingface_cli
huggingface-cli login
## 1. Set Up Environment

Follow the original MAE README instructions to set up the Python environment:

```
conda create -n MAE python=3.12
conda activate MAE
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install timm==1.0.15 tensorboard
```

Additionally, install the Hugging Face datasets package:

```
pip install datasets
```

## 2. Clone the MAE Repository and Apply Patches

Clone the repository and apply the compatibility patch:

```
git clone https://github.com/facebookresearch/mae.git
cd mae
git apply ../compatibility-fix.patch
```

## 3. Run Training with Hugging Face Dataset

Copy the `hf_mae_pretrain.py` script to the MAE directory:

```
cp ../hf_mae_pretrain.py ./
```

Run the MAE pretraining using the Hugging Face dataset:

```
python hf_mae_pretrain.py \
    --output_dir ./output_dir \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05
```

### Optional: Use Dynamic Tanh

If you want to use Dynamic Tanh instead of LayerNorm:

1. Copy the Dynamic Tanh implementation and patch:

```
cp ../dynamic_tanh.py ./
```

2. Add the `--use_dyt` flag when running training:

```
python hf_mae_pretrain.py \
    --output_dir ./output_dir \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --use_dyt
```

## 4. Distributed Training

For distributed training, you can use the `torchrun` command:

```
torchrun --nproc_per_node=8 hf_mae_pretrain.py \
    --output_dir ./output_dir \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05
```

## 5. Common Issues and Solutions

1. **Dataset column name mismatch**: The script automatically tries to detect the correct image and label column names in the dataset. If your dataset uses different column names, you may need to modify the script.

2. **Memory issues**: If you encounter memory issues, try reducing the batch size and adjusting the number of worker processes with `--num_workers`.

3. **Image format issues**: The script attempts to handle various image formats from Hugging Face datasets. If you encounter image loading errors, you may need to modify the `__getitem__` method in the `HFMappedDataset` class.

## 6. Evaluation

For fine-tuning and evaluation of pretrained models, refer to the original MAE documentation: [FINETUNE](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md). 