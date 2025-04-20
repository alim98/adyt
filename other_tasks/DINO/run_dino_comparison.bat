@echo off
echo Running DINO comparison of LayerNorm vs DyT vs ADyT...

REM Common parameters
set COMMON_ARGS=--arch vit_base ^
    --patch_size 16 ^
    --out_dim 65536 ^
    --norm_last_layer true ^
    --warmup_teacher_temp 0.04 ^
    --teacher_temp 0.07 ^
    --warmup_teacher_temp_epochs 50 ^
    --use_fp16 false ^
    --weight_decay 0.04 ^
    --weight_decay_end 0.4 ^
    --clip_grad 0.3 ^
    --batch_size_per_gpu 32 ^
    --epochs 400 ^
    --freeze_last_layer 3 ^
    --lr 0.00075 ^
    --warmup_epochs 10 ^
    --min_lr 2e-06 ^
    --global_crops_scale 0.25 1.0 ^
    --local_crops_scale 0.05 0.25 ^
    --local_crops_number 10 ^
    --seed 0 ^
    --num_workers 10 ^
    --optimizer adamw ^
    --momentum_teacher 0.996 ^
    --use_bn_in_head false ^
    --drop_path_rate 0.1

REM Directory paths
set DATA_PATH=C:\path\to\imagenet\train
set BASE_OUTPUT_DIR=C:\path\to\saving_dir

REM Run with LayerNorm (baseline)
echo Running DINO with LayerNorm...
set OUTPUT_DIR=%BASE_OUTPUT_DIR%\ln
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
torchrun --nproc_per_node=8 main_dino.py ^
    %COMMON_ARGS% ^
    --norm_type ln ^
    --data_path %DATA_PATH% ^
    --output_dir %OUTPUT_DIR%

REM Run with DynamicTanh
echo Running DINO with DynamicTanh...
set OUTPUT_DIR=%BASE_OUTPUT_DIR%\dyt
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
torchrun --nproc_per_node=8 main_dino.py ^
    %COMMON_ARGS% ^
    --norm_type dyt ^
    --data_path %DATA_PATH% ^
    --output_dir %OUTPUT_DIR%

REM Run with AdaptiveDynamicTanh
echo Running DINO with AdaptiveDynamicTanh...
set OUTPUT_DIR=%BASE_OUTPUT_DIR%\adyt
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
torchrun --nproc_per_node=8 main_dino.py ^
    %COMMON_ARGS% ^
    --norm_type adyt ^
    --lambda_factor 0.5 ^
    --smooth_factor 0.99 ^
    --alpha_min 0.1 ^
    --alpha_max 2.0 ^
    --data_path %DATA_PATH% ^
    --output_dir %OUTPUT_DIR%

echo All experiments completed!
pause 