@echo off
echo DiT Normalization Methods Comparison
echo ===================================
echo.
echo This will compare LayerNorm, DynamicTanh, and AdaptiveDynamicTanh
echo on the DiT model using your local ImageNet dataset.
echo.

REM Default parameters
set DATA_PATH=C:/Users/WINGPU/Desktop/DyT_2/other_tasks/DINO/data/ILSVRC/Data/CLS-LOC/train
set BATCH_SIZE=32
set MAX_STEPS=2000
set RESULTS_DIR=comparison_results

if "%1"=="" (
    echo Using default parameters:
    echo  - ImageNet path: %DATA_PATH%
    echo  - Batch size: %BATCH_SIZE%
    echo  - Max steps: %MAX_STEPS%
    echo.
    echo To customize, you can provide arguments:
    echo  run_comparison.bat [batch_size] [max_steps]
    echo.
    echo Example:
    echo  run_comparison.bat 16 1000
    echo.
    timeout /t 5
) else (
    set BATCH_SIZE=%1
    if not "%2"=="" set MAX_STEPS=%2
    echo Using custom parameters:
    echo  - ImageNet path: %DATA_PATH%
    echo  - Batch size: %BATCH_SIZE%
    echo  - Max steps: %MAX_STEPS%
    echo.
)

echo Running comparison...
echo.

python run_comparison.py --data-path "%DATA_PATH%" --batch-size %BATCH_SIZE% --max-steps %MAX_STEPS% --results-dir "%RESULTS_DIR%" --no-distributed

echo.
echo Comparison completed!
echo Results are saved in the %RESULTS_DIR% directory.
echo.
pause 