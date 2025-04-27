@echo off
echo Setting up Python path...
cd %~dp0

REM Check if mae directory exists and has content
if not exist "mae\util" (
    echo MAE directory is missing or empty. Cloning repository...
    if exist "mae" rd /s /q mae
    git clone https://github.com/facebookresearch/mae.git mae
    cd mae
    git apply ..\compatibility-fix.patch
    cd ..
    echo MAE repository cloned and patch applied.
)

set PYTHONPATH=%cd%\mae
echo Python path set to: %PYTHONPATH%

echo Running comparison...
python compare_mae_methods.py %*

echo Done! 