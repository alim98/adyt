@echo off
echo Setting up Python path...
set PYTHONPATH=%PYTHONPATH%;%cd%\mae
echo Python path set to: %PYTHONPATH%

echo Running comparison...
python compare_mae_methods.py %*

echo Done! 