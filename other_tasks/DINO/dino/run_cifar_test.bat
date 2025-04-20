@echo off
echo Running DINO test on CIFAR-10 with LN, DyT, and ADyT...

REM Create results directory
if not exist results\cifar10_dino mkdir results\cifar10_dino

REM Run the test
python test_dino_cifar.py

echo Test completed!
echo Results saved to .\results\cifar10_dino\
pause 