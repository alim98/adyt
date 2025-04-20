#!/bin/bash
echo "Running DINO test on CIFAR-10 with LN, DyT, and ADyT..."

# Create results directory
mkdir -p results/cifar10_dino

# Run the test
python test_dino_cifar.py

echo "Test completed!"
echo "Results saved to ./results/cifar10_dino/" 