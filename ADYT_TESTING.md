# Testing the AdaptiveDynamicTanh (ADyT) Module

This document provides guidelines for testing and validating the AdaptiveDynamicTanh (ADyT) module, which extends the original DynamicTanh approach with gradient-based adaptive alpha adjustment.

## Prerequisites

Ensure you have the following installed:
- Python 3.6+
- PyTorch 1.12+
- matplotlib
- tqdm (for progress bars)
- torchvision (for MNIST dataset, optional)

## Quick Start

To quickly test the ADyT module and verify its functionality:

```bash
# Run basic unit tests
python test_adyt.py

# Run a simple example with MNIST
python example_usage.py --use-adyt --epochs 3

# Compare with regular LayerNorm
python example_usage.py --epochs 3
```

## Testing Options

### 1. Unit Tests

The `test_adyt.py` script performs several basic tests:

- **Forward Pass Test**: Ensures the module correctly transforms input data
- **Gradient Adaptation Test**: Verifies alpha parameter adapts based on gradient norms
- **Conversion Function Test**: Tests the utility function that converts LayerNorm to ADyT
- **Visualization**: Generates a plot showing how alpha changes with different gradient norms
- **Integration Test**: Tests compatibility with models from the timm library (if available)

```bash
python test_adyt.py
```

### 2. Example Application

The `example_usage.py` script demonstrates how to use ADyT in a simple CNN model for MNIST classification:

```bash
# Run with AdaptiveDynamicTanh
python example_usage.py --use-adyt --lambda-factor 0.1 --smooth-factor 0.9

# Different hyperparameter settings
python example_usage.py --use-adyt --lambda-factor 0.2 --smooth-factor 0.8

# Compare with LayerNorm baseline
python example_usage.py
```

### 3. Comprehensive Benchmarking

The `benchmark_adyt.py` script provides more extensive benchmarking against both LayerNorm and the original DynamicTanh:

```bash
# Run full benchmarks (speed and accuracy)
python benchmark_adyt.py --model both --epochs 5

# Speed benchmark only
python benchmark_adyt.py --speed_only

# Accuracy benchmark only with more epochs
python benchmark_adyt.py --accuracy_only --epochs 10

# Test on a specific model
python benchmark_adyt.py --model transformer
```

## Understanding the Results

### Alpha Parameter Adaptation

One of the key features to verify is how the alpha parameter adapts during training. This can be observed in:

1. The generated `alpha_adaptation.png` visualization from unit tests
2. The printed gradient norm and alpha values during the gradient adaptation test

Expect to see:
- Smaller alpha values when gradient norms are large
- Larger alpha values when gradient norms are small

### Performance Comparison

When comparing ADyT to LayerNorm and standard DyT, look for:

1. **Accuracy**: ADyT should achieve comparable or slightly better accuracy
2. **Convergence**: ADyT may converge faster in some cases due to the adaptive mechanism
3. **Speed**: ADyT has slightly higher computational cost than standard DyT but remains efficient

## Debugging Issues

If you encounter problems:

1. **Module not found errors**: Ensure dynamic_tanh_adaptive.py is in your Python path
2. **CUDA errors**: Try running with --no-cuda to use CPU instead
3. **Gradient explosion**: Reduce the lambda factor (e.g., --lambda-factor 0.05)
4. **Slow adaptation**: Decrease the smooth factor (e.g., --smooth-factor 0.7)

## Reporting Results

When reporting test results, include:

1. The model architecture used
2. Hyperparameter settings (lambda_factor, smooth_factor)
3. Final training and test accuracy
4. Convergence speed (number of epochs to reach a threshold)
5. Any generated plots showing training curves

## Adding Custom Tests

To test with your own models:

1. Import the ADyT module and conversion function
2. Either create models with ADyT directly or use the conversion function
3. Update gradient norms in your training loop after loss.backward()
4. Compare performance with standard LayerNorm and DyT versions 