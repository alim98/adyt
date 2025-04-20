# Adaptive Dynamic Tanh (ADyT)

## Introduction

Adaptive Dynamic Tanh (ADyT) extends the original DynamicTanh (DyT) approach by dynamically adjusting the α parameter during training based on gradient information. While the original DyT uses a fixed learnable scalar parameter α, ADyT adapts this parameter throughout training to improve stability and convergence.

## Mathematical Formulation

The core idea of ADyT is to adjust the α parameter based on the gradient norm of the layer:

$$\alpha(t) = \alpha_0 \cdot \left(1 + \frac{\lambda}{\varepsilon + G_t}\right)$$

Where:
- $\alpha(t)$ is the dynamically adjusted alpha at time t
- $\alpha_0$ is the initial alpha value (learnable)
- $\lambda$ is a hyperparameter controlling the strength of adjustment
- $G_t$ is the Exponential Moving Average (EMA) of the gradient norm: $G_t = \beta G_{t-1} + (1-\beta) g_t$
- $g_t$ is the current gradient norm of the DyT layer parameters
- $\varepsilon$ is a small constant to prevent division by zero
- The computed value is clipped to $[\alpha_{min}, \alpha_{max}]$ to prevent extreme values

## Intuition and Rationale

The adaptive adjustment of α provides several benefits:

1. **Training Stability**: When gradients are large (indicating unstable or rapid updates), α is reduced, limiting extreme activation values and stabilizing training.

2. **Improved Convergence**: When gradients are small (indicating more stable training), α increases, allowing tanh to behave more linearly within a wider range, facilitating subtle adjustments and faster convergence.

3. **Self-Regulating Behavior**: The model automatically adapts its behavior based on the current training dynamics without requiring manual intervention.

4. **Value Clipping**: Prevents sudden large changes in the α parameter, further enhancing stability.

## Implementation Details

The implementation consists of several key components:

### `AdaptiveDynamicTanh` Class
This is the main class that implements the adaptive version of DynamicTanh.

```python
class AdaptiveDynamicTanh(nn.Module):
    def __init__(
        self, 
        normalized_shape,
        channels_last: bool = True,
        alpha_init_value: float = 0.5,
        lambda_factor: float = 0.1, 
        smooth_factor: float = 0.99,
        eps: float = 1e-6,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0
    ):
        # ...initialization code...
```

**Parameters:**
- `normalized_shape`: The shape of the input tensor's normalization dimensions
- `channels_last`: Whether the input has channels in the last dimension
- `alpha_init_value`: Initial value for the α parameter (α₀)
- `lambda_factor`: The λ hyperparameter controlling adjustment strength (typically 0.1-1.0)
- `smooth_factor`: Exponential moving average factor for gradient norm smoothing (β, typically 0.99)
- `eps`: Small epsilon value to prevent division by zero
- `alpha_min`: Minimum allowed value for α after adjustment (typically 0.1)
- `alpha_max`: Maximum allowed value for α after adjustment (typically 2.0)

### Key Methods

#### `compute_alpha()`
Calculates the current adaptive α value based on the smoothed gradient norm and applies clipping.

#### `update_grad_norm()`
Updates the internal smoothed gradient norm based on the current layer gradients. Uses exponential moving average for stability.

#### `forward(x)`
Applies the adaptive tanh transformation: weight * tanh(α(t) * x) + bias

### Utility Functions

#### `convert_ln_to_adyt(module, lambda_factor=0.1, smooth_factor=0.99, alpha_min=0.1, alpha_max=2.0)`
Recursively converts all LayerNorm layers in a model to AdaptiveDynamicTanh.

#### `update_adyt_grad_norms(model)`
Updates gradient norms in all AdaptiveDynamicTanh layers of a model. Should be called after `loss.backward()` but before `optimizer.step()`.

## Usage Guide

### Step 1: Convert your model to use ADyT

```python
import torch
from dynamic_tanh_adaptive import convert_ln_to_adyt

# Create your model
model = create_model()

# Convert all LayerNorm layers to AdaptiveDynamicTanh
model = convert_ln_to_adyt(
    model, 
    lambda_factor=0.5,  # Adjust based on task complexity
    smooth_factor=0.99, # Higher for smoother adaptation
    alpha_min=0.1,      # Lower bound for alpha
    alpha_max=2.0       # Upper bound for alpha
)
```

### Step 2: Modify your training loop

```python
from dynamic_tanh_adaptive import update_adyt_grad_norms

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update the gradient norms in all ADyT layers
        # This must be called after backward() but before optimizer.step()
        update_adyt_grad_norms(model)
        
        # Update model parameters
        optimizer.step()
```

### Step 3: Monitor performance

During training, you may want to monitor:
- The training stability (less fluctuation in loss)
- Convergence speed compared to standard DyT
- Final model performance

## Hyperparameter Recommendations

- **lambda_factor**: Start with 0.5, adjust based on task complexity
  - Higher values (0.5-1.0) for complex tasks with potentially large gradients
  - Lower values (0.1-0.3) for simpler tasks or if training becomes unstable
  
- **smooth_factor**: Use 0.99 for optimal stability
  - Increase to 0.995-0.999 for even more stable but slower adaptation
  - Decrease to 0.9-0.95 only if faster adaptation to gradient changes is needed

- **alpha_min/alpha_max**: The defaults (0.1/2.0) work well for most tasks
  - Widen the range for more flexible adaptation
  - Narrow the range for more stable training

## Potential Challenges and Solutions

### 1. Computational Overhead
Computing gradient norms adds some computational overhead. If this becomes an issue, consider:
- Updating less frequently (e.g., every N steps)
- Computing gradient norms on a subset of model parameters

### 2. Adaptation Instability
If the adaptation mechanism itself becomes unstable:
- Increase the smoothing factor
- Apply gradient clipping before computing norms
- Reduce the lambda_factor

### 3. Model Divergence
If training diverges:
- Start with a smaller lambda_factor
- Ensure gradient clipping is applied in the optimizer
- Consider a smaller learning rate initially

## Comparison with Original DyT

| Feature | Original DyT | Adaptive DyT (ADyT) |
|---------|-------------|---------------------|
| Formula | tanh(α * x) | tanh(α(t) * x) |
| α behavior | Learnable but fixed during each forward pass | Dynamically adjusted based on gradient norm |
| Parameters | α, weight, bias | α₀, weight, bias, λ, smoothing factor |
| Adaptability | Adapts during training via backprop | Additional real-time adaptation based on gradient information |
| Training integration | Drop-in replacement | Requires additional update step in training loop |
| Computational cost | Lower | Slightly higher due to gradient norm computation |

## Theoretical Analysis

The key insight of ADyT is connecting the scaling factor α with gradient behavior. This creates an implicit feedback loop:

1. When gradients are high → α decreases → activations are compressed → gradients stabilize
2. When gradients are low → α increases → activations expand → network can make more fine-grained updates

This behavior mimics some of the benefits of normalization layers while maintaining the computational efficiency of DyT.

## Conclusion

Adaptive Dynamic Tanh (ADyT) enhances the original DyT approach by making it responsive to the training dynamics through gradient-based adaptation. It aims to combine the computational efficiency of DyT with improved training stability and convergence speed. 