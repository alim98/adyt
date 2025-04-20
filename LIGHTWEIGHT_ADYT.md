# Lightweight Implementation of Adaptive Dynamic Tanh (ADyT)

This document outlines the recommended lightweight approach for implementing AdaptiveDynamicTanh (ADyT), which enhances DynamicTanh through efficient gradient-based dynamic adjustment of the alpha parameter.

## Lightweight Gradient-Based Approach

We recommend using **(1) layer-wise gradient norm** (only from the DyT layer) together with **(2) an Exponential Moving Average (EMA)**. This approach is efficient, computationally lightweight, and effective for stabilizing training.

### Step-by-Step Implementation

1. **Compute the gradient norm of only the DyT layer** (or block) each iteration.  
   - After backpropagation, extract \(\|\nabla_{W_{\text{DyT}}} L(\theta)\|\) — the L2-norm of only the DyT parameters' gradients (not the entire model).
   - This keeps computation minimal and focused on the parameters that matter for adaptation.

2. **Update this gradient norm via an EMA** to smooth out noise and avoid sudden oscillations:
   ```
   G_{t} = β · G_{t-1} + (1 - β) · ‖∇_{W_DyT} L(θ)‖
   ```
   - Here, \(\beta \in [0,1)\) is a smoothing factor (e.g., 0.9). 
   - \(G_{t}\) is the running average of the DyT-layer gradient norm.

3. **Adjust \(\alpha\) based on the EMA**:
   ```
   α(t) = α₀ · (1 + λ / G_{t})
   ```
   - \(\alpha_{0}\) is your base initialization (e.g., 0.5 or 1.0).  
   - \(\lambda\) is a hyperparameter controlling adjustment strength.  
   - If \(G_{t}\) grows large, \(\alpha(t)\) becomes smaller; if \(G_{t}\) is small, \(\alpha(t)\) grows.
   - This adaptive behavior helps maintain stable training.

### Implementation in PyTorch

```python
def update_grad_norm(self):
    """Update the smoothed gradient norm using only the gradient of this layer's parameters."""
    if not self.training or not self.adaptive_enabled:
        return
        
    # Compute the gradient norm from ONLY this layer's parameters
    total_norm = 0.0
    for p in [self.alpha_base, self.weight, self.bias]:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    grad_norm = total_norm ** 0.5
    
    # Apply exponential smoothing to avoid rapid fluctuations
    if self.grad_norm_smoothed == 0:
        self.grad_norm_smoothed = torch.tensor([grad_norm], device=self.alpha_base.device)
    else:
        self.grad_norm_smoothed = (
            self.smooth_factor * self.grad_norm_smoothed + 
            (1 - self.smooth_factor) * torch.tensor([grad_norm], device=self.alpha_base.device)
        )

def compute_alpha(self):
    """Compute the adaptive alpha value based on the layer-specific smoothed gradient norm."""
    if self.adaptive_enabled and self.grad_norm_smoothed > 0:
        # α(t) = α₀ * (1 + λ / G_t)
        return self.alpha_base * (1 + self.lambda_factor / (self.grad_norm_smoothed + self.eps))
    return self.alpha_base
```

## Benefits of this Approach

1. **Minimal Computational Overhead**
   - Computing the L2 norm of a single layer's gradient is cheap compared to the overall forward-backward pass
   - EMA updates require just a few extra floating-point operations each step
   - Significantly lighter than approaches that require computing statistics across batches

2. **Training Stability**
   - When gradients spike, \(\alpha\) is automatically reduced, preventing exploding activations
   - As training progresses and gradients shrink naturally, \(\alpha\) can increase to allow finer adjustments
   - The EMA smoothing prevents rapid oscillations in \(\alpha\) values

3. **Simple Integration**
   - The update code runs once per iteration after backward() but before optimizer.step()
   - No need for special initialization or complex infrastructure
   - Can be added to existing models with minimal modification

## Hyperparameter Tuning

- **Base alpha (\(\alpha_0\))**: Start with the same value used for DyT (typically 0.5 or 1.0)
- **Lambda factor (\(\lambda\))**: Controls adjustment strength. Recommended range: 0.05-0.2
- **Smooth factor (\(\beta\))**: Controls EMA decay. Recommended value: 0.9-0.95

## Usage in Training Loop

```python
# Training loop example
model = create_model()
model = convert_ln_to_adyt(model, lambda_factor=0.1, smooth_factor=0.9)
optimizer = torch.optim.AdamW(model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Update gradient norms in all AdaptiveDynamicTanh layers
        update_adyt_grad_norms(model)
        
        optimizer.step()
        optimizer.zero_grad()
```

This implementation achieves the benefits of adaptive normalization while maintaining computational efficiency. 