import torch
import torch.nn as nn


class AdaptiveDynamicTanh(nn.Module):
    """
    AdaptiveDynamicTanh (ADyT): An extension of DynamicTanh with gradient-based dynamic adjustment of alpha.
    
    This implementation adjusts the alpha parameter during training based on the layer-specific gradient norm:
    α(t) = α₀ * (1 + λ/(ε + G_t))
    
    Where G_t is the Exponential Moving Average (EMA) of the gradient norm of the DyT layer.
    """
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5, 
                 lambda_factor=0.1, smooth_factor=0.99, eps=1e-6, alpha_min=0.1, alpha_max=2.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.lambda_factor = lambda_factor
        self.smooth_factor = smooth_factor
        self.eps = eps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Base alpha parameter (learnable)
        self.alpha_base = nn.Parameter(torch.ones(1) * alpha_init_value)
        
        # Weights and bias (if using elementwise affine)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Register a buffer for tracking the gradient norm (not a learnable parameter)
        self.register_buffer("grad_norm_smoothed", torch.tensor([1.0]))
        
        # Flag to indicate if we're in training mode
        self.adaptive_enabled = True

    def compute_alpha(self):
        """
        Compute the adaptive alpha value based on the layer-specific smoothed gradient norm.
        Formula: α(t) = α₀ * (1 + λ/(ε + G_t))
        The result is clipped to [alpha_min, alpha_max] to prevent sudden large changes.
        """
        if self.adaptive_enabled and self.training:
            # α(t) = α₀ * (1 + λ/(ε + G_t))
            alpha = self.alpha_base * (1 + self.lambda_factor / (self.grad_norm_smoothed + self.eps))
            # Clip alpha to prevent extreme values
            return torch.clamp(alpha, self.alpha_min, self.alpha_max)
        return self.alpha_base
    
    def update_grad_norm(self):
        """
        Update the smoothed gradient norm using only the gradient of this layer's parameters.
        Uses Exponential Moving Average (EMA) for stability.
        """
        if not self.training or not self.adaptive_enabled:
            return
            
        # Compute the gradient norm from ONLY this layer's parameters
        total_norm = 0.0
        params = [self.alpha_base]
        if self.elementwise_affine:
            params.extend([self.weight, self.bias])
            
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        
        # Apply exponential smoothing to avoid rapid fluctuations
        self.grad_norm_smoothed = (
            self.smooth_factor * self.grad_norm_smoothed + 
            (1 - self.smooth_factor) * torch.tensor([grad_norm], device=self.alpha_base.device)
        )

    def forward(self, x):
        # Get current adaptive alpha value
        alpha = self.compute_alpha()
        
        # Apply tanh with adaptive alpha
        x = torch.tanh(alpha * x)
        
        # Apply weights and bias if using elementwise affine
        if self.elementwise_affine:
            return self.weight * x + self.bias
        else:
            return x

    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, "
                f"elementwise_affine={self.elementwise_affine}, "
                f"alpha_init_value={self.alpha_init_value}, "
                f"lambda_factor={self.lambda_factor}, "
                f"alpha_min={self.alpha_min}, "
                f"alpha_max={self.alpha_max}")


def convert_ln_to_adyt(module, lambda_factor=0.1, smooth_factor=0.99, alpha_min=0.1, alpha_max=2.0):
    """
    Recursively convert all LayerNorm layers in the module to AdaptiveDynamicTanh.
    
    Args:
        module (nn.Module): The module to convert
        lambda_factor (float): The λ parameter for adaptive alpha adjustment
        smooth_factor (float): The β parameter for EMA of gradient norm
        alpha_min (float): Minimum value for alpha after adjustment
        alpha_max (float): Maximum value for alpha after adjustment
    
    Returns:
        nn.Module: The converted module
    """
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = AdaptiveDynamicTanh(
            module.normalized_shape, 
            module.elementwise_affine,
            lambda_factor=lambda_factor,
            smooth_factor=smooth_factor,
            alpha_min=alpha_min,
            alpha_max=alpha_max
        )
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_adyt(child, lambda_factor, smooth_factor, alpha_min, alpha_max))
    del module
    return module_output


def update_adyt_grad_norms(model):
    """
    Update the gradient norm in all AdaptiveDynamicTanh layers of the model.
    
    This function should be called after loss.backward() but before optimizer.step()
    
    Args:
        model (nn.Module): The model containing AdaptiveDynamicTanh layers
    """
    for module in model.modules():
        if isinstance(module, AdaptiveDynamicTanh):
            module.update_grad_norm() 