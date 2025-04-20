import torch
import torch.nn as nn


class AdaptiveDynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, lambda_factor=0.5, smooth_factor=0.99, alpha_min=0.1, alpha_max=2.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.lambda_factor = lambda_factor
        self.smooth_factor = smooth_factor
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Initialize parameters
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
        # Initialize EMA of gradient norm
        self.register_buffer('grad_ema', torch.zeros(1))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

    def forward(self, x):
        # Clip alpha to stay within bounds
        alpha_clipped = torch.clamp(self.alpha, self.alpha_min, self.alpha_max)
        return self.weight * torch.tanh(alpha_clipped * x) + self.bias

    def update_grad_norm(self):
        if self.weight.grad is not None:
            current_grad_norm = torch.norm(self.weight.grad.data)
            if not self.initialized:
                self.grad_ema.copy_(current_grad_norm)
                self.initialized.fill_(True)
            else:
                self.grad_ema.mul_(self.smooth_factor).add_(current_grad_norm * (1 - self.smooth_factor))
            
            # Update alpha based on gradient norm
            eps = 1e-6  # Small constant for numerical stability
            self.alpha.data = self.alpha_init_value * (1 + self.lambda_factor / (eps + self.grad_ema))

    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, "
                f"alpha_init_value={self.alpha_init_value}, "
                f"lambda_factor={self.lambda_factor}, "
                f"smooth_factor={self.smooth_factor}, "
                f"alpha_min={self.alpha_min}, "
                f"alpha_max={self.alpha_max}")


def convert_ln_to_adyt(module, lambda_factor=0.5, smooth_factor=0.99, alpha_min=0.1, alpha_max=2.0):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = AdaptiveDynamicTanh(
            module.normalized_shape,
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
    for module in model.modules():
        if isinstance(module, AdaptiveDynamicTanh):
            module.update_grad_norm() 