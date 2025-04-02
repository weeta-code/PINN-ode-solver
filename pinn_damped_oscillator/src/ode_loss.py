import torch
import torch.nn as nn

class DampedOscillatorLoss(nn.Module):
    def __init__(self):
        super(DampedOscillatorLoss, self).__init__()
        
    def compute_residual_loss(self, model, z, xi):
        """Compute the physics-informed loss for the ODE residual"""
        x, dx_dz, d2x_dz2 = model.compute_derivatives(z, xi)
        
        # ODE residual: d²x/dz² + 2ξ·dx/dz + x = 0
        residual = d2x_dz2 + 2 * xi * dx_dz + x
        
        return torch.mean(residual ** 2)
    
    def compute_ic_loss(self, model, z0, xi0):
        """Compute the initial condition loss"""
        x0, dx_dz0, _ = model.compute_derivatives(z0, xi0)
        
        # Initial conditions: x(0) = 0.7, dx/dz(0) = 1.2
        ic_loss = (x0 - 0.7) ** 2 + (dx_dz0 - 1.2) ** 2
        
        return torch.mean(ic_loss)
    
    def forward(self, model, z, xi, z0, xi0):
        """Compute the total loss combining residual and initial conditions"""
        residual_loss = self.compute_residual_loss(model, z, xi)
        ic_loss = self.compute_ic_loss(model, z0, xi0)
        
        # Total loss (can be weighted if needed)
        total_loss = residual_loss + ic_loss
        
        return total_loss, residual_loss, ic_loss 