import torch
import torch.nn as nn

class DampedOscillatorPINN(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64], activation='tanh'):
        super(DampedOscillatorPINN, self).__init__()
        
        # Input layer (z, ξ) -> first hidden layer
        layers = [nn.Linear(2, hidden_layers[0])]
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        # Activation function
        if activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, z, xi):
        # Combine inputs
        x = torch.cat([z, xi], dim=1)
        
        # Forward pass through layers
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x
    
    def compute_derivatives(self, z, xi):
        """Compute first and second derivatives with respect to z"""
        z.requires_grad_(True)
        xi.requires_grad_(True)
        
        # Forward pass
        x = self.forward(z, xi)
        
        # First derivative
        dx_dz = torch.autograd.grad(x, z, grad_outputs=torch.ones_like(x),
                                  create_graph=True)[0]
        
        # Second derivative
        d2x_dz2 = torch.autograd.grad(dx_dz, z, grad_outputs=torch.ones_like(dx_dz),
                                     create_graph=True)[0]
        
        return x, dx_dz, d2x_dz2 