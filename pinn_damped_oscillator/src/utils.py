import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_training_data(n_points=1000, z_range=(0, 20), xi_range=(0.1, 0.4)):
    """Generate training data points"""
    z = torch.linspace(z_range[0], z_range[1], n_points).reshape(-1, 1)
    xi = torch.linspace(xi_range[0], xi_range[1], n_points).reshape(-1, 1)
    return z, xi

def train_model(model, loss_fn, optimizer, n_epochs=5000, n_points=1000):
    """Train the PINN model"""
    z, xi = generate_training_data(n_points)
    z0 = torch.zeros(1, 1)  # Initial condition point
    xi0 = torch.tensor([[0.25]])  # Example damping ratio
    
    history = {
        'total_loss': [],
        'residual_loss': [],
        'ic_loss': []
    }
    
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        
        # Compute losses
        total_loss, residual_loss, ic_loss = loss_fn(model, z, xi, z0, xi0)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store history
        history['total_loss'].append(total_loss.item())
        history['residual_loss'].append(residual_loss.item())
        history['ic_loss'].append(ic_loss.item())
        
    return history

def plot_training_history(history, save_path=None):
    """Plot training loss history"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(history['total_loss'], label='Total Loss')
    plt.semilogy(history['residual_loss'], label='Residual Loss')
    plt.semilogy(history['ic_loss'], label='IC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_solution_comparison(model, xi_values=[0.1, 0.25, 0.4], save_path=None):
    """Plot PINN solution vs analytical solution for different ξ values"""
    z = torch.linspace(0, 20, 1000).reshape(-1, 1)
    
    plt.figure(figsize=(12, 8))
    
    for xi in xi_values:
        xi_tensor = torch.ones_like(z) * xi
        with torch.no_grad():
            x_pred = model(z, xi_tensor)
        
        # Analytical solution for damped oscillator
        omega_d = np.sqrt(1 - xi**2)
        A = 0.7  # Initial displacement
        B = (1.2 + xi * A) / omega_d  # Initial velocity term
        x_analytical = np.exp(-xi * z.numpy()) * (A * np.cos(omega_d * z.numpy()) + 
                                                 B * np.sin(omega_d * z.numpy()))
        
        plt.plot(z.numpy(), x_pred.numpy(), '--', label=f'PINN (ξ={xi})')
        plt.plot(z.numpy(), x_analytical, '-', label=f'Analytical (ξ={xi})')
    
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('PINN vs Analytical Solution')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 