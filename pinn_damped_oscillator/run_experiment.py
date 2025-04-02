#!/usr/bin/env python3
"""
Run the PINN experiment for the damped harmonic oscillator.
This script loads the model, trains it, and generates visualizations.
"""

import os
import torch
import matplotlib.pyplot as plt
from src.pinn_model import DampedOscillatorPINN
from src.ode_loss import DampedOscillatorLoss
from src.utils import train_model, plot_training_history, plot_solution_comparison

def main():
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = DampedOscillatorPINN(hidden_layers=[64, 64, 64], activation='tanh')
    model = model.to(device)
    
    # Create loss function and optimizer
    loss_fn = DampedOscillatorLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training model...")
    history = train_model(model, loss_fn, optimizer, n_epochs=5000)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path='assets/loss_plot.png')
    
    # Plot solution comparison
    print("Plotting solution comparison...")
    plot_solution_comparison(model, xi_values=[0.1, 0.25, 0.4], 
                            save_path='assets/compare_plot.png')
    
    print("Experiment completed successfully!")
    print("Results saved in the 'assets' directory.")

if __name__ == "__main__":
    main() 