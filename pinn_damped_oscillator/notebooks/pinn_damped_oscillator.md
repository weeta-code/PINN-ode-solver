# Physics-Informed Neural Networks for Damped Harmonic Oscillator

This notebook demonstrates how to solve the damped harmonic oscillator equation using Physics-Informed Neural Networks (PINNs). We'll explore how neural networks can learn to approximate solutions to differential equations while respecting the underlying physics.

## The Physics: Damped Harmonic Oscillator

The damped harmonic oscillator is described by the second-order differential equation:

$$\frac{d^2x}{dz^2} + 2\xi\frac{dx}{dz} + x = 0$$

where:
- $x$ is the displacement
- $z$ is the time variable
- $\xi$ is the damping ratio

Initial conditions:
- $x(0) = 0.7$
- $\frac{dx}{dz}(0) = 1.2$

## What are PINNs?

Physics-Informed Neural Networks (PINNs) are a type of neural network that incorporates physical laws into their training process. Instead of learning from data alone, PINNs learn to satisfy both the differential equations and the boundary/initial conditions simultaneously.

The key advantages of PINNs are:
1. No need for labeled training data
2. Ability to solve complex differential equations
3. Continuous solution representation
4. Easy to evaluate at any point in the domain

## Setting up the Environment

```python
import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.pinn_model import DampedOscillatorPINN
from src.ode_loss import DampedOscillatorLoss
from src.utils import train_model, plot_training_history, plot_solution_comparison
```

## Setting up the PINN Model

We'll create a neural network with the following architecture:
- Input layer: [z, ξ] (2 neurons)
- Hidden layers: [64, 64, 64] neurons
- Output layer: x (1 neuron)
- Activation: Tanh

The loss function combines:
1. ODE residual loss
2. Initial condition loss

```python
# Create model and loss function
model = DampedOscillatorPINN(hidden_layers=[64, 64, 64], activation='tanh')
loss_fn = DampedOscillatorLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
history = train_model(model, loss_fn, optimizer, n_epochs=5000)

# Plot training history
plot_training_history(history, save_path='../assets/loss_plot.png')
```

## Results and Analysis

Let's compare our PINN solution with the analytical solution for different damping ratios (ξ). The analytical solution for a damped harmonic oscillator is:

$$x(z) = e^{-\xi z}(A\cos(\omega_d z) + B\sin(\omega_d z))$$

where:
- $\omega_d = \sqrt{1-\xi^2}$ is the damped natural frequency
- $A$ and $B$ are constants determined by initial conditions

```python
# Plot solutions for different damping ratios
plot_solution_comparison(model, xi_values=[0.1, 0.25, 0.4], 
                        save_path='../assets/compare_plot.png')
```

## Discussion

The PINN successfully learns to approximate the solution of the damped harmonic oscillator equation. We can observe that:

1. The solution closely matches the analytical solution for all tested damping ratios
2. The network correctly captures the decaying behavior of the oscillations
3. The initial conditions are well satisfied

## Future Extensions

This implementation could be extended in several ways:

1. **Variable Damping Ratio**: Implement meta-learning to handle varying ξ values more efficiently
2. **Forced Oscillations**: Add external forcing terms to the equation
3. **Parameter Estimation**: Use PINNs to estimate unknown parameters in the system
4. **Higher Dimensions**: Extend to coupled oscillators or spatial problems

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
2. Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks, 9(5), 987-1000. 