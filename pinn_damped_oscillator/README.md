# PINN Damped Oscillator

A Physics-Informed Neural Network (PINN) implementation for solving the damped harmonic oscillator equation. This project demonstrates how neural networks can learn to approximate solutions to differential equations while respecting the underlying physics.

## 🎯 Problem Statement

We solve the damped harmonic oscillator equation:

$$\frac{d^2x}{dz^2} + 2\xi\frac{dx}{dz} + x = 0$$

with initial conditions:
- $x(0) = 0.7$
- $\frac{dx}{dz}(0) = 1.2$

The damping ratio ξ varies in the range [0.1, 0.4], and we solve the equation on the domain z ∈ [0, 20].

## 🚀 Features

- PINN implementation using PyTorch
- Automatic differentiation for computing derivatives
- Visualization of solutions and training progress
- Comparison with analytical solutions
- Support for multiple damping ratios
- Google Colab-ready notebook

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- SciPy
- tqdm

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pinn_damped_oscillator.git
cd pinn_damped_oscillator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Quick Run

The simplest way to run the experiment is to use the provided script:

```bash
python run_experiment.py
```

This will:
1. Train the PINN model
2. Generate visualizations
3. Save results in the `assets` directory

### Option 2: Using the Markdown File

The project includes a markdown file (`notebooks/pinn_damped_oscillator.md`) that contains all the code and explanations. You can:

1. View it directly on GitHub
2. Convert it to a Jupyter notebook using the provided script:
```bash
cd notebooks
python convert_to_notebook.py
```

### Option 3: Using the Jupyter Notebook

1. Open the notebook in Google Colab or Jupyter:
```bash
jupyter notebook notebooks/pinn_damped_oscillator.ipynb
```

2. Follow the notebook cells to:
   - Understand the physics of damped oscillation
   - See how PINNs work
   - Train the model
   - Visualize results

## 📊 Results

The project generates several visualizations:
- Training loss curves
- Comparison between PINN and analytical solutions
- Solutions for different damping ratios

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

1. Report bugs and issues
2. Suggest new features
3. Improve documentation
4. Add more examples
5. Implement extensions (see below)

## 🔮 Future Extensions

Some interesting directions for extending this project:

1. **Meta-Learning**: Implement meta-learning to handle varying damping ratios more efficiently
2. **Forced Oscillations**: Add external forcing terms to the equation
3. **Parameter Estimation**: Use PINNs to estimate unknown parameters
4. **Higher Dimensions**: Extend to coupled oscillators or spatial problems
5. **Real-time Visualization**: Add interactive plots during training

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks, 9(5), 987-1000.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the PyTorch team for the amazing deep learning framework
- The physics-informed neural networks community for inspiration
- Contributors and users of this project 