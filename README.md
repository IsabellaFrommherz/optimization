# Optimization – Python Exercises

Solutions to exercises from the module "Optimization & Machine Learning" (M16, FernUni Switzerland, FS24).
Originally solved in MATLAB, reimplemented in Python using PyTorch.

### Gradient Descent
Minimization of a 2D point configuration function using plain Gradient Descent with PyTorch autograd.

### Optimization Algorithms
Implementation and comparison of 6 optimization algorithms:
- Newton-Raphson (with exact Hessian via PyTorch)
- Gradient Descent with Momentum
- ADAM (without stochastic gradients)
- Plain SGD
- SGD with Momentum
- ADAM (with stochastic gradients)

### Best Configuration (Step-Size Schedule)
All 6 methods combined with a step-size schedule to find the best possible configuration.
The learning rate decreases over time: `eta_t = eta / (1 + 0.001 * t)`

## Topics Covered
- Gradient Descent
- Stochastic Gradient Descent
- Momentum
- Adaptive Learning Rates (ADAM)
- Newton-Raphson with exact Hessian Matrix
- Automatic Differentiation with PyTorch
- Step-Size Schedules
- Convergence Analysis

## Requirements
- Python 3.x
- PyTorch
- Matplotlib

## Install Dependencies
pip install torch matplotlib
