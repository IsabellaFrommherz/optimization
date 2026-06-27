# Optimization – Python Exercises

Solutions to exercises from the module "Optimization & Machine Learning" (M16, FernUni Switzerland, FS24).
Implemented in Python using PyTorch.

### Gradient Descent
- Minimization of a 2D point configuration function
- Plain gradient descent with PyTorch autograd
- Visualization of optimized point distribution

### Optimization Algorithms
**Comparison of 6 Optimization Methods**
- Newton-Raphson (with exact Hessian via PyTorch)
- Gradient Descent with Momentum
- ADAM (without stochastic gradients)
- Plain SGD
- SGD with Momentum
- ADAM (with stochastic gradients)
- Convergence comparison and point distribution plots

### Step-Size Schedule
**Best Configuration with Step-Size Schedule**
- All 6 methods combined with step-size schedule: eta_t = eta / (1 + 0.001 * t)
- Tracking best solution found during optimization
- Comparison of best objective values per method

## Topics Covered
- Gradient Descent
- Stochastic Gradient Descent
- Momentum
- Adaptive Learning Rates (ADAM)
- Newton-Raphson with exact Hessian Matrix
- Automatic Differentiation with PyTorch
- Step-Size Schedules
- Convergence Analysis
- Numerical Optimization

## Requirements
- Python 3.x
- PyTorch
- Matplotlib

## Install Dependencies
pip install torch matplotlib
