import torch
import matplotlib.pyplot as plt

n = 20
torch.manual_seed(42)


# ── Objective function ────────────────────────────────────────────
def f(x):
    a = x[:, 0]  # extract a-coordinates of all points
    b = x[:, 1]  # extract b-coordinates of all points

    # clamp exponent to avoid exp() overflow
    exponent = torch.clamp((a ** 2 + 2 * b ** 2) ** 5, max=50)
    # term 1: penalizes points far from the origin
    term1 = torch.mean(torch.exp(exponent))

    # compute pairwise differences between all points
    diff = x[:, None, :] - x[None, :, :]
    # compute euclidean distances, add small value to avoid sqrt(0)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    # term 2: penalizes points that are too close to each other
    term2 = torch.mean(torch.exp(-10 * dist))

    return term1 + term2


# ── Stochastic objective function ─────────────────────────────────
def f_stoch(x):
    # sample random indices i, j, l uniformly from {0, ..., k-1}
    i = torch.randint(0, n, (1,)).item()
    j = torch.randint(0, n, (1,)).item()
    l = torch.randint(0, n, (1,)).item()

    a_i = x[i, 0]
    b_i = x[i, 1]
    exponent = torch.clamp((a_i ** 2 + 2 * b_i ** 2) ** 5, max=50)
    # stochastic term 1: single random point
    term1 = torch.exp(exponent)

    # stochastic term 2: single random pair of points
    diff = x[j] - x[l]
    dist = torch.sqrt((diff ** 2).sum() + 1e-8)
    term2 = torch.exp(-10 * dist)

    return term1 + term2


# ── Helper: fresh starting point ──────────────────────────────────
def init_x():
    # initialize points near origin for numerical stability
    x = torch.randn(n, 2) * 0.01
    # enable gradient tracking for autograd
    x.requires_grad_(True)
    return x


# ── Helper: compute gradient ──────────────────────────────────────
def get_grad(x, stochastic=False):
    # reset gradient to avoid accumulation
    if x.grad is not None:
        x.grad.zero_()
    # use stochastic or full objective function
    loss = f_stoch(x) if stochastic else f(x)
    # compute gradients via backpropagation
    loss.backward()
    # return loss value and a copy of the gradient
    return loss.item(), x.grad.clone()


# ── 1. Newton-Raphson ──────────────────────────────────────────────
def newton_raphson(iterations=100):
    x = init_x()
    losses = []
    for _ in range(iterations):
        loss_val, grad = get_grad(x)
        losses.append(loss_val)

        # compute exact Hessian matrix via PyTorch autograd
        x_fresh = x.detach().requires_grad_(True)
        hess = torch.autograd.functional.hessian(f, x_fresh)

        # reshape Hessian from (20,2,20,2) to (40,40)
        hess = hess.reshape(n * 2, n * 2)
        # flatten gradient from (20,2) to (40,)
        grad_flat = grad.reshape(-1)

        # regularization to ensure positive definiteness
        hess = hess + 0.1 * torch.eye(n * 2)

        # solve H*delta = grad instead of explicitly inverting H
        delta = torch.linalg.solve(hess, grad_flat)

        # clamp step size to prevent divergence
        delta = torch.clamp(delta, -0.1, 0.1)

        # update x without tracking this step
        with torch.no_grad():
            x -= delta.reshape(n, 2)

    return x.detach(), losses


# ── 2. Gradient Descent with Momentum ─────────────────────────────
def gd_momentum(eta=0.001, beta=0.9, iterations=1000):
    x = init_x()
    # initialize velocity to zero
    velocity = torch.zeros_like(x)
    losses = []
    for _ in range(iterations):
        loss_val, grad = get_grad(x)
        losses.append(loss_val)
        with torch.no_grad():
            # update velocity as weighted average of past gradients
            velocity = beta * velocity + (1 - beta) * grad
            x -= eta * velocity
    return x.detach(), losses


# ── 3. ADAM without stochastic gradients ──────────────────────────
def adam(eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8, iterations=1000):
    x = init_x()
    # initialize first and second moment estimates
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    losses = []
    for t in range(1, iterations + 1):
        loss_val, grad = get_grad(x)
        losses.append(loss_val)
        with torch.no_grad():
            # update biased moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            # bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            # update x with adaptive learning rate
            x -= eta * m_hat / (torch.sqrt(v_hat) + eps)
    return x.detach(), losses


# ── 4. Plain SGD ───────────────────────────────────────────────────
def plain_sgd(eta=0.001, iterations=1000):
    x = init_x()
    losses = []
    for _ in range(iterations):
        # use stochastic gradient
        loss_val, grad = get_grad(x, stochastic=True)
        losses.append(loss_val)
        with torch.no_grad():
            x -= eta * grad
    return x.detach(), losses


# ── 5. SGD with Momentum ───────────────────────────────────────────
def sgd_momentum(eta=0.001, beta=0.9, iterations=1000):
    x = init_x()
    velocity = torch.zeros_like(x)
    losses = []
    for _ in range(iterations):
        # use stochastic gradient
        loss_val, grad = get_grad(x, stochastic=True)
        losses.append(loss_val)
        with torch.no_grad():
            velocity = beta * velocity + (1 - beta) * grad
            x -= eta * velocity
    return x.detach(), losses


# ── 6. ADAM with stochastic gradients ─────────────────────────────
def adam_stoch(eta=0.01, beta1=0.9, beta2=0.999, eps=1e-8, iterations=1000):
    x = init_x()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    losses = []
    for t in range(1, iterations + 1):
        # use stochastic gradient
        loss_val, grad = get_grad(x, stochastic=True)
        losses.append(loss_val)
        with torch.no_grad():
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x -= eta * m_hat / (torch.sqrt(v_hat) + eps)
    return x.detach(), losses


# ── Run all methods ────────────────────────────────────────────────
print("Running Newton-Raphson...")
x1, l1 = newton_raphson()

print("Running GD with Momentum...")
x2, l2 = gd_momentum()

print("Running ADAM...")
x3, l3 = adam()

print("Running Plain SGD...")
x4, l4 = plain_sgd()

print("Running SGD with Momentum...")
x5, l5 = sgd_momentum()

print("Running ADAM (stochastic)...")
x6, l6 = adam_stoch()

# ── Plot 1: Loss curves ────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(l1, label="Newton-Raphson")
plt.plot(l2, label="GD with Momentum")
plt.plot(l3, label="ADAM")
plt.plot(l4, label="Plain SGD")
plt.plot(l5, label="SGD with Momentum")
plt.plot(l6, label="ADAM (stochastic)")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence Comparison")
plt.legend()
plt.yscale("log")
plt.grid(True)
plt.show()

# ── Plot 2: Best point distributions ──────────────────────────────
methods = [
    (x1, l1, "Newton-Raphson"),
    (x2, l2, "GD with Momentum"),
    (x3, l3, "ADAM"),
    (x4, l4, "Plain SGD"),
    (x5, l5, "SGD with Momentum"),
    (x6, l6, "ADAM (stochastic)"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for idx, (x_opt, loss, name) in enumerate(methods):
    x_np = x_opt.numpy()
    best = min(loss)
    axes[idx].scatter(x_np[:, 0], x_np[:, 1])
    axes[idx].set_title(f"{name}\nBest loss: {best:.4f}")
    axes[idx].grid(True)

plt.tight_layout()
plt.show()

# ── Report best values ─────────────────────────────────────────────
print("\nBest objective values:")
for x_opt, loss, name in methods:
    print(f"{name}: {min(loss):.6f}")