import torch
import matplotlib.pyplot as plt

n = 20

# ── Objective function ────────────────────────────────────────────
def f(x):
    a = x[:, 0]  # extract a-coordinates of all points
    b = x[:, 1]  # extract b-coordinates of all points

    # clamp exponent to avoid exp() overflow
    exponent = torch.clamp((a**2 + 2*b**2)**5, max=50)
    # term 1: penalizes points far from the origin
    term1 = torch.mean(torch.exp(exponent))

    # compute pairwise differences between all points
    diff = x[:, None, :] - x[None, :, :]
    # compute euclidean distances, add small value to avoid sqrt(0)
    dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)
    # term 2: penalizes points that are too close to each other
    term2 = torch.mean(torch.exp(-10 * dist))

    return term1 + term2

# initialize points near origin for numerical stability
torch.manual_seed(42)
x = torch.randn(n, 2) * 0.01
# enable gradient tracking for autograd
x.requires_grad_(True)

# ── Gradient Descent ──────────────────────────────────────────────
eta = 0.001      # learning rate
iterations = 1000

for i in range(iterations):
    loss = f(x)
    # compute gradients via backpropagation
    loss.backward()

    # update x without tracking this step
    with torch.no_grad():
        x -= eta * x.grad
    # reset gradient to avoid accumulation
    x.grad.zero_()

    if i % 500 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

# ── Plot ──────────────────────────────────────────────────────────
# detach from autograd and convert to numpy for plotting
x_np = x.detach().numpy()

plt.figure(figsize=(6,6))
plt.scatter(x_np[:, 0], x_np[:, 1])
plt.title("Gradient Descent - Optimized Points")
plt.xlabel("a")
plt.ylabel("b")
plt.grid(True)
plt.show()

print(f"Final loss: {f(x).item():.6f}")