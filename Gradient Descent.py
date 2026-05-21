import torch
import matplotlib.pyplot as plt

n = 20

# Input

# Starting point
torch.manual_seed(42)
x = torch.randn(n, 2) * 0.01
x.requires_grad_(True)

# Function to minimize
def f(x):
    a = x[:, 0]
    b = x[:, 1]

    # Term 1
    exponent = torch.clamp((a ** 2 + 2 * b ** 2) ** 5, max=50)
    term1 = torch.mean(torch.exp(exponent))

    # Term 2
    diff = x[:, None, :] - x[None, :, :]
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    term2 = torch.mean(torch.exp(-10 * dist))

    return term1 + term2


# Parameters
eta = 0.001 # Step size
N = 2000 # Number of steps

for i in range(N):
    loss = f(x)
    loss.backward()

    with torch.no_grad():
        x -= eta * x.grad
    x.grad.zero_()

    if i % 500 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

# Output

# Approximate local minimizer
x_np = x.detach().numpy()
print(x_np)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(x_np[:, 0], x_np[:, 1])
plt.title("Gradient Descent - Optimized Points")
plt.xlabel("a")
plt.ylabel("b")
plt.grid(True)
plt.show()