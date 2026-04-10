"""
02_linreg_sgd_vs_batch.py
=========================
Lab Topic Coverage:
- SGD vs batch gradient descent comparison

Outputs:
  outputs/week2_sgd_vs_batch.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 100
X = 2 * np.random.rand(N, 1)
y = 4 + 3 * X + np.random.randn(N, 1)
X_b = np.hstack([np.ones((N, 1)), X])
theta_true = np.array([[4.0], [3.0]])

def compute_mse(theta):
    return np.mean((X_b.dot(theta) - y)**2)

# ── Batch Gradient Descent ─────────────────────────────────────────────────
eta_bgd = 0.1
n_iterations = 50
theta_bgd = np.random.randn(2, 1)
bgd_path = [theta_bgd]
bgd_loss = [compute_mse(theta_bgd)]

for _ in range(n_iterations):
    gradients = 2/N * X_b.T.dot(X_b.dot(theta_bgd) - y)
    theta_bgd = theta_bgd - eta_bgd * gradients
    bgd_path.append(theta_bgd)
    bgd_loss.append(compute_mse(theta_bgd))

# ── Stochastic Gradient Descent ────────────────────────────────────────────
eta_sgd = 0.1
n_epochs = 50
theta_sgd = bgd_path[0].copy() # Start from exact same random init
sgd_path = [theta_sgd]
sgd_loss = [compute_mse(theta_sgd)]

# SGD decays LR slightly for convergence
def learning_schedule(t):
    t0, t1 = 5, 50
    return t0 / (t + t1)

np.random.seed(42)
t = 0
for epoch in range(n_epochs):
    # Just to keep total steps to 50 for direct plotting comparison, 
    # we take 1 stochastic step per epoch here instead of N steps, 
    # to show structural variance.
    # (Normally SGD does N steps per epoch)
    random_index = np.random.randint(N)
    xi = X_b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    gradients = 2 * xi.T.dot(xi.dot(theta_sgd) - yi)
    eta = learning_schedule(t)
    theta_sgd = theta_sgd - eta * gradients
    
    sgd_path.append(theta_sgd)
    sgd_loss.append(compute_mse(theta_sgd))
    t += 1

print("--- Batch vs SGD (Linear Regression) ---")
print(f"Final BGD loss: {bgd_loss[-1]:.4f}")
print(f"Final SGD loss: {sgd_loss[-1]:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────
bgd_path = np.array(bgd_path)
sgd_path = np.array(sgd_path)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: Loss over time
axes[0].plot(bgd_loss, "b-", linewidth=2, label="Batch GD (Smooth)")
axes[0].plot(sgd_loss, "r--", linewidth=1.5, label="Stochastic GD (Noisy steps)")
axes[0].set_xlabel("Iteration Step")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Loss Convergence")
axes[0].legend()

# Plot 2: Parameter space traversal
axes[1].plot(bgd_path[:,0], bgd_path[:,1], "b-o", linewidth=2, label="Batch GD")
axes[1].plot(sgd_path[:,0], sgd_path[:,1], "r-+", linewidth=1.5, label="SGD")
axes[1].plot(4.0, 3.0, "k*", markersize=12, label="True Theta")
axes[1].set_xlabel("Theta 0 (Bias)")
axes[1].set_ylabel("Theta 1 (Weight)")
axes[1].set_title("Parameter Space Traversal")
axes[1].legend()

plt.tight_layout()
out_plot = os.path.join(OUT_DIR, "week2_sgd_vs_batch.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"[+] Saved plot: {out_plot}")
