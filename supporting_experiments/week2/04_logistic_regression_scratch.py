"""
04_logistic_regression_scratch.py
=================================
Lab Topic Coverage:
- Logistic regression from scratch on a small binary dataset

Outputs:
  outputs/week2_logreg_boundary.png
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

# ── 1. Synthetic Binary Dataset ────────────────────────────────────────────
np.random.seed(42)
N = 200
# Two blob classes
X1 = np.random.randn(N//2, 2) + np.array([-1, -1])  # Class 0
X2 = np.random.randn(N//2, 2) + np.array([2, 2])    # Class 1
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(N//2), np.ones(N//2)]).reshape(-1, 1)

# Add bias
X_b = np.hstack([np.ones((N, 1)), X])

# ── 2. Logistic Regression (Math) ──────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_log_loss(theta, X_b, y):
    m = len(y)
    h = sigmoid(X_b.dot(theta))
    # Add epsilon to prevent log(0)
    eps = 1e-15
    return -1/m * np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))

eta = 0.5
epochs = 1000
theta = np.zeros((3, 1))

losses = []
for epoch in range(epochs):
    h = sigmoid(X_b.dot(theta))
    gradients = 1/N * X_b.T.dot(h - y)
    theta = theta - eta * gradients
    if epoch % 100 == 0:
        losses.append(compute_log_loss(theta, X_b, y))

preds = (sigmoid(X_b.dot(theta)) >= 0.5).astype(int)
accuracy = np.mean(preds == y) * 100
print("--- Logistic Regression (Scratch) ---")
print(f"Accuracy: {accuracy:.1f}%")

# ── 3. Plot Decision Boundary ──────────────────────────────────────────────
x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 200),
                       np.linspace(x1_min, x1_max, 200))
X_grid = np.c_[xx0.ravel(), xx1.ravel()]
X_grid_b = np.hstack([np.ones((X_grid.shape[0], 1)), X_grid])

Z = sigmoid(X_grid_b.dot(theta))
Z = Z.reshape(xx0.shape)

plt.figure(figsize=(6, 5))
# Contour map of probability
contour = plt.contourf(xx0, xx1, Z, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.3)
# Decision boundary line (p=0.5)
plt.contour(xx0, xx1, Z, levels=[0.5], linewidths=2, colors="k")

plt.scatter(X1[:, 0], X1[:, 1], c="red", marker="o", edgecolors="k", label="Class 0")
plt.scatter(X2[:, 0], X2[:, 1], c="blue", marker="s", edgecolors="k", label="Class 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Logistic Regression Boundary (Acc: {accuracy:.1f}%)")
plt.legend()
plt.tight_layout()

out_plot = os.path.join(OUT_DIR, "week2_logreg_boundary.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"[+] Saved plot: {out_plot}")
