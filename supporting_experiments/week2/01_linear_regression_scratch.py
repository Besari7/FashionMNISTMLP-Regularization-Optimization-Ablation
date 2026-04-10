"""
01_linear_regression_scratch.py
===============================
Lab Topic Coverage:
- Linear regression from scratch on a small synthetic dataset
- Closed-form (normal equation) solution
- Gradient descent solution
- Demonstration / note showing why MLE corresponds to MSE under Gaussian noise

Outputs:
  outputs/week2_linreg_fit.png
  outputs/week2_linreg_mle_note.txt
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

# ── 1. Synthetic Dataset ───────────────────────────────────────────────────
np.random.seed(42)
N = 100
# y = 3.5 * x + 2.0 + noise
X = 2 * np.random.rand(N, 1)        # [0, 2]
y = 2.0 + 3.5 * X + np.random.randn(N, 1) * 0.5  # Gaussian noise

# Add bias term (X_b has shape (N, 2))
X_b = np.hstack([np.ones((N, 1)), X])

# ── 2. Closed-Form Solution (Normal Equations) ─────────────────────────────
# theta = (X^T * X)^(-1) * X^T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# ── 3. Gradient Descent Solution ───────────────────────────────────────────
eta = 0.1       # learning rate
epochs = 100
m = N
theta_gd = np.random.randn(2, 1)

for epoch in range(epochs):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta_gd - eta * gradients

print("--- Linear Regression (Scratch) ---")
print(f"Ground Truth : bias = 2.000, w = 3.500")
print(f"Normal Eq.   : bias = {theta_best[0][0]:.3f}, w = {theta_best[1][0]:.3f}")
print(f"Grad Descent : bias = {theta_gd[0][0]:.3f}, w = {theta_gd[1][0]:.3f}")

# ── 4. Plot ────────────────────────────────────────────────────────────────
X_new = np.array([[0], [2]])
X_new_b = np.hstack([np.ones((2, 1)), X_new])
y_predict_best = X_new_b.dot(theta_best)
y_predict_gd   = X_new_b.dot(theta_gd)

plt.figure(figsize=(8, 5))
plt.plot(X, y, "b.", label="Data (y = 3.5x + 2 + N(0, 0.5))")
plt.plot(X_new, y_predict_best, "r-", linewidth=2, label="Normal Equation")
plt.plot(X_new, y_predict_gd, "g--", linewidth=2, label="Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Solutions")
plt.legend()
plt.tight_layout()

out_plot = os.path.join(OUT_DIR, "week2_linreg_fit.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"[+] Saved plot: {out_plot}")

# ── 5. MLE to MSE Explanation ──────────────────────────────────────────────
mle_note = """
MLE and MSE Equivalence under Gaussian Noise
============================================
Let target y_i = theta^T x_i + epsilon_i, where epsilon_i ~ N(0, sigma^2).
The probability of y_i given x_i is normally distributed:
p(y_i | x_i; theta) = (1 / sqrt(2*pi*sigma^2)) * exp( - (y_i - theta^T x_i)^2 / (2*sigma^2) )

To find the Maximum Likelihood Estimate (MLE) for theta, we maximize the log-likelihood (LL):
LL(theta) = sum [ log(p(y_i | x_i; theta)) ]
          = sum [ -log(sqrt(2*pi*sigma^2)) - (y_i - theta^T x_i)^2 / (2*sigma^2) ]

Notice that the first term is a constant with respect to theta.
Therefore, maximizing LL(theta) is exactly mathematically equivalent to MINIMIZING the second term:
minimize: sum [ (y_i - theta^T x_i)^2 ]

This is exactly the Mean Squared Error (MSE) cost function!
Thus: MLE under Gaussian Noise === Minimizing MSE.
"""
out_txt = os.path.join(OUT_DIR, "week2_linreg_mle_note.txt")
with open(out_txt, "w") as f:
    f.write(mle_note)
print(f"[+] Saved note: {out_txt}")
