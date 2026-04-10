"""
02_mlp_from_scratch.py
======================
Lab Topic Coverage:
- Small MLP from scratch (NumPy)
- Manual forward pass
- Manual loss computation (MSE)
- Manual backpropagation step-by-step
- Simple training loop on a toy dataset

Outputs:
  outputs/week3_mlp_scratch_loss.png
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

# ── Data (Toy regression dataset) ──────────────────────────────────────────
np.random.seed(42)
X = np.random.randn(100, 2)
# True function: non-linear
y = (np.sin(X[:, 0]) + np.cos(X[:, 1])).reshape(-1, 1)

# ── Initialization ─────────────────────────────────────────────────────────
input_size = 2
hidden_size = 10
output_size = 1

# Weights (He initialization)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, output_size))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ── Training Loop (Manual Backprop) ────────────────────────────────────────
epochs = 500
lr = 0.01
losses = []

for epoch in range(epochs):
    # --- FORWARD PASS ---
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = Z2  # Linear output
    
    # --- LOSS COMPUTATION (MSE) ---
    m = X.shape[0]
    loss = (1 / (2 * m)) * np.sum((A2 - y)**2)
    losses.append(loss)
    
    # --- BACKPROPAGATION (Chain Rule) ---
    # Gradient of loss w.r.t A2: (A2 - y) / m
    dZ2 = (A2 - y) / m
    
    # Gradients for Layer 2
    dW2 = A1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    
    # Gradient propagating through Layer 2 to A1
    dA1 = dZ2.dot(W2.T)
    
    # Gradient propagating through ReLU (Z1)
    dZ1 = dA1 * relu_deriv(Z1)
    
    # Gradients for Layer 1
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    
    # --- APPY GRADIENTS (SGD step) ---
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:<3} | MSE Loss: {loss:.5f}")

print(f"Final Epoch | MSE Loss: {losses[-1]:.5f}")

# ── Plot ───────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(losses, "b-", linewidth=2)
plt.title("MLP From Scratch - Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

out_plot = os.path.join(OUT_DIR, "week3_mlp_scratch_loss.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"[+] Saved plot: {out_plot}")
