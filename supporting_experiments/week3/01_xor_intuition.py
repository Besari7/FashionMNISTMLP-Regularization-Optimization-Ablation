"""
01_xor_intuition.py
===================
Lab Topic Coverage:
- XOR intuition demo showing linear inseparability
- Small hand-crafted minimum 2-layer MLP to solve XOR

Outputs:
  outputs/week3_xor_solution.png
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

# ── Data (XOR) ─────────────────────────────────────────────────────────────
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# ── Hand-Crafted MLP Weights (Solution to XOR) ─────────────────────────────
# Hidden Layer: 2 neurons
# Neuron 1: OR gate (w11=1, w21=1, b1=-0.5)
# Neuron 2: NAND gate (w12=-1, w22=-1, b2=1.5)
W1 = np.array([[1, -1],
               [1, -1]])
b1 = np.array([-0.5, 1.5])

# Output Layer: 1 neuron
# AND gate applied to the outputs of the hidden layer
W2 = np.array([[1],
               [1]])
b2 = np.array([-1.5])

def step_function(z):
    return (z >= 0).astype(int)

# ── Forward Pass ───────────────────────────────────────────────────────────
hidden_layer = step_function(X.dot(W1) + b1)
output_layer = step_function(hidden_layer.dot(W2) + b2).ravel()

print("--- XOR MLP Intuition (Manual Feedforward) ---")
print("Input X:")
print(X)
print("Target Y:")
print(y)
print("MLP Output:")
print(output_layer)

# ── Plot ───────────────────────────────────────────────────────────────────
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                     np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

h_grid = step_function(grid.dot(W1) + b1)
Z = step_function(h_grid.dot(W2) + b2)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], cmap="RdBu", alpha=0.3)
plt.scatter(X[y==0, 0], X[y==0, 1], c="red", marker="o", s=150, edgecolors="k", label="Class 0 (0,0 and 1,1)")
plt.scatter(X[y==1, 0], X[y==1, 1], c="blue", marker="s", s=150, edgecolors="k", label="Class 1 (0,1 and 1,0)")

plt.title("XOR Separated by a 2-Layer Perceptron")
plt.xlabel("Input x1")
plt.ylabel("Input x2")
plt.legend()
plt.tight_layout()

out_plot = os.path.join(OUT_DIR, "week3_xor_solution.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"[+] Saved plot: {out_plot}")
