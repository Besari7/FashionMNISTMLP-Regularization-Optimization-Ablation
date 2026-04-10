"""
04_backprop_memoization_demo.py
===============================
Week 3 coverage:
- Computational graph intuition
- Backpropagation with explicit memoization (cached intermediates)

Output:
  supporting_experiments/outputs/week3_memoization_demo.txt
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# Tiny 2-layer network for binary classification-style output
W1 = np.random.randn(2, 4) * 0.2
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.2
b2 = np.zeros((1, 1))

X = np.random.randn(32, 2)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float64).reshape(-1, 1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(np.float64)


# Forward pass with memoization cache
cache = {}
cache["Z1"] = X @ W1 + b1
cache["A1"] = relu(cache["Z1"])
cache["Z2"] = cache["A1"] @ W2 + b2
cache["Yhat"] = sigmoid(cache["Z2"])

# BCE loss
eps = 1e-12
loss = -np.mean(y * np.log(cache["Yhat"] + eps) + (1 - y) * np.log(1 - cache["Yhat"] + eps))

# Backward pass explicitly reusing cached tensors
m = X.shape[0]
dZ2 = (cache["Yhat"] - y) / m
dW2 = cache["A1"].T @ dZ2
db2 = np.sum(dZ2, axis=0, keepdims=True)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * relu_grad(cache["Z1"])
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0, keepdims=True)

grad_norms = {
    "dW1": float(np.linalg.norm(dW1)),
    "db1": float(np.linalg.norm(db1)),
    "dW2": float(np.linalg.norm(dW2)),
    "db2": float(np.linalg.norm(db2)),
}

report = f"""
================================================================
Week 3: Backprop with Explicit Memoization
================================================================
Cached keys used in backward:
- {list(cache.keys())}

Loss: {loss:.6f}
Gradient norms:
- dW1: {grad_norms['dW1']:.6f}
- db1: {grad_norms['db1']:.6f}
- dW2: {grad_norms['dW2']:.6f}
- db2: {grad_norms['db2']:.6f}

Why memoization matters:
- Backward pass needs intermediate forward values (A1, Z1, Yhat).
- Caching avoids recomputation and keeps chain-rule implementation clean.
================================================================
""".strip()

out_txt = os.path.join(OUT_DIR, "week3_memoization_demo.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"[+] Saved {out_txt}")
