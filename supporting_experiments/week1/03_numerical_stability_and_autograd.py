"""
03_numerical_stability_and_autograd.py
======================================
Week 1 coverage:
- Numerical overflow / underflow in naive computations
- Stable softmax / log-sum-exp
- Gradient-based optimization and PyTorch autograd

Output:
  supporting_experiments/outputs/week1_numerical_autograd_report.txt
"""

import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Overflow/underflow demo
x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)

# Naive softmax (can overflow)
def softmax_naive(z):
    ez = np.exp(z)
    return ez / np.sum(ez)

# Stable softmax
def softmax_stable(z):
    z_shift = z - np.max(z)
    ez = np.exp(z_shift)
    return ez / np.sum(ez)

naive_failed = False
try:
    sm_naive = softmax_naive(x)
except FloatingPointError:
    sm_naive = np.array([np.nan, np.nan, np.nan])
    naive_failed = True

sm_stable = softmax_stable(x)

# Autograd and gradient-based optimization demo
torch.manual_seed(42)
param = torch.tensor([5.0], requires_grad=True)
optimizer = torch.optim.SGD([param], lr=0.1)
loss_history = []

for _ in range(40):
    optimizer.zero_grad()
    loss = (param - 2.0) ** 2
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

report = f"""
================================================================
Week 1: Numerical Computation and Autograd
================================================================
Naive softmax input: {x.tolist()}
Naive softmax output: {sm_naive.tolist()} (failed={naive_failed})
Stable softmax output: {sm_stable.tolist()}
Stable softmax sum: {float(np.sum(sm_stable)):.6f}

Gradient-based optimization with autograd:
- Objective: (theta - 2)^2
- Initial theta: 5.0
- Final theta: {param.item():.6f}
- Initial loss: {loss_history[0]:.6f}
- Final loss: {loss_history[-1]:.6f}

Notes:
- Shifting logits by max(logit) avoids overflow in exp.
- Autograd builds a computational graph and computes gradients automatically.
================================================================
""".strip()

out_txt = os.path.join(OUT_DIR, "week1_numerical_autograd_report.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"[+] Saved {out_txt}")
