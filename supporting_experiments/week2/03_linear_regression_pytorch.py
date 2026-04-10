"""
03_linear_regression_pytorch.py
===============================
Lab Topic Coverage:
- Linear regression in PyTorch

Outputs:
  checkpoints/week2_linreg_torch.pt
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────
torch.manual_seed(42)
N = 100
X = 2 * torch.rand(N, 1)
y = 2.0 + 3.5 * X + torch.randn(N, 1) * 0.5

# ── Model ──────────────────────────────────────────────────────────────────
class PyTorchLinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 1 input, 1 output
        
    def forward(self, x):
        return self.linear(x)

model = PyTorchLinReg()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("--- Linear Regression (PyTorch) ---")
print(f"Target : bias = 2.0, weight = 3.5")
print(f"Init   : bias = {model.linear.bias.item():.3f}, weight = {model.linear.weight.item():.3f}")

# ── Train ──────────────────────────────────────────────────────────────────
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

print(f"Final  : bias = {model.linear.bias.item():.3f}, weight = {model.linear.weight.item():.3f}")
print(f"Loss   : {loss.item():.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
out_ckpt = os.path.join(CKPT_DIR, "week2_linreg_torch.pt")
torch.save(model.state_dict(), out_ckpt)
print(f"[+] Saved checkpoint: {out_ckpt}")
