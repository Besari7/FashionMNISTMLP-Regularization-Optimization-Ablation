"""
03_mlp_pytorch.py
=================
Lab Topic Coverage:
- Rewrite the precise same MLP in PyTorch

Outputs:
  outputs/week3_mlp_torch_loss.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data (Identical Toy regression dataset as NumPy) ───────────────────────
torch.manual_seed(42)
X = torch.randn(100, 2)
y = (torch.sin(X[:, 0]) + torch.cos(X[:, 1])).view(-1, 1)

# ── Model Definition ───────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

criterion = nn.MSELoss()
# Using simple SGD to exactly match the scratch NumPy implementation
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ── Training Loop ──────────────────────────────────────────────────────────
epochs = 500
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X)
    # PyTorch MSE averages the loss by default, multiplying by 0.5 to match custom math exactly
    loss = 0.5 * criterion(predictions, y) 
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:<3} | MSE Loss (PyTorch): {loss.item():.5f}")

print(f"Final Epoch | MSE Loss (PyTorch): {losses[-1]:.5f}")

# ── Plot ───────────────────────────────────────────────────────────────────
plt.figure()
plt.plot(losses, "m-", linewidth=2)
plt.title("MLP in PyTorch - Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

out_plot = os.path.join(OUT_DIR, "week3_mlp_torch_loss.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"[+] Saved plot: {out_plot}")
