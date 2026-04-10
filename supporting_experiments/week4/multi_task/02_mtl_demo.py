"""
02_mtl_demo.py
==============
Week 4 Topic: Multi-task learning

A lightweight multi-task neural network with one shared backbone and two output heads.
Task 1: Predict whether the number is > 0 (Binary Classification)
Task 2: Predict the exact quadrant of the 2D point (4-class Classification)

Outputs:
  outputs/week4_mtl_curves.png
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
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Synthetic MTL Dataset ───────────────────────────────────────────────
torch.manual_seed(42)
N = 1000
X = torch.randn(N, 2) * 2

# Task 1: Is X[:, 0] > 0? (Binary)
y_task1 = (X[:, 0] > 0).float().view(-1, 1)

# Task 2: Quadrant (0, 1, 2, 3) (Multi-class)
y_task2 = torch.zeros(N, dtype=torch.long)
y_task2[(X[:, 0] > 0) & (X[:, 1] > 0)] = 0  # Q1
y_task2[(X[:, 0] <= 0) & (X[:, 1] > 0)] = 1 # Q2
y_task2[(X[:, 0] <= 0) & (X[:, 1] <= 0)] = 2 # Q3
y_task2[(X[:, 0] > 0) & (X[:, 1] <= 0)] = 3  # Q4

# ── 2. Multi-Task Network Architecture ─────────────────────────────────────
class MTLNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared Trunk
        self.shared = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Head 1 (Binary)
        self.head1 = nn.Linear(16, 1)
        # Head 2 (4 classes)
        self.head2 = nn.Linear(16, 4)
        
    def forward(self, x):
        features = self.shared(x)
        out1 = self.head1(features)
        out2 = self.head2(features)
        return out1, out2

model = MTLNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Losses
crit1 = nn.BCEWithLogitsLoss()
crit2 = nn.CrossEntropyLoss()

# ── 3. Training Loop ───────────────────────────────────────────────────────
epochs = 200
history = {"loss": [], "acc1": [], "acc2": []}

print("--- Multi-Task Learning (MTL) ---")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    out1, out2 = model(X)
    
    # Joint Loss
    loss1 = crit1(out1, y_task1)
    loss2 = crit2(out2, y_task2)
    joint_loss = loss1 + loss2  # Equal weighting for simplicity
    
    joint_loss.backward()
    optimizer.step()
    
    # Metrics
    with torch.no_grad():
        acc1 = ((torch.sigmoid(out1) > 0.5) == y_task1).float().mean()
        acc2 = (out2.argmax(dim=1) == y_task2).float().mean()
        
    history["loss"].append(joint_loss.item())
    history["acc1"].append(acc1.item() * 100)
    history["acc2"].append(acc2.item() * 100)
    
    if (epoch+1) % 40 == 0:
        print(f"Epoch {epoch+1:3d} | Joint Loss: {joint_loss.item():.4f} | "
              f"T1 Acc: {acc1.item()*100:.1f}% | T2 Acc: {acc2.item()*100:.1f}%")

# ── 4. Plotting ────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("Epochs")
ax1.set_ylabel("Joint Loss", color="k")
ax1.plot(history["loss"], "k-", lw=2, label="Joint Loss (L1 + L2)")
ax1.tick_params(axis="y", labelcolor="k")

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy (%)", color="k")
ax2.plot(history["acc1"], "r--", lw=2, label="Task 1 (Positive X) Acc")
ax2.plot(history["acc2"], "b:", lw=2, label="Task 2 (Quadrant) Acc")
ax2.tick_params(axis="y", labelcolor="k")
ax2.set_ylim(0, 105)

fig.suptitle("Multi-Task Learning Convergence (Shared Backbone)")
fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "week4_mtl_curves.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[+] Saved plot -> {out_path}")
