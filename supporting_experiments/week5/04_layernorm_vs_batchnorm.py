"""
04_layernorm_vs_batchnorm.py — Layer Normalization vs Batch Normalization
==========================================================================
Compares BatchNorm1d and LayerNorm on a small MLP trained on FashionMNIST.

Key insight:
- BatchNorm normalises across the batch dimension (depends on batch stats).
- LayerNorm normalises across the feature dimension (independent of batch).
- LayerNorm is preferred for variable-length / small-batch / sequential data.
- BatchNorm is generally stronger for fixed-size batch training (CNNs, MLPs).

Run:  python supporting_experiments/week5/04_layernorm_vs_batchnorm.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "week5")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
SUBSET = 10_000
EPOCHS = 15
BATCH = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(norm_type="none"):
    layers = []
    dims = [784, 256, 128, 10]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:  # no norm on output
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(dims[i+1]))
            elif norm_type == "layernorm":
                layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def main():
    save_path = os.path.join(OUT_DIR, "layernorm_vs_batchnorm.png")
    if os.path.exists(save_path):
        print("[+] LayerNorm vs BatchNorm artifact found. Skipping.")
        return

    print("=> Running LayerNorm vs BatchNorm comparison...")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tfm)
    idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(SEED))[:SUBSET]
    train_dl = DataLoader(Subset(full, idx.tolist()), batch_size=BATCH, shuffle=True)
    test_dl = DataLoader(test, batch_size=BATCH, shuffle=False)

    configs = {
        "No Normalization": "none",
        "BatchNorm": "batchnorm",
        "LayerNorm": "layernorm",
    }

    results = {}
    for label, ntype in configs.items():
        torch.manual_seed(SEED)
        model = build_model(ntype).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()

        losses, accs = [], []
        for ep in range(EPOCHS):
            model.train()
            ep_loss, n = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(y)
                n += len(y)
            losses.append(ep_loss / n)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
            accs.append(100.0 * correct / total)

        results[label] = {"loss": losses, "acc": accs}
        print(f"  {label:20s}  final_acc={accs[-1]:.2f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    for (name, d), c in zip(results.items(), colors):
        ax1.plot(range(1, EPOCHS+1), d["loss"], color=c, lw=2, label=name)
        ax2.plot(range(1, EPOCHS+1), d["acc"],  color=c, lw=2, label=name)

    ax1.set_title("Training Loss", color="white", fontweight="bold")
    ax1.set_xlabel("Epoch", color="#cccccc")
    ax1.set_ylabel("CE Loss", color="#cccccc")
    ax1.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle="--")

    ax2.set_title("Test Accuracy", color="white", fontweight="bold")
    ax2.set_xlabel("Epoch", color="#cccccc")
    ax2.set_ylabel("Accuracy (%)", color="#cccccc")
    ax2.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle("LayerNorm vs BatchNorm — Normalisation Strategy Comparison",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved {save_path}")

if __name__ == "__main__":
    main()
