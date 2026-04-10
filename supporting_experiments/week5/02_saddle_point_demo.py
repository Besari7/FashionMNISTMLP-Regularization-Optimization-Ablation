"""
02_saddle_point_demo.py — Saddle Point Escape Visualization
=============================================================
Demonstrates how different optimizers handle a saddle-point surface
f(x,y) = x² - y²  (the classic saddle).  Momentum-based methods
escape the saddle far faster than vanilla SGD.

Run:  python supporting_experiments/week5/02_saddle_point_demo.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "week5")
os.makedirs(OUT_DIR, exist_ok=True)

def saddle(xy):
    return xy[0]**2 - xy[1]**2

def run_on_saddle(opt_class, opt_kwargs, steps=200, start=(0.01, 0.01)):
    xy = torch.tensor(list(start), dtype=torch.float32, requires_grad=True)
    opt = opt_class([xy], **opt_kwargs)
    path = [xy.detach().clone().numpy()]
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = saddle(xy)
        loss.backward()
        opt.step()
        path.append(xy.detach().clone().numpy())
        losses.append(loss.item())
    return np.array(path), losses

def main():
    save_path = os.path.join(OUT_DIR, "saddle_point_demo.png")
    if os.path.exists(save_path):
        print("[+] Saddle point demo artifact found. Skipping.")
        return

    print("=> Generating saddle point escape demo...")

    configs = {
        "SGD (lr=0.01)":     (torch.optim.SGD, {"lr": 0.01}),
        "SGD+Momentum":      (torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
        "Adam":              (torch.optim.Adam, {"lr": 0.01}),
    }

    x_range = np.linspace(-1.5, 1.5, 200)
    y_range = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 - Y**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # Left: trajectories on surface
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")
    ax1.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.6, linewidths=0.5)
    ax1.contourf(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.15)
    ax1.plot(0, 0, "wx", markersize=12, mew=2, label="Saddle Point")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    all_losses = {}
    for (name, (cls, kw)), c in zip(configs.items(), colors):
        path, losses = run_on_saddle(cls, kw, steps=150)
        ax1.plot(path[:, 0], path[:, 1], color=c, lw=2, alpha=0.8, label=name)
        ax1.plot(path[0, 0], path[0, 1], "o", color=c, ms=5)
        all_losses[name] = losses

    ax1.set_title("Saddle Point Trajectories  f(x,y) = x² − y²",
                  color="white", fontweight="bold", fontsize=11)
    ax1.set_xlabel("x", color="#cccccc")
    ax1.set_ylabel("y", color="#cccccc")
    ax1.tick_params(colors="#aaaaaa")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333355")
    ax1.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=8)

    # Right: loss over steps
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    for (name, losses), c in zip(all_losses.items(), colors):
        ax2.plot(losses, color=c, lw=2, label=name)
    ax2.set_title("Loss Over Steps (escape speed)", color="white",
                  fontweight="bold", fontsize=11)
    ax2.set_xlabel("Step", color="#cccccc")
    ax2.set_ylabel("Loss", color="#cccccc")
    ax2.tick_params(colors="#aaaaaa")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333355")
    ax2.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=8)
    ax2.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved {save_path}")

if __name__ == "__main__":
    main()
