"""
01_optimizer_landscape.py — 2D Loss Surface Optimizer Trajectories
===================================================================
Visualises how different optimizers (SGD, Momentum, NAG, RMSProp, Adam)
navigate a 2D Rosenbrock loss surface, illustrating path-dependent
behaviour and convergence properties.

This is a pedagogical/toy demonstration — not connected to FashionMNIST.

Run:  python supporting_experiments/week5/01_optimizer_landscape.py
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "outputs", "week5")
os.makedirs(OUT_DIR, exist_ok=True)

# Rosenbrock function:  f(x,y) = (1-x)^2 + 100*(y-x^2)^2
def rosenbrock(xy):
    x, y = xy[0], xy[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def run_optimizer_on_surface(opt_class, opt_kwargs, steps=500, start=(-1.5, 1.5)):
    """Run an optimizer on the Rosenbrock surface and record trajectory."""
    xy = torch.tensor(list(start), dtype=torch.float32, requires_grad=True)
    opt = opt_class([xy], **opt_kwargs)
    path = [xy.detach().clone().numpy()]

    for _ in range(steps):
        opt.zero_grad()
        loss = rosenbrock(xy)
        loss.backward()
        opt.step()
        path.append(xy.detach().clone().numpy())

    return np.array(path)

def main():
    save_path = os.path.join(OUT_DIR, "optimizer_landscape.png")
    if os.path.exists(save_path):
        print("[+] Optimizer landscape artifact found. Skipping.")
        return

    print("=> Generating 2D optimizer trajectory landscape...")

    optimizers = {
        "SGD (lr=0.001)":       (torch.optim.SGD,  {"lr": 0.001}),
        "SGD+Momentum (0.9)":   (torch.optim.SGD,  {"lr": 0.001, "momentum": 0.9}),
        "NAG (Nesterov)":       (torch.optim.SGD,  {"lr": 0.001, "momentum": 0.9, "nesterov": True}),
        "RMSProp":              (torch.optim.RMSprop, {"lr": 0.001}),
        "Adam":                 (torch.optim.Adam, {"lr": 0.01}),
    }

    # Meshgrid for contour
    x_range = np.linspace(-2.0, 2.0, 300)
    y_range = np.linspace(-1.0, 3.0, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Contour
    levels = np.logspace(0, 3.5, 30)
    ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.6, linewidths=0.5)
    ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.2)

    # Optimal point
    ax.plot(1, 1, "w*", markersize=15, zorder=10, label="Minimum (1,1)")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for (name, (cls, kw)), c in zip(optimizers.items(), colors):
        path = run_optimizer_on_surface(cls, kw, steps=400)
        ax.plot(path[:, 0], path[:, 1], color=c, lw=1.5, alpha=0.85, label=name)
        ax.plot(path[0, 0], path[0, 1], "o", color=c, markersize=6)
        ax.plot(path[-1, 0], path[-1, 1], "s", color=c, markersize=6)

    ax.set_title("Optimizer Trajectories on Rosenbrock Surface",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("x", color="#cccccc")
    ax.set_ylabel("y", color="#cccccc")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")
    ax.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved {save_path}")

if __name__ == "__main__":
    main()
