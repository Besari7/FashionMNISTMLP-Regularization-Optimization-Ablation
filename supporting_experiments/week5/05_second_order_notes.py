"""
05_second_order_notes.py — Second-Order Optimization Methods
==============================================================
Brief pedagogical demonstration of Newton's method vs L-BFGS vs Adam
on a simple 2D quadratic to illustrate convergence rate differences.

Second-order methods are NOT used in the main project because:
  1. Computing/storing the full Hessian is O(n²) in parameter count.
  2. For a 530K-parameter MLP, the Hessian would be ~2.1 TB (infeasible).
  3. Even quasi-Newton (L-BFGS) requires full-batch gradients, breaking
     the minibatch SGD paradigm that the project relies on.
  4. In practice, adaptive first-order methods (Adam/AdamW) approximate
     diagonal second-order information sufficiently for most tasks.

Run:  python supporting_experiments/week5/05_second_order_notes.py
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

def ill_conditioned_quadratic(xy, A=None):
    """f(x) = 0.5 * x^T A x  where A has eigenvalues 1 and 50 (ill-conditioned)."""
    if A is None:
        A = torch.tensor([[50.0, 0.0], [0.0, 1.0]])
    return 0.5 * xy @ A @ xy

def run_optimizer(opt_class, opt_kwargs, loss_fn, steps=100, start=(5.0, 5.0)):
    xy = torch.tensor(list(start), dtype=torch.float32, requires_grad=True)
    opt = opt_class([xy], **opt_kwargs)
    path = [xy.detach().clone().numpy()]
    losses = []

    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(xy)
        loss.backward()
        opt.step()
        path.append(xy.detach().clone().numpy())
        losses.append(loss.item())

    return np.array(path), losses

def run_lbfgs(loss_fn, steps=30, start=(5.0, 5.0)):
    """L-BFGS requires a closure."""
    xy = torch.tensor(list(start), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([xy], lr=0.5, max_iter=5)
    path = [xy.detach().clone().numpy()]
    losses = []

    for _ in range(steps):
        def closure():
            opt.zero_grad()
            loss = loss_fn(xy)
            loss.backward()
            return loss
        loss = opt.step(closure)
        path.append(xy.detach().clone().numpy())
        losses.append(loss.item() if loss is not None else 0.0)

    return np.array(path), losses

def main():
    save_path = os.path.join(OUT_DIR, "second_order_comparison.png")
    if os.path.exists(save_path):
        print("[+] Second-order comparison artifact found. Skipping.")
        return

    print("=> Generating second-order methods comparison...")

    A = torch.tensor([[50.0, 0.0], [0.0, 1.0]])
    loss_fn = lambda xy: ill_conditioned_quadratic(xy, A)

    # Meshgrid
    xr = np.linspace(-6, 6, 200)
    yr = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(xr, yr)
    Z = 0.5 * (50 * X**2 + Y**2)

    # Run optimizers
    configs = {
        "SGD (lr=0.01)": run_optimizer(torch.optim.SGD, {"lr": 0.01}, loss_fn, steps=80),
        "Adam (lr=0.5)": run_optimizer(torch.optim.Adam, {"lr": 0.5}, loss_fn, steps=80),
        "L-BFGS":        run_lbfgs(loss_fn, steps=30),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    # Left: trajectories
    levels = np.logspace(0, 3.5, 25)
    ax1.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5, linewidths=0.5)
    ax1.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.15)
    ax1.plot(0, 0, "w*", markersize=15, zorder=10, label="Minimum")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for (name, (path, _)), c in zip(configs.items(), colors):
        ax1.plot(path[:, 0], path[:, 1], color=c, lw=2, alpha=0.8, label=name)
        ax1.plot(path[0, 0], path[0, 1], "o", color=c, ms=6)

    ax1.set_title("Trajectories on Ill-Conditioned Quadratic\n(kappa = 50)",
                  color="white", fontweight="bold", fontsize=11)
    ax1.set_xlabel("x1 (high curvature)", color="#cccccc")
    ax1.set_ylabel("x2 (low curvature)", color="#cccccc")
    ax1.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=8)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)

    # Right: loss curves
    for (name, (_, losses)), c in zip(configs.items(), colors):
        ax2.plot(losses, color=c, lw=2, label=name)
    ax2.set_title("Convergence Speed", color="white", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Step", color="#cccccc")
    ax2.set_ylabel("Loss (log scale)", color="#cccccc")
    ax2.set_yscale("log")
    ax2.legend(facecolor="#1e1e3a", labelcolor="white", fontsize=8)
    ax2.grid(True, alpha=0.2, linestyle="--")

    fig.suptitle("First-Order vs Second-Order Methods — Ill-Conditioned Problem",
                 color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved {save_path}")

    # Write text summary
    txt_path = os.path.join(OUT_DIR, "second_order_notes.txt")
    notes = """
==========================================================================
  SECOND-ORDER OPTIMIZATION -- WHY NOT USED IN THE MAIN PROJECT
==========================================================================

  Newton's Method:
    - Update rule:  theta <- theta - H^{-1} grad L(theta)
    - Converges quadratically near optimum (very fast)
    - BUT: requires O(n^2) storage for Hessian (n = param count)
    - For our 530K-param MLP: Hessian = 530K x 530K x 4 bytes ~ 1.1 TB
    - Completely infeasible for modern deep learning

  L-BFGS (Limited-memory BFGS):
    - Approximates the inverse Hessian using m previous gradient pairs
    - Much more memory-efficient than full Newton (O(mn) vs O(n^2))
    - BUT: requires full-batch gradients for stable approximation
    - Breaks the minibatch SGD paradigm that enables large-scale training
    - Works well for small problems (< ~10K parameters)

  Why Adam/AdamW works well in practice:
    - Maintains per-parameter running variance (v_t) of gradients
    - Update effectively scales by 1/sqrt(v_t) ~ diagonal Hessian inverse
    - This is a crude second-order approximation (diagonal assumption)
    - Combined with momentum -> fast convergence on ill-conditioned problems
    - O(n) memory -- same as SGD (just 2 extra state vectors)

  Conclusion:
    Adaptive first-order methods (Adam/AdamW) approximate diagonal
    second-order information at first-order cost, making them the
    practical choice for deep learning. True second-order methods
    remain primarily of theoretical/pedagogical interest for modern
    neural networks.
==========================================================================
"""
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(notes.strip() + "\n")
    print(f"[+] Saved {txt_path}")

if __name__ == "__main__":
    main()
