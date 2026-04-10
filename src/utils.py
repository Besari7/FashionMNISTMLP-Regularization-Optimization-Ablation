"""
utils.py - Visualization & Analysis Utilities
===============================================

Plotting functions for the regularization ablation study:
    1. Training & validation curves (overfitting detection)
    2. Weight distribution histograms (L1 sparsity vs L2 shrinkage)
    3. Comparative summary bar charts
    4. Sparsity analysis
    5. Results table with multi-seed aggregation
"""

import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import random


# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

def set_seed(seed: int = 42):
    """Fix all sources of randomness for deterministic execution."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# COLOUR PALETTE (supports up to 8 experiments)
# ===========================================================================

COLORS = {
    "Baseline":          "#E74C3C",
    "L1":                "#3498DB",
    "L2":                "#2ECC71",
    "Dropout":           "#9B59B6",
    "BatchNorm":         "#F39C12",
    "L2+Dropout":        "#1ABC9C",
    "L2+BatchNorm":      "#E67E22",
    "Dropout+BatchNorm": "#34495E",
    "DataAug":           "#D35400",  # Pumpkin Orange
    "LabelSmoothing":    "#27AE60",  # Nephritis Green
}

def _color(name: str) -> str:
    return COLORS.get(name, "#7F8C8D")


# ===========================================================================
# 1. TRAINING CURVES
# ===========================================================================

def plot_training_curves(histories: Dict[str, Dict[str, List[float]]],
                         save_dir: str = "results",
                         figsize: Tuple[int, int] = (20, 10)):
    """
    Overlay training/validation loss and accuracy for all experiments.

    Left panel  - Loss (log scale) with train (solid) / val (dashed).
    Right panel - Accuracy with same convention.
    The gap between solid and dashed lines visualises overfitting.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Comparative Training Curves — "
                 "Regularization Ablation Study",
                 fontsize=16, fontweight="bold", y=1.02)

    for name, h in histories.items():
        c = _color(name)
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], "-",  color=c, alpha=0.8,
                     label=f"{name} (Train)", linewidth=2)
        axes[0].plot(epochs, h["val_loss"],   "--", color=c, alpha=0.6,
                     label=f"{name} (Val)",   linewidth=2)
        axes[1].plot(epochs, h["train_acc"],  "-",  color=c, alpha=0.8,
                     label=f"{name} (Train)", linewidth=2)
        axes[1].plot(epochs, h["val_acc"],    "--", color=c, alpha=0.6,
                     label=f"{name} (Val)",   linewidth=2)

    axes[0].set(xlabel="Epoch", ylabel="Cross-Entropy Loss",
                title="Training vs Validation Loss\n(Generalization Gap)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)",
                title="Training vs Validation Accuracy\n(Overfitting Detection)")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {path}")


# ===========================================================================
# 2. INDIVIDUAL TRAINING CURVES (per model, with generalization-gap fill)
# ===========================================================================

def plot_individual_training_curves(histories: Dict[str, Dict[str, List[float]]],
                                    save_dir: str = "results",
                                    figsize: Tuple[int, int] = (24, 10)):
    """One subplot per experiment: loss (top row) and accuracy (bottom row)."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    fig.suptitle("Individual Model Analysis — "
                 "Overfitting Dynamics",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, (name, h) in enumerate(histories.items()):
        c = _color(name)
        ep = range(1, len(h["train_loss"]) + 1)
        ax_t = axes[0, idx] if n > 1 else axes[0]
        ax_b = axes[1, idx] if n > 1 else axes[1]

        ax_t.plot(ep, h["train_loss"], "-",  color=c, lw=2, label="Train")
        ax_t.plot(ep, h["val_loss"],   "--", color=c, lw=2, alpha=.7,
                  label="Val")
        ax_t.fill_between(ep, h["train_loss"], h["val_loss"],
                          alpha=0.15, color=c, label="Gap")
        ax_t.set_title(name, fontsize=11, fontweight="bold")
        ax_t.set(xlabel="Epoch", ylabel="Loss")
        ax_t.legend(fontsize=7)
        ax_t.grid(True, alpha=0.3, linestyle="--")

        ax_b.plot(ep, h["train_acc"], "-",  color=c, lw=2, label="Train")
        ax_b.plot(ep, h["val_acc"],   "--", color=c, lw=2, alpha=.7,
                  label="Val")
        ax_b.fill_between(ep, h["val_acc"], h["train_acc"],
                          alpha=0.15, color=c)
        ax_b.set(xlabel="Epoch", ylabel="Accuracy (%)")
        ax_b.legend(fontsize=7)
        ax_b.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = os.path.join(save_dir, "individual_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {path}")


# ===========================================================================
# 3. WEIGHT DISTRIBUTION HISTOGRAMS
# ===========================================================================

def plot_weight_histograms(models: Dict[str, nn.Module],
                           save_dir: str = "results",
                           figsize: Tuple[int, int] = (24, 18)):
    """
    Grid of histograms: rows = experiments, columns = layers.

    Expected observations:
        Baseline  - wide, heavy-tailed distribution
        L1        - sharp spike at w=0 (Laplace prior / sparsity)
        L2        - narrow Gaussian bell around 0
        Dropout   - similar to Baseline (indirect regularization)
        BatchNorm - narrower, normalised distribution
    """
    os.makedirs(save_dir, exist_ok=True)
    n_models = len(models)
    n_layers = 4   # 3 hidden + 1 output
    layer_labels = ["Layer 1\n(784->512)", "Layer 2\n(512->256)",
                    "Layer 3\n(256->128)", "Output\n(128->10)"]

    fig, axes = plt.subplots(n_models, n_layers, figsize=figsize)
    fig.suptitle("Weight Distribution Histograms — "
                 "Regularization Effect",
                 fontsize=16, fontweight="bold", y=1.02)

    for row, (name, model) in enumerate(models.items()):
        c = _color(name)
        linear_weights = [m.weight.data.cpu().numpy().flatten()
                          for m in model.modules() if isinstance(m, nn.Linear)]
        for col in range(min(n_layers, len(linear_weights))):
            ax = axes[row, col] if n_models > 1 else axes[col]
            w = linear_weights[col]
            ax.hist(w, bins=100, color=c, alpha=0.75,
                    edgecolor="black", linewidth=0.3, density=True)
            mu, sigma = np.mean(w), np.std(w)
            sp = np.sum(np.abs(w) < 1e-3) / len(w) * 100
            ax.text(0.97, 0.97,
                    f"mu={mu:.4f}\nsig={sigma:.4f}\nSpar={sp:.1f}%",
                    transform=ax.transAxes, fontsize=7,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.8))
            ax.axvline(0, color="red", ls=":", alpha=0.5, lw=1)
            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
            if row == 0:
                ax.set_title(layer_labels[col], fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, "weight_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {path}")


# ===========================================================================
# 4. WEIGHT COMPARISON OVERLAY (single layer, all models)
# ===========================================================================

def plot_weight_comparison(models: Dict[str, nn.Module],
                           layer_idx: int = 0,
                           save_dir: str = "results",
                           figsize: Tuple[int, int] = (16, 8)):
    """Overlay weight histograms of a single layer across all experiments."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for name, model in models.items():
        lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if layer_idx < len(lins):
            w = lins[layer_idx].weight.data.cpu().numpy().flatten()
            ax.hist(w, bins=150, alpha=0.40, color=_color(name),
                    label=name, density=True, edgecolor="none")

    ax.set_title(f"Layer {layer_idx+1} Weight Distribution Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10, loc="upper right")
    ax.axvline(0, color="black", ls=":", alpha=0.5, lw=1.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.text(0.02, 0.97,
            "L1 -> Laplace prior: spike at w=0 (sparsity)\n"
            "L2 -> Gaussian prior: narrow bell around 0\n"
            "Dropout -> indirect effect on weights\n"
            "BatchNorm -> normalised distribution",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(save_dir, "weight_comparison_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {path}")


# ===========================================================================
# 5. COMPARATIVE SUMMARY BAR CHARTS
# ===========================================================================

def plot_comparative_summary(histories: Dict[str, Dict[str, List[float]]],
                             save_dir: str = "results",
                             figsize: Tuple[int, int] = (22, 8)):
    """Three-panel bar chart: final accuracy, overfitting gap, val loss."""
    os.makedirs(save_dir, exist_ok=True)
    names = list(histories.keys())
    n = len(names)
    colors = [_color(nm) for nm in names]

    train_acc = [histories[m]["train_acc"][-1] for m in names]
    val_acc   = [histories[m]["val_acc"][-1]   for m in names]
    gap       = [t - v for t, v in zip(train_acc, val_acc)]
    val_loss  = [histories[m]["val_loss"][-1]  for m in names]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Comparative Performance Summary",
                 fontsize=16, fontweight="bold", y=1.02)
    x = np.arange(n)

    # Panel 1 — accuracy
    axes[0].bar(x - 0.15, train_acc, 0.3, color=colors, alpha=0.9,
                edgecolor="black", lw=0.5, label="Train")
    bars = axes[0].bar(x + 0.15, val_acc, 0.3, color=colors, alpha=0.5,
                       edgecolor="black", lw=0.5, hatch="//", label="Val")
    for b in bars:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                     f"{b.get_height():.1f}%", ha="center", fontsize=7)
    axes[0].set(ylabel="Accuracy (%)",
                title="Final Accuracy\n(Train vs Validation)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y", linestyle="--")

    # Panel 2 — overfitting gap
    bars2 = axes[1].bar(x, gap, 0.6, color=colors, alpha=0.8,
                        edgecolor="black", lw=0.5)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                     f"{b.get_height():.2f}%", ha="center", fontsize=7,
                     fontweight="bold")
    axes[1].set(ylabel="Train Acc - Val Acc (%)",
                title="Overfitting Gap\n(lower = better generalisation)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].axhline(0, color="green", ls="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis="y", linestyle="--")

    # Panel 3 — val loss
    bars3 = axes[2].bar(x, val_loss, 0.6, color=colors, alpha=0.8,
                        edgecolor="black", lw=0.5)
    for b in bars3:
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{b.get_height():.4f}", ha="center", fontsize=7)
    axes[2].set(ylabel="Validation Loss",
                title="Final Validation Loss\n(CE only)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    path = os.path.join(save_dir, "comparative_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {path}")


# ===========================================================================
# 6. SPARSITY ANALYSIS
# ===========================================================================

def compute_sparsity(model: nn.Module,
                     threshold: float = 1e-3) -> Dict[str, float]:
    """
    Compute per-layer and total weight sparsity.

    Sparsity(W) = |{w : |w| < threshold}| / |W| * 100
    """
    sp = {}
    total_p = total_s = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            n = param.numel()
            s = torch.sum(torch.abs(param.data) < threshold).item()
            sp[name] = s / n * 100
            total_p += n
            total_s += s
    sp["TOTAL"] = total_s / total_p * 100 if total_p else 0
    return sp


def print_sparsity_report(models: Dict[str, nn.Module],
                          threshold: float = 1e-3):
    """Tabular sparsity report for all trained models."""
    print("\n" + "=" * 70)
    print(f"  SPARSITY ANALYSIS  (threshold: |w| < {threshold:.0e})")
    print("=" * 70)
    for name, model in models.items():
        sp = compute_sparsity(model, threshold)
        print(f"\n  Model: {name}")
        print(f"  {'─' * 55}")
        for layer, val in sp.items():
            star = " *" if val > 10 else ""
            print(f"    {layer:42s}  {val:6.2f}%{star}")
    print("=" * 70 + "\n")


def save_sparsity_metrics(models: Dict[str, nn.Module],
                          save_dir: str = "results",
                          threshold: float = 1e-3) -> Dict[str, Dict[str, float]]:
    """
    Persist sparsity metrics to JSON and CSV for report-ready evidence.

    CSV format is long-form: Experiment, Layer, SparsityPct.
    """
    os.makedirs(save_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        results[name] = compute_sparsity(model, threshold)

    json_path = os.path.join(save_dir, "sparsity_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(save_dir, "sparsity_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "Layer", "SparsityPct"])
        for exp_name, layer_map in results.items():
            for layer, val in layer_map.items():
                w.writerow([exp_name, layer, f"{val:.4f}"])

    print(f"  [+] Saved {json_path}")
    print(f"  [+] Saved {csv_path}")
    return results


# ===========================================================================
# 7. RESULTS TABLE
# ===========================================================================

def print_results_table(histories: Dict[str, Dict[str, List[float]]],
                        aggregated: Optional[Dict] = None):
    """
    Print final-epoch metrics and (optionally) multi-seed aggregated test
    results in a readable table.
    """
    print("\n" + "=" * 95)
    print("  ABLATION STUDY — FINAL RESULTS")
    print("=" * 95)
    print(f"  {'Model':<20s} | {'TrLoss':>10s} | {'VLoss':>10s} | "
          f"{'TrAcc':>8s} | {'VAcc':>8s} | {'Gap':>7s}", end="")
    if aggregated:
        print(f" | {'TestAcc (M+/-S)':>18s}", end="")
    print()
    print(f"  {'─' * 90}")

    for name, h in histories.items():
        tl = h["train_loss"][-1]
        vl = h["val_loss"][-1]
        ta = h["train_acc"][-1]
        va = h["val_acc"][-1]
        g  = ta - va
        line = (f"  {name:<20s} | {tl:>10.4f} | {vl:>10.4f} | "
                f"{ta:>7.2f}% | {va:>7.2f}% | {g:>6.2f}%")
        if aggregated and name in aggregated:
            a = aggregated[name]
            line += (f" | {a['test_acc_mean']:>6.2f} +/- "
                     f"{a['test_acc_std']:.2f}%")
        print(line)

    print("=" * 95)
    print("  Gap = Train Acc - Val Acc  (lower is better)")
    print("=" * 95 + "\n")


def save_generalization_gap_metrics(
    gap_summary: Dict[str, Dict[str, float]],
    save_dir: str = "results",
) -> None:
    """
    Save generalization-gap metrics (train vs val) to JSON and CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "generalization_gap_metrics.json")
    with open(json_path, "w") as f:
        json.dump(gap_summary, f, indent=2)

    csv_path = os.path.join(save_dir, "generalization_gap_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Experiment",
            "Train_Acc_Mean",
            "Train_Acc_Std",
            "Val_Acc_Mean",
            "Val_Acc_Std",
            "Gap_Mean",
            "Gap_Std",
            "Train_Loss_Mean",
            "Val_Loss_Mean",
        ])
        for exp_name, metrics in gap_summary.items():
            w.writerow([
                exp_name,
                f"{metrics['train_acc_mean']:.4f}",
                f"{metrics['train_acc_std']:.4f}",
                f"{metrics['val_acc_mean']:.4f}",
                f"{metrics['val_acc_std']:.4f}",
                f"{metrics['gap_mean']:.4f}",
                f"{metrics['gap_std']:.4f}",
                f"{metrics['train_loss_mean']:.6f}",
                f"{metrics['val_loss_mean']:.6f}",
            ])

    print(f"  [+] Saved {json_path}")
    print(f"  [+] Saved {csv_path}")


# ===========================================================================
# 8. ADVERSARIAL ROBUSTNESS (FGSM)
# ===========================================================================

def fgsm_attack(image: torch.Tensor,
                epsilon: float,
                data_grad: torch.Tensor) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) point-wise perturbation.
    Assumes input images are normalised to [-1, 1].
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # Clip to maintain [-1, 1] data range
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)
    return perturbed_image


def evaluate_fgsm_robustness(models: Dict[str, nn.Module],
                             test_loader: torch.utils.data.DataLoader,
                             device: torch.device,
                             epsilon: float = 0.05,
                             save_dir: str = "results") -> Dict[str, float]:
    """
    Evaluates adversarial robustness of all trained models using FGSM.
    Plots a bar chart comparing original test accuracy vs attacked accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    print(f"  Running FGSM Attack across {len(models)} models ...")

    for name, model in models.items():
        model.eval()
        correct_clean = 0
        correct_adv = 0
        total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            # Forward pass (clean)
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            
            # If the initial prediction is wrong, don't bother attacking
            # (standard practice in adversarial evaluation)
            mask_correct = init_pred.squeeze() == target
            if not mask_correct.any():
                continue
                
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()

            # Create adversarial examples
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            # Forward pass (adversarial)
            # No gradients needed for evaluation
            with torch.no_grad():
                output_adv = model(perturbed_data)
                
            # Tally metrics
            final_pred_adv = output_adv.max(1, keepdim=True)[1]
            correct_clean += mask_correct.sum().item()
            correct_adv += (final_pred_adv.squeeze()[mask_correct] == target[mask_correct]).sum().item()
            total += target.size(0)

        clean_acc = 100. * correct_clean / total if total > 0 else 0
        adv_acc = 100. * correct_adv / total if total > 0 else 0
        drop = clean_acc - adv_acc
        results[name] = {"clean_acc": clean_acc, "adv_acc": adv_acc}
        print(f"    {name:<20s} | Clean: {clean_acc:5.1f}% -> "
              f"FGSM: {adv_acc:5.1f}%  (Drop: {drop:5.1f}%)")

    # Plotting
    names = list(results.keys())
    clean_accs = [results[n]["clean_acc"] for n in names]
    adv_accs = [results[n]["adv_acc"] for n in names]
    
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, clean_accs, width, label='Clean Data', 
           color='#2C3E50', alpha=0.9, edgecolor='black')
    ax.bar(x + width/2, adv_accs, width, label=f'FGSM Attacked ($\\epsilon$={epsilon})', 
           color='#E74C3C', alpha=0.9, edgecolor='black', hatch='//')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Adversarial Robustness Comparison (FGSM $\\epsilon$={epsilon})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, drop in enumerate([c - a for c, a in zip(clean_accs, adv_accs)]):
        ax.text(i + width/2, adv_accs[i] + 1, f"-{drop:.1f}%", 
                ha='center', color='darkred', fontweight='bold', fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "fgsm_robustness_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {path}")
    
    return results
