"""
01_pseudo_labeling.py
=====================
Week 4 Topic: Semi-supervised learning

Demonstrates semi-supervised learning using a pseudo-labeling (self-training) approach.
Compares a baseline supervised model (trained on 50 labels) vs a semi-supervised model
(trained on 50 labels + pseudo-labels from 500 unlabeled points).

Outputs:
  outputs/week4_semi_supervised.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Data Setup (Small Labeled, Large Unlabeled) ─────────────────────────
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)

# Split indices
np.random.seed(42)
indices = np.random.permutation(len(X))

labeled_idx   = indices[:50]        # 50 labeled
unlabeled_idx = indices[50:550]     # 500 unlabeled
test_idx      = indices[550:]       # 450 test

X_L, y_L = X[labeled_idx], y[labeled_idx]
X_U = X[unlabeled_idx]
X_T, y_T = X[test_idx], y[test_idx]

# ── 2. Supervised Baseline ─────────────────────────────────────────────────
print("--- Semi-Supervised Pseudo-Labeling ---")
model_baseline = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=42)
model_baseline.fit(X_L, y_L)

acc_baseline = accuracy_score(y_T, model_baseline.predict(X_T))
print(f"Supervised Baseline Accuracy (n=50) : {acc_baseline*100:.2f}%")

# ── 3. Pseudo-Labeling (Self-Training) ─────────────────────────────────────
# Step 1: Predict on Unlabeled data using Baseline
pseudo_probs = model_baseline.predict_proba(X_U)
pseudo_labels = np.argmax(pseudo_probs, axis=1)
confidences = np.max(pseudo_probs, axis=1)

# Step 2: Keep only high-confidence predictions (thresh = 0.85)
confident_idx = confidences > 0.85
X_U_confident = X_U[confident_idx]
y_U_pseudo    = pseudo_labels[confident_idx]

# Step 3: Combine Labeled + High-Confidence Pseudo-Labeled
X_combined = np.vstack([X_L, X_U_confident])
y_combined = np.hstack([y_L, y_U_pseudo])

# Step 4: Train new model on combined data
model_semi = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=42)
model_semi.fit(X_combined, y_combined)

acc_semi = accuracy_score(y_T, model_semi.predict(X_T))
print(f"Pseudo-Labeled (n=50 + {sum(confident_idx)} pseudo) : {acc_semi*100:.2f}%")

# ── 4. Plotting Decision Boundaries ────────────────────────────────────────
def plot_boundaries(ax, model, title):
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 100), np.linspace(-1.0, 1.5, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X_L[y_L==0, 0], X_L[y_L==0, 1], c="red", marker="o", edgecolors="k", label="Class 0 (L)")
    ax.scatter(X_L[y_L==1, 0], X_L[y_L==1, 1], c="blue", marker="s", edgecolors="k", label="Class 1 (L)")
    ax.scatter(X_U[:, 0], X_U[:, 1], c="gray", marker=".", alpha=0.2, label="Unlabeled")
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_boundaries(axes[0], model_baseline, f"Baseline (Acc: {acc_baseline*100:.1f}%)")
plot_boundaries(axes[1], model_semi,     f"Pseudo-Labeling (Acc: {acc_semi*100:.1f}%)")

axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "week4_semi_supervised.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[+] Saved plot -> {out_path}")
