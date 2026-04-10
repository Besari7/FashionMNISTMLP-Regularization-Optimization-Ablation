"""
04_boosting_demo.py
===================
Week 4 Topic: Boosting

Demonstrates AdaBoost (Adaptive Boosting) stage-wise training,
showing how sequential weak learners focus on previous mistakes.

Outputs:
  outputs/week4_boosting_stages.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Challenging Dataset (Concentric Circles) ────────────────────────────
X, y = make_circles(n_samples=400, noise=0.15, factor=0.5, random_state=42)

# ── 2. Stage-wise AdaBoost ─────────────────────────────────────────────────
print("--- Boosting (AdaBoost) ---")
# Evaluated at these stages to show progression
stages = [1, 5, 20, 50]
models = []

for n in stages:
    # Base estimator: stump (depth=1)
    base = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(estimator=base, n_estimators=n, random_state=42)
    clf.fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    models.append((clf, acc))
    print(f"AdaBoost ({n:2d} stumps) Acc: {acc*100:.1f}%")

# ── 3. Plot Progression ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

for i, (n, (clf, acc)) in enumerate(zip(stages, models)):
    Z = clf.predict(grid).reshape(xx.shape)
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    axes[i].scatter(X[y==0, 0], X[y==0, 1], c="red", marker="o", edgecolors="k", s=20)
    axes[i].scatter(X[y==1, 0], X[y==1, 1], c="blue", marker="s", edgecolors="k", s=20)
    axes[i].set_title(f"Stage: {n} Stump(s)\nAcc: {acc*100:.1f}%")
    axes[i].set_xticks([])
    axes[i].set_yticks([])

fig.suptitle("Boosting progression: Sequential weak learners combining to solve a non-linear problem", y=1.05)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "week4_boosting_stages.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[+] Saved plot -> {out_path}")
