"""
03_bagging_demo.py
==================
Week 4 Topic: Bagging (Bootstrap Aggregating)

Builds an ensemble of 10 lightweight decision trees via bootstrap sampling.
Compares a single high-variance model vs the Bagged ensemble to show variance reduction.

Outputs:
  outputs/week4_bagging_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Noisy Dataset Setup ─────────────────────────────────────────────────
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, flip_y=0.2, random_state=42)

# Split
X_train, y_train = X[:300], y[:300]
X_test,  y_test  = X[300:], y[300:]

# ── 2. Single Model (Baseline High Variance) ───────────────────────────────
print("--- Bagging (Bootstrap Aggregating) ---")
# Fully grown tree = high variance / overfitting
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
acc_single = accuracy_score(y_test, tree.predict(X_test))
print(f"Single Decision Tree Acc: {acc_single*100:.1f}%")

# ── 3. Bagging Implementation ──────────────────────────────────────────────
n_estimators = 15
models = []

np.random.seed(42)
for i in range(n_estimators):
    # Bootstrap sample (sample with replacement)
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot, y_boot = X_train[indices], y_train[indices]
    
    t = DecisionTreeClassifier(max_depth=5, random_state=i)
    t.fit(X_boot, y_boot)
    models.append(t)

# Aggregate Predictions (Voting)
def bagging_predict(models, X):
    preds = np.array([m.predict(X) for m in models])
    # Majority vote
    return np.round(np.mean(preds, axis=0)).astype(int)

bagged_preds = bagging_predict(models, X_test)
acc_bagged = accuracy_score(y_test, bagged_preds)
print(f"Bagged Ensemble ({n_estimators} trees): {acc_bagged*100:.1f}%")

# ── 4. Plot Boundaries ─────────────────────────────────────────────────────
def plot_boundaries(ax, predict_fn, title):
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c="red", marker="o", edgecolors="k", alpha=0.8)
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c="blue", marker="s", edgecolors="k", alpha=0.8)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_boundaries(axes[0], tree.predict, f"Single Tree (Overfit Boundary) | {acc_single*100:.1f}%")
plot_boundaries(axes[1], lambda x: bagging_predict(models, x), f"Bagged Ensemble ({n_estimators} trees) | {acc_bagged*100:.1f}%")

fig.suptitle("Bagging explicitly reduces high variance in noisy decision boundaries")
out_path = os.path.join(OUT_DIR, "week4_bagging_comparison.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"[+] Saved plot -> {out_path}")
