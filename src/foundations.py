import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR  = os.path.join(ROOT, "results", "foundations")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "foundations")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _phase0_retraining_required() -> bool:
    """Return True when any Phase 0 training artifact is missing."""
    training_artifacts = [
        os.path.join(OUT_DIR, "logreg_metrics.json"),
        os.path.join(OUT_DIR, "depth_vs_width_metrics.json"),
        os.path.join(OUT_DIR, "ce_vs_mse_metrics.json"),
    ]
    return not all(os.path.exists(path) for path in training_artifacts)


def _ensure_cuda_for_retraining(stage_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage_name} retraining requires a CUDA-capable GPU. "
            "Artifacts can still be reused without retraining."
        )

def _get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

# ── 1. Logistic Regression ─────────────────────────────────────────────────
class LogisticRegression(nn.Module):
    def __init__(self, in_features: int = 784, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.linear(x)

def run_logistic_baseline():
    expected_out = os.path.join(OUT_DIR, "logreg_metrics.json")
    if os.path.exists(expected_out):
        print("  [+] Logistic baseline artifacts found. Skipping execution.")
        return

    print("  => Running Logistic Baseline...")
    BATCH_SIZE, EPOCHS, LR, SEED, TRAIN_FRAC = 256, 10, 1e-3, 42, 0.8
    torch.manual_seed(SEED)

    full_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=_get_transforms())
    test_ds    = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=_get_transforms())

    n_train = int(len(full_train) * TRAIN_FRAC)
    train_ds, val_ds = random_split(full_train, [n_train, len(full_train) - n_train],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = LogisticRegression().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    @torch.no_grad()
    def accuracy(loader):
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        return loss_sum / total, 100.0 * correct / total

    history = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(y)

        tr_loss = ep_loss / n_train
        val_loss, val_acc = accuracy(val_loader)
        history.append(dict(epoch=epoch, train_loss=round(tr_loss, 6),
                            val_loss=round(val_loss, 6), val_acc=round(val_acc, 3)))

    _, test_acc = accuracy(test_loader)
    
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "logreg.pt"))
    res = dict(test_acc=round(test_acc, 3), elapsed_s=round(time.time()-t0, 1), epochs=EPOCHS, history=history)
    with open(expected_out, "w") as f:
        json.dump(res, f, indent=2)

# ── 2. Activation Plots ────────────────────────────────────────────────────
def generate_activation_plots():
    expected_out = os.path.join(OUT_DIR, "activation_functions.png")
    if os.path.exists(expected_out):
        print("  [+] Activation plots artifact found. Skipping execution.")
        return

    print("  => Generating Activation Plots...")
    x = np.linspace(-4, 4, 400)
    activations = [
        ("ReLU", np.maximum(0, x), "#e74c3c"),
        ("Sigmoid", 1 / (1 + np.exp(-x)), "#3498db"),
        ("Tanh", np.tanh(x), "#2ecc71"),
        ("Leaky ReLU\n(α=0.1)", np.where(x >= 0, x, 0.1 * x), "#f39c12"),
        ("ELU\n(α=1.0)", np.where(x >= 0, x, 1.0 * (np.exp(x) - 1)), "#9b59b6"),
    ]

    fig = plt.figure(figsize=(16, 4))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    for i, (name, y, color) in enumerate(activations):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#16213e")
        ax.plot(x, y, color=color, lw=2.5)
        ax.axhline(0, color="#aaaaaa", lw=0.6, ls="--")
        ax.axvline(0, color="#aaaaaa", lw=0.6, ls="--")
        ax.set_title(name, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("x", color="#cccccc", fontsize=8)
        ax.set_ylabel("f(x)", color="#cccccc", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor("#333355")
        ax.set_xlim(-4, 4)

    fig.suptitle("Activation Function Gallery", color="white", fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(expected_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

# ── 3. Backprop Sanity Check ───────────────────────────────────────────────
def run_backprop_sanity_check():
    expected_out = os.path.join(OUT_DIR, "backprop_sanity.txt")
    if os.path.exists(expected_out):
        print("  [+] Backprop sanity check log found. Skipping execution.")
        return

    print("  => Running Backprop Sanity Check...")
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(784, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 10),
    )
    batch_size = 64
    x = torch.randn(batch_size, 784)
    y = torch.randint(0, 10, (batch_size,))
    
    loss = nn.CrossEntropyLoss()(model(x), y)
    model.zero_grad()
    loss.backward()

    lines = ["=" * 60, "  Backpropagation Sanity Check", f"  Mini-batch size : {batch_size}", f"  Forward loss    : {loss.item():.6f}", "=" * 60, "  Per-parameter gradient diagnostics:", "  " + "-" * 80]
    all_ok = True
    for name, param in model.named_parameters():
        g = param.grad
        if g is None:
            lines.append(f"  {name:<30} {str(list(param.shape)):<18} {'N/A':>12} {'NO':>10}")
            all_ok = False
        else:
            finite = bool(torch.isfinite(g).all())
            all_ok = all_ok and finite
            lines.append(f"  {name:<30} {str(list(param.shape)):<18} {g.norm().item():>12.6f} {'YES':>10}")

    lines.extend(["", f"  Overall check PASSED: {all_ok}", "=" * 60])
    with open(expected_out, "w") as f:
        f.write("\n".join(lines) + "\n")

# ── 4. Depth vs Width ──────────────────────────────────────────────────────
def run_depth_vs_width_comparison():
    expected_out = os.path.join(OUT_DIR, "depth_vs_width_metrics.json")
    if os.path.exists(expected_out):
        print("  [+] Depth vs Width artifacts found. Skipping execution.")
        return

    print("  => Running Depth vs Width Comparison...")
    BATCH, EPOCHS, LR, SEED = 256, 5, 1e-3, 42
    torch.manual_seed(SEED)
    full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=_get_transforms())
    tr_ds, va_ds = random_split(full, [48000, 12000], generator=torch.Generator().manual_seed(SEED))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False)

    def build_model(arch):
        if arch == "shallow": return nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
        if arch == "deeper":  return nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
        if arch == "wider":   return nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))

    @torch.no_grad()
    def eval_acc(model):
        model.eval()
        corr, tot = 0, 0
        for x, y in va_dl:
            corr += (model(x.to(device)).argmax(1) == y.to(device)).sum().item()
            tot += len(y)
        return 100.0 * corr / tot

    results = {}
    for arch in ["shallow", "deeper", "wider"]:
        m = build_model(arch).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()
        val_accs = []
        for _ in range(EPOCHS):
            m.train()
            for x, y in tr_dl:
                opt.zero_grad()
                crit(m(x.to(device)), y.to(device)).backward()
                opt.step()
            val_accs.append(eval_acc(m))
        results[arch] = dict(val_accs=val_accs, final_val_acc=val_accs[-1])

    with open(expected_out, "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#aaaaaa")
    colors = {"shallow": "#e74c3c", "deeper": "#3498db", "wider": "#2ecc71"}
    for arch, d in results.items():
        ax.plot(range(1, EPOCHS+1), d["val_accs"], color=colors[arch], lw=2, label=f"{arch}")
    ax.set_title("Depth vs Width", color="white")
    ax.legend(facecolor="#1e1e3a", labelcolor="white")
    plt.savefig(os.path.join(OUT_DIR, "depth_vs_width.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close()

# ── 5. CE vs MSE ───────────────────────────────────────────────────────────
def run_ce_vs_mse_comparison():
    expected_out = os.path.join(OUT_DIR, "ce_vs_mse_metrics.json")
    if os.path.exists(expected_out):
        print("  [+] CE vs MSE artifacts found. Skipping execution.")
        return

    print("  => Running CE vs MSE Comparison...")
    torch.manual_seed(42)
    full = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=_get_transforms())
    idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(42))[:10000]
    loader = DataLoader(Subset(full, idx.tolist()), batch_size=256, shuffle=True)
    
    def run_loss(name):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        accs = []
        for _ in range(10):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                if name == "CE": loss = nn.CrossEntropyLoss()(logits, y)
                else: loss = F.mse_loss(F.softmax(logits, dim=1), F.one_hot(y, 10).float())
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                accs.append(100.0 * sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x, y in loader) / 10000)
        return accs

    res = {"CE": run_loss("CE"), "MSE": run_loss("MSE")}
    with open(expected_out, "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.plot(range(1, 11), res["CE"], "#e74c3c", lw=2, label="CrossEntropy")
    ax.plot(range(1, 11), res["MSE"], "#3498db", lw=2, label="MSE")
    ax.legend(facecolor="#1e1e3a", labelcolor="white")
    plt.savefig(os.path.join(OUT_DIR, "ce_vs_mse_curves.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close()

# ── Master Runner ──────────────────────────────────────────────────────────
def run_foundations_phase():
    """Phase 0: Execute methodological foundations conditionally."""
    print("\n" + "="*50)
    print(" PHASE 0: METHODOLOGICAL FOUNDATIONS")
    print("="*50)

    if _phase0_retraining_required():
        _ensure_cuda_for_retraining("Phase 0")
    
    run_logistic_baseline()
    generate_activation_plots()
    run_backprop_sanity_check()
    run_depth_vs_width_comparison()
    run_ce_vs_mse_comparison()
    
    print("--------------------------------------------------")
    print(" Phase 0 artifacts are ready in results/foundations/")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_foundations_phase()
