"""
optimization.py - Phase 2: Optimization / Training Stability
=============================================================

Lightweight, controlled experiments on the project backbone to compare
optimization and stability choices without retraining the full ablation
matrix.

Phase outputs:
  results/optimization/*.png
  results/optimization/*_metrics.json
  results/optimization/optimization_metrics.json
  checkpoints/optimization/*.pt
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results", "optimization")
CKPT_DIR = os.path.join(ROOT, "checkpoints", "optimization")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# Runtime controls (kept lightweight)
SEED = 42
SUBSET = 8_000
EPOCHS = 8
DEFAULT_BATCH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_cuda_for_retraining(stage_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage_name} retraining requires a CUDA-capable GPU. "
            "Artifacts can still be reused without retraining."
        )


def _phase2_retraining_required() -> bool:
    group_specs = {
        "optimizer_comparison": [
            "SGD", "SGD+Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam", "AdamW"
        ],
        "init_comparison": ["Default", "Xavier", "He"],
        "lr_schedule_comparison": ["None", "Step", "Cosine", "Warmup+Cosine"],
        "grad_clip_comparison": ["NoClip", "Clip1.0"],
        "batch_size_comparison": ["BS=32", "BS=128", "BS=512"],
        "normalization_stability_comparison": ["None", "BatchNorm", "LayerNorm"],
    }

    prefix_map = {
        "optimizer_comparison": "opt_",
        "init_comparison": "init_",
        "lr_schedule_comparison": "sched_",
        "grad_clip_comparison": "clip_",
        "batch_size_comparison": "batch_",
        "normalization_stability_comparison": "norm_",
    }

    for group_key, labels in group_specs.items():
        checkpoint_names = [f"{prefix_map[group_key]}{_slug(label)}.pt" for label in labels]
        if _try_load_cached_group(group_key, checkpoint_names) is None:
            return True
    return False


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace("+", "plus")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("=", "")
        .replace(" ", "_")
    )


def _transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])


def _get_loaders(batch_size: int = DEFAULT_BATCH) -> Tuple[DataLoader, DataLoader]:
    full = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=_transform()
    )
    test = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=_transform()
    )

    idx = torch.randperm(
        len(full), generator=torch.Generator().manual_seed(SEED)
    )[:SUBSET]

    train_loader = DataLoader(
        Subset(full, idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test,
        batch_size=DEFAULT_BATCH,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def _build_model(init_mode: str = "he", norm_type: str = "none") -> nn.Module:
    dims = [784, 512, 256, 128, 10]
    layers: List[nn.Module] = []

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))

        if i < len(dims) - 2:
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(out_dim))
            elif norm_type == "layernorm":
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        if init_mode == "he":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif init_mode == "xavier":
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif init_mode == "default":
            # Keep PyTorch defaults unchanged.
            pass
        else:
            raise ValueError(f"Unknown init mode: {init_mode}")

    return model.to(DEVICE)


def _make_optimizer(name: str, model: nn.Module):
    n = name.lower()
    if n == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.01)
    if n == "sgd+momentum":
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if n == "nesterov":
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    if n == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=0.01)
    if n == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=0.001)
    if n == "adam":
        return torch.optim.Adam(model.parameters(), lr=0.001)
    if n == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=0.001)
    raise ValueError(f"Unknown optimizer: {name}")


def _make_scheduler(name: str, optimizer: torch.optim.Optimizer):
    n = name.lower()
    if n == "none":
        return None
    if n == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    if n == "warmup+cosine":
        warmup_epochs = 2

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")


def _evaluate_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(1, total)


def _train_one_config(
    model: nn.Module,
    optimizer_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scheduler_name: str = "none",
    clip_norm: float = None,
) -> Tuple[List[float], List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(optimizer_name, model)
    scheduler = _make_scheduler(scheduler_name, optimizer)

    train_losses: List[float] = []
    test_accs: List[float] = []

    for _epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            n += y.size(0)

        if scheduler is not None:
            scheduler.step()

        train_losses.append(running_loss / max(1, n))
        test_accs.append(_evaluate_accuracy(model, test_loader))

    return train_losses, test_accs


def _plot_comparison(results: Dict[str, Dict[str, List[float]]], title: str, save_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#d1495b", "#30638e", "#00798c", "#edae49", "#66a182", "#5a189a", "#1b4332"]

    for idx, (name, data) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        epochs = range(1, len(data["train_loss"]) + 1)
        ax1.plot(epochs, data["train_loss"], lw=2, color=c, label=name)
        ax2.plot(epochs, data["test_acc"], lw=2, color=c, label=name)

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved {save_path}")


def _group_paths(group_key: str, checkpoint_names: List[str]) -> Tuple[str, str, List[str]]:
    plot_path = os.path.join(OUT_DIR, f"{group_key}.png")
    metrics_path = os.path.join(OUT_DIR, f"{group_key}_metrics.json")
    ckpt_paths = [os.path.join(CKPT_DIR, name) for name in checkpoint_names]
    return plot_path, metrics_path, ckpt_paths


def _try_load_cached_group(group_key: str, checkpoint_names: List[str]):
    plot_path, metrics_path, ckpt_paths = _group_paths(group_key, checkpoint_names)
    if os.path.exists(plot_path) and os.path.exists(metrics_path) and all(os.path.exists(p) for p in ckpt_paths):
        with open(metrics_path, "r") as f:
            cached = json.load(f)
        print(f"  [+] {group_key}: cached artifacts found, skipping.")
        return cached
    return None


def _save_group_metrics(group_key: str, results: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    summary = {name: round(values["test_acc"][-1], 4) for name, values in results.items()}
    metrics_path = os.path.join(OUT_DIR, f"{group_key}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [+] Saved {metrics_path}")
    return summary


def _save_model_checkpoint(name: str, model: nn.Module) -> None:
    path = os.path.join(CKPT_DIR, name)
    torch.save(model.state_dict(), path)


def run_optimizer_comparison():
    group_key = "optimizer_comparison"
    labels = [
        "SGD",
        "SGD+Momentum",
        "Nesterov",
        "AdaGrad",
        "RMSProp",
        "Adam",
        "AdamW",
    ]
    checkpoint_names = [f"opt_{_slug(label)}.pt" for label in labels]
    cached = _try_load_cached_group(group_key, checkpoint_names)
    if cached is not None:
        return cached

    print("  => Running optimizer comparison...")
    train_loader, test_loader = _get_loaders(DEFAULT_BATCH)

    results: Dict[str, Dict[str, List[float]]] = {}
    for label in labels:
        _set_seed(SEED)
        model = _build_model(init_mode="he", norm_type="none")
        losses, accs = _train_one_config(model, label, train_loader, test_loader)
        results[label] = {"train_loss": losses, "test_acc": accs}
        _save_model_checkpoint(f"opt_{_slug(label)}.pt", model)
        print(f"    {label:16s} final_acc={accs[-1]:.2f}%")

    _plot_comparison(
        results,
        f"Optimizer Comparison (Project Backbone, {SUBSET} subset)",
        os.path.join(OUT_DIR, f"{group_key}.png"),
    )
    return _save_group_metrics(group_key, results)


def run_init_comparison():
    group_key = "init_comparison"
    labels = ["Default", "Xavier", "He"]
    checkpoint_names = [f"init_{_slug(label)}.pt" for label in labels]
    cached = _try_load_cached_group(group_key, checkpoint_names)
    if cached is not None:
        return cached

    print("  => Running initialization comparison...")
    train_loader, test_loader = _get_loaders(DEFAULT_BATCH)

    mapping = {
        "Default": "default",
        "Xavier": "xavier",
        "He": "he",
    }

    results: Dict[str, Dict[str, List[float]]] = {}
    for label in labels:
        _set_seed(SEED)
        model = _build_model(init_mode=mapping[label], norm_type="none")
        losses, accs = _train_one_config(model, "AdamW", train_loader, test_loader)
        results[label] = {"train_loss": losses, "test_acc": accs}
        _save_model_checkpoint(f"init_{_slug(label)}.pt", model)
        print(f"    {label:16s} final_acc={accs[-1]:.2f}%")

    _plot_comparison(
        results,
        "Initialization Comparison (Default vs Xavier vs He)",
        os.path.join(OUT_DIR, f"{group_key}.png"),
    )
    return _save_group_metrics(group_key, results)


def run_lr_schedule_comparison():
    group_key = "lr_schedule_comparison"
    labels = ["None", "Step", "Cosine", "Warmup+Cosine"]
    checkpoint_names = [f"sched_{_slug(label)}.pt" for label in labels]
    cached = _try_load_cached_group(group_key, checkpoint_names)
    if cached is not None:
        return cached

    print("  => Running learning-rate schedule comparison...")
    train_loader, test_loader = _get_loaders(DEFAULT_BATCH)

    results: Dict[str, Dict[str, List[float]]] = {}
    for label in labels:
        _set_seed(SEED)
        model = _build_model(init_mode="he", norm_type="none")
        losses, accs = _train_one_config(
            model,
            "AdamW",
            train_loader,
            test_loader,
            scheduler_name=label,
        )
        pretty = {
            "None": "No Schedule",
            "Step": "StepLR",
            "Cosine": "CosineAnnealingLR",
            "Warmup+Cosine": "Warmup+Cosine",
        }[label]
        results[pretty] = {"train_loss": losses, "test_acc": accs}
        _save_model_checkpoint(f"sched_{_slug(label)}.pt", model)
        print(f"    {pretty:20s} final_acc={accs[-1]:.2f}%")

    _plot_comparison(
        results,
        "Learning-Rate Scheduling Comparison",
        os.path.join(OUT_DIR, f"{group_key}.png"),
    )
    return _save_group_metrics(group_key, results)


def run_grad_clip_comparison():
    group_key = "grad_clip_comparison"
    labels = ["NoClip", "Clip1.0"]
    checkpoint_names = [f"clip_{_slug(label)}.pt" for label in labels]
    cached = _try_load_cached_group(group_key, checkpoint_names)
    if cached is not None:
        return cached

    print("  => Running gradient clipping comparison...")
    train_loader, test_loader = _get_loaders(DEFAULT_BATCH)

    settings = [
        ("No Clipping", None, "NoClip"),
        ("Gradient Clip (max_norm=1.0)", 1.0, "Clip1.0"),
    ]

    results: Dict[str, Dict[str, List[float]]] = {}
    for label, clip_norm, ckpt_label in settings:
        _set_seed(SEED)
        model = _build_model(init_mode="he", norm_type="none")
        losses, accs = _train_one_config(
            model,
            "AdamW",
            train_loader,
            test_loader,
            clip_norm=clip_norm,
        )
        results[label] = {"train_loss": losses, "test_acc": accs}
        _save_model_checkpoint(f"clip_{_slug(ckpt_label)}.pt", model)
        print(f"    {label:28s} final_acc={accs[-1]:.2f}%")

    _plot_comparison(
        results,
        "Gradient Clipping Comparison",
        os.path.join(OUT_DIR, f"{group_key}.png"),
    )
    return _save_group_metrics(group_key, results)


def run_batch_size_comparison():
    group_key = "batch_size_comparison"
    labels = ["BS=32", "BS=128", "BS=512"]
    checkpoint_names = [f"batch_{_slug(label)}.pt" for label in labels]
    cached = _try_load_cached_group(group_key, checkpoint_names)
    if cached is not None:
        return cached

    print("  => Running mini-batch size comparison...")

    batch_sizes = [32, 128, 512]
    pretty_labels = ["BS=32", "BS=128", "BS=512"]

    results: Dict[str, Dict[str, List[float]]] = {}
    for bs, label in zip(batch_sizes, pretty_labels):
        _set_seed(SEED)
        train_loader, test_loader = _get_loaders(bs)
        model = _build_model(init_mode="he", norm_type="none")
        losses, accs = _train_one_config(model, "AdamW", train_loader, test_loader)
        results[label] = {"train_loss": losses, "test_acc": accs}
        _save_model_checkpoint(f"batch_{_slug(label)}.pt", model)
        print(f"    {label:10s} final_acc={accs[-1]:.2f}%")

    _plot_comparison(
        results,
        "Mini-Batch Size Comparison (SGD noise vs stability)",
        os.path.join(OUT_DIR, f"{group_key}.png"),
    )
    return _save_group_metrics(group_key, results)


def run_normalization_stability_comparison():
    group_key = "normalization_stability_comparison"
    labels = ["None", "BatchNorm", "LayerNorm"]
    checkpoint_names = [f"norm_{_slug(label)}.pt" for label in labels]
    cached = _try_load_cached_group(group_key, checkpoint_names)
    if cached is not None:
        return cached

    print("  => Running normalization stability comparison...")
    train_loader, test_loader = _get_loaders(DEFAULT_BATCH)

    mapping = {
        "None": "none",
        "BatchNorm": "batchnorm",
        "LayerNorm": "layernorm",
    }

    results: Dict[str, Dict[str, List[float]]] = {}
    for label in labels:
        _set_seed(SEED)
        model = _build_model(init_mode="he", norm_type=mapping[label])
        losses, accs = _train_one_config(model, "AdamW", train_loader, test_loader)
        results[label] = {"train_loss": losses, "test_acc": accs}
        _save_model_checkpoint(f"norm_{_slug(label)}.pt", model)
        print(f"    {label:12s} final_acc={accs[-1]:.2f}%")

    _plot_comparison(
        results,
        "Normalization for Training Stability (None vs BN vs LN)",
        os.path.join(OUT_DIR, f"{group_key}.png"),
    )
    return _save_group_metrics(group_key, results)


def run_optimization_phase():
    """Phase 2: Optimization / Training Stability."""
    print("\n" + "=" * 60)
    print("  PHASE 2: OPTIMIZATION / TRAINING STABILITY")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Budget: subset={SUBSET}, epochs={EPOCHS}, batch={DEFAULT_BATCH}")
    print("=" * 60)

    if _phase2_retraining_required():
        _ensure_cuda_for_retraining("Phase 2")

    _set_seed(SEED)

    combined = {
        "optimizer": run_optimizer_comparison(),
        "initialization": run_init_comparison(),
        "lr_schedule": run_lr_schedule_comparison(),
        "grad_clip": run_grad_clip_comparison(),
        "batch_size": run_batch_size_comparison(),
        "normalization": run_normalization_stability_comparison(),
    }

    combined_path = os.path.join(OUT_DIR, "optimization_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  [+] Saved {combined_path}")

    print("-" * 60)
    print("  Phase 2 artifacts are ready in results/optimization/")
    print("  Phase 2 checkpoints are ready in checkpoints/optimization/")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    run_optimization_phase()
