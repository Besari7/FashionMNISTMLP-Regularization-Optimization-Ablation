"""
train.py - Core Regularization/Generalization + Final Robustness
=================================================================

Implements two pipeline components used by run.py:

    Phase 1      Core regularization/generalization study
    Final Phase  FGSM robustness summary from saved checkpoints

Related phases live in separate modules:

    Phase 0  src/foundations.py
    Phase 2  src/optimization.py

Usage (from project root):
    python run.py

Key methodological safeguards
-----------------------------
* Fair Dropout evaluation  - training accuracy is computed in eval() mode
  via a separate forward pass (Dropout disabled, BN uses running stats).
* CE-only loss logging     - L1 penalty is added to the backward loss but
  only the cross-entropy component is recorded, enabling apples-to-apples
  comparison with validation loss.
* Bias-excluded penalties  - L1 and L2 (weight-decay) are applied only to
  weight tensors, not biases or BN affine parameters.
* AdamW optimizer          - decoupled weight decay (Loshchilov & Hutter,
  2019) avoids the well-known interaction between Adam's adaptive LR and
  L2 gradient penalty.
"""

import os
import sys
import json
import time
import copy
import csv

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Ensure src/ is importable when run via ``python run.py``
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_DIM, NUM_CLASSES, TRAIN_SIZE, VAL_SIZE,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, PATIENCE, NUM_WORKERS,
    PIN_MEMORY, NON_BLOCKING,
    SEEDS, GRID_SEARCH_EPOCHS,
    DROPOUT_SEARCH_SPACE, L1_SEARCH_SPACE, L2_SEARCH_SPACE,
    RESULTS_DIR, CHECKPOINTS_DIR, DATA_DIR, DATASET_NAME,
    AUGMENTATION_ROTATION, AUGMENTATION_PADDING,
    LABEL_SMOOTHING, FGSM_EPSILON,
)
from models import (
    get_model, compute_l1_penalty, create_optimizer, count_parameters,
)
from utils import (
    set_seed,
    plot_training_curves,
    plot_individual_training_curves,
    plot_weight_histograms,
    plot_weight_comparison,
    plot_comparative_summary,
    print_sparsity_report,
    print_results_table,
    save_generalization_gap_metrics,
    save_sparsity_metrics,
    evaluate_fgsm_robustness,
)


# ===========================================================================
# DATA LOADING
# ===========================================================================

def get_data_loaders(batch_size: int = BATCH_SIZE,
                     seed: int = 42,
                     augment: bool = False):
    """
    Load Fashion-MNIST and split into train / val / test DataLoaders.

    Preprocessing pipeline:
        1. ToTensor()           -> [0, 1]
        2. Normalize(0.5, 0.5)  -> [-1, 1]  (zero-centred)
        3. Flatten              -> 784-dim vector (MLP input)

    Parameters
    ----------
    augment : bool
        If True, the *training* set uses random rotation and random crop
        as a data-augmentation regularization strategy.  Validation and
        test sets are never augmented.

    Split strategy:
        60 K training images  ->  48 K train  +  12 K validation
        10 K test images      ->  held out for final evaluation only
    """
    # --- standard (non-augmented) transform ---
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])

    # --- augmented training transform ---
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(AUGMENTATION_ROTATION),
            transforms.RandomCrop(28, padding=AUGMENTATION_PADDING),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
    else:
        train_transform = base_transform

    # --- Instantiate datasets with distinct transforms ---
    train_ds_with_aug = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform,
    )
    train_ds_no_aug = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=base_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=base_transform,
    )

    # --- Generate split indices deterministically ---
    indices = torch.randperm(
        len(train_ds_with_aug), 
        generator=torch.Generator().manual_seed(seed)
    ).tolist()
    
    train_idx = indices[:TRAIN_SIZE]
    val_idx = indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    # --- Subsets ensure exact isolation of transforms ---
    train_dataset = torch.utils.data.Subset(train_ds_with_aug, train_idx)
    val_dataset = torch.utils.data.Subset(train_ds_no_aug, val_idx)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY and torch.cuda.is_available()),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


# ===========================================================================
# EVALUATION (shared by training-accuracy pass and validation)
# ===========================================================================

@torch.no_grad()
def evaluate(model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device):
    """
    Evaluate *in eval mode* with no gradient computation.

    model.eval() ensures:
        - Dropout layers pass all activations (no masking).
        - BatchNorm uses accumulated running mean/variance.

    Returns
    -------
    (avg_ce_loss, accuracy_percent)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in data_loader:
        data = data.to(device, non_blocking=NON_BLOCKING)
        target = target.to(device, non_blocking=NON_BLOCKING)
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return running_loss / total, 100.0 * correct / total


# ===========================================================================
# SINGLE-EPOCH TRAINING
# ===========================================================================

def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    l1_lambda: float = 0.0):
    """
    Standard forward-backward-update loop for one epoch.

    Returns
    -------
    avg_ce_loss : float
        Average *cross-entropy-only* loss across the epoch.
        The L1 penalty (if any) is used for gradient computation
        but is **not** included in the logged loss so that training
        loss and validation loss are directly comparable.
    """
    model.train()
    running_ce_loss = 0.0
    total = 0

    for data, target in train_loader:
        data = data.to(device, non_blocking=NON_BLOCKING)
        target = target.to(device, non_blocking=NON_BLOCKING)

        optimizer.zero_grad()
        output = model(data)

        # Cross-entropy loss (the only component we log)
        ce_loss = criterion(output, target)

        # Total loss for backprop = CE  +  optional L1 penalty
        total_loss = ce_loss
        if l1_lambda > 0:
            total_loss = total_loss + compute_l1_penalty(model, l1_lambda)

        total_loss.backward()
        optimizer.step()

        running_ce_loss += ce_loss.item() * data.size(0)
        total += target.size(0)

    return running_ce_loss / total


# ===========================================================================
# FULL MODEL TRAINING (with early stopping + checkpointing)
# ===========================================================================

def train_model(model: nn.Module,
                experiment_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                l1_lambda: float = 0.0,
                weight_decay: float = 0.0,
                lr: float = LEARNING_RATE,
                num_epochs: int = NUM_EPOCHS,
                patience: int = PATIENCE,
                checkpoint_dir: str = CHECKPOINTS_DIR,
                seed: int = 42,
                label_smoothing: float = 0.0):
    """
    Train a single model with early stopping.

    The best checkpoint (lowest validation loss) is saved to disk
    and re-loaded into the model before returning, so that downstream
    test evaluation uses the best-epoch weights, not the last-epoch ones.

    Parameters
    ----------
    label_smoothing : float
        If > 0, the *training* criterion uses soft targets via
        ``CrossEntropyLoss(label_smoothing=...)``.  Evaluation always
        uses standard (hard-target) CE for fair cross-experiment
        comparison.

    Returns
    -------
    history   : dict  – {train_loss, val_loss, train_acc, val_acc}
    best_epoch: int   – 1-indexed epoch with lowest validation loss
    """
    model = model.to(device)

    # Separate criteria: smoothed for training, standard for evaluation
    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # always hard targets
    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    safe_name = experiment_name.replace("+", "_plus_").replace(" ", "_")
    ckpt_path = os.path.join(checkpoint_dir, f"{safe_name}_seed{seed}.pt")

    sep = '\u2500' * 60
    print(f"\n{sep}")
    print(f"  Training: {experiment_name}  (seed={seed})")
    print(f"  Parameters: {count_parameters(model):,}  |  Device: {device}")
    print(f"  L1 lambda={l1_lambda}  |  Weight decay (L2)={weight_decay}"
          f"  |  Label smoothing={label_smoothing}")
    print(sep)

    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        # --- train one epoch (uses smoothed criterion if applicable) ---
        train_loss = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device,
            l1_lambda,
        )

        # --- FAIR training accuracy (eval mode, standard CE, no grad) ---
        _, train_acc = evaluate(model, train_loader, eval_criterion, device)

        # --- validation (always standard CE) ---
        val_loss, val_acc = evaluate(model, val_loader, eval_criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # --- early stopping bookkeeping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- periodic logging ---
        gap = train_acc - val_acc
        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            star = " *" if epoch == best_epoch else ""
            print(
                f"  Epoch [{epoch:3d}/{num_epochs}] | "
                f"TrL: {train_loss:.4f} | VL: {val_loss:.4f} | "
                f"TrA: {train_acc:.2f}% | VA: {val_acc:.2f}% | "
                f"Gap: {gap:.2f}%{star}"
            )

        if epochs_no_improve >= patience:
            print(f"  >> Early stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch})")
            break

    # reload best checkpoint into model
    if best_state is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "best_epoch": int(best_epoch),
            "checkpoint_format": "state+meta_v1",
        }, ckpt_path)

        meta_path = ckpt_path.replace(".pt", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"best_epoch": int(best_epoch)}, f, indent=2)

        model.load_state_dict(best_state)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s  |  Best epoch: {best_epoch}")

    return history, best_epoch


# ===========================================================================
# HYPERPARAMETER GRID SEARCH
# ===========================================================================

def grid_search(device, train_loader, val_loader):
    """
    Simple grid search over key regularization hyperparameters.

    Searches
    --------
    - Dropout probability : {0.2, 0.5}
    - L2 weight decay     : {1e-4, 1e-3}
    - L1 lambda           : {1e-4, 1e-3}

    Each candidate is trained for a reduced number of epochs (default 20)
    with a single seed.  The value yielding the lowest final validation
    loss is selected.

    Returns
    -------
    dict with keys 'dropout_prob', 'l2_lambda', 'l1_lambda' and
    'search_details' (per-candidate validation losses).
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: HYPERPARAMETER GRID SEARCH")
    print("=" * 60)

    details = {}

    # --- Dropout probability ---
    print("\n  [1/3] Searching Dropout probability ...")
    best_dp, best_dp_val = DROPOUT_SEARCH_SPACE[0], float("inf")
    for dp in DROPOUT_SEARCH_SPACE:
        set_seed(42)
        m = get_model("dropout", dropout_prob=dp)
        h, _ = train_model(
            m, f"grid_dp{dp}", train_loader, val_loader, device,
            num_epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_EPOCHS,
            checkpoint_dir=os.path.join(CHECKPOINTS_DIR, "grid_search"),
        )
        vl = min(h["val_loss"])
        details[f"dropout_p={dp}"] = vl
        print(f"    p={dp}  ->  best val loss = {vl:.4f}")
        if vl < best_dp_val:
            best_dp_val = vl
            best_dp = dp

    # --- L2 weight decay ---
    print("\n  [2/3] Searching L2 lambda (weight decay) ...")
    best_l2, best_l2_val = L2_SEARCH_SPACE[0], float("inf")
    for l2 in L2_SEARCH_SPACE:
        set_seed(42)
        m = get_model("baseline")
        h, _ = train_model(
            m, f"grid_l2_{l2}", train_loader, val_loader, device,
            weight_decay=l2,
            num_epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_EPOCHS,
            checkpoint_dir=os.path.join(CHECKPOINTS_DIR, "grid_search"),
        )
        vl = min(h["val_loss"])
        details[f"l2_lambda={l2}"] = vl
        print(f"    lambda={l2}  ->  best val loss = {vl:.4f}")
        if vl < best_l2_val:
            best_l2_val = vl
            best_l2 = l2

    # --- L1 lambda ---
    print("\n  [3/3] Searching L1 lambda ...")
    best_l1, best_l1_val = L1_SEARCH_SPACE[0], float("inf")
    for l1 in L1_SEARCH_SPACE:
        set_seed(42)
        m = get_model("baseline")
        h, _ = train_model(
            m, f"grid_l1_{l1}", train_loader, val_loader, device,
            l1_lambda=l1,
            num_epochs=GRID_SEARCH_EPOCHS, patience=GRID_SEARCH_EPOCHS,
            checkpoint_dir=os.path.join(CHECKPOINTS_DIR, "grid_search"),
        )
        vl = min(h["val_loss"])
        details[f"l1_lambda={l1}"] = vl
        print(f"    lambda={l1}  ->  best val loss = {vl:.4f}")
        if vl < best_l1_val:
            best_l1_val = vl
            best_l1 = l1

    best_params = {
        "dropout_prob": best_dp,
        "l2_lambda": best_l2,
        "l1_lambda": best_l1,
        "search_details": details,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "best_hyperparams.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\n  Selected:  dropout_p={best_dp}  |  "
          f"l2={best_l2}  |  l1={best_l1}")
    print(f"  Saved to {RESULTS_DIR}/best_hyperparams.json")

    return best_params


# ===========================================================================
# FULL ABLATION STUDY
# ===========================================================================

def _safe_experiment_name(experiment_name: str) -> str:
    return experiment_name.replace("+", "_plus_").replace(" ", "_")


def _build_experiment_matrix(best_dp: float,
                             best_l1: float,
                             best_l2: float):
    """Return the fixed regularization/generalization experiment matrix."""
    return {
        "Baseline": dict(
            model_type="baseline", model_kw={},
            l1_lambda=0.0, weight_decay=0.0,
            augment=False, label_smoothing=0.0),
        "L1": dict(
            model_type="baseline", model_kw={},
            l1_lambda=best_l1, weight_decay=0.0,
            augment=False, label_smoothing=0.0),
        "L2": dict(
            model_type="baseline", model_kw={},
            l1_lambda=0.0, weight_decay=best_l2,
            augment=False, label_smoothing=0.0),
        "Dropout": dict(
            model_type="dropout", model_kw={"dropout_prob": best_dp},
            l1_lambda=0.0, weight_decay=0.0,
            augment=False, label_smoothing=0.0),
        "BatchNorm": dict(
            model_type="batchnorm", model_kw={},
            l1_lambda=0.0, weight_decay=0.0,
            augment=False, label_smoothing=0.0),
        "L2+Dropout": dict(
            model_type="dropout", model_kw={"dropout_prob": best_dp},
            l1_lambda=0.0, weight_decay=best_l2,
            augment=False, label_smoothing=0.0),
        "L2+BatchNorm": dict(
            model_type="batchnorm", model_kw={},
            l1_lambda=0.0, weight_decay=best_l2,
            augment=False, label_smoothing=0.0),
        "Dropout+BatchNorm": dict(
            model_type="dropout_batchnorm",
            model_kw={"dropout_prob": best_dp},
            l1_lambda=0.0, weight_decay=0.0,
            augment=False, label_smoothing=0.0),
        "DataAug": dict(
            model_type="baseline", model_kw={},
            l1_lambda=0.0, weight_decay=0.0,
            augment=True, label_smoothing=0.0),
        "LabelSmoothing": dict(
            model_type="baseline", model_kw={},
            l1_lambda=0.0, weight_decay=0.0,
            augment=False, label_smoothing=LABEL_SMOOTHING),
    }


def _expected_core_artifacts():
    """Files required to consider Phase 1 complete and reusable."""
    return [
        os.path.join(RESULTS_DIR, "best_hyperparams.json"),
        os.path.join(RESULTS_DIR, "test_metrics.csv"),
        os.path.join(RESULTS_DIR, "training_curves.png"),
        os.path.join(RESULTS_DIR, "individual_training_curves.png"),
        os.path.join(RESULTS_DIR, "weight_histograms.png"),
        os.path.join(RESULTS_DIR, "weight_comparison_overlay.png"),
        os.path.join(RESULTS_DIR, "comparative_summary.png"),
        os.path.join(RESULTS_DIR, "generalization_gap_metrics.json"),
        os.path.join(RESULTS_DIR, "generalization_gap_metrics.csv"),
        os.path.join(RESULTS_DIR, "sparsity_metrics.json"),
        os.path.join(RESULTS_DIR, "sparsity_metrics.csv"),
    ]


def _expected_core_checkpoints(experiment_matrix):
    """All checkpoint paths expected for the regularization matrix."""
    paths = []
    for exp_name in experiment_matrix:
        safe_name = _safe_experiment_name(exp_name)
        for seed in SEEDS:
            paths.append(
                os.path.join(CHECKPOINTS_DIR, f"{safe_name}_seed{seed}.pt")
            )
    return paths


def _core_phase_complete(experiment_matrix) -> bool:
    expected = _expected_core_artifacts() + _expected_core_checkpoints(
        experiment_matrix
    )
    return all(os.path.exists(path) for path in expected)


def _ensure_cuda_for_retraining(stage_name: str) -> None:
    """Fail fast when retraining is required but CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{stage_name} retraining requires a CUDA-capable GPU. "
            "Artifacts can still be reused without retraining."
        )


def _load_best_hyperparams_if_available():
    path = os.path.join(RESULTS_DIR, "best_hyperparams.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_checkpoint_if_exists(model: nn.Module,
                               experiment_name: str,
                               seed: int,
                               device: torch.device):
    ckpt_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{_safe_experiment_name(experiment_name)}_seed{seed}.pt",
    )
    if not os.path.exists(ckpt_path):
        return False, -1

    state = torch.load(ckpt_path, map_location=device)
    best_epoch = -1

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        best_epoch = int(state.get("best_epoch", -1))
    else:
        # Backward compatibility: old checkpoints saved plain state_dict only.
        model.load_state_dict(state)
        meta_path = ckpt_path.replace(".pt", ".meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                best_epoch = int(meta.get("best_epoch", -1))

    return True, best_epoch


def run_core_regularization_generalization_phase():
    """
    Phase 1: Core Regularization / Generalization Study.

    Reuses existing artifacts/checkpoints when complete, and retrains only
    missing experiment-seed checkpoints when needed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  PHASE 1: CORE REGULARIZATION / GENERALIZATION STUDY")
    print("=" * 60)
    print(f"  Device        : {device}")
    print(f"  Dataset       : {DATASET_NAME}")
    print(f"  Architecture  : MLP  784 -> 512 -> 256 -> 128 -> 10")
    print(f"  Max epochs    : {NUM_EPOCHS}  (early stopping, patience={PATIENCE})")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Seeds         : {SEEDS}")
    print("=" * 60)

    # Load data once for optional grid search fallback
    set_seed(SEEDS[0])
    train_loader, val_loader, _ = get_data_loaders(seed=SEEDS[0])
    print(f"  Train: {TRAIN_SIZE:,}  |  Val: {VAL_SIZE:,}")

    bp = _load_best_hyperparams_if_available()
    if bp is None:
        print("\n  No cached hyperparameter selection found. Running grid search...")
        bp = grid_search(device, train_loader, val_loader)
    else:
        print("\n  Using cached best hyperparameters from results/best_hyperparams.json")

    best_dp = bp["dropout_prob"]
    best_l2 = bp["l2_lambda"]
    best_l1 = bp["l1_lambda"]

    experiment_matrix = _build_experiment_matrix(best_dp, best_l1, best_l2)

    force_phase1_retrain = os.environ.get("FORCE_PHASE1_RETRAIN", "0") == "1"
    core_complete = _core_phase_complete(experiment_matrix)
    retraining_needed = (not core_complete) or force_phase1_retrain

    if retraining_needed:
        _ensure_cuda_for_retraining("Phase 1")

    # Hard skip when all core outputs are already complete.
    if core_complete and not force_phase1_retrain:
        print("  [+] Phase 1 artifacts and checkpoints are complete. Skipping retraining.")
        print("=" * 60 + "\n")
        return

    if force_phase1_retrain:
        print("  [!] FORCE_PHASE1_RETRAIN=1 -> retraining all Phase 1 runs to refresh metrics.")

    print("\n" + "=" * 60)
    print(f"  PHASE 1 EXECUTION: {len(experiment_matrix)} experiments x {len(SEEDS)} seeds")
    print("=" * 60)

    all_results = {}
    first_histories = {}
    first_models = {}
    criterion = nn.CrossEntropyLoss()

    for exp_name, cfg in experiment_matrix.items():
        seed_runs = []

        for seed in SEEDS:
            set_seed(seed)
            tl, vl, tel = get_data_loaders(
                seed=seed, augment=cfg.get("augment", False),
            )

            model = get_model(cfg["model_type"], **cfg["model_kw"]).to(device)

            loaded = False
            ckpt_best_epoch = -1
            if not force_phase1_retrain:
                loaded, ckpt_best_epoch = _load_checkpoint_if_exists(
                    model, exp_name, seed, device
                )

            if loaded:
                print(f"\n  [reuse] {exp_name} (seed={seed}) checkpoint found. Skipping training.")
                history = None
                best_ep = ckpt_best_epoch
            else:
                history, best_ep = train_model(
                    model, exp_name, tl, vl, device,
                    l1_lambda=cfg["l1_lambda"],
                    weight_decay=cfg["weight_decay"],
                    label_smoothing=cfg.get("label_smoothing", 0.0),
                    seed=seed,
                )

            # Test evaluation always uses best available checkpointed model.
            test_loss, test_acc = evaluate(model, tel, criterion, device)
            print(f"  >> Test  |  Loss: {test_loss:.4f}  |  Acc: {test_acc:.2f}%")

            # Train/val evaluation for generalization-gap evidence
            train_loss_eval, train_acc_eval = evaluate(
                model, tl, criterion, device
            )
            val_loss_eval, val_acc_eval = evaluate(
                model, vl, criterion, device
            )

            seed_runs.append(dict(
                seed=seed,
                history=history,
                best_epoch=best_ep,
                test_loss=test_loss,
                test_acc=test_acc,
                train_loss=train_loss_eval,
                train_acc=train_acc_eval,
                val_loss=val_loss_eval,
                val_acc=val_acc_eval,
            ))

            if seed == SEEDS[0]:
                if history is not None:
                    first_histories[exp_name] = history
                first_models[exp_name] = model

        all_results[exp_name] = seed_runs

    print("\n" + "=" * 60)
    print("  MULTI-SEED TEST RESULTS  (Mean +/- Std)")
    print("=" * 60)

    aggregated = {}
    for exp_name, runs in all_results.items():
        accs = [r["test_acc"] for r in runs]
        losses = [r["test_loss"] for r in runs]
        valid_best_epochs = [r["best_epoch"] for r in runs if r["best_epoch"] > 0]
        best_epoch_mean = (
            float(np.mean(valid_best_epochs)) if valid_best_epochs else -1.0
        )

        agg = dict(
            test_acc_mean=np.mean(accs),
            test_acc_std=np.std(accs),
            test_loss_mean=np.mean(losses),
            test_loss_std=np.std(losses),
            best_epoch_mean=best_epoch_mean,
        )
        aggregated[exp_name] = agg
        best_epoch_display = f"{best_epoch_mean:.0f}" if best_epoch_mean > 0 else "missing"
        print(
            f"  {exp_name:20s} | "
            f"Acc: {agg['test_acc_mean']:.2f} +/- {agg['test_acc_std']:.2f}% | "
            f"Loss: {agg['test_loss_mean']:.4f} +/- {agg['test_loss_std']:.4f} | "
            f"BestEp: {best_epoch_display}"
        )
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Experiment", "Test_Acc_Mean", "Test_Acc_Std",
            "Test_Loss_Mean", "Test_Loss_Std", "Best_Epoch_Mean"
        ])
        for exp, a in aggregated.items():
            best_epoch_cell = (
                f"{a['best_epoch_mean']:.0f}"
                if a["best_epoch_mean"] > 0
                else "missing"
            )
            w.writerow([
                exp,
                f"{a['test_acc_mean']:.4f}",
                f"{a['test_acc_std']:.4f}",
                f"{a['test_loss_mean']:.6f}",
                f"{a['test_loss_std']:.6f}",
                best_epoch_cell,
            ])
    print(f"\n  Saved {csv_path}")

    gap_summary = {}
    for exp_name, runs in all_results.items():
        train_accs = [r["train_acc"] for r in runs]
        val_accs = [r["val_acc"] for r in runs]
        train_losses = [r["train_loss"] for r in runs]
        val_losses = [r["val_loss"] for r in runs]
        gaps = [ta - va for ta, va in zip(train_accs, val_accs)]

        gap_summary[exp_name] = dict(
            train_acc_mean=float(np.mean(train_accs)),
            train_acc_std=float(np.std(train_accs)),
            val_acc_mean=float(np.mean(val_accs)),
            val_acc_std=float(np.std(val_accs)),
            gap_mean=float(np.mean(gaps)),
            gap_std=float(np.std(gaps)),
            train_loss_mean=float(np.mean(train_losses)),
            val_loss_mean=float(np.mean(val_losses)),
            seed_metrics=[
                dict(
                    seed=r["seed"],
                    train_acc=r["train_acc"],
                    val_acc=r["val_acc"],
                    gap=r["train_acc"] - r["val_acc"],
                    train_loss=r["train_loss"],
                    val_loss=r["val_loss"],
                )
                for r in runs
            ],
        )

    save_generalization_gap_metrics(gap_summary, save_dir=RESULTS_DIR)
    save_sparsity_metrics(first_models, save_dir=RESULTS_DIR)

    # Regenerate plots only when first-seed histories were produced in this run.
    if len(first_histories) == len(experiment_matrix):
        print("  Generating plots from current run histories ...")
        plot_training_curves(first_histories, save_dir=RESULTS_DIR)
        plot_individual_training_curves(first_histories, save_dir=RESULTS_DIR)
        plot_weight_histograms(first_models, save_dir=RESULTS_DIR)
        plot_weight_comparison(first_models, layer_idx=0, save_dir=RESULTS_DIR)
        plot_comparative_summary(first_histories, save_dir=RESULTS_DIR)
        print_sparsity_report(first_models)
        print_results_table(first_histories, aggregated)
    else:
        print("  [i] Full first-seed histories were reused from checkpoints.")
        print("  [i] Keeping existing curve artifacts to avoid unnecessary retraining.")

    print("\n" + "=" * 60)
    print("  PHASE 1 COMPLETED")
    print(f"  Results  -> {RESULTS_DIR}/")
    print(f"  Checkpts -> {CHECKPOINTS_DIR}/")
    print("=" * 60 + "\n")


def _load_models_for_final_phase(device: torch.device):
    """Load seed-42 checkpoints for all regularization experiments."""
    bp = _load_best_hyperparams_if_available()
    if bp is None:
        return {}

    matrix = _build_experiment_matrix(
        bp["dropout_prob"],
        bp["l1_lambda"],
        bp["l2_lambda"],
    )

    models = {}
    seed = SEEDS[0]
    for exp_name, cfg in matrix.items():
        model = get_model(cfg["model_type"], **cfg["model_kw"]).to(device)
        if _load_checkpoint_if_exists(model, exp_name, seed, device):
            models[exp_name] = model
        else:
            print(f"  [warn] Missing checkpoint for final phase: {exp_name} (seed={seed})")
    return models


def run_final_robustness_summary_phase():
    """
    Final Phase: Robustness / Summary Artifacts.

    Runs FGSM robustness using cached seed-42 checkpoints from Phase 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print(f"  FINAL PHASE: ROBUSTNESS / SUMMARY ARTIFACTS  (eps={FGSM_EPSILON})")
    print("=" * 60)

    fgsm_plot = os.path.join(RESULTS_DIR, "fgsm_robustness_comparison.png")
    fgsm_json = os.path.join(RESULTS_DIR, "fgsm_metrics.json")

    if os.path.exists(fgsm_plot) and os.path.exists(fgsm_json):
        print("  [+] Final-phase FGSM artifacts already exist. Skipping.")
        print("=" * 60 + "\n")
        return

    _, _, test_loader = get_data_loaders(seed=SEEDS[0], augment=False)
    models = _load_models_for_final_phase(device)
    if not models:
        print("  [warn] No checkpoints available for final-phase FGSM evaluation.")
        print("  [warn] Run Phase 1 first to generate checkpoints.")
        print("=" * 60 + "\n")
        return

    fgsm_results = evaluate_fgsm_robustness(
        models, test_loader, device,
        epsilon=FGSM_EPSILON, save_dir=RESULTS_DIR,
    )
    with open(fgsm_json, "w") as f:
        json.dump(fgsm_results, f, indent=2)
    print(f"  Saved {fgsm_json}")

    print("=" * 60)
    print("  FINAL PHASE COMPLETED")
    print("=" * 60 + "\n")

def run_ablation_study():
    """
    Backward-compatible wrapper for legacy calls.

    Executes the new split flow:
    1) Core Regularization / Generalization Study
    2) Final Robustness / Summary Artifacts
    """
    run_core_regularization_generalization_phase()
    run_final_robustness_summary_phase()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_ablation_study()
