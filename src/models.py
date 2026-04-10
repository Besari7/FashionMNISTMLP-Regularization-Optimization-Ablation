"""
models.py - Neural Network Model Definitions
==============================================

Comparative Analysis of L1, L2, Dropout and Batch Normalization
on Overfitting in Neural Networks.

All variants share the same overparameterized MLP skeleton:
    Input(784) -> FC(512) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU -> FC(10)
    ~530 K trainable parameters (deliberately large relative to 48 K training
    samples so that overfitting is clearly observable).

Design rationale
----------------
- L1 / L2 regularization are *training-level* techniques (penalty in loss or
  optimizer weight-decay), NOT architectural changes.  Therefore they do not
  require separate model classes; the BaselineMLP architecture is reused.
- Dropout and Batch Normalization *do* modify the network graph, so they get
  dedicated classes (DropoutMLP, BatchNormMLP, DropoutBatchNormMLP).
"""

import torch
import torch.nn as nn


# ===========================================================================
# 1. BASELINE MLP - Unregularized Reference
# ===========================================================================

class BaselineMLP(nn.Module):
    """
    Unregularized baseline MLP.

    Purpose
    -------
    Establish a reference for pure overfitting behaviour.
    With ~530 K parameters trained on 48 K samples the model enters a
    high-variance regime: training loss approaches zero while validation
    loss increases after a few epochs.

    Architecture
    ------------
    Input(784) -> FC(512) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU -> FC(10)
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming (He) uniform init — suited to ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===========================================================================
# 2. DROPOUT MLP - Stochastic Regularization
# ===========================================================================

class DropoutMLP(nn.Module):
    """
    MLP with Dropout regularization after each hidden-layer activation.

    Dropout (Srivastava et al., 2014) randomly zeroes neurons with
    probability *p* during training, implicitly training an exponential
    ensemble of thinned sub-networks.

    PyTorch uses *inverted* dropout so no rescaling is needed at test time.
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10,
                 dropout_prob: float = 0.5, **kwargs):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===========================================================================
# 3. BATCH NORMALIZATION MLP
# ===========================================================================

class BatchNormMLP(nn.Module):
    """
    MLP with Batch Normalization (Ioffe & Szegedy, 2015).

    BN is inserted *before* ReLU (pre-activation placement as proposed in
    the original paper).  Learnable affine parameters (gamma, beta) preserve
    the network's representational power.

    At test time BN uses exponential-moving-average statistics accumulated
    during training, removing the mini-batch noise that acts as a mild
    regularizer.
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10, **kwargs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===========================================================================
# 4. DROPOUT + BATCH NORMALIZATION MLP (Combined)
# ===========================================================================

class DropoutBatchNormMLP(nn.Module):
    """
    MLP combining Dropout *and* Batch Normalization.

    Layer order: Linear -> BN -> ReLU -> Dropout.
    Dropout is placed *after* activation so that BN statistics are computed
    on the full (non-masked) activations, following the recommendation of
    Ioffe & Szegedy (2015) and Li et al. (2019, "Understanding the
    Disharmony between Dropout and Batch Normalization").
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10,
                 dropout_prob: float = 0.5, **kwargs):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

# Model registry ----------------------------------------------------------

MODEL_REGISTRY = {
    "baseline": BaselineMLP,
    "dropout": DropoutMLP,
    "batchnorm": BatchNormMLP,
    "dropout_batchnorm": DropoutBatchNormMLP,
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Factory: create a model instance by type string."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Valid types: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)


# L1 penalty (bias-excluded) ----------------------------------------------

def compute_l1_penalty(model: nn.Module, lambda_l1: float) -> torch.Tensor:
    """
    Compute L1 penalty over *weight* parameters only.

    Bias exclusion rationale
    ------------------------
    Biases control activation thresholds and do not contribute to
    inter-feature co-adaptation.  Penalising them toward zero can
    harm model expressiveness without meaningful regularization benefit.
    Batch-norm affine parameters (gamma/beta) are also excluded.

    Formula
    -------
        Omega_L1(theta) = lambda * sum_l sum_ij |W^(l)_ij|
    """
    bn_param_ids = set()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            bn_param_ids.update(id(p) for p in m.parameters())

    device = next(model.parameters()).device
    l1_norm = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if "weight" in name and id(param) not in bn_param_ids:
            l1_norm = l1_norm + torch.sum(torch.abs(param))
    return lambda_l1 * l1_norm


# Optimizer factory (AdamW, bias-excluded weight decay) --------------------

def create_optimizer(model: nn.Module, lr: float,
                     weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer with *decoupled* weight decay applied
    only to weight tensors (biases and BN params get zero decay).

    Why AdamW over Adam?
    --------------------
    Standard Adam applies L2 as a gradient penalty, which interacts
    poorly with adaptive learning rates.  AdamW (Loshchilov & Hutter,
    2019) decouples weight decay from the gradient update, yielding
    more principled regularization.
    """
    bn_param_ids = set()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            bn_param_ids.update(id(p) for p in m.parameters())

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or id(param) in bn_param_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


# Parameter counting -------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick self-test -----------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  MODEL SANITY CHECK")
    print("=" * 50)
    dummy = torch.randn(4, 784)
    for name, cls in MODEL_REGISTRY.items():
        kw = {"dropout_prob": 0.5} if "dropout" in name else {}
        m = cls(**kw)
        out = m(dummy)
        print(f"  {name:25s} | params: {count_parameters(m):>8,} | out: {out.shape}")
    print("  All models OK.")
