# FashionMNIST MLP: Regularization & Optimization Ablation

Deep Learning course project: a controlled, methodology-first ablation of regularization and optimization methods on a fixed FashionMNIST MLP backbone.

## Project Focus

This repository answers one coherent question:

How do regularization and optimization choices change overfitting behavior, training stability, and final generalization on a fixed FashionMNIST MLP?

The project stays on one backbone (FashionMNIST + overparameterized MLP) and compares methods in a controlled way instead of changing architectures between phases.

## Unified Project Phases

The full pipeline is executed by `python run.py` in this exact order:

1. Phase 0: Methodological Foundations
2. Phase 1: Core Regularization / Generalization Study
3. Phase 2: Optimization / Training Stability
4. Final Phase: Robustness / Summary Artifacts

All phases are artifact-aware and skip when outputs already exist.

## Why This Phase Design

1. Phase 0 establishes baseline methodology checks and foundational sanity experiments.
2. Phase 1 is the core controlled ablation for regularization/generalization.
3. Phase 2 isolates optimization variables on the same project backbone with lightweight runtime.
4. Final Phase computes robustness summary artifacts from saved checkpoints without full retraining.

This makes the repository presentation-ready, sequential, and non-contradictory.

## Repository Structure

```text
DL_Project/
  run.py
  src/
    config.py
    foundations.py
    models.py
    train.py
    optimization.py
    utils.py
  results/
    foundations/
    optimization/
    best_hyperparams.json
    test_metrics.csv
    training_curves.png
    individual_training_curves.png
    comparative_summary.png
    weight_histograms.png
    weight_comparison_overlay.png
    fgsm_robustness_comparison.png
    fgsm_metrics.json
  checkpoints/
    foundations/
    optimization/
    grid_search/
    *.pt
  supporting_experiments/
    README.md
    week1/
    week2/
    week3/
    week4/
    week5/
    outputs/
    checkpoints/
  responsibilities/
    <student_number>.md
  REPORT.md
  requirements.txt
  environment.yml
```

Main code uses project-native naming (`src/optimization.py`, `results/optimization/`, `checkpoints/optimization/`) and does not use week-coded names.

## Foundational Coverage Completion

The previously missing core foundations are now explicitly covered with
lightweight supporting scripts:

1. Linear algebra eigendecomposition and SVD:
  - `supporting_experiments/week1/01_linear_algebra_eig_svd.py`
2. Bayes rule and information theory (entropy, cross-entropy, KL):
  - `supporting_experiments/week1/02_probability_bayes_information.py`
3. Numerical overflow/underflow plus stable softmax and autograd demo:
  - `supporting_experiments/week1/03_numerical_stability_and_autograd.py`
4. Standard error and Bayesian estimation:
  - `supporting_experiments/week2/05_standard_error_and_bayesian_estimation.py`
5. Explicit backprop memoization demo:
  - `supporting_experiments/week3/04_backprop_memoization_demo.py`
6. Self-supervised pretext task demo:
  - `supporting_experiments/week4/self_supervised/06_rotation_pretext_demo.py`

## Week 5 Integration Map

### Main Codebase Integration (`src/optimization.py`)

Integrated directly into Phase 2:

1. Optimization framing through empirical risk minimization with minibatch updates.
2. Optimizer comparison on project setup:
   - SGD
   - SGD + Momentum
   - Nesterov
   - AdaGrad
   - RMSProp
   - Adam
   - AdamW
3. Initialization comparison:
   - Default
   - Xavier / Glorot
   - He / Kaiming
4. Learning-rate scheduling comparison:
   - No schedule
   - StepLR
   - CosineAnnealingLR
   - Warmup + Cosine
5. Gradient clipping comparison (no clip vs max_norm=1.0).
6. Mini-batch size comparison (BS=32, 128, 512).
7. Normalization stability comparison (None vs BatchNorm vs LayerNorm).

### Supporting Experiments (`supporting_experiments/week5/`)

Kept separate only for pedagogical/toy topics that do not naturally belong in the main benchmark loop:

1. 2D optimizer trajectory landscape.
2. Saddle-point escape visualization.
3. Ill-conditioned second-order demonstration and notes.

These scripts are standalone by design and do not write into the main `results/` tree.

## Method Choice Rationale

1. Main model family remains fixed to isolate method effects.
2. AdamW is retained as the main Phase 1 optimizer because decoupled weight decay makes L2 comparison cleaner.
3. Phase 2 includes broader optimizer family to justify choices against alternatives.
4. BatchNorm appears in both regularization and optimization perspectives.
5. LayerNorm is included in Phase 2 only as a stability comparison, not as a new project backbone.
6. Second-order methods are documented with controlled support experiments due scalability limits in the full model.

## Hyperparameter Tuning Protocol

Phase 1 uses explicit grid search before multi-seed training:

1. Dropout probability: {0.2, 0.5}
2. L1 lambda: {1e-4, 1e-3}
3. L2 lambda: {1e-4, 1e-3}

Selection criterion: lowest validation loss.

Saved artifact: `results/best_hyperparams.json`.

## Which Methods Run Simultaneously

In Phase 1, explicit combined-method variants are trained as separate controlled experiments:

1. L2 + Dropout
2. L2 + BatchNorm
3. Dropout + BatchNorm

This directly answers interaction effects instead of only single-method comparisons.

## Comparison Criteria

The project compares methods with consistent criteria:

1. Validation loss
2. Generalization gap (train vs validation accuracy)
3. Test accuracy/loss (multi-seed aggregation)
4. Best epoch under early stopping
5. FGSM robustness in Final Phase
6. Weight distribution/sparsity diagnostics
7. Generalization gap metrics (results/generalization_gap_metrics.json)
8. Sparsity metrics (results/sparsity_metrics.json)

## Checklist Coverage Map (Weeks 1-5)

For the professor rubric, the complete checklist-to-evidence matrix is in
REPORT.md Section 6.0. Highlights below point to the most direct evidence.

1. Mathematical foundations: supporting_experiments/week1/01_linear_algebra_eig_svd.py
  and supporting_experiments/week1/03_numerical_stability_and_autograd.py
2. Autograd vs manual gradients: supporting_experiments/week3/02_mlp_from_scratch.py
  and supporting_experiments/week3/03_mlp_pytorch.py
3. XOR MLP: supporting_experiments/week3/01_xor_intuition.py
4. Depth vs width: src/foundations.py
5. Regularization ablation matrix: src/train.py (Phase 1)
6. Optimization comparisons: src/optimization.py (Phase 2)
7. Bias-variance evidence: results/generalization_gap_metrics.json and results/test_metrics.csv
8. Hyperparameter tuning: results/best_hyperparams.json and Phase 2 comparison plots

## Runtime and Artifact Reuse Policy

To avoid unnecessary retraining:

1. Phase 1 loads existing checkpoints per experiment/seed and skips retraining where possible.
2. If Phase 1 artifacts are complete, the entire phase is skipped.
3. Final Phase (FGSM) runs from saved checkpoints and skips if robustness artifacts already exist.
4. Phase 2 (optimization) uses lightweight subset/epoch budgets and group-level cache checks.
5. Missing optimization groups are regenerated independently without rerunning complete Phase 2.

## GPU Usage

Retraining is GPU-only by project policy.

1. If a phase needs to retrain missing artifacts/checkpoints, it now requires CUDA and stops with a clear error when no GPU is available.
2. If artifacts are already complete, phases still skip/reuse normally without retraining.
3. Data loading remains CUDA-friendly (`pin_memory`, non-blocking transfer) during training runs.

## Expected Optimization Artifacts

Phase 2 writes to:

1. `results/optimization/optimizer_comparison.png`
2. `results/optimization/init_comparison.png`
3. `results/optimization/lr_schedule_comparison.png`
4. `results/optimization/grad_clip_comparison.png`
5. `results/optimization/batch_size_comparison.png`
6. `results/optimization/normalization_stability_comparison.png`
7. `results/optimization/*_metrics.json`
8. `results/optimization/optimization_metrics.json`
9. `checkpoints/optimization/*.pt`

## Run

```bash
python run.py
```

This runs all phases in sequence with skip/reuse logic.

### Force Phase 1 Retrain On GPU (Windows PowerShell)

Use the project venv interpreter explicitly to avoid accidentally using system Python.

```powershell
Set-Location c:\Users\berke\DL_Project
$env:FORCE_PHASE1_RETRAIN='1'
c:/Users/berke/DL_Project/.venv/Scripts/python.exe run.py
```

Or use the launcher script:

```powershell
Set-Location c:\Users\berke\DL_Project
powershell -ExecutionPolicy Bypass -File .\scripts\retrain_phase1_gpu.ps1
```

Expected startup logs now include active interpreter and CUDA status. If you do not see `.venv\\Scripts\\python.exe` in that log, the wrong Python was used.

## Supporting Experiments

See `supporting_experiments/README.md` for standalone script execution and expected outputs.

## References

1. Srivastava et al. (2014), Dropout.
2. Ioffe and Szegedy (2015), Batch Normalization.
3. Loshchilov and Hutter (2019), AdamW.
4. Kingma and Ba (2015), Adam.
5. Glorot and Bengio (2010), Xavier initialization.
6. He et al. (2015), Kaiming initialization.
7. Pascanu et al. (2013), Gradient clipping.
8. Goodfellow, Bengio, Courville (2016), Deep Learning.
