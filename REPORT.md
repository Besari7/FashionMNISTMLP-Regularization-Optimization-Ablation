# REPORT - FashionMNIST MLP: Regularization & Optimization Ablation

Course: Deep Learning (Graduate Level)
Project Title: FashionMNIST MLP: Regularization & Optimization Ablation

---

## 1. Project Scope and Objective

This project focuses on one coherent methodology question:

How do regularization and optimization choices affect overfitting, training stability, and final generalization on a fixed FashionMNIST MLP setup?

The repository intentionally keeps a single backbone (FashionMNIST + MLP family) to avoid confounding architectural effects.

---

## 2. Unified Pipeline Structure

Execution order follows run.py exactly:

1. Phase 0: Methodological Foundations
2. Phase 1: Core Regularization / Generalization Study
3. Phase 2: Optimization / Training Stability
4. Final Phase: Robustness / Summary Artifacts

This phase naming is consistent across run.py, README.md, and this report.

---

## 3. Main Pipeline Summary

### 3.1 Phase 0: Methodological Foundations

Covered in src/foundations.py:

1. Logistic regression baseline on FashionMNIST.
2. Activation gallery (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU).
3. Backprop sanity check.
4. Depth-vs-width comparison.
5. Cross-entropy vs MSE comparison.

### 3.2 Phase 1: Core Regularization / Generalization Study

Main controlled experiment matrix:

1. Baseline
2. L1
3. L2
4. Dropout
5. BatchNorm
6. L2+Dropout
7. L2+BatchNorm
8. Dropout+BatchNorm
9. DataAug
10. LabelSmoothing

Hyperparameter tuning protocol (grid search, validation loss criterion):

1. Dropout: {0.2, 0.5}
2. L1: {1e-4, 1e-3}
3. L2: {1e-4, 1e-3}

### 3.3 Phase 2: Optimization / Training Stability

Implemented in src/optimization.py with lightweight budget:

- subset=8000, epochs=8, seed=42

Compared methods:

1. Optimizers: SGD, SGD+Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW
2. Initialization: Default, Xavier, He
3. LR schedules: None, StepLR, CosineAnnealingLR, Warmup+Cosine
4. Gradient clipping: no clip vs max_norm=1.0
5. Minibatch-size comparison: BS=32, 128, 512
6. Normalization stability: None vs BatchNorm vs LayerNorm

### 3.4 Final Phase: Robustness / Summary Artifacts

FGSM robustness is computed from saved checkpoints (no full retraining) and exported as figure + JSON summary.

---

## 4. Dataset Quality Assessment

Dataset: FashionMNIST

1. Modality and scale:
- Grayscale images, 28x28 resolution, flattened to 784 features for MLP input.
- This resolution is sufficient for class semantics while remaining computationally tractable.

2. Split quality and leakage control:
- Train/val/test protocol uses 48k/12k/10k.
- Validation is used for model selection and early stopping.
- Test set is reserved for final reporting only.

3. Class-balance suitability:
- FashionMNIST is class-balanced by design (10 classes with near-uniform support), reducing label-frequency bias in comparison studies.

4. Difficulty profile:
- Harder than digit-only MNIST because of visually similar classes (for example, T-shirt vs Shirt, Pullover vs Coat), which makes regularization effects measurable.

5. Preprocessing quality controls:
- Main normalization maps pixel values to approximately [-1, 1] using mean=0.5, std=0.5.
- Deterministic transforms on validation/test; augmentation is train-only when enabled.

6. Why this dataset is appropriate:
- It is complex enough to expose overfitting but light enough for repeated controlled ablations and multi-seed analysis.

---

## 5. Method Rationale and Selection Logic

This section explains why each method was selected over alternatives.

1. Baseline:
- Required reference point for measuring overfitting gap and relative gains from all regularizers.

2. L1 regularization:
- Encourages sparsity by penalizing absolute weights.
- Useful for observing parameter-pruning tendency, but may reduce raw accuracy if over-applied.
- Bayesian view: L1 corresponds to a Laplace prior on weights,
  $$p(w) \propto \exp(-\lambda |w|)$$
  which favors many near-zero coefficients and can increase bias when too strong.

3. L2 regularization (AdamW weight decay):
- Penalizes large weights smoothly and generally preserves capacity better than strong L1.
- AdamW is used to decouple weight decay from adaptive gradient scaling, avoiding the classical Adam-L2 coupling issue.
- Bayesian view: L2 corresponds to a Gaussian prior on weights,
  $$p(w) \propto \exp(-\lambda w^2)$$
  which shrinks parameters smoothly and usually trades a smaller variance increase for less bias than strong L1.

4. Dropout:
- Stochastic masking reduces co-adaptation and acts as implicit model averaging.
- Selected to test robust performance under high-capacity MLP overfitting.

5. BatchNorm:
- Stabilizes hidden activation distributions and improves optimization dynamics.
- Used both as an optimization aid and as a mild regularizer through minibatch statistics.

6. Combined methods (L2+Dropout, L2+BatchNorm, Dropout+BatchNorm):
- Explicitly test whether mechanisms are complementary or interfering.
- These combinations answer simultaneous-method behavior directly, not only single-method performance.

7. Data augmentation:
- Increases effective data diversity and tests regularization through input-space perturbation.

8. Label smoothing:
- Reduces overconfidence by softening targets; intended to improve calibration and stability.

9. Optimization-phase method set:
- Momentum and Nesterov included to evaluate acceleration effects.
- AdaGrad/RMSProp/Adam included for adaptive learning-rate behavior.
- Step/Cosine/Warmup+Cosine included to compare schedule families under same backbone.

---

### 5.1 Statistical Alignment (MLE to Loss)

This project explicitly ties the loss functions to Maximum Likelihood Estimation (MLE)
assumptions from the lectures and labs.

1. Regression with Gaussian noise (MSE):
	Assume $y = f_\theta(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$. The
	negative log-likelihood is proportional to the sum of squared errors:
	$$
	-\log p(y|x,\theta) \propto \sum_i \left(y_i - f_\theta(x_i)\right)^2
	$$
	This is demonstrated in supporting_experiments/week2/01_linear_regression_scratch.py.

2. Classification with categorical likelihood (CE):
	For logits $z$ and class probabilities $p_k = \mathrm{softmax}(z)_k$, the
	categorical negative log-likelihood yields cross-entropy:
	$$
	-\log p(y|x,\theta) = -\sum_k y_k \log p_k
	$$
	This is used in the main pipeline (CrossEntropyLoss in src/train.py) and in
	supporting_experiments/week2/04_logistic_regression_scratch.py.

### 5.2 Explicit Hyperparameters and Training Protocol

Core training protocol (Phase 1):

1. Optimizer: AdamW (decoupled weight decay).
2. Batch size: 128.
3. Max epochs: 50 with early stopping (patience=7, best val loss).
4. Learning rate: 1e-3.
5. Seeds: 42, 123, 999.

Regularization and robustness specifics:

1. Dropout grid: {0.2, 0.5} -> selected 0.2.
2. L1 grid: {1e-4, 1e-3} -> selected 1e-4.
3. L2 grid: {1e-4, 1e-3} -> selected 1e-3.
4. Label smoothing: 0.1.
5. Data augmentation: RandomRotation=15 degrees, RandomCrop=28 with padding=4.
6. FGSM epsilon: 0.05 (inputs are normalized to [-1, 1]).

Phase 2 (optimization) uses a lightweight budget: subset=8000, epochs=8, seed=42.

## 6. Coverage Mapping (How Each Topic Was Applied)

### 6.0 Checklist Compliance Matrix (Weeks 1-5)

1. Mathematical Foundations and Implementation
- Linear algebra ops and tensor/matrix multiplications: supporting_experiments/week1/01_linear_algebra_eig_svd.py and tensor ops in src/train.py.
- Eigendecomposition/SVD: supporting_experiments/week1/01_linear_algebra_eig_svd.py.
- Automatic differentiation vs manual gradients: supporting_experiments/week3/02_mlp_from_scratch.py (manual) and supporting_experiments/week3/03_mlp_pytorch.py (autograd).
- Numerical stability (overflow/underflow, stable softmax): supporting_experiments/week1/03_numerical_stability_and_autograd.py.
- MLE alignment for loss: supporting_experiments/week2/01_linear_regression_scratch.py (MSE) and supporting_experiments/week2/04_logistic_regression_scratch.py (CE).

2. Model Architecture and Core Logic
- MLP construction and non-linear capability (XOR): supporting_experiments/week3/01_xor_intuition.py.
- Activation functions (ReLU, Leaky ReLU, ELU, Sigmoid, Tanh): src/foundations.py activation gallery.
- Loss functions: CrossEntropyLoss in src/train.py and BCEWithLogitsLoss in supporting_experiments/week4/multi_task/02_mtl_demo.py.
- Depth vs width comparison: src/foundations.py (depth-vs-width experiment).

3. Regularization Techniques (Overfitting Control)
- L2 weight decay: create_optimizer in src/models.py and Phase 1 matrix in src/train.py.
- L1 sparsity: compute_l1_penalty in src/models.py and Phase 1 matrix in src/train.py.
- Dropout (inverted): DropoutMLP and DropoutBatchNormMLP in src/models.py.
- Normalization: BatchNorm in src/models.py and LayerNorm in src/optimization.py.
- Label smoothing: CrossEntropyLoss(label_smoothing=0.1) in src/train.py.
- Early stopping: train_model in src/train.py (patience=7, val loss).
- Data augmentation: get_data_loaders in src/train.py (rotation and crop).

4. Optimization and Training Dynamics
- Weight initialization: Xavier/He in src/optimization.py; He init in src/models.py.
- Momentum/Nesterov: optimizer comparison in src/optimization.py.
- Adaptive learning rates: AdaGrad/RMSProp/Adam/AdamW in src/optimization.py.
- LR schedulers: StepLR, Cosine, Warmup+Cosine in src/optimization.py.
- Gradient clipping: grad clip comparison in src/optimization.py.

5. Empirical Analysis and Evaluation
- Bias-variance analysis: generalization gap and seed variance in results/generalization_gap_metrics.json and results/test_metrics.csv.
- Ablation study: Phase 1 experiment matrix in src/train.py with baseline and controlled variants.
- Hyperparameter tuning: grid search in src/train.py plus Phase 2 comparisons (LR, batch size, schedulers).

### Week 1

1. Linear algebra (vectors/matrices/tensors): practical matrix/tensor operations in supporting and main code.
2. Eigendecomposition and SVD: supporting_experiments/week1/01_linear_algebra_eig_svd.py.
3. Probability, Bayes rule, information theory: supporting_experiments/week1/02_probability_bayes_information.py.
4. Numerical overflow/underflow and stability: supporting_experiments/week1/03_numerical_stability_and_autograd.py.
5. Autograd: demonstrated in supporting script and used throughout main PyTorch training loops.

### Week 2

1. Generalization and overfitting: core target of Phase 1 ablations.
2. Point estimation and MLE: week2 linear regression supporting scripts.
3. Standard error and Bayesian estimation: supporting_experiments/week2/05_standard_error_and_bayesian_estimation.py.
4. SGD/minibatch logic: week2 SGD-vs-batch script plus main DataLoader-based training.
5. Linear and logistic regression: covered in week2 scripts and Phase 0 baseline.

### Week 3

1. MLP architecture and hidden-layer design: main backbone and week3 scripts.
2. Activations: foundations activation gallery and model definitions.
3. BCE/CCE with logits: BCEWithLogitsLoss in supporting MTL demo; CrossEntropyLoss in main pipeline.
4. Backprop/chain rule/computational graph: manual and autograd implementations.
5. Memoization: supporting_experiments/week3/04_backprop_memoization_demo.py caches forward intermediates for backward pass.

### Week 4

1. L1/L2, dropout, batchnorm, early stopping, data augmentation, label smoothing: all in main pipeline.
2. Semi-supervised learning: pseudo-labeling demo.
3. Self-supervised learning: rotation pretext demo.
4. Bagging/boosting: dedicated supporting demos.
5. Adversarial training and FGSM robustness: supporting adversarial-training demo and final-phase FGSM evaluation.

### Week 5

1. Xavier/He initialization: optimization phase comparisons.
2. Optimization challenges (ill-conditioning/saddles): supporting week5 demos.
3. Momentum/Nesterov and adaptive optimizers: optimization phase.
4. LR scheduling and clipping: optimization phase.
5. LayerNorm vs BatchNorm: optimization phase + supporting comparison script.

---

## 7. Simultaneous Runs and Evaluation Criteria

### Simultaneous method runs

Phase 1 explicitly runs these combined variants:

1. L2 + Dropout
2. L2 + BatchNorm
3. Dropout + BatchNorm

### Evaluation criteria

1. Validation loss
2. Generalization gap
3. Test accuracy/loss with multi-seed aggregation
4. Best epoch under early stopping
5. FGSM robustness drop
6. Weight distribution and sparsity diagnostics
7. Generalization gap metrics (results/generalization_gap_metrics.json)
8. Sparsity metrics (results/sparsity_metrics.json)

---

## 8. Key Findings and Results Synthesis

Source artifacts:

- results/test_metrics.csv
- results/best_hyperparams.json
- results/fgsm_metrics.json
- results/generalization_gap_metrics.json
- results/sparsity_metrics.json

### 8.1 Hyperparameter search outcomes

Selected best values:

1. Dropout probability = 0.2
2. L2 lambda = 0.001
3. L1 lambda = 0.0001

Validation-loss evidence from search:

- dropout 0.2 better than 0.5
- L2 0.001 slightly better than 0.0001
- L1 0.0001 substantially better than 0.001

### 8.2 Phase 1 complete results table (mean over 3 seeds)

| Experiment | Test Acc Mean | Test Acc Std | Test Loss Mean | Test Loss Std | Best Epoch Mean |
|---|---:|---:|---:|---:|---:|
| Baseline | 88.1833 | 0.1584 | 0.343765 | 0.005671 | 9 |
| L1 | 87.5533 | 0.1713 | 0.342931 | 0.002901 | 22 |
| L2 | 87.9367 | 0.1533 | 0.343522 | 0.004638 | 8 |
| Dropout | 88.5233 | 0.4778 | 0.332482 | 0.002017 | 21 |
| BatchNorm | 88.2733 | 0.4203 | 0.330304 | 0.008186 | 5 |
| L2+Dropout | 88.4633 | 0.0685 | 0.330859 | 0.003211 | 15 |
| L2+BatchNorm | 88.4800 | 0.4830 | 0.327156 | 0.005596 | 5 |
| Dropout+BatchNorm | 89.0267 | 0.2917 | 0.313405 | 0.002720 | 13 |
| DataAug | 84.9167 | 0.3311 | 0.407344 | 0.008887 | missing* |
| LabelSmoothing | 88.6967 | 0.1517 | 0.382151 | 0.006885 | missing* |

*missing: Legacy checkpoints do not contain recoverable best_epoch metadata for all three seeds; full refresh requires retraining those missing seed checkpoints.

Observations from full table:

1. Best overall test accuracy: Dropout+BatchNorm = 89.0267.
2. Best single regularizer by test accuracy: LabelSmoothing = 88.6967.
3. L2+Dropout has the strongest cross-seed stability with std = 0.0685.
4. DataAug is the lowest-accuracy variant here, while still helping generalization-gap behavior.

### 8.3 Main interpretation

1. Combined methods can outperform single methods on this setup (Dropout+BatchNorm best mean accuracy).
2. L2 variants remain competitive and stable, especially when combined.
3. Strong augmentation can reduce raw test accuracy in this exact MLP setup, despite regularization benefits.
4. Early stopping is important because best epochs vary substantially by method.

### 8.4 Robustness findings (FGSM)

From results/fgsm_metrics.json:

1. Clean accuracy leader remains Dropout+BatchNorm (89.29 on seed-42 model set).
2. Under FGSM, accuracy drops are method-dependent; L2+Dropout preserves adversarial accuracy relatively well (70.27).
3. Robustness ranking differs from clean ranking, reinforcing the need for final-phase robustness reporting.

### 8.5 Bias-Variance Analysis

In lecture notation, bias and variance are:

$$
	ext{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta, \qquad
	ext{Var}(\hat{\theta}) = \mathbb{V}[\hat{\theta}]
$$

In this project we do not observe true population parameter $\theta$ directly, so we use
practical proxies:

1. Variance proxy: seed-wise test accuracy std (results/test_metrics.csv).
2. Overfitting/variance proxy: train-val accuracy gap (results/generalization_gap_metrics.json).
3. Bias proxy: underfitting tendency reflected by lower train accuracy and higher train loss.

Method-level evidence:

1. Baseline: gap_mean = 3.856 and std = 0.1584.
   - High gap indicates overfitting pressure (higher variance behavior).
2. L1: gap_mean = 2.461 with very high sparsity TOTAL = 93.319%.
   - Strong capacity restriction increases bias risk (lower flexibility) while reducing variance pressure.
3. L2+Dropout: gap_mean = 3.123 and the lowest seed variance std = 0.0685.
   - This is the strongest stability point in Phase 1 and a key trade-off result.
4. Dropout+BatchNorm: best accuracy (89.0267) with moderate std = 0.2917.
   - Lower bias than stronger regularizers while keeping variance controlled enough for top performance.

Conclusion:

1. Strong regularization can reduce variance but may increase bias (for example L1).
2. Weak/no regularization reduces bias but can increase variance and overfitting (baseline gap).
3. Combined methods (especially L2+Dropout and Dropout+BatchNorm) give the best bias-variance trade-off on this setup.

### 8.6 Phase 2 optimization winners (from results/optimization/optimization_metrics.json)

1. Best optimizer: Nesterov (83.96), followed by AdamW (83.80).
2. Best initialization: He (83.80), slightly above Default (83.76) and Xavier (83.38).
3. Best LR scheduler: Warmup+Cosine (85.77), then CosineAnnealingLR (85.56).
4. Best batch size: BS=128 (83.80), better than BS=32 (83.43) and BS=512 (83.46).
5. Best normalization: BatchNorm (84.50), above None (83.80) and LayerNorm (82.42).
6. Gradient clipping was not beneficial in this setup (No Clipping 83.80 vs Clip 83.12).

---

## 9. Runtime Management and Reuse Strategy

1. Artifact-aware skip logic is active across phases.
2. Phase 1 reuses checkpoints and retrains only missing experiment/seed combinations.
3. Phase 2 caches group artifacts and reruns only missing groups.
4. Final phase computes FGSM summaries from saved checkpoints.

This design avoids unnecessary heavy retraining and keeps the repository reproducible.

---

## 10. Limitations

1. Single dataset and fixed architecture by design.
2. Some theoretical topics are represented with compact supporting demos rather than large-scale experiments.
3. Findings are methodologically strong for this benchmark but should not be over-generalized to all domains.

---

## 11. Conclusion

The project now provides:

1. Coherent phase-based methodology.
2. Full Week 1-5 coverage across main and supporting paths.
3. Explicit method rationale, tuning logic, simultaneous-method analysis, and results synthesis.
4. Artifact-aware reproducibility without changing the main project structure.

---

## References

1. Srivastava et al. (2014), Dropout.
2. Ioffe and Szegedy (2015), Batch Normalization.
3. Loshchilov and Hutter (2019), AdamW.
4. Kingma and Ba (2015), Adam.
5. Glorot and Bengio (2010), Xavier initialization.
6. He et al. (2015), Kaiming initialization.
7. Pascanu et al. (2013), Gradient clipping.
8. Goodfellow, Bengio, Courville (2016), Deep Learning.