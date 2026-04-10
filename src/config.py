"""
config.py - Experiment Configuration
======================================

Central configuration for all hyperparameters, search spaces, and paths.
All experiments share these settings to ensure fair comparison.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "FashionMNIST"
INPUT_DIM = 784          # 28 x 28 flattened
NUM_CLASSES = 10         # 10 clothing categories
TRAIN_SIZE = 48000       # training subset
VAL_SIZE = 12000         # validation subset

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7             # early-stopping patience (epochs)
NUM_WORKERS = 0          # DataLoader workers (0 for Windows compatibility)
PIN_MEMORY = True        # enables faster host->GPU transfer when CUDA is used
NON_BLOCKING = True      # non_blocking tensor transfer for CUDA pipelines

# ---------------------------------------------------------------------------
# Multi-Seed Reproducibility
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 999]

# ---------------------------------------------------------------------------
# Hyperparameter Grid Search
# ---------------------------------------------------------------------------
GRID_SEARCH_EPOCHS = 20  # reduced epoch budget for search phase
DROPOUT_SEARCH_SPACE = [0.2, 0.5]
L1_SEARCH_SPACE = [1e-4, 1e-3]
L2_SEARCH_SPACE = [1e-4, 1e-3]

# ---------------------------------------------------------------------------
# Data Augmentation (extra experiment variant)
# ---------------------------------------------------------------------------
AUGMENTATION_ROTATION = 15   # degrees for RandomRotation
AUGMENTATION_PADDING = 4     # pixels for RandomCrop padding

# ---------------------------------------------------------------------------
# Label Smoothing (extra experiment variant)
# ---------------------------------------------------------------------------
LABEL_SMOOTHING = 0.1        # soft-target smoothing factor

# ---------------------------------------------------------------------------
# FGSM Adversarial Robustness Evaluation
# ---------------------------------------------------------------------------
FGSM_EPSILON = 0.05          # perturbation budget for FGSM attack
