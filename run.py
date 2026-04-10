#!/usr/bin/env python
"""
run.py - Project Entry Point
==============================

Usage (from project root):
    python run.py

Executes the full project pipeline in order:
    Phase 0      Methodological Foundations
    Phase 1      Core Regularization / Generalization Study
    Phase 2      Optimization / Training Stability
    Final Phase  Robustness / Summary Artifacts

All phases are artifact-aware and skip when outputs already exist.
"""

import sys
import os
import io
import multiprocessing as mp
import torch

# Force UTF-8 encoding for Windows console (to support box-drawing characters like \u2500)
if sys.platform == "win32":
    # Keep output line-buffered so long training logs appear immediately.
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
        write_through=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
        write_through=True,
    )

# Import from the src package for editor/static analysis compatibility
from src.train import (
    run_core_regularization_generalization_phase,
    run_final_robustness_summary_phase,
)
from src.foundations import run_foundations_phase
from src.optimization import run_optimization_phase


def _ensure_expected_interpreter() -> None:
    """Fail fast on Windows if run.py is not started with project venv Python."""
    if sys.platform != "win32":
        return

    expected = os.path.normcase(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"))
    )
    current = os.path.normcase(os.path.abspath(sys.executable))

    if os.path.exists(expected) and current != expected:
        raise RuntimeError(
            "Wrong Python interpreter detected.\n"
            f"Expected: {expected}\n"
            f"Current : {current}\n"
            "Please run with: c:/Users/berke/DL_Project/.venv/Scripts/python.exe run.py"
        )


def _force_multiprocessing_venv_executable() -> None:
    """Ensure multiprocessing children use the active interpreter path."""
    if sys.platform == "win32":
        mp.set_executable(sys.executable)


def _print_runtime_diagnostics() -> None:
    """Print active interpreter and CUDA info for launch debugging."""
    print("=" * 60)
    print("RUNTIME DIAGNOSTICS")
    print("=" * 60)
    print(f"Python executable : {sys.executable}")
    print(f"Torch version     : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device       : {torch.cuda.get_device_name(0)}")
    print(f"FORCE_PHASE1_RETRAIN={os.environ.get('FORCE_PHASE1_RETRAIN', '0')}")
    print("=" * 60)

if __name__ == "__main__":
    _ensure_expected_interpreter()
    _force_multiprocessing_venv_executable()
    _print_runtime_diagnostics()

    # Phase 0: Methodological Foundations
    run_foundations_phase()

    # Phase 1: Core Regularization / Generalization Study
    run_core_regularization_generalization_phase()

    # Phase 2: Optimization / Training Stability
    run_optimization_phase()

    # Final Phase: Robustness / Summary Artifacts
    run_final_robustness_summary_phase()
