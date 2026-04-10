# Supporting Experiments

This folder contains isolated, lightweight scripts for pedagogical or toy
demonstrations that are intentionally kept outside the main pipeline.

These scripts are standalone and do not modify main project code in `src/`,
main result artifacts in `results/`, or main checkpoints in `checkpoints/`.

This area also includes foundational coverage scripts (Week 1 and selected
Week 2/3/4 topics) that are intentionally kept separate from the main
pipeline because they are pedagogical and lightweight.

## Directory Structure
```
supporting_experiments/
  outputs/           ← Generated figures and text logs for supporting scripts
    week5/           ← Week 5 supporting outputs only
  checkpoints/       ← Any explicitly requested checkpoint files
  week1/
    01_linear_algebra_eig_svd.py
    02_probability_bayes_information.py
    03_numerical_stability_and_autograd.py
  week2/
    01_linear_regression_scratch.py
    02_linreg_sgd_vs_batch.py
    03_linear_regression_pytorch.py
    04_logistic_regression_scratch.py
    05_standard_error_and_bayesian_estimation.py
  week3/
    01_xor_intuition.py        ← Hand-crafted 2-layer MLP solves XOR
    02_mlp_from_scratch.py
    03_mlp_pytorch.py
    04_backprop_memoization_demo.py
  week4/
    semi_supervised/01_pseudo_labeling.py
    self_supervised/06_rotation_pretext_demo.py
    multi_task/02_mtl_demo.py
    bagging/03_bagging_demo.py
    boosting/04_boosting_demo.py
    adversarial_training/05_adv_training_demo.py
  week5/
    01_optimizer_landscape.py      ← 2D loss surface trajectories
    02_saddle_point_demo.py        ← Saddle point escape
    03_batch_size_effect.py        ← Minibatch size comparison
    04_layernorm_vs_batchnorm.py   ← LayerNorm vs BatchNorm
    05_second_order_notes.py       ← Newton/L-BFGS demo + notes
```

## How to Run
From the repository root, run each script with the project Python environment:

```powershell
# Week 1
python supporting_experiments\week1\01_linear_algebra_eig_svd.py
python supporting_experiments\week1\02_probability_bayes_information.py
python supporting_experiments\week1\03_numerical_stability_and_autograd.py

# Week 2
python supporting_experiments\week2\01_linear_regression_scratch.py
python supporting_experiments\week2\02_linreg_sgd_vs_batch.py
python supporting_experiments\week2\03_linear_regression_pytorch.py
python supporting_experiments\week2\04_logistic_regression_scratch.py
python supporting_experiments\week2\05_standard_error_and_bayesian_estimation.py

# Week 3
python supporting_experiments\week3\01_xor_intuition.py
python supporting_experiments\week3\02_mlp_from_scratch.py
python supporting_experiments\week3\03_mlp_pytorch.py
python supporting_experiments\week3\04_backprop_memoization_demo.py

# Week 4
python supporting_experiments\week4\semi_supervised\01_pseudo_labeling.py
python supporting_experiments\week4\self_supervised\06_rotation_pretext_demo.py
python supporting_experiments\week4\multi_task\02_mtl_demo.py
python supporting_experiments\week4\bagging\03_bagging_demo.py
python supporting_experiments\week4\boosting\04_boosting_demo.py
python supporting_experiments\week4\adversarial_training\05_adv_training_demo.py

# Week 5
python supporting_experiments\week5\01_optimizer_landscape.py
python supporting_experiments\week5\02_saddle_point_demo.py
python supporting_experiments\week5\03_batch_size_effect.py
python supporting_experiments\week5\04_layernorm_vs_batchnorm.py
python supporting_experiments\week5\05_second_order_notes.py
```

## Output Locations

All generated artifacts from this folder are written under:

1. `supporting_experiments/outputs/`
2. `supporting_experiments/checkpoints/` (only when explicitly used)

No supporting script should write into `results/optimization/` or
`checkpoints/optimization/`.
