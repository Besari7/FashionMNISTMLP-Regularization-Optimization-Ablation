"""
05_standard_error_and_bayesian_estimation.py
===========================================
Week 2 coverage:
- Point estimation (theta vs theta_hat)
- Standard error
- Bayesian estimation (Beta-Bernoulli conjugacy)

Output:
  supporting_experiments/outputs/week2_standard_error_bayes_report.txt
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# Point estimation and standard error for mean
theta_true = 3.0
sigma_true = 2.0
n = 200
samples = np.random.normal(loc=theta_true, scale=sigma_true, size=n)

theta_hat = np.mean(samples)
sample_std = np.std(samples, ddof=1)
standard_error = sample_std / np.sqrt(n)

# Bayesian estimation with Beta-Bernoulli
# prior Beta(alpha, beta), posterior Beta(alpha + heads, beta + tails)
alpha_prior, beta_prior = 2.0, 2.0
trials = 120
p_true = 0.65
obs = np.random.binomial(1, p_true, size=trials)
heads = int(np.sum(obs))
tails = trials - heads

alpha_post = alpha_prior + heads
beta_post = beta_prior + tails
posterior_mean = alpha_post / (alpha_post + beta_post)
mle = heads / trials

report = f"""
================================================================
Week 2: Estimation, Standard Error, Bayesian Statistics
================================================================
Point estimation (Gaussian mean):
- True theta: {theta_true}
- Estimated theta_hat: {theta_hat:.6f}
- Sample std: {sample_std:.6f}
- Standard error (SE): {standard_error:.6f}

Beta-Bernoulli Bayesian estimation:
- Prior: Beta({alpha_prior}, {beta_prior})
- Observations: heads={heads}, tails={tails}
- MLE: {mle:.6f}
- Posterior: Beta({alpha_post}, {beta_post})
- Posterior mean: {posterior_mean:.6f}

Notes:
- SE quantifies uncertainty of theta_hat as an estimator of theta.
- Bayesian estimate combines prior belief and observed data.
================================================================
""".strip()

out_txt = os.path.join(OUT_DIR, "week2_standard_error_bayes_report.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"[+] Saved {out_txt}")
