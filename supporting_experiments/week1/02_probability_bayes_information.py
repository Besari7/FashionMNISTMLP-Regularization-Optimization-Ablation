"""
02_probability_bayes_information.py
===================================
Week 1 coverage:
- Probability distributions
- Bayes rule
- Information theory (entropy, cross-entropy, KL)

Output:
  supporting_experiments/outputs/week1_probability_info_report.txt
"""

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Bayes rule example: medical test
prior_disease = 0.01
sensitivity = 0.95
false_positive = 0.05

p_pos = sensitivity * prior_disease + false_positive * (1 - prior_disease)
posterior = (sensitivity * prior_disease) / p_pos

# Information theory on discrete distributions
p = np.array([0.7, 0.2, 0.1], dtype=np.float64)
q = np.array([0.6, 0.3, 0.1], dtype=np.float64)

entropy_p = -np.sum(p * np.log2(p))
cross_entropy_pq = -np.sum(p * np.log2(q))
kl_pq = np.sum(p * np.log2(p / q))

report = f"""
================================================================
Week 1: Probability, Bayes, Information Theory
================================================================
Bayes rule example (medical test):
- P(Disease) = {prior_disease}
- P(Pos | Disease) = {sensitivity}
- P(Pos | NoDisease) = {false_positive}
- P(Disease | Pos) = {posterior:.6f}

Information theory:
- p = {p.tolist()}
- q = {q.tolist()}
- H(p) = {entropy_p:.6f} bits
- CE(p,q) = {cross_entropy_pq:.6f} bits
- KL(p||q) = {kl_pq:.6f} bits

Identity check: CE(p,q) = H(p) + KL(p||q)
Difference = {abs(cross_entropy_pq - (entropy_p + kl_pq)):.12f}
================================================================
""".strip()

out_txt = os.path.join(OUT_DIR, "week1_probability_info_report.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"[+] Saved {out_txt}")
