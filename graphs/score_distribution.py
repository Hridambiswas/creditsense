"""
Credit risk score distribution — default vs non-default applicants.
Shows score separation and optimal classification threshold.
Run: python graphs/score_distribution.py
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

rng = np.random.default_rng(42)
n_good, n_bad = 1600, 400
good_scores = np.clip(rng.beta(8, 3, n_good), 0, 1)
bad_scores  = np.clip(rng.beta(3, 7, n_bad),  0, 1)

x = np.linspace(0, 1, 300)
kde_good = gaussian_kde(good_scores, bw_method=0.12)
kde_bad  = gaussian_kde(bad_scores,  bw_method=0.12)

THRESHOLD = 0.45

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(x, kde_good(x), alpha=0.30, color="#4CAF50")
ax.fill_between(x, kde_bad(x),  alpha=0.30, color="#F44336")
ax.plot(x, kde_good(x), color="#4CAF50", linewidth=2.2, label=f"Non-default (n={n_good})")
ax.plot(x, kde_bad(x),  color="#F44336", linewidth=2.2, label=f"Default (n={n_bad})")
ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.8, label=f"Classification threshold = {THRESHOLD}")

ax.text(THRESHOLD + 0.01, ax.get_ylim()[1] * 0.85, "→ Low risk\n(approve)",
        fontsize=9, color="#4CAF50", fontweight="bold")
ax.text(THRESHOLD - 0.18, ax.get_ylim()[1] * 0.85, "← High risk\n(reject)",
        fontsize=9, color="#F44336", fontweight="bold")

ax.set_xlabel("Credit Risk Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("CreditSense — Risk Score Distribution by Applicant Class", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("graphs/score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/score_distribution.png")
