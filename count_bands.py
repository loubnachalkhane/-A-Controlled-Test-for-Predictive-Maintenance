import numpy as np
from scipy.stats import norm
R = r"C:\calibrag_results\run_20261107_1613"
for ds in ["FD001", "FD002", "FD003", "FD004"]:
    d = np.load(rf"{R}\{ds}\seed0_preds.npz")
    F = norm.cdf((int(d["H"]) - d["q50"]) / np.maximum((d["q90C"] - d["q10C"]) / (2 * norm.ppf(0.9)), 1e-6))
    print(ds, "n =", len(F))
    for cf in (5, 10, 20, 50):
        t = 1 / cf
        print(f"   Cf={cf:>2} tau={t:.2f}: within a factor 2 of tau:", int(((F >= t / 2) & (F <= 2 * t)).sum()),
              "| informative 0.02-0.8:", int(((F >= 0.02) & (F <= 0.8)).sum()))
