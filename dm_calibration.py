from pathlib import Path

import numpy as np
import pandas as pd

from tf_helpers import H as T

dm_test = T["dm_test"]
res = Path("results")
res.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
reps = 200
pd.set_option("display.width", 200)


def ma(n, h):
    return np.convolve(
        rng.normal(size=n + h - 1), np.ones(h) / np.sqrt(h), "valid"
    )


def simulate(h, E, rho, n, delta, similar=0.0, own=0.7):
    S_base, S_a, S_b = ma(n, h), ma(n, h), ma(n, h)
    la, lb, ids = [], [], []
    for e in range(E):
        mix = lambda S: np.sqrt(rho) * S + np.sqrt(1 - rho) * ma(n, h)
        base, part_a, part_b_free = mix(S_base), mix(S_a), mix(S_b)
        part_b = np.sqrt(similar) * part_a + np.sqrt(1 - similar) * part_b_free
        la.append(np.abs(base + own * part_a))
        lb.append(np.abs(base + own * (1 - delta) * part_b))
        ids += [e] * n
    return np.concatenate(la), np.concatenate(lb), np.array(ids)


def rejection(h, E, rho, n, delta, similar=0.0):
    rej, gains = 0, []
    for _ in range(reps):
        la, lb, ids = simulate(h, E, rho, n, delta, similar)
        r = dm_test(la, lb, ids, h)
        rej += bool(np.isfinite(r["p_value"]) and r["p_value"] < 0.05)
        gains.append(r["rel_gain_pct"])
    return rej / reps, float(np.mean(gains))


rows = []
print(
    "TABLE 1: SIZE. A and B equally good, share of runs with p < 0.05 (should "
    "be about 0.05)"
)
for h in (1, 31, 115):
    for E, rho in ((1, 0.0), (8, 0.0), (8, 0.5)):
        rate, _ = rejection(h, E, rho, 800, 0.0)
        rows.append(
            {
                "table": "size",
                "h": h,
                "entities": E,
                "shared share": rho,
                "n per entity": 800,
                "delta": 0.0,
                "reject rate": rate,
                "mean gain %": np.nan,
            }
        )
        print(
            f"  h = {h:>3}, E = {E}, shared share = {rho}: rejects in "
            f"{rate:.1%} of {reps} runs"
        )

print(
    "\nTABLE 2: SMALL GAINS. B almost a copy of A but slightly better (E = 1, "
    "h = 31, 95% of the own errors are shared)"
)
for n in (500, 2000, 8000):
    for delta in (0.0, 0.01, 0.02, 0.05):
        rate, gain = rejection(31, 1, 0.0, n, delta, similar=0.95)
        rows.append(
            {
                "table": "power",
                "h": 31,
                "entities": 1,
                "shared share": 0.0,
                "n per entity": n,
                "delta": delta,
                "reject rate": rate,
                "mean gain %": gain,
            }
        )
        print(
            f"  n = {n:>4}, B's error shrunk by {delta:.0%}: average gain "
            f"{gain:5.2f}% -> rejects in {rate:.1%}"
        )
pd.DataFrame(rows).to_csv(res / "dm_calibration.csv", index=False)
print(f"\nsaved {res / 'dm_calibration.csv'}")
print(
    "Reading: rates well above 5% in table 1 mean the p-values are too small "
    "(typical with common events between entities "
    "and long horizons). In table 2 note how small the gain can be while the "
    "test already rejects often."
)
