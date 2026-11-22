from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

clean = Path("clean_data")
res = Path("results")
datasets = ["ClusterTrace", "RepoHealth", "EPA-Air", "GDELT", "StudentLife"]
bin_counts = (100, 200, 400)
n_perm = 1000
rng = np.random.default_rng(0)
pd.set_option("display.width", 200)


def ranks(a):
    return stats.rankdata(a)


def partial_rho(x, y, Z):
    D = np.column_stack(
        [np.ones(len(x))] + [ranks(Z[:, j]) for j in range(Z.shape[1])]
    )
    P = np.eye(len(x)) - D @ np.linalg.pinv(D)
    rx, ry = P @ ranks(x), P @ ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def entity_result(t_num, y_num, t_txt, n_bins):
    t0, t1 = t_num.min(), t_num.max()
    width = (t1 - t0) / n_bins
    b = np.minimum(((t_num - t0) / width).astype(int), n_bins - 1)
    ybar = pd.Series(y_num).groupby(b).mean().reindex(range(n_bins)).ffill()
    nobs = np.bincount(b, minlength=n_bins).astype(float)
    tt = t_txt[(t_txt >= t0) & (t_txt <= t1)]
    cnt = np.bincount(
        np.minimum(((tt - t0) / width).astype(int), n_bins - 1),
        minlength=n_bins,
    ).astype(float)
    vol = ybar.diff().abs().to_numpy()
    x_full, y_full = cnt[:-1], vol[1:]
    Z_full = np.column_stack([nobs[1:], nobs[:-1]])
    ok = np.isfinite(y_full)
    if ok.sum() < 40 or np.std(x_full[ok]) == 0 or np.std(y_full[ok]) == 0:
        return None
    y_, Z_ = y_full[ok], Z_full[ok]
    raw = float(stats.spearmanr(x_full[ok], y_)[0])
    part = partial_rho(x_full[ok], y_, Z_)
    if not np.isfinite(part):
        return None
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = int(rng.integers(1, len(x_full)))
        null[i] = partial_rho(np.roll(x_full, s)[ok], y_, Z_)
    p = (1 + np.sum(np.abs(null[np.isfinite(null)]) >= abs(part))) / (
        1 + np.isfinite(null).sum()
    )
    return raw, part, p


rows = []
for ds in datasets:
    fn, ft = clean / f"{ds}_numeric.csv", clean / f"{ds}_text.csv"
    if not (fn.exists() and ft.exists()):
        continue
    num = pd.read_csv(fn, usecols=["entity_id", "date_time", "target"])
    txt = pd.read_csv(ft, usecols=["entity_id", "date_time"])
    num["t"] = pd.to_datetime(num["date_time"], utc=True).astype("int64")
    txt["t"] = pd.to_datetime(txt["date_time"], utc=True).astype("int64")
    for n_bins in bin_counts:
        out = []
        for e, g in num.groupby("entity_id"):
            tt = txt.loc[txt["entity_id"] == e, "t"].to_numpy()
            if len(g) < 60 or len(tt) < 20 or g["t"].max() <= g["t"].min():
                continue
            r = entity_result(
                g["t"].to_numpy(float),
                g["target"].to_numpy(),
                tt.astype(float),
                n_bins,
            )
            if r:
                out.append(r)
        if not out:
            continue
        raw, part, p = (np.array(c) for c in zip(*out))
        rows.append(
            {
                "dataset": ds,
                "bins": n_bins,
                "entities": len(out),
                "median rho (raw)": np.median(raw),
                "median rho (density removed)": np.median(part),
                "share positive": float(np.mean(part > 0)),
                "Fisher p (circular shift)": stats.combine_pvalues(
                    p, method="fisher"
                )[1],
            }
        )
if rows:
    tbl = pd.DataFrame(rows)
    tbl.to_csv(res / "h5_followup.csv", index=False)
    print(tbl.round(4).to_string(index=False))
    print(
        "\nIf 'density removed' is much smaller than 'raw' (or its p is "
        "large) the link was an artefact of how many "
        "numeric observations fall in each bin. A real effect should survive "
        "all three bin sizes with the same sign."
    )
else:
    print("nothing to test: clean_data files not found")
