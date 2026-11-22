import re
from pathlib import Path

import numpy as np
import pandas as pd

from ts_common import _lgbm, fit_model, grouped_time_cv, make_features, metrics

clean = Path("clean_data")
out = Path("results")
out.mkdir(exist_ok=True)

datasets = ["RepoHealth", "EPA-Air"]
models = ["Ridge", "LightGBM"]
history = re.compile(r"^(lag_\d+|mean_\d+|std_\d+|tmean_\w+|tstd_\w+)$")
pd.set_option("display.width", 200)


def feature_sets(cols):
    hist = [c for c in cols if history.match(c)]
    has_latest = ["lag_1"] + [
        c for c in hist if c.startswith(("mean_", "std_", "tmean_", "tstd_"))
    ]
    return {
        "all features": list(cols),
        "no lag_1": [c for c in cols if c != "lag_1"],
        "no latest value": [c for c in cols if c not in has_latest],
        "no target history": [c for c in cols if c not in hist],
    }


def fit_predict(model, Xtr, ytr, ent, Xev):
    if model == "LightGBM":
        return _lgbm().fit(Xtr, ytr).predict(Xev)
    return fit_model(model, Xtr, ytr, ent, delta=False).predict(Xev)


def train_mean_baseline(tr, ev):
    return ev["entity_id"].map(tr.groupby("entity_id")["target"].mean()).values


rows_t1, rows_t2 = [], []
for ds in datasets:
    path = clean / f"{ds}_numeric.csv"
    if not path.exists():
        print(f"\n===== {ds}: {path} not found, skipped =====")
        continue
    data, cols = make_features(pd.read_csv(path))
    train = data[data["split"] == "train"].reset_index(drop=True)
    val = data[data["split"] == "val"].reset_index(drop=True)
    sets = feature_sets(cols)
    print(
        f"\n===== {ds}: train {len(train)}, val {len(val)}, {len(cols)} "
        f"features ====="
    )
    for name, fs in sets.items():
        print(
            f"  {name:18s} {len(fs):3d} features"
            + (
                ""
                if name == "all features"
                else f"  (removed: {sorted(set(cols) - set(fs))[:12]}"
                f"{'...' if len(set(cols) - set(fs)) > 12 else ''})"
            )
        )

    folds = list(grouped_time_cv(train))
    cv_mae = {(s, m): [] for s in sets for m in models}
    cv_mae[("baseline", "last value")], cv_mae[("baseline", "train mean")] = (
        [],
        [],
    )
    oos = {
        k: np.full(len(train), np.nan)
        for k in list(cv_mae)
        if k[0] != "baseline"
    }
    oos_last, oos_mean = np.full(len(train), np.nan), np.full(
        len(train), np.nan
    )
    for a, b in folds:
        tr, ev = train.iloc[a], train.iloc[b]
        base = train_mean_baseline(tr, ev)
        cv_mae[("baseline", "last value")].append(
            metrics(ev, ev["lag_1"].values, base)[0]["MAE"]
        )
        cv_mae[("baseline", "train mean")].append(
            metrics(ev, base, base)[0]["MAE"]
        )
        oos_last[b], oos_mean[b] = ev["lag_1"].values, base
        for s, fs in sets.items():
            for m in models:
                pred = fit_predict(
                    m, tr[fs], tr["target"], tr["entity_id"], ev[fs]
                )
                cv_mae[(s, m)].append(metrics(ev, pred, base)[0]["MAE"])
                oos[(s, m)][b] = pred
    base_v = train_mean_baseline(train, val)
    val_mae = {
        ("baseline", "last value"): metrics(val, val["lag_1"].values, base_v)[
            0
        ]["MAE"],
        ("baseline", "train mean"): metrics(val, base_v, base_v)[0]["MAE"],
    }
    for s, fs in sets.items():
        for m in models:
            val_mae[(s, m)] = metrics(
                val,
                fit_predict(
                    m, train[fs], train["target"], train["entity_id"], val[fs]
                ),
                base_v,
            )[0]["MAE"]

    ref = {m: np.mean(cv_mae[("all features", m)]) for m in models}
    tab = []
    for (s, m), v in cv_mae.items():
        row = {
            "dataset": ds,
            "feature set": s,
            "model": m,
            "CV MAE": float(np.mean(v)),
            "CV MAE std": float(np.std(v)),
            "validation MAE": val_mae[(s, m)],
        }
        row["CV change vs all features %"] = (
            100 * (row["CV MAE"] / ref[m] - 1) if m in ref else np.nan
        )
        tab.append(row)
    t1 = pd.DataFrame(tab)
    rows_t1.append(t1)
    print(
        f"\nTEST 1 ({ds}): MAE, lower is better. 'change' = % worse (+) or "
        f"better (-) than the same model with all features"
    )
    print(
        t1[
            [
                "feature set",
                "model",
                "CV MAE",
                "CV MAE std",
                "CV change vs all features %",
                "validation MAE",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    ok = ~np.isnan(oos_last)
    dev = (train["lag_1"] - train["mean_20"]).abs() / train["std_20"].replace(
        0, np.nan
    )
    ok &= dev.notna().to_numpy()
    d = dev.to_numpy()
    q50, q90 = np.nanpercentile(d[ok], [50, 90])
    groups = {
        "near (lowest 50%)": ok & (d <= q50),
        "mid (50-90%)": ok & (d > q50) & (d <= q90),
        "far (top 10%)": ok & (d > q90),
    }
    y = train["target"].to_numpy()
    for g, mask in groups.items():
        r = {
            "dataset": ds,
            "group": g,
            "n rows": int(mask.sum()),
            "MAE last value": np.abs(y[mask] - oos_last[mask]).mean(),
            "MAE train mean": np.abs(y[mask] - oos_mean[mask]).mean(),
        }
        for m in models:
            for s in ("all features", "no latest value"):
                p = oos[(s, m)]
                r[f"MAE {m} / {s}"] = np.abs(y[mask] - p[mask]).mean()
            p = oos[("all features", m)]
            r[f"{m} (all) beats last value, share of rows"] = float(
                (
                    np.abs(y[mask] - p[mask])
                    < np.abs(y[mask] - oos_last[mask])
                ).mean()
            )
        rows_t2.append(r)
    t2 = pd.DataFrame([r for r in rows_t2 if r["dataset"] == ds])
    print(
        f"\nTEST 2 ({ds}): pooled out-of-sample MAE by how far lag_1 is from "
        f"its recent average"
    )
    print(t2.drop(columns="dataset").round(3).T.to_string(header=False))

if rows_t1:
    pd.concat(rows_t1).to_csv(out / "lag_ablation.csv", index=False)
    pd.DataFrame(rows_t2).to_csv(
        out / "lag_ablation_hard_cases.csv", index=False
    )
    print(
        f"\nsaved {out / 'lag_ablation.csv'} and "
        f"{out / 'lag_ablation_hard_cases.csv'}. Test data was NOT used."
    )
    print(
        "How to read: 'no latest value' close to 'all features' = the model "
        "does not live on the latest value. "
        "In the far group the model should still beat 'last value' (share of "
        "rows above 50%, MAE below)."
    )
