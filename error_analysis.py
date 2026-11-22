from pathlib import Path

import numpy as np
import pandas as pd

from tf_helpers import H as T
from ts_common import make_features

steps_ahead, to_ns = T["steps_ahead"], T["to_ns"]
clean = Path("clean_data")
res = Path("results")

runs = [
    ("ClusterTrace_h12", "ClusterTrace", 12),
    ("RepoHealth_h744", "RepoHealth", 744),
    ("EPA-Air_h168", "EPA-Air", 168),
    ("GDELT_h336", "GDELT", 336),
    ("StudentLife_h744", "StudentLife", 744),
]
model_preference = ["VotingEnsemble", "LightGBM", "Ridge", "RandomForest"]
text_sets = ["+ TF-IDF", "+ RecAvg", "+ LatestText"]
n_boot = 1000
rng = np.random.default_rng(0)
pd.set_option("display.width", 220)


def thirds(x, names=("low", "mid", "high")):
    x = pd.Series(x)
    out = pd.Series(index=x.index, dtype=object)
    ok = x.notna()
    if ok.sum() < 30 or x[ok].nunique() < 2:
        return out
    codes = pd.qcut(x[ok], 3, labels=False, duplicates="drop")
    k = int(codes.max()) + 1 if codes.notna().any() else 0
    if k < 2:
        return out
    use = names if k == 3 else (names[0], names[-1])
    out[ok] = [use[int(c)] for c in codes]
    return out


def text_recency_hours(sub, texts):
    t_cut = to_ns(sub["t_cut"])
    t_txt, e_txt = to_ns(texts["date_time"]), texts["entity_id"].to_numpy()
    out = np.full(len(sub), np.nan)
    for e, idx in sub.groupby("entity_id").indices.items():
        tt = np.sort(t_txt[e_txt == e])
        if len(tt) == 0:
            continue
        k = np.searchsorted(tt, t_cut[idx], side="right") - 1
        ok = k >= 0
        out[idx[ok]] = (t_cut[idx[ok]] - tt[k[ok]]) / np.timedelta64(1, "h")
    return out


def analyse(run, ds, H):
    f = res / f"{run}_cv_predictions.csv"
    if not (f.exists() and (clean / f"{ds}_numeric.csv").exists()):
        print(f"{run}: files missing, skipped")
        return []
    pred = pd.read_csv(f)
    data, _ = make_features(
        pd.read_csv(clean / f"{ds}_numeric.csv"), horizon_hours=H
    )
    sub = data.iloc[pred["row"].to_numpy()].reset_index(drop=True)
    assert (
        sub["entity_id"].to_numpy() == pred["entity_id"].to_numpy()
    ).all() and np.allclose(
        sub["target"].to_numpy(), pred["y_true"].to_numpy()
    ), (
        "the saved predictions do not match the cleaned data: "
        "run clean_data.py and text_features.py again together"
    )
    model = next(
        (m for m in model_preference if f"numeric only | {m}" in pred.columns),
        None,
    )
    if model is None:
        print(f"{run}: no model found in the predictions, skipped")
        return []
    y = pred["y_true"].to_numpy()
    loss_num = np.abs(y - pred[f"numeric only | {model}"].to_numpy())
    loss_txt = {
        s: np.abs(y - pred[f"{s} | {model}"].to_numpy())
        for s in text_sets
        if f"{s} | {model}" in pred.columns
    }
    texts = pd.read_csv(clean / f"{ds}_text.csv")
    texts["date_time"] = pd.to_datetime(texts["date_time"], utc=True)

    strata = {}
    strata["gap before origin"] = thirds(sub["gap_prev_h"])
    if "lead_h" in sub:
        strata["lead time"] = thirds(sub["lead_h"])
    rec = text_recency_hours(sub, texts)
    lab = thirds(rec, ("recent", "mid", "old"))
    lab[np.isnan(rec)] = "no text yet"
    strata["text recency"] = lab
    n_rows = data.groupby("entity_id")["target"].size()
    if len(n_rows) >= 3:
        strata["entity length"] = thirds(
            sub["entity_id"].map(n_rows), ("short", "mid", "long")
        )
    med = pd.Series(y).groupby(sub["entity_id"].to_numpy()).transform("median")
    mad = (pd.Series(y) - med).abs().groupby(
        sub["entity_id"].to_numpy()
    ).transform("median") * 1.4826
    strata["target size (outcome!)"] = pd.Series(
        np.where((np.abs(y - med) > 3 * mad) & (mad > 0), "extreme", "typical")
    )

    h = max(1, int(np.median(steps_ahead(sub))))
    B = max(2 * h, 20)
    key = (
        sub["entity_id"].astype(str)
        + "#"
        + (sub.groupby("entity_id").cumcount() // B).astype(str)
    )
    block = pd.factorize(key)[0]
    K = block.max() + 1

    out = []
    print(
        f"\n==================== {run}  (model: {model}, {len(y)} rows, "
        f"blocks of {B} rows) ===================="
    )
    for s, lt in loss_txt.items():
        for sname, labels in strata.items():
            for g in pd.unique(labels.dropna()):
                m = (labels == g).to_numpy()
                if m.sum() < 30:
                    continue
                bn = np.bincount(
                    block, weights=np.where(m, loss_num, 0), minlength=K
                )
                bt = np.bincount(
                    block, weights=np.where(m, lt, 0), minlength=K
                )
                gain = 100 * (1 - bt.sum() / bn.sum())
                active = np.where(bn > 0)[0]
                if len(active) >= 5:
                    draws = rng.choice(active, size=(n_boot, len(active)))
                    boot = 100 * (1 - bt[draws].sum(1) / bn[draws].sum(1))
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                else:
                    lo = hi = np.nan
                out.append(
                    {
                        "run": run,
                        "model": model,
                        "text set": s,
                        "stratum": sname,
                        "group": g,
                        "n": int(m.sum()),
                        "numeric MAE": loss_num[m].mean(),
                        "text MAE": lt[m].mean(),
                        "gain %": gain,
                        "CI low": lo,
                        "CI high": hi,
                        "CI excludes 0": (
                            bool(lo > 0 or hi < 0)
                            if np.isfinite(lo)
                            else False
                        ),
                    }
                )
    tbl = pd.DataFrame(out)
    if len(tbl):
        view = tbl[
            tbl["text set"]
            == ("+ RecAvg" if "+ RecAvg" in loss_txt else list(loss_txt)[0])
        ]
        print(
            f"text set shown: {view['text set'].iloc[0]} (all sets are in the "
            f"CSV)"
        )
        print(
            view[
                [
                    "stratum",
                    "group",
                    "n",
                    "numeric MAE",
                    "text MAE",
                    "gain %",
                    "CI low",
                    "CI high",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    return out


rows = []
for run, ds, H in runs:
    rows += analyse(run, ds, H)
if rows:
    tbl = pd.DataFrame(rows)
    tbl.to_csv(res / "error_analysis.csv", index=False)
    n_sig = int(tbl["CI excludes 0"].sum())
    pos = int(((tbl["CI excludes 0"]) & (tbl["gain %"] > 0)).sum())
    print(
        f"\nsaved {res / 'error_analysis.csv'}: {len(tbl)} cells, {n_sig} "
        f"with an interval that excludes 0 "
        f"({pos} where text helped, {n_sig - pos} where text hurt). By chance "
        f"alone about 5% of cells would. "
        f"A real effect shows up as the same direction across strata and runs."
    )
