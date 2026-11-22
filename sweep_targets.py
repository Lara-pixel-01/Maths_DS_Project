from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ts_common import _lgbm, fit_model, grouped_time_cv, make_features

runs = [
    ("RepoHealth", "new_issues", 744),
    ("RepoHealth", "merged_prs", 744),
    ("RepoHealth", "issue_comment_volume", 744),
    ("ClusterTrace", "machine_load_1", 12),
    ("ClusterTrace", "n_job", 12),
    ("ClusterTrace", "n_task", 12),
    ("ClusterTrace", "machine_cpu_usr", 12),
]
n_splits = 6
model_names = ["Ridge", "RandomForest", "LightGBM"]
sets = ["numeric only", "+ TF-IDF", "+ RecAvg"]
predict_delta = True
prescreen = True
prescreen_limit = 0.95
prescreen_only = False
prescreen_datasets = ("ClusterTrace", "RepoHealth")
clean = Path("clean_data")
out = Path("results")
out.mkdir(exist_ok=True)
_src = Path(__file__).with_name("text_features.py").read_text(encoding="utf-8")
_helpers = {
    "__name__": "text_features_helpers",
    "__file__": "text_features.py",
}
exec(compile(_src.split('pd.set_option("display.width", 220)')[0],"text_features.py","exec",),_helpers,)
build_windows, load_embeddings, recavg = ( _helpers["build_windows"],_helpers["load_embeddings"],_helpers["recavg"],)
pca_block, tfidf_block = _helpers["pca_block"], _helpers["tfidf_block"]
steps_ahead, dm_tests, window_hours = (_helpers["steps_ahead"],_helpers["dm_tests"],_helpers["window_hours"],)
_helpers["model_names"], _helpers["sets"] = model_names, sets


def persistence_r2(num):
    r2 = []
    for _, g in num[num["split"] == "train"].groupby("entity_id"):
        y = g.sort_values("date_time")["target"].to_numpy()
        if len(y) < 3 or np.var(y) == 0:
            continue
        r2.append(1 - np.var(np.diff(y)) / np.var(y))
    return float(np.median(r2)) if r2 else np.nan


def run_one(ds, target, H):
    run = f"{ds}_{target}"
    data, cols = make_features(
        pd.read_csv(clean / f"{run}_numeric.csv"), horizon_hours=H
    )
    if predict_delta:
        data["target"] = data["target"] - data["lag_1"]
    texts = pd.read_csv(clean / f"{run}_text.csv")
    texts["date_time"] = pd.to_datetime(texts["date_time"], utc=True)
    texts = (
        texts.dropna(subset=["text_clean"])
        .sort_values(["entity_id", "date_time"])
        .reset_index(drop=True)
    )
    T = texts["text_clean"].astype(str).to_numpy()

    Malpha, _ = build_windows(data, texts, window_hours[ds])
    M = Malpha.copy()
    M.data[:] = 1.0
    print(
        f"rows: {len(data)} | entities: {data['entity_id'].nunique()} | "
        f"texts: {len(texts)} | "
        f"window: {window_hours[ds]} h | rows with text in the window: "
        f"{(Malpha.getnnz(axis=1) > 0).mean():.1%}"
    )
    R_rec = recavg(Malpha, load_embeddings(ds, texts))
    X_numeric = data[cols]
    train_pos = np.where(data["split"].to_numpy() == "train")[0]

    def build_X(set_name, fit_pos):
        if set_name == "numeric only":
            return X_numeric
        Z, prefix = (
            (pca_block(R_rec, fit_pos), "rec")
            if set_name == "+ RecAvg"
            else (tfidf_block(fit_pos, M, T), "svd")
        )
        if Z is None:
            return X_numeric
        return pd.concat(
            [
                X_numeric,
                pd.DataFrame(
                    Z,
                    index=data.index,
                    columns=[f"{prefix}_{i}" for i in range(Z.shape[1])],
                ),
            ],
            axis=1,
        )

    def fit_predict(name, X, fit_pos, ev_pos):
        Xf, yf = X.iloc[fit_pos], data["target"].iloc[fit_pos]
        if name == "LightGBM":
            model = _lgbm().fit(Xf, yf)
        else:
            model = fit_model(name, Xf, yf, data["entity_id"].iloc[fit_pos])
        return model.predict(X.iloc[ev_pos])

    train_df = data.iloc[train_pos].reset_index(drop=True)
    oos = []
    for k, (a, b) in enumerate(grouped_time_cv(train_df, n_splits=n_splits)):
        fit_pos, ev_pos = train_pos[a], train_pos[b]
        preds = {}
        for s in sets:
            X = build_X(s, fit_pos)
            for name in model_names:
                preds[(s, name)] = fit_predict(name, X, fit_pos, ev_pos)
        oos.append((ev_pos, preds))
        print(f"  fold {k + 1}/{n_splits} done")

    pos_all = np.concatenate([p for p, _ in oos])
    assert len(np.unique(pos_all)) == len(pos_all), "a row was validated twice"
    order = np.argsort(pos_all, kind="stable")
    pos_all = pos_all[order]
    y_true, ents = (
        data["target"].to_numpy()[pos_all],
        data["entity_id"].to_numpy()[pos_all],
    )
    pool = {
        k: np.concatenate([pr[k] for _, pr in oos])[order] for k in oos[0][1]
    }
    h_steps = max(1, int(np.median(steps_ahead(data.iloc[pos_all]))))
    print(
        f"pooled out-of-sample rows: {len(y_true)} | forecast horizon = about "
        f"{h_steps} observation(s) ahead"
    )
    pd.DataFrame(
        {
            "row": pos_all,
            "entity_id": ents,
            "date_time": data["date_time"].to_numpy()[pos_all],
            "y_true": y_true,
            **{f"{s_} | {m}": v for (s_, m), v in pool.items()},
        }
    ).to_csv(out / f"{run}_h{H}_sweep_predictions.csv", index=False)

    def mae(pred):
        return float(
            pd.Series(np.abs(y_true - pred)).groupby(ents).mean().mean()
        )

    persistence = (
        np.zeros(len(y_true))
        if predict_delta
        else data["lag_1"].to_numpy()[pos_all]
    )
    print(f"persistence (predict 'no change') MAE: {mae(persistence):.4f}")
    tests = pd.DataFrame(dm_tests(pool, y_true, ents, h_steps, run))
    rows = []
    for name in model_names:
        m = {s: mae(pool[(s, name)]) for s in sets}
        t = tests[
            (tests["hypothesis"] == "H1")
            & (tests["comparison"] == f"{name}: numeric only  vs  + RecAvg")
        ].iloc[0]
        rows.append(
            {
                "Dataset": ds,
                "Target": target,
                "Model": name,
                "Numeric MAE": m["numeric only"],
                "TF-IDF MAE": m["+ TF-IDF"],
                "RecAvg MAE": m["+ RecAvg"],
                "RecAvg Gain %": 100 * (1 - m["+ RecAvg"] / m["numeric only"]),
                "DM p-value (RecAvg vs Numeric)": t["p_value"],
                "Mode": "delta" if predict_delta else "level",
            }
        )
    return rows


pd.set_option("display.width", 200)
keep = []
print(
    "PRE-SCREEN (persistence R2 on the train part = 1 - Var(y_t - y_t-1) / "
    "Var(y_t), median over entities; "
    f"limit {prescreen_limit})"
    if prescreen
    else "PRE-SCREEN is off"
)
for ds, target, H in runs:
    if not (
        (clean / f"{ds}_{target}_numeric.csv").exists()
        and (clean / f"{ds}_{target}_text.csv").exists()
    ):
        print(
            f"NOT RUN {ds}/{target}: clean files missing (run clean_data.py "
            f"first)"
        )
        continue
    if prescreen and ds in prescreen_datasets:
        r2 = persistence_r2(
            pd.read_csv(
                clean / f"{ds}_{target}_numeric.csv",
                usecols=["entity_id", "date_time", "target", "split"],
            )
        )
        if r2 > prescreen_limit:
            print(f"SKIPPING {target}: persistence R2 = {r2:.3f}")
            continue
        print(f"keeping  {target}: persistence R2 = {r2:.3f}")
    keep.append((ds, target, H))


summary = []
if not prescreen_only:
    for ds, target, H in keep:
        print(
            f"\n==================== {ds} / {target} @ {H}h "
            f"({'delta' if predict_delta else 'level'}) ===================="
        )
        summary += run_one(ds, target, H)

if summary:
    tbl = pd.DataFrame(summary)
    p = tbl["DM p-value (RecAvg vs Numeric)"]
    ok = p.notna()
    tbl["FDR-corrected p-value"] = np.nan
    tbl.loc[ok, "FDR-corrected p-value"] = multipletests(
        p[ok], method="fdr_bh"
    )[1]
    cols = ["Dataset","Target","Model","Numeric MAE","TF-IDF MAE","RecAvg MAE","RecAvg Gain %","DM p-value (RecAvg vs Numeric)","FDR-corrected p-value","Mode",]
    tbl = tbl[cols]
    path = out / "sweep_summary.csv"
    tbl.to_csv(path, mode="a", header=not path.exists(), index=False)
    print("\n==================== SWEEP SUMMARY ====================")
    print(
        f"FDR family = {int(ok.sum())} RecAvg-vs-numeric tests of this call. "
        f"Gain % > 0: RecAvg has the lower error."
    )
    print(tbl.round(4).to_string(index=False))
    print(f"\nappended {len(tbl)} rows to {path.resolve()}")
    print(
        "READ THIS: read p together with the gain (a gain of 0.3% can be "
        "'significant' with many pooled rows) and "
        "with the persistence MAE printed above. Untuned LightGBM and 6 "
        "folds: do not compare these MAEs with "
        "the tables of text_features.py."
    )
