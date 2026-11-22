from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from ts_common import _lgbm, fit_model,grouped_time_cv,inner_splits,make_features

runs = [("RepoHealth", "new_issues", 744), ("RepoHealth", "merged_prs", 744)]
numeric_models = ["LightGBM", "Ridge"]
text_sets = ["RecAvg", "TF-IDF"]
stage2 = ["Ridge", "LightGBM", "Gated"]
stage2_lgbm = dict(
    objective="l1",
    n_estimators=150,
    learning_rate=0.03,
    num_leaves=7,
    min_child_samples=50,
    reg_lambda=10.0,
)
outer_splits = 6
clean = Path("clean_data")
out = Path("results")
out.mkdir(exist_ok=True)

_src = Path(__file__).with_name("text_features.py").read_text(encoding="utf-8")
_helpers = {
    "__name__": "text_features_helpers",
    "__file__": "text_features.py",
}
exec(
    compile(
        _src.split('pd.set_option("display.width", 220)')[0],
        "text_features.py",
        "exec",
    ),
    _helpers,
)
build_windows, load_embeddings, recavg = (
    _helpers["build_windows"],
    _helpers["load_embeddings"],
    _helpers["recavg"],
)
pca_block, tfidf_block, steps_ahead = (
    _helpers["pca_block"],
    _helpers["tfidf_block"],
    _helpers["steps_ahead"],
)
dm_test, window_hours = _helpers["dm_test"], _helpers["window_hours"]


def gated_correction(Ztr, resid, ent_tr, Zev, rp_ev):
    df = pd.DataFrame({"entity_id": ent_tr, "target": resid})
    idx, lab = [], []
    for a, b in grouped_time_cv(df, n_splits=2):
        if len(a) < 30:
            continue
        rp = fit_model(
            "Ridge", Ztr.iloc[a], pd.Series(resid[a]), ent_tr[a]
        ).predict(Ztr.iloc[b])
        idx.append(b)
        lab.append((np.abs(resid[b] - rp) < np.abs(resid[b])).astype(int))
    if not idx:
        return np.zeros(len(Zev))
    idx, lab = np.concatenate(idx), np.concatenate(lab)
    if len(np.unique(lab)) < 2:
        p = np.full(len(Zev), float(lab.mean()))
    else:
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
        ).fit(Ztr.iloc[idx], lab)
        p = clf.predict_proba(Zev)[:, 1]
    return np.clip(2 * (p - 0.5), 0, 1) * rp_ev


def run_one(ds, target, H):
    run = f"{ds}_{target}"
    data, cols = make_features(
        pd.read_csv(clean / f"{run}_numeric.csv"), horizon_hours=H
    )
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
    R_rec = recavg(Malpha, load_embeddings(ds, texts))
    print(
        f"rows: {len(data)} | entities: {data['entity_id'].nunique()} | "
        f"texts: {len(texts)} | "
        f"rows with text in the window: "
        f"{(Malpha.getnnz(axis=1) > 0).mean():.1%}"
    )

    X_num, y, ent = (
        data[cols],
        data["target"].to_numpy(),
        data["entity_id"].to_numpy(),
    )
    split = data["split"].to_numpy()
    train_pos, val_pos = (np.where(split == s)[0] for s in ("train", "val"))

    def fit_numeric(name, pos):
        if name == "LightGBM":
            return _lgbm().fit(X_num.iloc[pos], y[pos])
        return fit_model("Ridge", X_num.iloc[pos], pd.Series(y[pos]), ent[pos])

    def evaluate(fit_pos, ev_pos):
        assert set(ev_pos).isdisjoint(
            fit_pos
        ), "evaluation rows must not be training rows"
        Z = {
            "RecAvg": pca_block(R_rec, fit_pos),
            "TF-IDF": tfidf_block(fit_pos, M, T),
        }
        zcols = {
            k: [f"z{i}" for i in range(v.shape[1])]
            for k, v in Z.items()
            if v is not None
        }
        inner = list(
            grouped_time_cv(
                data.iloc[fit_pos].reset_index(drop=True),
                n_splits=inner_splits,
            )
        )
        res = {}
        for name in numeric_models:
            oof_pos, oof_pred = [], []
            for a, b in inner:
                m = fit_numeric(name, fit_pos[a])
                oof_pos.append(fit_pos[b])
                oof_pred.append(m.predict(X_num.iloc[fit_pos[b]]))
            oof_pos, oof_pred = np.concatenate(oof_pos), np.concatenate(
                oof_pred
            )
            assert (
                len(np.unique(oof_pos)) == len(oof_pos)
                and np.isin(oof_pos, fit_pos).all()
            )
            resid = y[oof_pos] - oof_pred
            p_num = fit_numeric(name, fit_pos).predict(X_num.iloc[ev_pos])
            res[(name, "numeric")] = p_num
            for k in text_sets:
                if Z[k] is None:
                    rps = {s2: np.zeros(len(ev_pos)) for s2 in stage2}
                else:
                    Ztr = pd.DataFrame(Z[k][oof_pos], columns=zcols[k])
                    Zev = pd.DataFrame(Z[k][ev_pos], columns=zcols[k])
                    rp_ridge = fit_model(
                        "Ridge", Ztr, pd.Series(resid), ent[oof_pos]
                    ).predict(Zev)
                    rps = {}
                    if "Ridge" in stage2:
                        rps["Ridge"] = rp_ridge
                    if "LightGBM" in stage2:
                        rps["LightGBM"] = (
                            _lgbm(**stage2_lgbm).fit(Ztr, resid).predict(Zev)
                        )
                    if "Gated" in stage2:
                        rps["Gated"] = gated_correction(
                            Ztr, resid, ent[oof_pos], Zev, rp_ridge
                        )
                for s2, rp in rps.items():
                    res[(name, k, s2)] = p_num + rp
                    res[(name, k, s2, "resid")] = rp
        return res

    train_df = data.iloc[train_pos].reset_index(drop=True)
    oos = []
    for k, (a, b) in enumerate(
        grouped_time_cv(train_df, n_splits=outer_splits)
    ):
        oos.append((train_pos[b], evaluate(train_pos[a], train_pos[b])))
        print(f"  fold {k + 1}/{outer_splits} done")
    pos_all = np.concatenate([p for p, _ in oos])
    assert len(np.unique(pos_all)) == len(pos_all), "a row was validated twice"
    order = np.argsort(pos_all, kind="stable")
    pos_all = pos_all[order]
    pool = {
        k: np.concatenate([r[k] for _, r in oos])[order] for k in oos[0][1]
    }
    y_true, ents = y[pos_all], ent[pos_all]
    h_steps = max(1, int(np.median(steps_ahead(data.iloc[pos_all]))))
    print(
        f"pooled out-of-sample rows: {len(y_true)} | forecast horizon = about "
        f"{h_steps} observation(s) ahead"
    )

    def mae(pred):
        return float(
            pd.Series(np.abs(y_true - pred)).groupby(ents).mean().mean()
        )

    val = evaluate(train_pos, val_pos)
    y_val, ent_val = y[val_pos], ent[val_pos]

    def mae_val(pred):
        return float(
            pd.Series(np.abs(y_val - pred)).groupby(ent_val).mean().mean()
        )

    rows = []
    for name in numeric_models:
        e_num = y_true - pool[(name, "numeric")]
        for k in text_sets:
            for s2 in stage2:
                e_two = y_true - pool[(name, k, s2)]
                t = dm_test(np.abs(e_num), np.abs(e_two), ents, h_steps)
                rp = pool[(name, k, s2, "resid")]
                rows.append(
                    {
                        "Dataset": ds,
                        "Target": target,
                        "Numeric model": name,
                        "Text": k,
                        "Stage 2": s2,
                        "Numeric MAE": mae(pool[(name, "numeric")]),
                        "Two-stage MAE": mae(pool[(name, k, s2)]),
                        "Gain %": 100
                        * (
                            1
                            - mae(pool[(name, k, s2)])
                            / mae(pool[(name, "numeric")])
                        ),
                        "DM p-value": t["p_value"],
                        "entities_better": t["entities_better"],
                        "residual_R2": 1
                        - np.sum((e_num - rp) ** 2) / np.sum(e_num**2),
                        "VAL Numeric MAE": mae_val(val[(name, "numeric")]),
                        "VAL Two-stage MAE": mae_val(val[(name, k, s2)]),
                    }
                )
    return rows


rows = []
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
    print(
        f"\n==================== {ds} / {target} @ {H}h ===================="
    )
    rows += run_one(ds, target, H)

if rows:
    pd.set_option("display.width", 250)
    tbl = pd.DataFrame(rows)
    ok = tbl["DM p-value"].notna()
    tbl["FDR p-value"] = np.nan
    tbl.loc[ok, "FDR p-value"] = multipletests(
        tbl.loc[ok, "DM p-value"], method="fdr_bh"
    )[1]
    tbl["VAL Gain %"] = 100 * (
        1 - tbl["VAL Two-stage MAE"] / tbl["VAL Numeric MAE"]
    )
    tbl.to_csv(out / "residual_summary.csv", index=False)
    print(
        "\n==================== TWO-STAGE vs NUMERIC ONLY ===================="
    )
    print(
        f"FDR family = {int(ok.sum())} tests of this run. Gain % > 0: the "
        f"two-stage model has the lower error. "
        "residual_R2 > 0: the text explains part of the numeric model's error."
    )
    show = ["Target","Numeric model","Text","Stage 2","Numeric MAE","Two-stage MAE","Gain %","DM p-value","FDR p-value","entities_better","residual_R2","VAL Gain %",]
    print(tbl[show].round(4).to_string(index=False))
    print(f"\nsaved {out / 'residual_summary.csv'}")
    print(
        "READ THIS: exploratory run. Count it in the multiple-testing story "
        "of the report, read p together with "
        "the gain, and look at 'VAL Gain %': a gain in CV that is not there "
        "on validation is not reliable. "
        "Stage-1 LightGBM is untuned, so do not compare these MAEs with "
        "text_features.py."
    )
