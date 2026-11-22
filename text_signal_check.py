from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from tf_helpers import H as T
from ts_common import grouped_time_cv, make_features

build_windows, load_embeddings, recavg = T["build_windows"], T["load_embeddings"], T["recavg"]
pca_block, tfidf_block, window_hours_dict = T["pca_block"], T["tfidf_block"], T["WINDOW_HOURS"]
window_hours_dict.setdefault("FNSPID_abs", 7 * 24)

clean = Path("clean_data")
res = Path("results")

runs = [
    ("RepoHealth", "new_issues", 744), 
    ("RepoHealth", "merged_prs", 744), 
    ("RepoHealth", "issue_comment_volume", 744),
    ("ClusterTrace", "machine_cpu", 12), 
    ("ClusterTrace", "n_job", 12), 
    ("EPA-Air", "pm2_5", 168),
    ("GDELT", "AvgTone", 336), 
    ("StudentLife", "sleep_duration", 744)
]

n_splits = 6
n_perm = 100
rng = np.random.default_rng(0)
pd.set_option("display.width", 220)


def make_classifier():
    return make_pipeline(
        StandardScaler(), 
        FunctionTransformer(np.clip, kw_args={"a_min": -5, "a_max": 5}),
        LogisticRegression(C=0.1, max_iter=500)
    )


def pooled_auc(folds, x_by_fold, y):
    score, truth = [], []
    for (fit, ev), x in zip(folds, x_by_fold):
        if x is None or len(np.unique(y[fit])) < 2:
            return np.nan
        m = make_classifier().fit(x[fit], y[fit])
        score.append(m.predict_proba(x[ev])[:, 1])
        truth.append(y[ev])
    truth = np.concatenate(truth)
    return roc_auc_score(truth, np.concatenate(score)) if len(np.unique(truth)) == 2 else np.nan


def run_one(ds, target, h):
    data, cols = make_features(pd.read_csv(clean / f"{ds}_{target}_numeric.csv"), horizon_hours=h)
    texts = pd.read_csv(clean / f"{ds}_{target}_text.csv")
    texts["date_time"] = pd.to_datetime(texts["date_time"], utc=True)
    texts = texts.dropna(subset=["text_clean"]).sort_values(["entity_id", "date_time"]).reset_index(drop=True)
    
    m_alpha, _ = build_windows(data, texts, window_hours_dict[ds])
    m = m_alpha.copy()
    m.data[:] = 1.0
    r_rec = recavg(m_alpha, load_embeddings(ds, texts))
    t_texts = texts["text_clean"].astype(str).to_numpy()

    keep = (data["target"].to_numpy() != data["lag_1"].to_numpy())
    y = (data["target"].to_numpy() > data["lag_1"].to_numpy()).astype(int)
    x_num = data[cols].to_numpy(float)
    
    train_pos = np.where((data["split"].to_numpy() == "train") & keep)[0]
    train_df = data.iloc[train_pos].reset_index(drop=True)
    
    folds = []
    for a, b in grouped_time_cv(train_df, n_splits=n_splits):
        folds.append((train_pos[a], train_pos[b]))
    if not folds:
        return None
        
    z_dict = {"RecAvg": [], "TF-IDF": []}
    for fit, _ in folds:
        z_dict["RecAvg"].append(pca_block(r_rec, fit))
        z_dict["TF-IDF"].append(tfidf_block(fit, m, t_texts))
        
    ent = data["entity_id"].to_numpy()
    groups = [np.where(ent == e)[0] for e in pd.unique(ent)]

    def shifted(zs):
        perm = np.arange(len(data))
        for idx in groups:
            perm[idx] = np.roll(idx, int(rng.integers(1, max(2, len(idx)))))
        return [None if z is None else z[perm] for z in zs]

    auc_num = pooled_auc(folds, [x_num] * len(folds), y)
    
    row = {
        "dataset": ds, 
        "target": target, 
        "n rows": int(len(train_pos)), 
        "share up": float(y[train_pos].mean()),
        "AUC numeric": auc_num
    }
    
    for name in ("RecAvg", "TF-IDF"):
        zs = z_dict[name]
        if any(z is None for z in zs):
            continue
            
        both = lambda z_list: [np.hstack([x_num, z]) for z in z_list]
        auc_txt = pooled_auc(folds, zs, y)
        auc_both = pooled_auc(folds, both(zs), y)
        
        null_txt, null_delta = [], []
        for _ in range(n_perm):
            zs_perm = shifted(zs)
            null_txt.append(pooled_auc(folds, zs_perm, y))
            null_delta.append(pooled_auc(folds, both(zs_perm), y) - auc_num)
            
        p = lambda obs, null: (1 + np.sum(np.array(null) >= obs)) / (1 + len(null))
        
        row.update({
            f"AUC text only ({name})": auc_txt, 
            f"p ({name} text only)": p(auc_txt, null_txt),
            f"AUC numeric+text ({name})": auc_both, 
            f"AUC change ({name})": auc_both - auc_num,
            f"p ({name} AUC change)": p(auc_both - auc_num, null_delta)
        })
    return row


rows = []
for ds, target, h in runs:
    if not ((clean / f"{ds}_{target}_numeric.csv").exists() and (clean / f"{ds}_{target}_text.csv").exists()):
        print(f"NOT RUN {ds}/{target}: clean files missing")
        continue
    print(f"\n==================== {ds} / {target} ====================")
    res_row = run_one(ds, target, h)
    if res_row:
        rows.append(res_row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res_row.items()})
        
if rows:
    tbl = pd.DataFrame(rows)
    tbl.to_csv(res / "text_signal_check.csv", index=False)
    print("\n==================== SUMMARY ====================")
    print(tbl.round(4).to_string(index=False))
    print(f"\nsaved {res / 'text_signal_check.csv'}. AUC 0.5 = coin flip. Look at the AUC change: signal that the "
          f"numbers do not already have. With {len(tbl)} runs x 2 text types x 2 tests, a few p < 0.05 are expected by "
          "chance.")