import hashlib
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from statsmodels.stats.multitest import multipletests

from ts_common import PREDICT_DELTA, fit_model, grouped_time_cv, make_features, metrics, models

clean = Path("clean_data")
out = Path("results")
out.mkdir(exist_ok=True)

RUNS = [("RepoHealth", 31 * 24), ("GDELT", 14 * 24)]
OPTIONAL = ["ClusterTrace", "StudentLife"]
WINDOW_HOURS = {"RepoHealth": 31 * 24, "EPA-Air": 7 * 24, "GDELT": 14 * 24, "ClusterTrace": 12, "StudentLife": 31 * 24, "FNSPID_abs": 7 * 24}
MAX_TEXTS_PER_WINDOW = 50
N_COMP = 20
MAX_VOCAB = 3000
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_COLUMN = "text_raw"
MODEL_NAMES = ["Ridge", "RandomForest", "LightGBM", "VotingEnsemble"]
SETS = ["numeric only", "+ TF-IDF", "+ RecAvg", "+ LatestText", "+ LLM"]
LLM_CODES = {"RepoHealth": list(range(0, 6)), "GDELT": list(range(-2, 3))}
LOSS = "abs"
EVALUATE_TEST = True


def to_ns(times):
    idx = pd.DatetimeIndex(times)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.values.astype("datetime64[ns]")


def build_windows(data, texts, hours):
    t_target, t_cut = to_ns(data["date_time"]), to_ns(data["t_cut"])
    t_txt = to_ns(texts["date_time"])
    e_rows, e_txt = data["entity_id"].to_numpy(), texts["entity_id"].to_numpy()
    window = np.timedelta64(int(round(hours * 3600)), "s")
    sigma = float(hours)
    indptr, indices, values, running = [np.zeros(1, dtype=np.int64)], [], [], 0
    latest = np.full(len(data), -1, dtype=np.int64)

    for e in pd.unique(e_rows):
        r = np.where(e_rows == e)[0]
        assert (np.diff(r) == 1).all(), "rows must be sorted by entity"
        t_idx = np.where(e_txt == e)[0]
        counts = np.zeros(len(r), dtype=np.int64)
        if len(t_idx) > 0:
            tt, tc, tr = t_txt[t_idx], t_cut[r], t_target[r]
            hi = np.searchsorted(tt, tc, side="right")
            lo = np.searchsorted(tt, tc - window, side="right")
            lo = np.maximum(lo, hi - MAX_TEXTS_PER_WINDOW)
            counts = hi - lo
            latest[r] = np.where(counts > 0, t_idx[np.maximum(hi - 1, 0)], -1)
            offset = np.cumsum(counts) - counts
            local = np.arange(counts.sum()) + np.repeat(lo - offset, counts)
            dt_hours = (np.repeat(tr, counts) - tt[local]) / np.timedelta64(1, "h")
            indices.append(t_idx[local])
            values.append(np.exp(-(dt_hours / sigma) ** 2))
        indptr.append(running + np.cumsum(counts))
        running += int(counts.sum())

    ind = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
    val = np.concatenate(values) if values else np.zeros(0)
    return sparse.csr_matrix((val, ind, np.concatenate(indptr)), shape=(len(data), len(texts))), latest


def load_embeddings(ds, texts):
    docs = texts[EMBED_COLUMN].astype(str).tolist()
    tag = hashlib.md5("\n".join(docs).encode("utf-8")).hexdigest()[:8]
    path = clean / f"{ds}_embeddings_{EMBED_COLUMN}_{tag}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    print(f"  computing {len(docs)} embeddings with {EMBED_MODEL} (one time, then cached)...")
    E = SentenceTransformer(EMBED_MODEL).encode(docs, batch_size=64, show_progress_bar=True)
    E = np.asarray(E, dtype="float32")
    np.save(path, E)
    return E


def recavg(Malpha, E):
    total = Malpha @ E
    weight = np.asarray(Malpha.sum(axis=1)).ravel()
    return total / np.maximum(weight, 1e-12)[:, None]


def pca_block(R, fit_pos):
    k = min(N_COMP, R.shape[1], len(fit_pos) - 1)
    if k < 2:
        return None
    return PCA(n_components=k, random_state=0).fit(R[fit_pos]).transform(R)

def tfidf_block(fit_pos, M, T):
    used = np.unique(M[fit_pos].indices)
    if len(used) < 10:
        return None
    vec = TfidfVectorizer(max_features=MAX_VOCAB, min_df=3)
    try:
        vec.fit(T[used])
    except ValueError:
        return None
    counts = np.asarray(M.sum(axis=1)).ravel()
    R = sparse.diags(1 / np.maximum(counts, 1)) @ (M @ vec.transform(T))
    k = min(N_COMP, R.shape[1] - 1, len(fit_pos) - 1)
    if k < 2:
        return None
    return TruncatedSVD(n_components=k, random_state=0).fit(R[fit_pos]).transform(R)

def llm_block(data, texts_all, llm, ds, hours):
    assert len(llm) == len(texts_all) and (llm["entity_id"].astype(str).to_numpy() == texts_all["entity_id"].astype(str).to_numpy()).all() \
        and (llm["date_time"].astype(str).to_numpy() == texts_all["date_time"].astype(str).to_numpy()).all(), \
        "llm_features.csv does not match <dataset>_text.csv (different rows or order): run llm_extract.py again"
    keep = texts_all["text_clean"].notna().to_numpy()
    right = pd.DataFrame({"entity_id": llm["entity_id"].astype(str).to_numpy()[keep],"date_time": to_ns(pd.to_datetime(llm["date_time"], utc=True))[keep],
                        "llm_code": llm["llm_code"].to_numpy(float)[keep]}).sort_values("date_time", kind="stable")
    t_cut = to_ns(data["t_cut"])
    assert not pd.isna(t_cut).any(), "a row has no forecast origin time"
    left = pd.DataFrame({"pos": np.arange(len(data)), "entity_id": data["entity_id"].astype(str).to_numpy(),
                        "t_cut": t_cut}).sort_values("t_cut", kind="stable")
    merged = pd.merge_asof(left, right, left_on="t_cut", right_on="date_time", by="entity_id", direction="backward",
                        tolerance=pd.Timedelta(hours=hours)).sort_values("pos")
    assert (merged["pos"].to_numpy() == np.arange(len(data))).all()
    code = merged["llm_code"].to_numpy()
    out = pd.DataFrame(index=data.index)
    for c in LLM_CODES[ds][1:]:
        out[f"llm_{'m' if c < 0 else ''}{abs(c)}"] = (code == c).astype(float)
    out["llm_has_text"] = (~np.isnan(code)).astype(float)
    return out




def show(tbl, label):
    base = float(tbl.loc[tbl["set"] == "baseline", "MAE"].iloc[0])
    p = tbl[tbl["set"] != "baseline"].pivot(index="model", columns="set", values="MAE")[SETS]
    print(f"\n{label}   (baseline train mean MAE = {base:.4f})")
    print(p.round(4).to_string())
    print("gain in MAE compared with 'numeric only' (%, positive = text helped):")
    print(((1 - p.div(p["numeric only"], axis=0)) * 100).round(2).to_string())


def steps_ahead(data):
    out = np.ones(len(data))
    t_all, c_all = to_ns(data["date_time"]), to_ns(data["t_cut"])
    for idx in data.groupby("entity_id").indices.values():
        t, c = t_all[idx], c_all[idx]
        out[idx] = np.searchsorted(t, t, side="right") - np.searchsorted(t, c, side="right")
    return out


def dm_test(loss_a, loss_b, entity_ids, h_steps):
    la, lb = np.asarray(loss_a, float), np.asarray(loss_b, float)
    d, n = la - lb, len(la)
    res = dict(n=n, rel_gain_pct=100 * (la.mean() - lb.mean()) / la.mean() if la.mean() > 0 else np.nan,
               dm_stat=np.nan, p_value=np.nan, entities_better="")
    ents = np.asarray(entity_ids)
    _, first = np.unique(ents, return_index=True)
    bounds = list(np.sort(first)) + [n]
    blocks = [d[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    better = sum(b.mean() > 0 for b in blocks)
    res["entities_better"] = f"{better}/{len(blocks)}"
    if n < 30 or not np.isfinite(d).all():
        return res
    dc = [b - d.mean() for b in blocks]
    lags = min(max(2 * h_steps, int(np.ceil(n ** (1 / 3)))), n // 4)

    def gamma(k):
        return sum(float(np.dot(b[k:], b[:len(b) - k])) for b in dc if len(b) > k) / n

    var = gamma(0) + 2 * sum((1 - k / (lags + 1)) * gamma(k) for k in range(1, lags + 1))
    if var <= 0:
        return res
    stat = d.mean() / np.sqrt(var / n)
    h = max(int(h_steps), 1)
    factor = (n + 1 - 2 * h + h * (h - 1) / n) / n 
    if factor > 0:
        stat *= np.sqrt(factor)
    res["dm_stat"], res["p_value"] = stat, 2 * student_t.sf(abs(stat), df=n - 1)
    return res


def dm_tests(pool, y_true, ents, h_steps, run):
    err = {k: y_true - p for k, p in pool.items()}
    loss = {k: (np.abs(e) if LOSS == "abs" else e ** 2) for k, e in err.items()}
    rows = []

    def add(hyp, label, key_a, key_b):
        rows.append(dict(run=run, hypothesis=hyp, comparison=label, **dm_test(loss[key_a], loss[key_b], ents, h_steps)))

    for model in MODEL_NAMES:
        for s_ in SETS[1:]:
            add("H1", f"{model}: numeric only  vs  {s_}", ("numeric only", model), (s_, model))
        for a, b in combinations(SETS[1:], 2):
            add("H2", f"{model}: {a}  vs  {b}", (a, model), (b, model))
    if "VotingEnsemble" in MODEL_NAMES:
        for s_ in SETS:
            for m in ("RandomForest", "LightGBM"):
                if m in MODEL_NAMES:
                    add("E1", f"{s_}: {m}  vs  VotingEnsemble", (s_, m), (s_, "VotingEnsemble"))
    return rows




pd.set_option("display.width", 220)

passed = None
gate_path = out / "gate.csv"
gate = pd.read_csv(gate_path).set_index("dataset")
passed = {d for d in gate.index if bool(gate.loc[d, "passed"])}

runs = []
for ds, H in RUNS:
    ok_gate = ds not in OPTIONAL or (passed is not None and ds in passed)
    ok_files = (clean / f"{ds}_numeric.csv").exists() and (clean / f"{ds}_text.csv").exists()
    if ok_gate and ok_files:
        runs.append((ds, H))
    else:
        print(f"skipping {ds} (gate failed or files missing)")
print("runs:", [f"{d} @ {'next obs' if h is None else str(h) + 'h'}" for d, h in runs])

all_tests, validation, emb_cache = [], {}, {}
for ds, H in runs:
    run = f"{ds}_next" if H is None else f"{ds}_h{H}"
    print(f"\n==================== {run} ====================")
    data, cols = make_features(pd.read_csv(clean / f"{ds}_numeric.csv"), horizon_hours=H)
    texts = pd.read_csv(clean / f"{ds}_text.csv")
    texts_all = texts[["entity_id", "date_time", "text_clean"]].copy()
    texts["date_time"] = pd.to_datetime(texts["date_time"], utc=True)
    texts = texts.dropna(subset=["text_clean"]).sort_values(["entity_id", "date_time"]).reset_index(drop=True)
    T = texts["text_clean"].astype(str).to_numpy()

    Malpha, latest = build_windows(data, texts, WINDOW_HOURS[ds])
    M = Malpha.copy()
    M.data[:] = 1.0
    print(f"rows: {len(data)} | texts: {len(texts)} | window: {WINDOW_HOURS[ds]} h | "
          f"rows with at least one text in the window: {(Malpha.getnnz(axis=1) > 0).mean():.1%}")



    if ds not in emb_cache:
        emb_cache[ds] = load_embeddings(ds, texts)
    R_rec = recavg(Malpha, emb_cache[ds])
    R_latest = np.zeros((len(data), emb_cache[ds].shape[1]), dtype="float32")
    has_text = latest >= 0
    R_latest[has_text] = emb_cache[ds][latest[has_text]]
    X_numeric = data[cols]
    llm_X = None
    if "+ LLM" in SETS:
        llm_path = clean / f"{ds}_llm_features.csv"
        if not llm_path.exists():
            raise SystemExit(f"{llm_path} not found: run llm_extract.py first")
        llm_X = llm_block(data, texts_all, pd.read_csv(llm_path), ds, WINDOW_HOURS[ds])
        print(f"rows with an LLM code in the window: {llm_X['llm_has_text'].mean():.1%} | "
              f"share of rows per code: " + ", ".join(f"{c}: {v:.1%}" for c, v in llm_X.drop(columns='llm_has_text').mean().items()))

    split = data["split"].to_numpy()
    train_pos, val_pos, test_pos = (np.where(split == s)[0] for s in ("train", "val", "test"))
    skipped = set()

    def build_X(set_name, fit_pos):
        if set_name == "numeric only":
            return X_numeric
        if set_name == "+ LLM":
            return pd.concat([X_numeric, llm_X], axis=1)
        if set_name == "+ RecAvg":
            Z, prefix = pca_block(R_rec, fit_pos), "rec"
        elif set_name == "+ LatestText":
            Z, prefix = pca_block(R_latest, fit_pos), "lat"
        else:
            Z, prefix = tfidf_block(fit_pos, M, T), "svd"
        if Z is None:
            skipped.add(set_name)
            return X_numeric
        return pd.concat([X_numeric, pd.DataFrame(Z, index=data.index, columns=[f"{prefix}_{i}" for i in range(Z.shape[1])])], axis=1)

    def evaluate(fit_pos, ev_pos):
        fit, ev = data.iloc[fit_pos], data.iloc[ev_pos]
        base = ev["entity_id"].map(fit.groupby("entity_id")["target"].mean()).values
        rows = [dict(set="baseline", model="train mean", **metrics(ev, base, base)[0]),
                dict(set="baseline", model="last value", **metrics(ev, ev["lag_1"].values, base)[0])]
        per_entity = []
        preds = {("baseline", "train mean"): base, ("baseline", "last value"): ev["lag_1"].values}
        for s in SETS:
            X = build_X(s, fit_pos)
            cache = {}
            for name in MODEL_NAMES:
                model = fit_model(name, X.iloc[fit_pos], fit["target"], fit["entity_id"], cache=cache,
                                delta=PREDICT_DELTA)
                pred = model.predict(X.iloc[ev_pos])
                preds[(s, name)] = pred
                summary, by_e = metrics(ev, pred, base)
                rows.append(dict(set=s, model=name, **summary))
                per_entity.append(by_e.rename("MAE").reset_index().assign(set=s, model=name))
        preds = {k: v for k, v in preds.items() if k[0] != "baseline"}
        return pd.DataFrame(rows), pd.concat(per_entity), preds

    train_df = data.iloc[train_pos].reset_index(drop=True)
    folds, oos = [], []
    for k, (a, b) in enumerate(grouped_time_cv(train_df)):
        tbl, _, preds = evaluate(train_pos[a], train_pos[b])
        folds.append(tbl.assign(fold=k))
        oos.append((train_pos[b], preds))
        print(f"  fold {k + 1} done")
    cv_long = pd.concat(folds)
    cv_long.to_csv(out / f"{run}_text_cv.csv", index=False)
    cv_mean = cv_long.groupby(["set", "model"], as_index=False)[["MAE", "RMSE", "R2"]].mean()
    show(cv_mean[cv_mean["model"] != "last value"], f"CROSS VALIDATION ({cv_long['fold'].nunique()} walk-forward folds, mean)")
    lv = cv_mean[cv_mean["model"] == "last value"].iloc[0]
    print(f"baseline 'last value' (the value at the origin): MAE {lv['MAE']:.4f}, R2 {lv['R2']:.3f}")

    pos_all = np.concatenate([p for p, _ in oos])
    assert len(np.unique(pos_all)) == len(pos_all), "a row was validated twice"
    order = np.argsort(pos_all, kind="stable")
    pos_all = pos_all[order]
    y_true, ents = data["target"].to_numpy()[pos_all], data["entity_id"].to_numpy()[pos_all]
    pool = {k: np.concatenate([pr[k] for _, pr in oos])[order] for k in oos[0][1]}
    pd.DataFrame({"row": pos_all, "entity_id": ents, "date_time": data["date_time"].to_numpy()[pos_all],
                  "y_true": y_true, **{f"{s_} | {m}": v for (s_, m), v in pool.items()}}
                 ).to_csv(out / f"{run}_cv_predictions.csv", index=False)
    h_steps = max(1, int(np.median(steps_ahead(data.iloc[pos_all]))))
    print(f"pooled out-of-sample rows: {len(y_true)} | forecast horizon = about {h_steps} observation(s) ahead")
    all_tests += dm_tests(pool, y_true, ents, h_steps, run)

    val_tbl, val_ent, _ = evaluate(train_pos, val_pos)
    val_tbl.to_csv(out / f"{run}_text_val.csv", index=False)
    val_ent.to_csv(out / f"{run}_text_entity_val.csv", index=False)
    show(val_tbl[val_tbl["model"] != "last value"], "VALIDATION (trained on train)")
    validation[run] = val_tbl

    for s in SETS:
        X = build_X(s, train_pos)
        m = models()["LightGBM"].fit(X.iloc[train_pos], data["target"].iloc[train_pos])
        gain = pd.Series(m.booster_.feature_importance("gain"), index=m.booster_.feature_name())
        gain = (gain / gain.sum()).sort_values(ascending=False)
        text_share = gain[[c for c in gain.index if c.startswith(("rec_", "svd_", "lat_", "llm_"))]].sum()
        print(f"\nLightGBM importance, set '{s}': share of total gain coming from text columns = {text_share:.1%}")
        print(gain.head(8).round(4).to_string())
        gain.to_csv(out / f"{run}_lgbm_importance_{s.replace('+ ', 'plus_').replace(' ', '_')}.csv")

    if EVALUATE_TEST:
        test_tbl, test_ent, _ = evaluate(np.concatenate([train_pos, val_pos]), test_pos)
        test_tbl.to_csv(out / f"{run}_text_test.csv", index=False)
        test_ent.to_csv(out / f"{run}_text_entity_test.csv", index=False)
        show(test_tbl[test_tbl["model"] != "last value"], "TEST (final result, trained on train + val)")

    if skipped:
        print(f"\nNOTE: too little text to build {sorted(skipped)} in some fits (numeric features only were used there).")


tests = pd.DataFrame(all_tests)
for hyp in tests["hypothesis"].unique():
    mask = (tests["hypothesis"] == hyp) & tests["p_value"].notna()
    tests.loc[mask, "p_fdr"] = multipletests(tests.loc[mask, "p_value"], method="fdr_bh")[1]
tests.to_csv(out / "hypothesis_tests_dm.csv", index=False)
print("\n==================== HYPOTHESIS TESTS (Diebold-Mariano on pooled out-of-sample errors) ====================")
print(f"loss = {LOSS}. rel_gain_pct > 0 means the SECOND set / model has the lower error (% of the first one's loss).")
print("p_value = DM p-value (two-sided).  p_fdr = Benjamini-Hochberg adjusted p-value inside each family (H1, H2, E1).")
print("entities_better = in how many entities the second set / model has the lower mean loss.")
cols_out = ["run", "hypothesis", "comparison", "n", "rel_gain_pct", "dm_stat", "p_value", "p_fdr", "entities_better"]
print(tests[cols_out].round(4).to_string(index=False))
for hyp, grp in tests.groupby("hypothesis"):
    sig = grp[grp["p_fdr"] < 0.05]
    print(f"  {hyp}: {len(grp)} tests, {len(sig)} with p_fdr < 0.05 "
          f"({int((sig['rel_gain_pct'] > 0).sum())} where the second set is better, "
          f"{int((sig['rel_gain_pct'] < 0).sum())} where it is worse)")
prim = tests[(tests["hypothesis"] == "H1") & tests["comparison"].str.endswith("numeric only  vs  + LLM")].copy()
if len(prim):
    ok = prim["p_value"].notna()
    prim.loc[ok, "p_fdr_llm_only"] = multipletests(prim.loc[ok, "p_value"], method="fdr_bh")[1]
    prim.to_csv(out / "llm_primary_tests.csv", index=False)
    print(f"\n-------- PRIMARY QUESTION: '+ LLM' vs 'numeric only' ({len(prim)} tests = models x runs, FDR over these only) --------")
    print(prim[["run", "comparison", "n", "rel_gain_pct", "p_value", "p_fdr_llm_only", "entities_better"]].round(4).to_string(index=False))
print("READ THIS: with thousands of pooled rows even a tiny difference can become 'significant'. Always read p_fdr "
      "together with rel_gain_pct (a gain of 0.3% is statistically visible but practically negligible) and with "
      "entities_better. The test treats entities as independent; entities that share events or weather make "
      "the p-values too small.")

if validation:
    fig, axes = plt.subplots(1, len(validation), figsize=(5.5 * len(validation), 4.8), squeeze=False)
    for ax, (run, tbl) in zip(axes.ravel(), validation.items()):
        p = tbl[~tbl["set"].eq("baseline")].pivot(index="model", columns="set", values="MAE")[SETS]
        p.plot.bar(ax=ax, rot=0)
        ax.axhline(float(tbl.loc[tbl["model"] == "train mean", "MAE"].iloc[0]), color="gray", ls="--",
                   label="baseline: train mean")
        ax.set_title(run, fontsize=9)
        ax.set_ylabel("validation MAE (lower is better)")
        ax.legend(fontsize=6)
    fig.suptitle("Numeric only vs the text sets")
    fig.tight_layout()
    fig.savefig(out / "fig_killer_plot.png", dpi=150)
    plt.close(fig)

print(f"\nSaved to {out.resolve()}")
print("Test data was NOT used." if not EVALUATE_TEST else "Test data was used (final result).")