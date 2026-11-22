import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests

clean = Path("clean_data")
res = Path("results")
res.mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

runs = {
    "ClusterTrace": ("ClusterTrace_h12", 38),
    "RepoHealth": ("RepoHealth_h744", 31),
    "EPA-Air": ("EPA-Air_h168", 21),
    "GDELT": ("GDELT_h336", 115),
    "StudentLife": ("StudentLife_h744", 31),
}
irregularity_type = {
    "RepoHealth": "trigger",
    "GDELT": "trigger",
    "EPA-Air": "artifact",
}
text_sets = ["+ TF-IDF", "+ RecAvg", "+ LatestText"]
n_bins = 200
max_lag = 3
max_gaps = 5000
beyond = 10

rows = []
pd.set_option("display.width", 220)


def add(hyp, scope, detail, p, stat=np.nan):
    rows.append(
        {
            "hypothesis": hyp,
            "scope": scope,
            "detail": detail,
            "statistic": stat,
            "p_value": p,
        }
    )


def fisher(ps):
    ps = [p for p in ps if np.isfinite(p)]
    return stats.combine_pvalues(ps, method="fisher")[1] if ps else np.nan


def h3():
    print(
        "\n==================== H3: does the gain from text differ between "
        "datasets / types? ===================="
    )
    for s in text_sets:
        by_ds = {}
        for ds, (run, _) in runs.items():
            f = res / f"{run}_text_cv.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d["set"] != "baseline"]
            p = d.pivot_table(
                index=["fold", "model"], columns="set", values="MAE"
            )
            if s in p.columns:
                by_ds[ds] = (
                    (100 * (1 - p[s] / p["numeric only"]))
                    .groupby("fold")
                    .mean()
                    .to_numpy()
                )
        usable = {k: v for k, v in by_ds.items() if len(v) >= 2}
        if len(usable) < 2:
            print(f"{s}: fewer than 2 datasets with results, skipped")
            continue
        h, p = stats.kruskal(*usable.values())
        med = {k: round(float(np.median(v)), 2) for k, v in usable.items()}
        print(
            f"{s} across datasets: H = {h:.2f}, p = {p:.4f} | median fold "
            f"gain % per dataset: {med}"
        )
        add("H3", f"{s} by dataset", f"median fold gain %: {med}", p, h)
        if p < 0.05:
            names = list(usable)
            pairs = [
                (a, b) for i, a in enumerate(names) for b in names[i + 1 :]
            ]
            pp = [
                stats.mannwhitneyu(usable[a], usable[b]).pvalue
                for a, b in pairs
            ]
            adj = multipletests(pp, method="holm")[1]
            for (a, b), q in zip(pairs, adj):
                print(f"    {a} vs {b}: Holm-adjusted p = {q:.4f}")
        types = {}
        for ds, v in usable.items():
            if ds in irregularity_type:
                types.setdefault(irregularity_type[ds], []).append(v)
        types = {k: np.concatenate(v) for k, v in types.items()}
        if len(types) >= 2:
            h2, p2 = stats.kruskal(*types.values())
            print(
                f"{s} across irregularity types {sorted(types)}: H = "
                f"{h2:.2f}, p = {p2:.4f} "
                f"(only datasets with a known type)"
            )
            add("H3", f"{s} by type", f"types: {sorted(types)}", p2, h2)


def gaps_hours(df):
    t = pd.to_datetime(df["date_time"], utc=True)
    g = t.groupby(df["entity_id"]).diff().dt.total_seconds() / 3600
    g = g[g > 0].dropna()
    return g.to_numpy()


def h4():
    print(
        "\n==================== H4: do inter-arrival times differ between "
        "datasets? ===================="
    )
    rng = np.random.default_rng(0)
    for kind, suffix in (
        ("numeric observations", "numeric"),
        ("texts", "text"),
    ):
        samples, table = {}, []
        for ds in runs:
            f = clean / f"{ds}_{suffix}.csv"
            if not f.exists():
                continue
            g = gaps_hours(pd.read_csv(f, usecols=["entity_id", "date_time"]))
            if len(g) < 10:
                continue
            table.append(
                {
                    "dataset": ds,
                    "n gaps": len(g),
                    "median h": np.median(g),
                    "mean h": g.mean(),
                    "CV": g.std() / g.mean(),
                    "p90 h": np.percentile(g, 90),
                }
            )
            samples[ds] = rng.choice(g, min(len(g), max_gaps), replace=False)
        if len(samples) < 2:
            print(f"{kind}: fewer than 2 datasets, skipped")
            continue
        try:
            h, p = stats.kruskal(*samples.values())
        except ValueError:
            print(f"\n{kind}: all gaps are identical, Kruskal-Wallis skipped")
            continue
        n = sum(len(v) for v in samples.values())
        eps2 = (h - len(samples) + 1) / (n - len(samples))
        print(f"\nInter-arrival times of the {kind} (hours):")
        print(pd.DataFrame(table).round(3).to_string(index=False))
        print(
            f"Kruskal-Wallis: H = {h:.1f}, p = {p:.3g}, epsilon^2 = "
            f"{eps2:.3f}  (CV = 1: Poisson-like, > 1: bursty)"
        )
        add("H4", f"gaps between {kind}", f"epsilon^2 = {eps2:.3f}", p, h)


def granger_p(vol, vcount):
    d = pd.DataFrame(
        {"vol": pd.Series(vol).diff(), "cnt": pd.Series(vcount).diff()}
    ).dropna()
    if len(d) < 30 or d["cnt"].std() == 0 or d["vol"].std() == 0:
        return np.nan
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = grangercausalitytests(d[["vol", "cnt"]], maxlag=max_lag)
    except Exception:
        return np.nan
    return min(1.0, min(out[k][0]["ssr_ftest"][1] for k in out) * max_lag)


def h5():
    print(
        "\n==================== H5: does more text come before higher target "
        "volatility? ===================="
    )
    out = []
    for ds in runs:
        fn, ft = clean / f"{ds}_numeric.csv", clean / f"{ds}_text.csv"
        if not (fn.exists() and ft.exists()):
            continue
        num = pd.read_csv(fn, usecols=["entity_id", "date_time", "target"])
        txt = pd.read_csv(ft, usecols=["entity_id", "date_time"])
        num["t"] = pd.to_datetime(num["date_time"], utc=True).astype("int64")
        txt["t"] = pd.to_datetime(txt["date_time"], utc=True).astype("int64")
        rhos, p_sp, p_gr = [], [], []
        for e, g in num.groupby("entity_id"):
            tt = txt.loc[txt["entity_id"] == e, "t"].to_numpy()
            t0, t1 = g["t"].min(), g["t"].max()
            if t1 <= t0 or len(g) < 60 or len(tt) < 20:
                continue
            width = (t1 - t0) / n_bins
            b = np.minimum(
                ((g["t"].to_numpy() - t0) / width).astype(int), n_bins - 1
            )
            ybar = (
                pd.Series(g["target"].to_numpy())
                .groupby(b)
                .mean()
                .reindex(range(n_bins))
                .ffill()
            )
            tt = tt[(tt >= t0) & (tt <= t1)]
            cnt = np.bincount(
                np.minimum(((tt - t0) / width).astype(int), n_bins - 1),
                minlength=n_bins,
            )
            vol = ybar.diff().abs().to_numpy()
            x, y = cnt[:-1], vol[1:]
            ok = np.isfinite(y)
            if ok.sum() < 30 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                continue
            r, p = stats.spearmanr(x[ok], y[ok])
            rhos.append(r)
            p_sp.append(p)
            p_gr.append(granger_p(vol, cnt))
        if not rhos:
            print(f"{ds}: no entity with enough data, skipped")
            continue
        out.append(
            {
                "dataset": ds,
                "entities": len(rhos),
                "median rho": np.median(rhos),
                "share rho>0": np.mean(np.array(rhos) > 0),
                "Fisher p (Spearman)": fisher(p_sp),
                "Fisher p (Granger)": fisher(p_gr),
            }
        )
    if not out:
        print("nothing to test")
        return
    tbl = pd.DataFrame(out)
    for col in ("Fisher p (Spearman)", "Fisher p (Granger)"):
        ok = tbl[col].notna()
        tbl.loc[ok, col.replace("Fisher p", "FDR p")] = multipletests(
            tbl.loc[ok, col], method="fdr_bh"
        )[1]
    print(tbl.round(4).to_string(index=False))
    print(
        "rho > 0: more texts in a bin go with higher volatility in the NEXT "
        "bin. Fisher: entities treated as independent."
    )
    for _, r in tbl.iterrows():
        add(
            "H5",
            f"{r['dataset']} Spearman",
            f"median rho {r['median rho']:.3f}, {int(r['entities'])} entities",
            r["Fisher p (Spearman)"],
        )
        add(
            "H5",
            f"{r['dataset']} Granger",
            f"{int(r['entities'])} entities",
            r["Fisher p (Granger)"],
        )


def ljung_box(r, first, m):
    r = np.asarray(r, float) - np.mean(r)
    n, denom = len(r), np.sum(r**2)
    if n < first + m + 20 or denom == 0:
        return np.nan
    q = (
        n
        * (n + 2)
        * sum(
            (np.sum(r[k:] * r[:-k]) / denom) ** 2 / (n - k)
            for k in range(first, first + m)
        )
    )
    return float(stats.chi2.sf(q, m))


def h6():
    print(
        "\n==================== H6: are the out-of-sample residuals "
        "uncorrelated beyond the horizon? ===================="
    )
    out = []
    for ds, (run, h) in runs.items():
        f = res / f"{run}_cv_predictions.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for col in [c for c in d.columns if c.startswith("numeric only | ")]:
            p_in, p_out = [], []
            for _, g in d.groupby("entity_id", sort=False):
                r = g["y_true"].to_numpy() - g[col].to_numpy()
                p_in.append(ljung_box(r, 1, 10))
                p_out.append(ljung_box(r, h + 1, beyond))
            if not any(np.isfinite(p_out)):
                continue
            out.append(
                {
                    "dataset": ds,
                    "model": col.split("| ")[1],
                    "horizon h": h,
                    "entities": int(np.isfinite(p_out).sum()),
                    "share rejecting, lags 1-10 (expected)": np.nanmean(
                        np.array(p_in) < 0.05
                    ),
                    f"share rejecting, lags h+1..h+{beyond}": np.nanmean(
                        np.array(p_out) < 0.05
                    ),
                    "Fisher p (beyond h)": fisher(p_out),
                }
            )
    if not out:
        print("nothing to test (no cv_predictions files found)")
        return
    tbl = pd.DataFrame(out)
    ok = tbl["Fisher p (beyond h)"].notna()
    tbl.loc[ok, "FDR p"] = multipletests(
        tbl.loc[ok, "Fisher p (beyond h)"], method="fdr_bh"
    )[1]
    print(tbl.round(4).to_string(index=False))
    print(
        "Rejecting beyond the horizon means the numeric model leaves "
        "structure in its errors (it could be improved)."
    )
    for _, r in tbl.iterrows():
        add(
            "H6",
            f"{r['dataset']} / {r['model']}",
            f"{int(r['entities'])} entities, h = {int(r['horizon h'])}",
            r["Fisher p (beyond h)"],
        )


h3()
h4()
h5()
h6()

if rows:
    tbl = pd.DataFrame(rows)
    tbl["p_fdr"] = np.nan
    for hyp in tbl["hypothesis"].unique():
        m = (tbl["hypothesis"] == hyp) & tbl["p_value"].notna()
        if m.any():
            tbl.loc[m, "p_fdr"] = multipletests(
                tbl.loc[m, "p_value"], method="fdr_bh"
            )[1]
    tbl.to_csv(res / "extra_hypotheses.csv", index=False)
    print(
        f"\nsaved {res / 'extra_hypotheses.csv'} ({len(tbl)} tests, "
        f"Benjamini-Hochberg inside each hypothesis)"
    )
