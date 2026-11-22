from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ts_common import fit_model,grouped_time_cv,make_features,metrics,models,predict_delta

matplotlib.use("Agg")

clean = Path("clean_data")
out = Path("results")
out.mkdir(exist_ok=True)

main = ["RepoHealth", "EPA-Air", "GDELT"]
optional = ["ClusterTrace", "StudentLife"]
gate_r2 = 0.05
evaluate_test = False


def evaluate(train, ev, cols, ds):
    base = (
        ev["entity_id"].map(train.groupby("entity_id")["target"].mean()).values
    )
    preds = {
        "baseline: train mean": base,
        "baseline: last value": ev["lag_1"].values,
    }
    if ds == "EPA-Air":
        preds["baseline: same time yesterday"] = ev["lag_3"].values
    cache = {}
    for name in models():
        preds[name] = fit_model(
            name,
            train[cols],
            train["target"],
            train["entity_id"],
            cache=cache,
            delta=predict_delta,
        ).predict(ev[cols])
    return pd.DataFrame(
        {name: metrics(ev, p, base)[0] for name, p in preds.items()}
    ).T


def plot_table(table, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [
        "lightgray" if m.startswith("baseline") else "tab:blue"
        for m in table.index
    ]
    for ax, m in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.barh(table.index, table[m], color=colors)
        ax.set_title(
            m + (" (higher is better)" if m == "R2" else " (lower is better)")
        )
        ax.invert_yaxis()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


pd.set_option("display.width", 200)
gate_rows, all_val = [], []

for ds in main + optional:
    path = clean / f"{ds}_numeric.csv"
    if not path.exists():
        print(
            f"\n===== {ds}: {path} not found, skipped (run clean_data.py "
            f"first) ====="
        )
        continue
    print(f"\n===== {ds} =====")
    data, cols = make_features(pd.read_csv(path))
    train = data[data["split"] == "train"].reset_index(drop=True)
    val = data[data["split"] == "val"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    print(
        f"rows: train {len(train)}, val {len(val)}, test {len(test)} | "
        f"entities {data['entity_id'].nunique()} | features {len(cols)}"
    )

    folds = pd.concat([evaluate(train.iloc[a], train.iloc[b], cols, ds) for a, b in grouped_time_cv(train)])
    cv = folds.groupby(level=0).agg(["mean", "std"])[["MAE", "R2"]]
    cv.columns = ["cv_MAE_mean", "cv_MAE_std", "cv_R2_mean", "cv_R2_std"]
    cv = cv.sort_values("cv_MAE_mean")
    print("\nCROSS VALIDATION on the training part (walk-forward folds):")
    print(cv.round(4).to_string())
    cv.to_csv(out / f"{ds}_cv.csv")

    best = cv.loc[~cv.index.str.startswith("baseline"), "cv_R2_mean"].max()
    passed = best >= gate_r2
    print(f"\n>>> {ds}: best CV R2 of the numeric models = {best:.4f}")
    if ds in optional and not passed:
        print(
            f"WARNING: {ds} FAILED the gate (CV R2 < {gate_r2}). Exclude it "
            f"from the final project."
        )
    gate_rows.append(
        dict(
            dataset=ds,
            best_cv_R2=best,
            optional=ds in optional,
            passed=bool(passed or ds in main),
        )
    )

    val_table = evaluate(train, val, cols, ds).sort_values("MAE")
    print("\nVALIDATION (trained on train, scored on val; sorted by MAE):")
    print(val_table.round(4).to_string())
    val_table.to_csv(out / f"{ds}_validation.csv")
    plot_table(
        val_table,
        f"{ds}: validation (numeric only)",
        out / f"fig_{ds}_validation.png",
    )
    all_val.append(
        val_table.assign(dataset=ds)
        .reset_index()
        .rename(columns={"index": "method"})
    )

    ridge = models()["Ridge"].fit(train[cols], train["target"])
    lgbm = models()["LightGBM"].fit(train[cols], train["target"])
    gain = pd.Series(
        lgbm.booster_.feature_importance("gain"),
        index=lgbm.booster_.feature_name(),
    )
    imp = pd.DataFrame(
        {
            "ridge_coef (scaled)": pd.Series(
                ridge.regressor_[-1].coef_, index=cols
            ),
            "lightgbm_gain_share": gain / gain.sum(),
        }
    ).sort_values("lightgbm_gain_share", ascending=False)
    print("\nMOST IMPORTANT FEATURES (top 10 by LightGBM gain):")
    print(imp.head(10).round(4).to_string())
    imp.to_csv(out / f"{ds}_feature_importance.csv")

    if evaluate_test:
        test_table = evaluate(
            pd.concat([train, val]), test, cols, ds
        ).sort_values("MAE")
        print("\nTEST (final result, trained on train + val):")
        print(test_table.round(4).to_string())
        test_table.to_csv(out / f"{ds}_test.csv")

gate = pd.DataFrame(gate_rows)
gate.to_csv(out / "gate.csv", index=False)
pd.concat(all_val).to_csv(out / "validation_all.csv", index=False)
print(
    "\n===== GATE SUMMARY (best cross-validated R2 of the numeric models) "
    "====="
)
print(gate.round(4).to_string(index=False))
print(
    "Test data was NOT used."
    if not evaluate_test
    else "Test data was used (final result)."
)
