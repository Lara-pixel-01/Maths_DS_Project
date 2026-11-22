import pandas as pd
import warnings 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import re
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
 

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop = set(ENGLISH_STOP_WORDS)


datasets = Path('time_imm')

output_path = Path("eda_outputs")
output_path.mkdir(exist_ok=True)

list_csvs = list(datasets.rglob('time_series.csv'))


print(f"Amount of csv files {len(list_csvs)}")

targets = {"FNSPID": "close", "GDELT": "AvgTone", "EPA-AIR":"pm2_5"}
max_text_per_entity = 2000


def to_ns(times):
    return pd.DatetimeIndex(times).tz_convert(None).values.astype("datetime64[ns]")

def adf_p(values):
    try:
        return adfuller(values[-20000:])[1]
    except Exception:
        return np.nan


def get_time_column(df: pd.DataFrame):
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'stamp' in col.lower():
            parse_time = pd.to_datetime(df[col], errors='coerce', utc=True)
            if parse_time.notna().sum() > (len(df) * 0.9):
                return col, parse_time

    return None, None



def gap_stats(times:pd.Series):
    time = times.dropna().sort_values()
    gaps = time.diff().dt.total_seconds().dropna() / 3600.0

    if gaps.empty:
        return {"mean_gap": np.nan, 
                "median_gap": np.nan, 
                "max_gap": np.nan, 
                "cv_gap": np.nan, 
                "big_gap_share": np.nan,
                "zero_gap_share": np.nan
        }, gaps

    mean = gaps.mean()

    stats = {
        "mean_gap": mean, 
        "median_gap": gaps.median(), 
        "max_gap": gaps.max(), 
        "cv_gap": gaps.std() / mean if mean > 0 else np.nan, 
        "big_gap_share": (gaps > (5 * gaps.median())).mean(),
        "zero_gap_share": (gaps==0).mean()
    }

    return stats, gaps

text_rows = []
gap_store = {}
summary = []
schema_rows = []
problems = []

for f in list_csvs:

    chunks = f.parts
    dataset_name = chunks[1]
    entity_name = chunks[3]
    entity_id = dataset_name + "/" + entity_name

    df = pd.read_csv(f)
    row_count = len(df)

    time_col, times =  get_time_column(df)


    is_sorted = False
    n_dup_ts = 0
    t_min = None
    t_max = None
    gstats = {}


    data_cols = []

    if times is not None:
        is_sorted = bool(times.dropna().is_monotonic_increasing)
        n_dup_ts = int(times.duplicated().sum())
        t_min = times.min()
        t_max = times.max()

        gstats, gaps = gap_stats(times)
        gap_store[entity_id] = gaps

    else: 
        problems.append((entity_id, "time_series.csv", "no date found"))

    for col in df.columns:
        if col not in (time_col, "record_id"):
            data_cols.append(col)

    missing_vals = df[data_cols].isna().mean().mean() * 100

    numeric_col = df[data_cols].select_dtypes(include="number").columns.to_list()
    n_dup_rows = int(df.duplicated().sum())

    txt_path = f.parent / 'text.csv'
    
    text_df = pd.read_csv(txt_path)
    n_text = len(text_df)

    text_time_cols,  text_times = get_time_column(text_df)
    if text_times is not None and times is not None:
        inside = ((text_times >= t_min) & (text_times <= t_max)).mean()
    else:
        inside = np.nan

    text_rows.append({
        "entity_id":  entity_id,
        "Dataset": dataset_name,
        "n_text": n_text,
        "text_per_obs": n_text / max(row_count, 1),
        "share_texts_inside_numeric_span": inside,
    })

    summary.append({
        "entity_id": entity_id,
        "Dataset": dataset_name,
        "Entity": entity_name,
        "RowCount": row_count,
        "n_cols": len(df.columns),
        "time_col": time_col,
        "t_min": t_min,
        "t_max": t_max,
        "span_days": (t_max - t_min).total_seconds() / 86400 if pd.notna(t_min) else np.nan,
        "is_sorted": is_sorted,
        "n_dup_ts": n_dup_ts,
        "n_dup_rows": n_dup_rows,
        "MissingVals": missing_vals,
        "n_numeric_cols": len(numeric_col),
        "n_text": n_text,
        **gstats,
    })

    for col in df.columns:
        schema_rows.append({
            "entity_id": entity_id,
            "Dataset": dataset_name,
            "column": col,
            "dtype": str(df[col].dtype),
            "is_time": col == time_col,
            "missing_pct": df[col].isna().mean() * 100,
            "n_unique": df[col].nunique(),
        })

summary_df = pd.DataFrame(summary)
schema_df = pd.DataFrame(schema_rows)
text_df_all = pd.DataFrame(text_rows)

summary_df.to_csv(output_path / "inventory_per_entity.csv", index = False)
schema_df.to_csv(output_path / "schema_per_column.csv", index=False)
text_df_all.to_csv(output_path / "text_stats_per_entity.csv", index=False)
pd.DataFrame(problems, columns=["entity_id", "file", "problems"]).to_csv(output_path / "problems.csv", index=False)

print(summary_df[["entity_id", "RowCount", "span_days", "MissingVals", "median_gap", "cv_gap", "n_text"]].head(15))
print("\nEntities without a date column:", int(summary_df["time_col"].isna().sum()))
print("Entities NOT sorted by time   :", int((summary_df["is_sorted"] == False).sum()))
print(f"Problems: {len(problems)}")


per_dataset = summary_df.groupby("Dataset").agg(
    n_entities=("entity_id", "count"),
    total_rows=("RowCount", "sum"),
    median_rows=("RowCount", "median"),
    median_gap_h=("median_gap", "median"),
    median_cv_gap=("cv_gap", "median"),
    median_big_gap_share=("big_gap_share", "median"),
    mean_missing_pct=("MissingVals", "mean"),
    total_texts=("n_text", "sum"),
).round(3)
per_dataset.to_csv(output_path / "summary_per_dataset.csv")
print("\n Per Dataset")
print(per_dataset)


boxplots = [
    ("RowCount", "Observations per entity (log)", "fig_series_length.png", True, None),
    ("median_gap", "Median gap per entity, hours (log)", "fig_median_gap.png", True, None),
    ("cv_gap", "Irregularity: CV of gaps (dashed line = 1)", "fig_gap_cv.png", False, 1),
    ("MissingVals", "Missing values in data columns (%)", "fig_missingness.png", False, None),
]

for column, title, file_name, use_log, hline in boxplots:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=summary_df, x="Dataset", y=column, ax=ax)
    if use_log:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, ls="--", c="gray")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path / file_name, dpi=150)
    plt.close(fig)


names = sorted(summary_df["Dataset"].unique())
n_cols = 3
n_rows = int(np.ceil(len(names) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.6 * n_rows), squeeze=False)
for ax, ds in zip(axes.ravel(), names):
    ids = summary_df.loc[summary_df["Dataset"] == ds, "entity_id"]
    all_gaps = [gap_store[i] for i in ids if i in gap_store]
    if all_gaps:
        pooled = pd.concat(all_gaps, ignore_index=True)
        pooled = pooled[pooled > 0]
        if len(pooled):
            ax.hist(np.log10(pooled), bins=50)
    ax.set_title(ds)
    ax.set_xlabel("log10(gap in hours)")
for ax in axes.ravel()[len(names):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(output_path / "fig_gap_histograms.png", dpi=150)
plt.close(fig)


if not text_df_all.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=text_df_all, x="Dataset", y="n_text", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Texts per entity (log)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path / "fig_text_volume.png", dpi=150)
    plt.close(fig)




cand = per_dataset.copy()
cand["rank_rows"] = cand["median_rows"].rank(ascending=False)
cand["rank_text"] = cand["total_texts"].rank(ascending=False)
cand["rank_missing"] = cand["mean_missing_pct"].rank(ascending=True)
cand["score"] = cand[["rank_rows", "rank_text", "rank_missing"]].mean(axis=1)
cand = cand.sort_values("score")
cand.to_csv(output_path / "deep_dive_candidates.csv")
print("\n Candidates")
print(cand[["median_rows", "total_texts", "mean_missing_pct", "score"]])

print(f"\nDone. Tables and figures are in: {output_path.resolve()}")


s = pd.read_csv("eda_outputs/schema_per_column.csv")
s = s[s["Dataset"].isin(["FNSPID", "GDELT", "EPA-Air"])]
print(s.groupby(["Dataset", "column"]).agg(
    dtype=("dtype", "first"),
    missing_pct=("missing_pct", "mean"),
    n_unique=("n_unique", "median"),
).round(1).to_string())



p3_rows = []
p3_longest = {}

for ds, targ in targets.items():
    print(f"\n {ds}: target = {targ}")

    for f in sorted((datasets / ds).rglob("time_series.csv")):
        df = pd.read_csv(f)

        time_col, times = get_time_column(df)

        if time_col is None or targ not in df.columns:
            continue


        s = pd.Series(df[targ].values, index=pd.DatetimeIndex(times)).dropna()
        s = s[s.index.notna()].sort_index()

        if len(s) < 20:
            continue

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        row = {
            "Dataset": ds,
            "Entity": f.parent.name,
            "n_observed": len(s),
            "target_missing_pct": df[targ].isna().mean() * 100,
            "mean": s.mean(),
            "std": s.std(),
            "skew": s.skew(),
            "outlier_share": ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() if iqr > 0 else np.nan,
            "adf_p_level": adf_p(s.values),
            "adf_p_change": adf_p(s.diff().dropna().values),
        }

        txt_path = f.parent / "text.csv"
        if txt_path.exists():
            _, text_times = get_time_column(pd.read_csv(txt_path))
            if text_times is not None:
                text_ns = np.sort(to_ns(text_times.dropna()))
                up_to = np.searchsorted(text_ns, to_ns(s.index), side="right")
                per_gap = np.diff(up_to)
                row["share_gaps_without_text"] = (per_gap == 0).mean()
                row["texts_per_gap"] = per_gap.mean()
    
        p3_rows.append(row)
    
        if ds not in p3_longest or len(s) > len(p3_longest[ds][1]):
            p3_longest[ds] = (f.parent.name, s, df[targ].isna().values)


p3 = pd.DataFrame(p3_rows)

p3.to_csv(output_path / "part3_target_summary.csv", index=False)
 
print("\n Target Summary (median over entities")
cols = ["n_observed", "target_missing_pct", "skew", "outlier_share",
        "adf_p_level", "adf_p_change", "share_gaps_without_text", "texts_per_gap"]
print(p3.groupby("Dataset")[cols].median().round(3).T)
 
print("\nShare of entities that are stationary (ADF p < 0.05):")
print(pd.DataFrame({
    "on the values": (p3["adf_p_level"] < 0.05).groupby(p3["Dataset"]).mean(),
    "on the changes": (p3["adf_p_change"] < 0.05).groupby(p3["Dataset"]).mean(),
}).round(2))
 
print("\nMissing % of the target along the timeline (longest entity):")
for ds, (name, s, is_na) in p3_longest.items():
    chunks = [round(c.mean() * 100) for c in np.array_split(is_na, 10)]
    print(f"  {ds:8s} ({name}): {chunks}")
 
fig, axes = plt.subplots(2, len(p3_longest), figsize=(5 * len(p3_longest), 6), squeeze=False)
for i, (ds, (name, s, _)) in enumerate(p3_longest.items()):
    axes[0, i].plot(s.index, s.values, lw=0.7)
    axes[0, i].set_title(f"{ds}/{name}: {targets[ds]}", fontsize=9)
    axes[1, i].hist(s.values, bins=50)
    axes[1, i].set_title("distribution of values", fontsize=9)
fig.tight_layout()
fig.savefig(output_path / "part3_target_overview.png", dpi=150)
plt.close(fig)
 
 
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(w for w in s.split() if w not in stop and len(w) > 2)
 
 
def top_words(vectorizer, texts, use_mean=False, n=15):
    X = vectorizer.fit_transform(texts)
    score = X.mean(axis=0).A1 if use_mean else X.sum(axis=0).A1
    words = np.array(vectorizer.get_feature_names_out())
    best = np.argsort(score)[::-1][:n]
    return pd.Series(score[best].round(4), index=words[best])
 
 
for ds in targets:
    print(f"\nText: {ds}")
    parts = []
    for f in sorted((datasets / ds).rglob("text.csv")):
        tdf = pd.read_csv(f)
        raw = tdf[tdf.columns[-1]].dropna().astype(str).head(max_text_per_entity)
        parts.append(pd.DataFrame({"entity_id": f"{ds}/{f.parent.name}", "raw": raw.values}))
    if not parts:
        print("no text found")
        continue
 
    texts = pd.concat(parts, ignore_index=True)
    texts["clean"] = texts["raw"].map(clean_text)
 
    print(f"texts: {len(texts)} | words per text: {texts['raw'].str.split().str.len().mean():.1f} before, "
          f"{texts['clean'].str.split().str.len().mean():.1f} after cleaning")
    print(f"empty after cleaning: {(texts['clean'] == '').mean():.1%} | "
          f"duplicates: {texts['clean'].duplicated().mean():.1%}")
    print("BEFORE:", texts["raw"].iloc[0][:150])
    print("AFTER :", texts["clean"].iloc[0][:150])
 
    good = texts.loc[texts["clean"] != "", "clean"]
    if len(good) < 10:
        print("too few usable texts for BoW / TF-IDF")
        continue
 
    bow = top_words(CountVectorizer(max_features=5000, min_df=2), good)
    tfidf = top_words(TfidfVectorizer(max_features=5000, min_df=2), good, use_mean=True)
    print("\nTop words, BoW (total count)   |   Top words, TF-IDF (average weight)")
    print(pd.DataFrame({"bow_word": bow.index, "bow_count": bow.values,
                        "tfidf_word": tfidf.index, "tfidf_weight": tfidf.values}).to_string(index=False))
    pd.DataFrame({"bow_word": bow.index, "bow_count": bow.values,
                  "tfidf_word": tfidf.index, "tfidf_weight": tfidf.values}).to_csv(
        output_path / f"part3_top_words_{ds}.csv", index=False)
 
print(f"\n Files are in: {output_path.resolve()}")
