import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

root = Path("time_imm")
out = Path("clean_data")
out.mkdir(exist_ok=True)

max_missing = 0.95
min_rows = 100

config = {
    "RepoHealth": {
        "targets": {
            "new_issues": "log1p",
            "merged_prs": "log1p",
            "issue_comment_volume": "log1p",
        },
        "legacy": "new_issues",
        "drop": [],
    },
    "EPA-Air": {
        "targets": {"pm2_5": "log1p"},
        "legacy": "pm2_5",
        "drop": ["aqi", "ozone"],
    },
    "GDELT": {"targets": {"AvgTone": None}, "legacy": "AvgTone", "drop": []},
    "ClusterTrace": {
        "targets": {
            "machine_cpu": None,
            "machine_load_1": None,
            "n_job": "log1p",
            "n_task": "log1p",
            "machine_cpu_usr": None,
        },
        "legacy": "machine_cpu",
        "drop": [],
    },
    "StudentLife": {
        "targets": {"sleep_duration": None},
        "legacy": "sleep_duration",
        "drop": [],
    },
}

domain_stop = {
    "article",
    "articles",
    "highlights",
    "highlight",
    "including",
    "overall",
    "efforts",
    "reports",
    "report",
    "weather",
}
keep = {"not", "no", "nor", "never", "none"}
stop = (set(ENGLISH_STOP_WORDS) | domain_stop) - keep


def get_time_column(df):
    for col in df.columns:
        if (
            "date" in col.lower()
            or "time" in col.lower()
            or "stamp" in col.lower()
        ):
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            if parsed.notna().sum() > len(df) * 0.9:
                return col, parsed
    return None, None


def scan(files, target, drop):
    n_rows, n_na, numeric = 0, {}, set()
    for f in files:
        df = pd.read_csv(f)
        if target not in df.columns:
            raise ValueError(
                f"Target '{target}' not found in {f}. Columns are: "
                f"{list(df.columns)}"
            )
        n_rows += len(df)
        for c in df.columns:
            n_na[c] = n_na.get(c, 0) + int(df[c].isna().sum())
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric.add(c)
    share = {c: k / n_rows for c, k in n_na.items()}
    too_empty = [
        c for c, s in share.items() if s > max_missing and c != target
    ]
    skip = set(drop) | set(too_empty) | {"record_id", target}
    mask_cols = [c for c in numeric if n_na[c] > 0 and c not in skip]
    return sorted(too_empty), sorted(mask_cols)


def clean_numeric(f, ds, cfg, drop_cols, mask_cols):
    target = cfg["target"]
    df = pd.read_csv(f)
    time_col, times = get_time_column(df)
    df = df.drop(columns=[time_col]).assign(date_time=times)

    df = df.drop(columns=["record_id"] + list(drop_cols), errors="ignore")
    df = (
        df.dropna(subset=["date_time"])
        .sort_values("date_time", kind="stable")
        .drop_duplicates()
    )

    n_dup = int(df["date_time"].duplicated().sum())
    if n_dup:
        df = df.groupby("date_time", as_index=False).mean(numeric_only=True)

    for c in mask_cols:
        if c in df.columns:
            last_seen = (
                df["date_time"]
                .where(df[c].notna())
                .ffill()
                .fillna(df["date_time"].iloc[0])
            )
            df[f"{c}_missing_hours"] = (
                df["date_time"] - last_seen
            ).dt.total_seconds() / 3600

    df = df.dropna(subset=[target])

    if cfg["transform"] == "log_return":
        df = df[df[target] > 0]
        df["target"] = np.log(df[target]).diff()
    elif cfg["transform"] == "abs_log_return":
        df = df[df[target] > 0]
        df["target"] = np.log(df[target]).diff().abs()
    elif cfg["transform"] == "log1p":
        df = df[df[target] >= 0]
        df["target"] = np.log1p(df[target])
    else:
        df["target"] = df[target]
    df = df.drop(columns=[target]).dropna(subset=["target"])

    others = [c for c in df.columns if c not in ("date_time", "target")]
    df[others] = df[others].ffill().fillna(0)

    pos = np.arange(len(df)) / max(len(df), 1)
    df["split"] = np.where(
        pos < 0.6, "train", np.where(pos < 0.8, "val", "test")
    )
    df["entity_id"] = f"{ds}/{f.parent.name}"
    first = ["date_time", "entity_id", "target", "split"]
    return (
        df[first + [c for c in df.columns if c not in first]].reset_index(
            drop=True
        ),
        n_dup,
    )


def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(w for w in s.split() if w not in stop and len(w) > 2)


for ds, cfg in config.items():
    folder = cfg.get("folder", ds)
    files = sorted((root / folder).rglob("time_series.csv"))
    print(f"\n===== {ds} =====")
    if not files:
        print(f"no files found in {root / folder}, skipped")
        continue

    parts = []
    for f in sorted((root / folder).rglob("text.csv")):
        tdf = pd.read_csv(f)
        time_col, times = get_time_column(tdf)
        text_col = [
            c for c in tdf.columns if c not in (time_col, "record_id")
        ][-1]
        t = pd.DataFrame(
            {
                "date_time": times,
                "entity_id": f"{ds}/{f.parent.name}",
                "raw": tdf[text_col],
            }
        )
        parts.append(t.dropna().drop_duplicates(subset=["raw"]))
    txt_out = None
    if not parts:
        print("no text files found")
    else:
        txt = pd.concat(parts, ignore_index=True)
        n_before = len(txt)
        empty_info = (
            txt["raw"]
            .astype(str)
            .str.contains("not mention|no mention", case=False)
            .mean()
        )
        print(
            f"texts that say the article does not mention the topic: "
            f"{empty_info:.1%}"
        )
        txt["text_clean"] = txt["raw"].astype(str).map(clean_text)
        txt = txt[txt["text_clean"] != ""].sort_values(
            ["entity_id", "date_time"]
        )
        txt_out = txt.rename(columns={"raw": "text_raw"})[
            ["date_time", "entity_id", "text_clean", "text_raw"]
        ]
        print(
            f"text (column '{text_col}'): {n_before} texts after removing "
            f"empty/duplicates, "
            f"{len(txt)} left after cleaning, "
            f"{txt['text_clean'].str.split().str.len().mean():.1f} words each"
        )
        print("BEFORE:", txt["raw"].iloc[0][:110])
        print("AFTER :", txt["text_clean"].iloc[0][:110])
        print(
            "top words:",
            txt["text_clean"]
            .str.split()
            .explode()
            .value_counts()
            .head(10)
            .to_dict(),
        )

    for target, transform in cfg["targets"].items():
        tcfg = {"target": target, "transform": transform}
        print(f"\n--- {ds} / {target} (transform: {transform}) ---")
        try:
            too_empty, mask_cols = scan(files, target, cfg["drop"])
        except ValueError as e:
            print(f"skipped: {e}")
            continue
        print(
            f"columns dropped (more than {max_missing:.0%} empty): "
            f"{too_empty or 'none'}"
        )
        print(
            f"columns that get a *_missing_hours mask: {mask_cols or 'none'}"
        )
        tables, dups = [], 0
        for f in files:
            t, n_dup = clean_numeric(
                f, ds, tcfg, cfg["drop"] + too_empty, mask_cols
            )
            if len(t) < min_rows:
                print(f"  skipped {f.parent.name}: only {len(t)} usable rows")
                continue
            tables.append(t)
            dups += n_dup
        if not tables:
            print("no entity has enough usable rows for this target, skipped")
            continue
        num = pd.concat(tables, ignore_index=True)
        names = [f"{ds}_{target}"] + ([ds] if target == cfg["legacy"] else [])
        for name in names:
            num.to_csv(out / f"{name}_numeric.csv", index=False)
            if txt_out is not None:
                txt_out.to_csv(out / f"{name}_text.csv", index=False)
        print(
            f"numeric: {len(num)} rows, {num['entity_id'].nunique()} "
            f"entities, duplicate timestamps merged: {dups}"
        )
        print(
            "rows per split:",
            num["split"].value_counts().to_dict(),
            "| target mean/std:",
            round(num["target"].mean(), 3),
            round(num["target"].std(), 3),
        )
        print("saved:", ", ".join(f"{n}_numeric.csv" for n in names))

print(f"\nDone. Clean files are in: {out.resolve()}")
