import argparse
import csv
import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
default_model = "qwen3.5:4b"
prompt_version = "int-v1"
max_tries = 3
max_chars = 2000
clean = Path("clean_data")

prompts = {
    "RepoHealth": (
        "Read this GitHub commit/PR text. Output only a single number:\n"
        "0 = general maintenance\n1 = new feature\n2 = bug fix\n3 = urgent "
        "bug/crash\n"
        "4 = documentation\n5 = testing\nText: {text}"
    ),
    "GDELT": (
        "Read this news text. Output only a single number for sentiment:\n"
        "-2 = very negative\n-1 = slightly negative\n0 = neutral\n1 = "
        "slightly positive\n2 = very positive\n"
        "Text: {text}"
    ),
}
valid = {"RepoHealth": set(range(0, 6)), "GDELT": set(range(-2, 3))}


def parse_code(content, ds):
    s = str(content).replace("−", "-").replace("–", "-")
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    code = int(m.group(0))
    return code if code in valid[ds] else None


def ask(model, ds, text):
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "think": False,
        "options": {
            "num_predict": 5,
            "temperature": 0.0,
            "seed": 0,
            "num_ctx": 2048,
        },
        "messages": [
            {
                "role": "user",
                "content": prompts[ds].format(text=text[:max_chars]),
            }
        ],
    }
    t0, raw = time.time(), ""
    for _ in range(max_tries):
        try:
            r = requests.post(
                f"{ollama_url}/api/chat", json=payload, timeout=180
            )
            if (
                r.status_code == 400
                and "think" in r.text.lower()
                and "think" in payload
            ):
                payload.pop("think")
                r = requests.post(
                    f"{ollama_url}/api/chat", json=payload, timeout=180
                )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            return parse_code(raw, ds), raw, time.time() - t0
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(1)
    return None, "<no answer from the server>", time.time() - t0


def cache_key(model, ds, text):
    return hashlib.sha1(
        f"{model}|{prompt_version}|{ds}|{text}".encode("utf-8")
    ).hexdigest()


def load_cache(path):
    cache = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) == 4 and row[0] != "key":
                    try:
                        cache[row[0]] = int(row[3])
                    except ValueError:
                        pass
    return cache


def check_server(model):
    try:
        names = [
            m["name"]
            for m in requests.get(f"{ollama_url}/api/tags", timeout=5)
            .json()
            .get("models", [])
        ]
    except requests.RequestException:
        raise SystemExit(
            f"Cannot reach Ollama at {ollama_url}. Start it (`ollama serve`) "
            f"and try again."
        )
    if model not in names and f"{model}:latest" not in names:
        raise SystemExit(
            f"Model '{model}' is not installed. Run:  ollama pull "
            f"{model}\nInstalled: {names or 'none'}"
        )


def run(ds, model, test, workers, n_test):
    f = clean / f"{ds}_text.csv"
    if not f.exists():
        print(f"{ds}: {f} not found, skipped")
        return
    df = pd.read_csv(f)
    texts = df["text_raw"].astype(str).tolist()
    keys = [cache_key(model, ds, t) for t in texts]
    cache_path, log_path = (
        clean / f"{ds}_llm_cache.csv",
        clean / f"{ds}_llm_failures.log",
    )
    cache = load_cache(cache_path)
    if not cache_path.exists():
        with open(cache_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(
                ["key", "date_time", "entity_id", "llm_code"]
            )
    first_row = {}
    for i, k in enumerate(keys):
        first_row.setdefault(k, i)
    todo = {k: i for k, i in first_row.items() if k not in cache}
    done_before = len(first_row) - len(todo)
    if test:
        todo = dict(list(todo.items())[:n_test])
    print(
        f"\n===== {ds}: {len(texts)} texts, {len(first_row)} different, "
        f"{done_before} already in the cache, "
        f"{len(todo)} to do with {model} ====="
    )

    lock, times, bad = threading.Lock(), [], []

    def work(k, i):
        code, raw, secs = ask(model, ds, texts[i])
        with lock:
            if code is not None:
                with open(cache_path, "a", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(
                        [
                            k,
                            df["date_time"].iloc[i],
                            df["entity_id"].iloc[i],
                            code,
                        ]
                    )
                cache[k] = code
            else:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(
                        f"{df['entity_id'].iloc[i]}\t{df['date_time'].iloc[i]}\t"
                        f"raw answer: {raw!r}\n"
                    )
                bad.append(raw)
        return i, code, raw, secs

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, k, i) for k, i in todo.items()]
            try:
                for fut in tqdm(
                    as_completed(futs), total=len(futs), desc=ds, unit="text"
                ):
                    i, code, raw, secs = fut.result()
                    times.append(secs)
                    if test:
                        print(
                            f"\n{texts[i][:150]!r}\n  raw answer {raw!r} -> "
                            f"code {code}  ({secs:.1f}s)"
                        )
            except KeyboardInterrupt:
                ex.shutdown(wait=False, cancel_futures=True)
                print(
                    "\nStopped. Everything finished so far is saved in the "
                    "cache: run the script again to continue."
                )
                raise
    finally:
        if times:
            print(
                f"{len(times)} texts asked, {len(bad)} answers could not be "
                f"read, {np.mean(times):.2f} s per text on average"
                + (
                    f" -> the full run would take about "
                    f"{np.mean(times) * len(first_row) / 3600 / max(workers, 1):.1f} h"
                    if test
                    else ""
                )
            )
            if bad:
                print(
                    f"first unreadable answers: {bad[:3]}  (all in "
                    f"{log_path.name}). Empty answers usually mean the "
                    "reasoning mode is still on."
                )
    if test:
        return
    out = pd.DataFrame(
        {
            "date_time": df["date_time"],
            "entity_id": df["entity_id"],
            "llm_code": pd.Series([cache.get(k, np.nan) for k in keys]).astype(
                "Int64"
            ),
        }
    )
    out.to_csv(clean / f"{ds}_llm_features.csv", index=False)
    missing = int(out["llm_code"].isna().sum())
    print(
        f"saved clean_data/{ds}_llm_features.csv ({len(out)} rows, {missing} "
        f"without a code)"
    )
    print(
        "share of each code:\n"
        + out["llm_code"]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        .round(3)
        .to_string()
    )
    if missing:
        print("Run the script again to retry the texts that failed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Label texts with a local Ollama model "
        "(one integer per text)"
    )
    ap.add_argument("--model", default=default_model)
    ap.add_argument(
        "--dataset",
        choices=list(prompts),
        help="only this dataset (default: both)",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="try 20 texts per dataset, print them, do not write the CSV",
    )
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel requests (needs OLLAMA_NUM_PARALLEL on the server)",
    )
    args = ap.parse_args()
    check_server(args.model)
    for ds in ([args.dataset] if args.dataset else list(prompts)):
        run(ds, args.model, args.test, args.workers, args.n_test)
