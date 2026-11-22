import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

meanings = {
    "RepoHealth": {
        0: "general maintenance",
        1: "new feature",
        2: "bug fix",
        3: "urgent bug/crash",
        4: "documentation",
        5: "testing",
    },
    "GDELT": {
        -2: "very negative",
        -1: "slightly negative",
        0: "neutral",
        1: "slightly positive",
        2: "very positive",
    },
}
N = 20
clean, res = Path("clean_data"), Path("results")
res.mkdir(exist_ok=True)

for ds, meaning in meanings.items():
    ft, fl = clean / f"{ds}_text.csv", clean / f"{ds}_llm_features.csv"
    if not (ft.exists() and fl.exists()):
        print(f"{ds}: files missing, skipped")
        continue
    txt, llm = pd.read_csv(ft), pd.read_csv(fl)
    assert len(txt) == len(llm), "the two files have different numbers of rows"
    df = pd.DataFrame(
        {
            "entity_id": txt["entity_id"],
            "date_time": txt["date_time"],
            "code": llm["llm_code"],
            "meaning": llm["llm_code"].map(meaning),
            "text": txt["text_raw"].astype(str),
        }
    )
    s = df.sample(N, random_state=0).reset_index(drop=True)
    s["looks right? (y/n)"] = ""
    s.to_csv(res / f"label_check_{ds}.csv", index=False)
    print(
        "\n" + "=" * 100 + f"\n{ds}: {N} random texts. "
        f"For each one ask yourself: would I give this code?\n" + "=" * 100
    )
    for i, r in s.iterrows():
        shown = r["text"].replace("\n", " ")
        print(
            f"\n[{i + 1}] CODE {r['code']} = {r['meaning']}\n    "
            f"{shown[:500]}{'...' if len(shown) > 500 else ''}"
        )
    print(f"\nsaved {res / f'label_check_{ds}.csv'}")
