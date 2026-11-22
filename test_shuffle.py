import pandas as pd
for ds in ["FNSPID", "GDELT", "EPA-Air"]:
    d = pd.read_csv(f"clean_data/{ds}_numeric.csv", parse_dates=["date_time"])
    ok = all(g.loc[g.split == "train", "date_time"].max() < g.loc[g.split == "val", "date_time"].min()
             and g.loc[g.split == "val", "date_time"].max() < g.loc[g.split == "test", "date_time"].min()
             for _, g in d.groupby("entity_id"))
    print(ds, "train < val < test for every entity:", ok)