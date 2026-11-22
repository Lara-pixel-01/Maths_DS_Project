import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

n_splits = 8
inner_splits = 3
n_iter = 10
lags = (1, 2, 3, 5, 7)
row_windows = (5, 20)
time_windows = ("1D", "7D")
log_cols = {"volume", "NumArticles", "NumMentions", "NumSources"}
use_query_time = True
predict_delta = False
lgbm_objective = "regression_l1"


def make_features(df, horizon_hours=None):
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    df = df.sort_values(["entity_id", "date_time"]).reset_index(drop=True)
    n = len(df)
    g = df.groupby("entity_id")
    y = df["target"].to_numpy()
    t_ns = df["date_time"].dt.tz_convert(None).to_numpy().astype("datetime64[ns]")

    O = pd.DataFrame(index=df.index)
    for k in lags:
        O[f"lag_{k}"] = g["target"].shift(k - 1)
    for w in row_windows:
        O[f"mean_{w}"] = g["target"].transform(lambda s, w=w: s.rolling(w).mean())
        O[f"std_{w}"] = g["target"].transform(lambda s, w=w: s.rolling(w).std())
    
    for win in time_windows:
        m, s, c = (np.full(n, np.nan) for _ in range(3))
        for idx in g.indices.values():
            roll = pd.Series(y[idx], index=pd.DatetimeIndex(t_ns[idx])).rolling(win)
            m[idx], s[idx], c[idx] = roll.mean().to_numpy(), roll.std().to_numpy(), roll.count().to_numpy()
        O[f"tmean_{win}"], O[f"tstd_{win}"], O[f"tcount_{win}"] = m, np.nan_to_num(s), c
        
    O["gap_prev_h"] = g["date_time"].diff().dt.total_seconds() / 3600
    for c in [c for c in df.columns if c not in ("date_time", "entity_id", "target", "split")]:
        v = df[c]
        if c in log_cols or c.endswith("_missing_hours"):
            v = np.log1p(v.clip(lower=0))
        O[f"{c}_lag1"] = v

    src = np.full(n, -1)
    delta = None if horizon_hours is None else np.timedelta64(int(round(horizon_hours * 3600)), "s")
    for idx in g.indices.values():
        if delta is None:
            src[idx[1:]] = idx[:-1]
        else:
            k = np.searchsorted(t_ns[idx], t_ns[idx] - delta, side="right")
            ok = k >= 1
            src[idx[ok]] = idx[k[ok] - 1]
            
    rows = np.where(src >= 0)[0]
    origin = src[rows]

    F = O.iloc[origin].reset_index(drop=True)
    F.index = rows
    if use_query_time:
        F["lead_h"] = (t_ns[rows] - t_ns[origin]) / np.timedelta64(1, "h")
        tt = df["date_time"].iloc[rows]
        for name, value, period in (("hour", tt.dt.hour, 24), ("weekday", tt.dt.dayofweek, 7),
                                    ("month", tt.dt.month - 1, 12), ("doy", tt.dt.dayofyear - 1, 366)):
            F[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
            F[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
            
    F.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in F.columns]

    meta = df.loc[rows, ["entity_id", "date_time", "target", "split"]].copy()
    meta["t_cut"] = df["date_time"].iloc[origin].to_numpy()
    keep = F.notna().all(axis=1)
    data = pd.concat([meta, F], axis=1)[keep]
    return data.reset_index(drop=True), list(F.columns)


def grouped_time_cv(df, n_splits=n_splits):
    size = df.groupby("entity_id")["target"].transform("size")
    pos = df.groupby("entity_id").cumcount() / size
    step = 1 / (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_idx = np.where(pos < k * step)[0]
        val_idx = np.where((pos >= k * step) & (pos < (k + 1) * step))[0]
        if len(train_idx) and len(val_idx):
            yield train_idx, val_idx


clip = 5.0

def linear(model):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), FunctionTransformer(np.clip, kw_args={"a_min": -clip, "a_max": clip}), model),
        transformer=StandardScaler())


def _rf():
    return RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=42)


def _lgbm(**params):
    settings = dict(objective=lgbm_objective, n_estimators=200, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    settings.update(params)
    return LGBMRegressor(**settings)


def models():
    return {
        "LinearRegression": linear(LinearRegression()),
        "Ridge": linear(Ridge(alpha=10.0)),
        "Lasso": linear(Lasso(alpha=0.01)),
        "ElasticNet": linear(ElasticNet(alpha=0.01, l1_ratio=0.5)),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42),
        "RandomForest": _rf(),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42),
        "LightGBM": _lgbm(),
        "VotingEnsemble": VotingRegressor([("rf", _rf()), ("lgbm", _lgbm())]),
    }


lgbm_search = {"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.05, 0.1],
               "n_estimators": [100, 200, 300], "min_child_samples": [20, 50, 100],
               "subsample": [0.7, 0.9, 1.0], "subsample_freq": [1],
               "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [0, 1, 10]}
ridge_alphas = [0.1, 1, 10, 100, 1000, 10000]
max_lr = 0.3


def tune_lgbm(X, y, entity_ids):
    tmp = pd.DataFrame({"entity_id": np.asarray(entity_ids), "target": np.asarray(y)})
    inner = list(grouped_time_cv(tmp, inner_splits))
    if not inner:
        return None
        
    coarse = RandomizedSearchCV(LGBMRegressor(objective=lgbm_objective, random_state=42, verbose=-1), lgbm_search, n_iter=n_iter, cv=inner, scoring="neg_mean_absolute_error", random_state=42, n_jobs=1, refit=False).fit(X, y)
    best = coarse.best_params_
    leaves, lr = int(best["num_leaves"]), float(best["learning_rate"])
    fine_grid = {"num_leaves": sorted({max(2, int(round(leaves * 0.5))), leaves, int(round(leaves * 2))}),
                 "learning_rate": sorted({lr * 0.5, lr, min(lr * 2, max_lr)})}
    fixed = {k: v for k, v in best.items() if k not in fine_grid}
    
    fine = GridSearchCV(LGBMRegressor(objective=lgbm_objective, random_state=42, verbose=-1, **fixed), fine_grid, cv=inner, scoring="neg_mean_absolute_error", n_jobs=1, refit=False).fit(X, y)
    return {**fixed, **fine.best_params_}


class _ShiftBack:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return np.asarray(self.model.predict(X), float) + X["lag_1"].to_numpy(float)


def fit_model(name, X, y, entity_ids, cache=None, delta=False):
    if delta:
        shifted = np.asarray(y, float) - X["lag_1"].to_numpy(float)
        return _ShiftBack(fit_model(name, X, shifted, entity_ids, cache=cache, delta=False))
        
    cache = {} if cache is None else cache
    if name == "Ridge":
        tmp = pd.DataFrame({"entity_id": np.asarray(entity_ids), "target": np.asarray(y)})
        inner = list(grouped_time_cv(tmp, inner_splits))
        if not inner:
            return models()["Ridge"].fit(X, y)
        search = GridSearchCV(linear(Ridge()), {"regressor__ridge__alpha": ridge_alphas}, cv=inner, scoring="neg_mean_absolute_error", n_jobs=1, refit=True)
        return search.fit(X, y).best_estimator_
        
    if name in ("LightGBM", "VotingEnsemble"):
        if "lgbm" not in cache:
            cache["lgbm"] = tune_lgbm(X, y, entity_ids)
        params = cache["lgbm"] or {}
        if name == "LightGBM":
            return _lgbm(**params).fit(X, y)
        return VotingRegressor([("rf", _rf()), ("lgbm", _lgbm(**params))]).fit(X, y)
        
    return models()[name].fit(X, y)


def metrics(ev, pred, base):
    err = ev["target"].values - pred
    by_e = pd.DataFrame({"e": ev["entity_id"].values, "ae": np.abs(err), "se": err ** 2}).groupby("e").mean()
    sse_base = ((ev["target"].values - base) ** 2).sum()
    summary = {"MAE": by_e["ae"].mean(), "MSE": by_e["se"].mean(), "RMSE": np.sqrt(by_e["se"]).mean(), "R2": 1 - (err ** 2).sum() / sse_base}
    return summary, by_e["ae"]