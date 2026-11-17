"""Revision pipeline for 'Tuning Without Skill?' (BEU Fen Bilimleri Dergisi, major revision).

Features, models, search spaces and walk-forward logic are taken unchanged from the
author's notebook (19.ipynb).  The only change to the search itself is that it is run by an
explicit loop over the same ParameterSampler draws that RandomizedSearchCV uses, which
reproduces RandomizedSearchCV exactly while letting us (i) count failed candidates,
(ii) record the inner-CV score of the default configuration and (iii) optionally add the
default configuration as an extra candidate.

Usage:
  python study.py run   --data-dir DATA --out OUT [--seed 42] [--n-iter 12] [--with-default]
                        [--start 2015-01-01] [--end 2026-10-12] [--years 3] [--test-len 63]
  python study.py leaky --data-dir DATA --out OUT [--end 2026-10-12] [--years 3]
  python study.py stationarity --data-dir DATA --out OUT
"""
import argparse, json, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.base import clone
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

UNIVERSE = {
    "AAPL": "Equity", "MSFT": "Equity", "NVDA": "Equity", "AMZN": "Equity",
    "JPM": "Equity", "XOM": "Equity", "JNJ": "Equity", "TSLA": "Equity",
    "SPY": "Index/ETF", "QQQ": "Index/ETF", "IWM": "Index/ETF", "EEM": "Index/ETF",
    "^GDAXI": "Index/ETF", "XU100.IS": "Index/ETF",
    "TLT": "Bond", "GLD": "Commodity", "USO": "Commodity",
    "EURUSD=X": "FX", "BTC-USD": "Crypto", "ETH-USD": "Crypto",
}

# ----------------------------------------------------------------------------- data
def load_dir(path, tickers):
    out = {}
    for t in tickers:
        for name in (t, t.replace("^", "").replace("=", "_"), t.replace("^", "")):
            f = os.path.join(path, f"{name}.csv")
            if os.path.exists(f):
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                df.columns = [str(c).title() for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()]
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_localize(None)
                out[t] = df.sort_index()
                break
    return out

def synthetic(tickers, n=780, seed=0, start="2023-10-12"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    out = {}
    for t in tickers:
        h = np.zeros(n); h[0] = 1e-4; r = np.zeros(n)
        for i in range(1, n):
            h[i] = 2e-6 + 0.08 * r[i-1] ** 2 + 0.9 * h[i-1]
            r[i] = np.sqrt(h[i]) * rng.standard_normal() + 3e-4
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, .003, n))
        hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, .004, n)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, .004, n)))
        v = rng.lognormal(15, .3, n) * (1 + 5 * np.abs(r))
        out[t] = pd.DataFrame(dict(Open=o, High=hi, Low=lo, Close=c, Volume=v), index=idx)
    return out

# ----------------------------------------------------------------------------- features (unchanged)
def make_features(df, leak_control=False, same_day=False):
    """Row t: features known at close of t-1, target = sign of r_t.
    same_day=True skips the one-day shift (deliberately leaky protocol)."""
    o, h, l, c, v = (df[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    r = np.log(c / c.shift(1))
    F = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5):
        F[f"ret_l{k}"] = r.shift(k - 1)
    for w in (5, 10, 20):
        F[f"retmean_{w}"] = r.rolling(w).mean()
    for w in (5, 10, 20):
        F[f"vol_{w}"] = r.rolling(w).std()
    F["volratio_5_20"] = F["vol_5"] / F["vol_20"]
    F["volratio_10_20"] = F["vol_10"] / F["vol_20"]
    d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / dn.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    F["rsi14"] = (100 - 100 / (1 + rs)) / 100 - 0.5
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    F["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    F["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20) - 0.5
    F["dist_sma20"] = c / m20 - 1
    F["dist_sma50"] = c / c.rolling(50).mean() - 1
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    F["stoch_k"] = (c - ll) / (hh - ll).replace(0, np.nan) - 0.5
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    F["atr14_pct"] = tr.rolling(14).mean() / c
    F["hl_range"] = (h - l) / c
    F["oc_body"] = (c - o) / o
    F["upper_shadow"] = (h - np.maximum(o, c)) / c
    F["lower_shadow"] = (np.minimum(o, c) - l) / c
    F["gap"] = np.log(o / c.shift(1))
    lv = np.log(v.replace(0, np.nan))
    F["dlogvol"] = lv.diff()
    F["relvol_20"] = lv - np.log(v.rolling(20).mean())
    if not same_day:
        F = F.shift(1)
    y = (r > 0).astype(int)
    if leak_control:
        F["LEAK_r_t"] = r
    F = F.replace([np.inf, -np.inf], np.nan)
    ok = F.notna().all(axis=1) & r.notna()
    return F[ok], y[ok], r[ok]

def raw_price_features(df):
    """Raw OHLCV levels of day t-1 (what many leaky studies feed in), plus lags."""
    F = pd.DataFrame(index=df.index)
    for k in ["Open", "High", "Low", "Close", "Volume"]:
        for lag in (1, 2, 3):
            F[f"{k}_l{lag}"] = df[k].astype(float).shift(lag)
    r = np.log(df["Close"] / df["Close"].shift(1))
    y = (r > 0).astype(int)
    ok = F.notna().all(axis=1) & r.notna()
    return F[ok], y[ok], r[ok]

# ----------------------------------------------------------------------------- models (unchanged)
def model_zoo(seed):
    return {
        "KNN": (Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier())]),
                {"m__n_neighbors": [5, 11, 21, 41, 81], "m__weights": ["uniform", "distance"],
                 "m__p": [1, 2]}),
        "RF": (RandomForestClassifier(random_state=seed, n_jobs=1),
               {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 5, 8, None],
                "min_samples_leaf": [1, 5, 10, 20], "max_features": ["sqrt", 0.5, 1.0]}),
        "GBM": (XGBClassifier(random_state=seed, n_jobs=1, verbosity=0, eval_metric="logloss"),
                {"n_estimators": [50, 100, 200, 400], "max_depth": [1, 2, 3, 4],
                 "learning_rate": [0.005, 0.01, 0.03, 0.1], "subsample": [0.6, 0.8, 1.0],
                 "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10],
                 "reg_lambda": [1, 10, 50]}),
        "ANN": (Pipeline([("sc", StandardScaler()),
                          ("m", MLPClassifier(random_state=seed, max_iter=300, early_stopping=False))]),
                {"m__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)],
                 "m__activation": ["relu", "tanh"], "m__alpha": [1e-4, 1e-2, 1e-1, 1.0],
                 "m__learning_rate_init": [1e-3, 3e-3, 1e-2]}),
    }

def inner_cv_score(est, params, X, y, cv):
    """Mean balanced accuracy over chronological inner folds; failed fits score 0 (as error_score=0.0)."""
    scores, failed = [], 0
    for tr, va in cv.split(X):
        try:
            m = clone(est).set_params(**params).fit(X.iloc[tr], y.iloc[tr])
            scores.append(balanced_accuracy_score(y.iloc[va], m.predict(X.iloc[va])))
        except Exception:
            scores.append(0.0); failed += 1
    return float(np.mean(scores)), failed

def tune(est, space, X, y, n_iter, inner_splits, gap, seed, with_default):
    """Identical to RandomizedSearchCV(n_iter, TimeSeriesSplit(gap), balanced_accuracy,
    random_state=seed, error_score=0.0, refit=True); optionally prepends the default config."""
    cv = TimeSeriesSplit(n_splits=inner_splits, gap=gap)
    cands = list(ParameterSampler(space, n_iter=n_iter, random_state=seed))
    default_score, default_failed = inner_cv_score(est, {}, X, y, cv)
    if with_default:
        cands = [{}] + cands
    res = []
    for p in cands:
        s, f = (default_score, default_failed) if p == {} else inner_cv_score(est, p, X, y, cv)
        res.append((s, f))
    scores = np.array([s for s, _ in res])
    best = int(np.argmax(scores))       # first among ties, as RandomizedSearchCV rank 'min'
    bp = cands[best]
    model = clone(est).set_params(**bp).fit(X, y)
    n_failed_cands = int(sum(1 for _, f in res if f > 0))
    n_failed_fits = int(sum(f for _, f in res))
    return model, bp, float(scores[best]), default_score, n_failed_cands, n_failed_fits, (with_default and best == 0)

def walk_forward(X, y, r, init_train, test_len, gap, n_iter, inner_splits, models, seed, with_default):
    n = len(X); zoo = model_zoo(seed); preds = []; chosen = []
    a = init_train; fold = 0
    while a + gap < n:
        te = np.arange(a + gap, min(a + gap + test_len, n)); tr = np.arange(0, a)
        Xtr, ytr = X.iloc[tr], y.iloc[tr]; Xte, yte = X.iloc[te], y.iloc[te]
        base = pd.DataFrame(dict(date=X.index[te], y=yte.values, r=r.iloc[te].values, fold=fold))
        base["pred_AlwaysUp"] = 1
        base["pred_Majority"] = int(ytr.mean() >= .5)
        base["pred_Persist"] = (X["ret_l1"].iloc[te].values > 0).astype(int)
        for m in models:
            est, space = zoo[m]
            d = clone(est).fit(Xtr, ytr)
            base[f"pred_{m}_def"] = d.predict(Xte); base[f"prob_{m}_def"] = d.predict_proba(Xte)[:, 1]
            t, bp, bs, ds, nfc, nff, picked_def = tune(est, space, Xtr, ytr, n_iter, inner_splits, gap, seed, with_default)
            base[f"pred_{m}_tuned"] = t.predict(Xte); base[f"prob_{m}_tuned"] = t.predict_proba(Xte)[:, 1]
            chosen.append(dict(fold=fold, model=m, n_train=len(tr), inner_bacc=bs, default_inner_bacc=ds,
                               outer_bacc_tuned=balanced_accuracy_score(yte, base[f"pred_{m}_tuned"]),
                               outer_bacc_def=balanced_accuracy_score(yte, base[f"pred_{m}_def"]),
                               failed_candidates=nfc, failed_fits=nff, picked_default=bool(picked_def),
                               params=json.dumps({k: (v.item() if isinstance(v, np.generic) else v) for k, v in bp.items()}, default=str)))
        preds.append(base); a += test_len; fold += 1
    return pd.concat(preds, ignore_index=True), pd.DataFrame(chosen)

# ----------------------------------------------------------------------------- run modes
def eval_window(df, end, years, start=None):
    end = pd.Timestamp(end)
    lo = pd.Timestamp(start) if start else end - pd.Timedelta(days=int(365.25 * years))
    df = df[(df.index >= lo - pd.Timedelta(days=60 if not start else 0)) & (df.index <= end)]
    X, y, r = make_features(df)
    keep = X.index >= lo
    return X[keep], y[keep], r[keep], df

def _run_asset(job):
    t, df, a = job
    t0 = time.time()
    X, y, r, _ = eval_window(df, a["end"], a["years"], a.get("start"))
    if len(X) < a["init_train"] + a["test_len"] + a["gap"] + 20:
        return t, None, None, dict(asset=t, skipped=True, n_obs=len(X))
    pr, ch = walk_forward(X, y, r, a["init_train"], a["test_len"], a["gap"], a["n_iter"], a["inner_splits"],
                          a["models"], a["seed"], a["with_default"])
    pr.insert(0, "asset", t); ch.insert(0, "asset", t)
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10, random_state=a["seed"], n_jobs=1)
    rf.fit(X.iloc[:a["init_train"]], y.iloc[:a["init_train"]])
    meta = dict(asset=t, cls=UNIVERSE.get(t, "Other"), n_obs=len(X), start=str(X.index[0].date()),
                end=str(X.index[-1].date()), up_share=float(y.mean()), n_test=len(pr),
                n_folds=int(pr["fold"].nunique()), seconds=round(time.time() - t0),
                fi=json.dumps(dict(zip(X.columns, rf.feature_importances_.round(5).tolist()))))
    print(f"{t:10s} obs={len(X):5d} test={len(pr):5d} folds={meta['n_folds']:3d} {meta['seconds']:6d}s", flush=True)
    return t, pr, ch, meta

def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    data = synthetic(a.tickers.split(","), n=a.syn_n) if a.synthetic else load_dir(a.data_dir, a.tickers.split(","))
    cfg = dict(end=a.end, years=a.years, start=a.start, init_train=a.init_train, test_len=a.test_len, gap=a.gap,
               n_iter=a.n_iter, inner_splits=a.inner_splits, models=a.models.split(","), seed=a.seed,
               with_default=a.with_default)
    json.dump(dict(cfg, data_dir=a.data_dir, synthetic=a.synthetic), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    jobs = [(t, df, cfg) for t, df in data.items()]
    with Pool(a.procs) as pool:
        res = pool.map(_run_asset, jobs, chunksize=1)
    P = pd.concat([p for _, p, _, _ in res if p is not None], ignore_index=True)
    C = pd.concat([c for _, _, c, _ in res if c is not None], ignore_index=True)
    P.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    C.to_csv(os.path.join(a.out, "chosen_params.csv"), index=False)
    pd.DataFrame([m for *_, m in res]).to_csv(os.path.join(a.out, "assets.csv"), index=False)
    print("saved", a.out)

def _leaky_asset(job):
    """Accuracy of default models under three leaky protocols vs. the honest walk-forward."""
    t, df, a = job
    _, _, _, dfw = eval_window(df, a["end"], a["years"])
    lo = pd.Timestamp(a["end"]) - pd.Timedelta(days=int(365.25 * a["years"]))
    out = []
    protos = {
        "L1_shuffledKFold_stationary": make_features(dfw),
        "L2_shuffledKFold_rawprices": raw_price_features(dfw),
        "L3_sameday_features_walkforward": make_features(dfw, same_day=True),
    }
    zoo = model_zoo(a["seed"])
    if len(protos["L1_shuffledKFold_stationary"][0]) < 300:
        return out
    honest_idx = protos["L1_shuffledKFold_stationary"][0].index
    honest_idx = honest_idx[honest_idx >= lo]
    for name, (X, y, r) in protos.items():
        keep = X.index.isin(honest_idx); X, y = X[keep], y[keep]
        for m in a["models"]:
            est = zoo[m][0]
            if name.startswith("L3"):
                accs = []; n = len(X); s = a["init_train"]
                yp = []; yt = []
                while s + 5 < n:
                    te = np.arange(s + 5, min(s + 5 + 63, n))
                    mdl = clone(est).fit(X.iloc[:s], y.iloc[:s])
                    yp += list(mdl.predict(X.iloc[te])); yt += list(y.iloc[te]); s += 63
                acc = accuracy_score(yt, yp); bacc = balanced_accuracy_score(yt, yp)
            else:
                cv = KFold(5, shuffle=True, random_state=a["seed"]); yp = np.zeros(len(y), int)
                for tr, te in cv.split(X):
                    yp[te] = clone(est).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
                acc = accuracy_score(y, yp); bacc = balanced_accuracy_score(y, yp)
            out.append(dict(asset=t, protocol=name, model=m, acc=acc, bacc=bacc, n=len(y)))
    print("leaky", t, flush=True)
    return out

def cmd_leaky(a):
    os.makedirs(a.out, exist_ok=True)
    data = synthetic(a.tickers.split(","), n=a.syn_n) if a.synthetic else load_dir(a.data_dir, a.tickers.split(","))
    cfg = dict(end=a.end, years=a.years, init_train=a.init_train, models=a.models.split(","), seed=a.seed)
    with Pool(a.procs) as pool:
        res = pool.map(_leaky_asset, [(t, df, cfg) for t, df in data.items()], chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(a.out, "leaky.csv"), index=False)

def cmd_stationarity(a):
    from statsmodels.tsa.stattools import adfuller, kpss
    os.makedirs(a.out, exist_ok=True)
    data = synthetic(a.tickers.split(","), n=a.syn_n) if a.synthetic else load_dir(a.data_dir, a.tickers.split(","))
    rows = []
    for t, df in data.items():
        X, y, r, _ = eval_window(df, a.end, a.years)
        if len(X) < 300: continue
        for col in X.columns:
            x = X[col].values
            try: p_adf = adfuller(x, autolag="AIC")[1]
            except Exception: p_adf = np.nan
            try: p_kpss = kpss(x, regression="c", nlags="auto")[1]
            except Exception: p_kpss = np.nan
            rows.append(dict(asset=t, feature=col, p_adf=p_adf, p_kpss=p_kpss))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "stationarity.csv"), index=False)
    print("stationarity rows", len(rows))

# ----------------------------------------------------------------------------- overlapping-horizon leak (added)
def _overlap_asset(job):
    """Target = sign of the cumulative return over t..t+h-1 (overlapping labels).  Same features (t-1).
    Compare shuffled 5-fold CV with walk-forward (purge = h) on identical rows."""
    t, df, a = job
    h = a["h"]
    _, _, _, dfw = eval_window(df, a["end"], a["years"])
    lo = pd.Timestamp(a["end"]) - pd.Timedelta(days=int(365.25 * a["years"]))
    X, y1, r = make_features(dfw)
    if len(X) < 300: return []
    fwd = r.rolling(h).sum().shift(-(h - 1))
    ok = fwd.notna() & (X.index >= lo)
    X, y = X[ok], (fwd[ok] > 0).astype(int)
    zoo = model_zoo(a["seed"]); out = []
    for m in a["models"]:
        est = zoo[m][0]
        cv = KFold(5, shuffle=True, random_state=a["seed"]); yp = np.zeros(len(y), int)
        for tr, te in cv.split(X):
            yp[te] = clone(est).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        out.append(dict(asset=t, protocol=f"L4_shuffledKFold_h{h}", model=m, acc=accuracy_score(y, yp),
                        bacc=balanced_accuracy_score(y, yp), n=len(y)))
        n = len(X); s = a["init_train"]; ypw, ytw = [], []
        while s + h < n:
            te = np.arange(s + h, min(s + h + 63, n))
            mdl = clone(est).fit(X.iloc[:s], y.iloc[:s])
            ypw += list(mdl.predict(X.iloc[te])); ytw += list(y.iloc[te]); s += 63
        out.append(dict(asset=t, protocol=f"L4_walkforward_h{h}", model=m, acc=accuracy_score(ytw, ypw),
                        bacc=balanced_accuracy_score(ytw, ypw), n=len(ytw), up_share=float(np.mean(ytw))))
    print("overlap", t, flush=True)
    return out

def cmd_overlap(a):
    os.makedirs(a.out, exist_ok=True)
    data = load_dir(a.data_dir, a.tickers.split(","))
    cfg = dict(end=a.end, years=a.years, init_train=a.init_train, models=a.models.split(","), seed=a.seed, h=10)
    with Pool(a.procs) as pool:
        res = pool.map(_overlap_asset, [(t, df, cfg) for t, df in data.items()], chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(a.out, "overlap.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "leaky", "stationarity", "overlap"])
    ap.add_argument("--out", required=True); ap.add_argument("--data-dir"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--syn-n", type=int, default=780)
    ap.add_argument("--tickers", default=",".join(UNIVERSE))
    ap.add_argument("--end", default="2026-10-12"); ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--init-train", type=int, default=252); ap.add_argument("--test-len", type=int, default=63)
    ap.add_argument("--gap", type=int, default=5); ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--inner-splits", type=int, default=3); ap.add_argument("--models", default="KNN,RF,GBM,ANN")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--with-default", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    {"run": cmd_run, "leaky": cmd_leaky, "stationarity": cmd_stationarity, "overlap": cmd_overlap}[a.mode](a)

