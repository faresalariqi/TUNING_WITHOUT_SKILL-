"""Statistics for the revision.  Reads run folders written by study.py and writes JSON/CSV summaries.

  python stats_rev.py main  RUN_DIR OUT_DIR          # all tables/tests for one run
  python stats_rev.py seeds OUT_DIR RUN_DIR [RUN_DIR ...]
  python stats_rev.py regimes RUN_DIR OUT_DIR       # extended-sample down/up-market breakdown
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score

MODELS = ["KNN", "RF", "GBM", "ANN"]
RULES = ["AlwaysUp", "Persist"] + [f"{m}_{k}" for m in MODELS for k in ("def", "tuned")]
ACTIVE = ["Persist"] + [f"{m}_{k}" for m in MODELS for k in ("def", "tuned")]   # 9 non-trivial rules
CRYPTO = {"BTC-USD", "ETH-USD"}
RNG_SEED = 0

# ---------------------------------------------------------------- basic metrics
def mcc(y, p):
    return matthews_corrcoef(y, p) if len(set(p)) > 1 and len(set(y)) > 1 else 0.0

def strategy(g, col, cost_bps=5.0, ann=252):
    """Position decided from information at the close of t-1 and held from close t-1 to close t:
    +1 if the rule predicts 'up', -1 otherwise.  Turnover_t = |pos_t - pos_{t-1}| (entry = 1 unit,
    long<->short reversal = 2 units); net_t = pos_t * r_t - turnover_t * cost."""
    pos = np.where(g[col].values == 1, 1.0, -1.0)
    turn = np.abs(np.diff(np.r_[0.0, pos]))
    gross = pos * g["r"].values
    net = gross - turn * cost_bps * 1e-4
    sh = lambda x: np.sqrt(ann) * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    return sh(gross), sh(net), turn.mean()

def per_asset_metrics(P):
    recs = []
    for t, g in P.groupby("asset"):
        g = g.sort_values("date")
        ann_cal = 365 if t in CRYPTO else 252
        for m in RULES:
            p = g[f"pred_{m}"].values; y = g["y"].values
            prob = g[f"prob_{m}"].values if f"prob_{m}" in g else p.astype(float)
            try: auc = roc_auc_score(y, prob)
            except Exception: auc = np.nan
            sg, sn, to = strategy(g, f"pred_{m}")
            sg_c, sn_c, _ = strategy(g, f"pred_{m}", ann=ann_cal)
            recs.append(dict(asset=t, method=m, n=len(g), acc=accuracy_score(y, p), bacc=balanced_accuracy_score(y, p),
                             mcc=mcc(y, p), auc=auc, sharpe_gross=sg, sharpe_net=sn, sharpe_gross_cal=sg_c,
                             sharpe_net_cal=sn_c, turnover=to, up_pred=p.mean(), up_actual=y.mean()))
    return pd.DataFrame(recs)

# ---------------------------------------------------------------- tests
def hac_var(d, lag=None):
    d = np.asarray(d, float) - np.mean(d); n = len(d)
    if lag is None: lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = d @ d / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1); v += 2 * w * (d[k:] @ d[:-k]) / n
    return v

def dm_test(loss_a, loss_b):
    """Diebold-Mariano with Newey-West HAC variance; d = loss_a - loss_b. Returns (mean d, t, two-sided p)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float); n = len(d)
    v = hac_var(d)
    if v <= 0: return d.mean(), 0.0, 1.0
    tstat = d.mean() / np.sqrt(v / n)
    return d.mean(), tstat, 2 * stats.norm.sf(abs(tstat))

def pt_test(y, p):
    """Pesaran-Timmermann (1992). H0: predicted and realised signs independent. One-sided p (positive)."""
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    py, pp = y.mean(), p.mean(); ph = (y == p).mean()
    pstar = py * pp + (1 - py) * (1 - pp)
    vph = pstar * (1 - pstar) / n
    vps = ((2 * py - 1) ** 2 * pp * (1 - pp) / n + (2 * pp - 1) ** 2 * py * (1 - py) / n
           + 4 * py * pp * (1 - py) * (1 - pp) / n ** 2)
    if vph - vps <= 0: return 0.0, 1.0
    s = (ph - pstar) / np.sqrt(vph - vps)
    return s, stats.norm.sf(s)

def phi_test(y, p):
    """Original manuscript test: n*phi^2 ~ chi2_1 under no association, one-sided for phi>0."""
    phi = mcc(y, p); n = len(y); z = np.sqrt(n) * phi
    return phi, stats.norm.sf(z)

def circular_shift_test(y, p, B=999, minshift=21, seed=RNG_SEED):
    """Dependence-aware randomisation test of MCC: predictions are circularly shifted relative to
    outcomes, which keeps the serial dependence of both series but breaks any alignment.  One-sided."""
    rng = np.random.default_rng(seed); n = len(y); obs = mcc(y, p)
    if n < 3 * minshift: return obs, np.nan
    shifts = rng.integers(minshift, n - minshift, B)
    null = np.array([mcc(y, np.roll(p, s)) for s in shifts])
    return obs, (1 + np.sum(null >= obs)) / (B + 1)

def circular_shift_exact(y, p, minshift=21):
    """Exact version: enumerate every admissible rotation. With fixed margins MCC is monotone in the
    number of true positives, so TP(s) for all shifts comes from one circular cross-correlation (FFT)."""
    y = np.asarray(y, float); p = np.asarray(p, float); n = len(y)
    if n < 3 * minshift or p.std() == 0: return mcc(y.astype(int), p.astype(int)), np.nan, np.nan
    tp = np.rint(np.real(np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(p)))))   # tp[s] = sum_t y_t p_{t-s}
    obs = tp[0]; null = tp[minshift:n - minshift + 1]
    pval = (1 + np.sum(null >= obs)) / (1 + len(null))
    return mcc(y.astype(int), p.astype(int)), pval, 1 / (1 + len(null))

def hac_assoc_test(y, p):
    """Regression y_t = a + b*p_t + e_t, one-sided test of b > 0 with Newey-West HAC standard error.
    Under H0 (independence of forecast and outcome) b = 0; robust to serial dependence."""
    y = np.asarray(y, float); p = np.asarray(p, float); n = len(y)
    if p.std() == 0: return 0.0, 1.0
    x = p - p.mean(); b = (x @ (y - y.mean())) / (x @ x)
    e = (y - y.mean()) - b * x
    u = x * e
    se = np.sqrt(hac_var(u) * n) / (x @ x)
    tstat = b / se
    return tstat, stats.norm.sf(tstat)

def holm(pvals):
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * p[k]); adj[k] = min(1.0, run)
    return adj

# ---------------------------------------------------------------- bootstrap
def mbb_indices(n, block, rng):
    nb = int(np.ceil(n / block)); starts = rng.integers(0, n - block + 1, nb)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]

def cluster_bootstrap(P, func, B=600, block=20, seed=RNG_SEED):
    """Resample assets with replacement, then 20-day moving blocks within each asset (as in the
    manuscript).  func(sub_df) -> scalar or dict computed per asset; we average over sampled assets."""
    rng = np.random.default_rng(seed)
    groups = {t: g.sort_values("date").reset_index(drop=True) for t, g in P.groupby("asset")}
    names = list(groups); out = []
    for _ in range(B):
        vals = []
        for t in rng.choice(names, len(names), replace=True):
            g = groups[t]; idx = mbb_indices(len(g), block, rng)
            vals.append(func(g.iloc[idx]))
        out.append(pd.DataFrame(vals).mean() if isinstance(vals[0], dict) else np.mean(vals))
    return pd.DataFrame(out) if isinstance(out[0], pd.Series) else np.array(out)

# ---------------------------------------------------------------- main analysis
def analyse(run_dir, out_dir, B=600, light=False):
    os.makedirs(out_dir, exist_ok=True)
    P = pd.read_csv(os.path.join(run_dir, "predictions.csv"), parse_dates=["date"])
    C = pd.read_csv(os.path.join(run_dir, "chosen_params.csv"))
    A = pd.read_csv(os.path.join(run_dir, "assets.csv"))
    R = {}
    M = per_asset_metrics(P); M.to_csv(os.path.join(out_dir, "metrics_by_asset.csv"), index=False)
    T4 = M.groupby("method")[["acc", "bacc", "mcc", "auc", "sharpe_gross", "sharpe_net", "sharpe_gross_cal",
                              "sharpe_net_cal", "turnover", "up_pred"]].mean().loc[RULES]
    up = M[M.method == "AlwaysUp"].set_index("asset")
    T4["n_acc_gt_up"] = [int((M[M.method == m].set_index("asset").acc > up.acc).sum()) for m in RULES]
    T4["n_sharpe_gt_up"] = [int((M[M.method == m].set_index("asset").sharpe_net > up.sharpe_net).sum()) for m in RULES]
    T4["n_sharpe_cal_gt_up"] = [int((M[M.method == m].set_index("asset").sharpe_net_cal > up.sharpe_net_cal).sum()) for m in RULES]
    R["n_assets"] = int(P.asset.nunique()); R["n_oos_per_model"] = int(len(P))
    R["oos_span"] = [str(P.date.min().date()), str(P.date.max().date())]
    R["up_share_oos"] = float(P.y.mean())
    R["oos_per_asset"] = P.groupby("asset").size().to_dict()

    # --- bootstrap: CI of mean accuracy and of paired difference vs AlwaysUp
    def f_acc(g):
        d = {f"acc_{m}": (g[f"pred_{m}"].values == g.y.values).mean() for m in RULES}
        for m in ACTIVE:
            d[f"dup_{m}"] = d[f"acc_{m}"] - d["acc_AlwaysUp"]
        for mm in MODELS:
            d[f"dbacc_{mm}"] = (balanced_accuracy_score(g.y, g[f"pred_{mm}_tuned"]) -
                                balanced_accuracy_score(g.y, g[f"pred_{mm}_def"])) if g.y.nunique() > 1 else 0.0
        return d
    BS = cluster_bootstrap(P, f_acc, B=B)
    ci = lambda s, q=(2.5, 97.5): [float(np.percentile(s, q[0])), float(np.percentile(s, q[1]))]
    T4["acc_ci_lo"] = [ci(BS[f"acc_{m}"])[0] for m in RULES]; T4["acc_ci_hi"] = [ci(BS[f"acc_{m}"])[1] for m in RULES]
    vs_up = []
    for m in ACTIVE:
        s = BS[f"dup_{m}"]; obs = float((M[M.method == m].set_index("asset").acc - up.acc).mean())
        vs_up.append(dict(method=m, mean_diff_pp=100 * obs, ci_lo_pp=100 * ci(s)[0], ci_hi_pp=100 * ci(s)[1],
                          p_boot_one_sided=float((np.sum(s >= 0) + 1) / (len(s) + 1))))
    vs_up = pd.DataFrame(vs_up); vs_up.to_csv(os.path.join(out_dir, "vs_alwaysup_paired.csv"), index=False)
    T4.to_csv(os.path.join(out_dir, "table4.csv"))

    # --- per-asset tests (171)
    tests = []
    for t, g in P.groupby("asset"):
        g = g.sort_values("date"); y = g.y.values
        for m in ACTIVE:
            p = g[f"pred_{m}"].values
            phi, p_phi = phi_test(y, p); s_pt, p_pt = pt_test(y, p)
            _, p_cs, p_floor = circular_shift_exact(y, p)
            t_hac, p_hac = hac_assoc_test(y, p)
            dm_d, dm_t, dm_p = dm_test((p != y).astype(float), (1 != y).astype(float))  # loss model - loss up
            tests.append(dict(asset=t, method=m, mcc=phi, p_phi=p_phi, pt_stat=s_pt, p_pt=p_pt, p_circ=p_cs, p_circ_floor=p_floor,
                              t_hac=t_hac, p_hac=p_hac,
                              acc_minus_up=-dm_d, dm_t=-dm_t, p_dm_two=dm_p,
                              p_dm_better=stats.norm.sf(-dm_t), p_dm_worse=stats.norm.cdf(-dm_t)))
    TT = pd.DataFrame(tests)
    for col in ["p_phi", "p_pt", "p_circ", "p_hac", "p_dm_better", "p_dm_worse"]:
        if TT[col].notna().all(): TT[col + "_holm"] = holm(TT[col].values)
    TT.to_csv(os.path.join(out_dir, "per_asset_tests.csv"), index=False)
    R["tests"] = {c: dict(n=int(len(TT)), nominal=int((TT[c] < .05).sum()),
                          holm=int((TT[c + "_holm"] < .05).sum()) if c + "_holm" in TT else None,
                          min_holm=float(TT[c + "_holm"].min()) if c + "_holm" in TT else None)
                  for c in ["p_phi", "p_pt", "p_circ", "p_hac", "p_dm_better", "p_dm_worse"] if TT[c].notna().all()}
    R["nominal_phi_list"] = TT[TT.p_phi < .05][["asset", "method", "mcc", "p_phi"]].to_dict("records")
    R["dm_worse_by_method"] = TT.groupby("method").apply(lambda g: int((g.p_dm_worse < .05).sum())).to_dict()

    # --- tuned vs default
    tv = []
    for mm in MODELS:
        d = (M[M.method == f"{mm}_tuned"].set_index("asset").bacc - M[M.method == f"{mm}_def"].set_index("asset").bacc) * 100
        dm = (M[M.method == f"{mm}_tuned"].set_index("asset").mcc - M[M.method == f"{mm}_def"].set_index("asset").mcc)
        s = BS[f"dbacc_{mm}"] * 100
        try: w = stats.wilcoxon(d.values).pvalue
        except Exception: w = np.nan
        sd = d.std(ddof=1); n = len(d)
        mde = (stats.t.ppf(.975, n - 1) + stats.t.ppf(.80, n - 1)) * sd / np.sqrt(n)
        lo90, hi90 = ci(s, (5, 95)); lo95, hi95 = ci(s)
        # paired forecast comparison per asset (DM on 0/1 loss, tuned vs default)
        dmp = []
        for t, g in P.groupby("asset"):
            g = g.sort_values("date")
            _, tt, pp = dm_test((g[f"pred_{mm}_tuned"] != g.y).astype(float), (g[f"pred_{mm}_def"] != g.y).astype(float))
            dmp.append(pp)
        tv.append(dict(model=mm, mean_d=d.mean(), median_d=d.median(), wins=int((d > 0).sum()), n=n, wilcoxon_p=w,
                       ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90,
                       tost_equiv_1pp=bool(lo90 > -1 and hi90 < 1), tost_equiv_2pp=bool(lo90 > -2 and hi90 < 2),
                       mde_pp=mde, d_min=d.min(), d_max=d.max(), mean_dmcc=dm.mean(),
                       dm_nominal=int(np.sum(np.array(dmp) < .05)), dm_holm=int(np.sum(holm(dmp) < .05))))
    TV = pd.DataFrame(tv)

    # --- inner vs outer (selected config) and the same gap for the unselected default config
    g1 = C.groupby("model").agg(inner=("inner_bacc", "mean"), outer=("outer_bacc_tuned", "mean"),
                                def_inner=("default_inner_bacc", "mean"), def_outer=("outer_bacc_def", "mean"),
                                n_sel=("inner_bacc", "size"), failed_cands=("failed_candidates", "sum"),
                                folds_with_fail=("failed_candidates", lambda x: int((x > 0).sum())),
                                picked_default=("picked_default", "sum"))
    g1["gap_tuned_pp"] = 100 * (g1.inner - g1.outer); g1["gap_def_pp"] = 100 * (g1.def_inner - g1.def_outer)
    g1["selection_optimism_pp"] = g1.gap_tuned_pp - g1.gap_def_pp
    g1["corr_inner_outer"] = C.groupby("model").apply(lambda g: g.inner_bacc.corr(g.outer_bacc_tuned))
    # bootstrap CI (over assets) of the selection optimism
    rng = np.random.default_rng(RNG_SEED); assets = C.asset.unique(); so = {m: [] for m in MODELS}
    per = C.assign(gap_t=C.inner_bacc - C.outer_bacc_tuned, gap_d=C.default_inner_bacc - C.outer_bacc_def)
    for _ in range(B):
        samp = pd.concat([per[per.asset == a] for a in rng.choice(assets, len(assets))])
        for m, g in samp.groupby("model"): so[m].append(100 * (g.gap_t.mean() - g.gap_d.mean()))
    g1["sel_opt_ci_lo"] = [np.percentile(so[m], 2.5) for m in g1.index]
    g1["sel_opt_ci_hi"] = [np.percentile(so[m], 97.5) for m in g1.index]
    TV = TV.merge(g1.reset_index().rename(columns={"model": "model"}), on="model")
    TV.to_csv(os.path.join(out_dir, "table5_tuning.csv"), index=False)

    # --- ex-post winner, prediction bias, chosen params, importances
    var = [f"{m}_{k}" for m in MODELS for k in ("def", "tuned")]
    W = M[M.method.isin(var)].pivot(index="asset", columns="method", values="bacc")
    R["expost_best_bacc"] = float(W.max(axis=1).mean()); R["avg_variant_bacc"] = float(W.mean(axis=1).mean())
    R["up_pred_share"] = M.groupby("method").up_pred.mean().to_dict()
    sel = {}
    for m, g in C.groupby("model"):
        dl = [json.loads(x) for x in g.params]; keys = sorted({k for d in dl for k in d})
        sel[m] = {c: pd.Series([str(d.get(c)) for d in dl]).value_counts(normalize=True).round(3).to_dict() for c in keys}
    R["selected_params"] = sel
    fi = pd.DataFrame([json.loads(x) for x in A.dropna(subset=["fi"]).fi]).mean()
    R["fi_min"] = [fi.idxmin(), float(fi.min())]; R["fi_max"] = [fi.idxmax(), float(fi.max())]; R["fi_uniform"] = 1 / len(fi)
    R["class_mcc"] = M[M.method.isin(var)].merge(A[["asset", "cls"]], on="asset").groupby("cls").mcc.mean().to_dict()
    R["mcc_range"] = [float(M[M.method.isin(var)].mcc.min()), float(M[M.method.isin(var)].mcc.max())]
    R["pos_mcc_counts"] = M[M.method.isin(var)].groupby("method").apply(lambda g: int((g.mcc > 0).sum())).to_dict()
    json.dump(R, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    return T4, vs_up, TV, R

def seeds(out_dir, run_dirs):
    rows = []
    for rd in run_dirs:
        cfg = json.load(open(os.path.join(rd, "config.json")))
        P = pd.read_csv(os.path.join(rd, "predictions.csv"), parse_dates=["date"])
        M = per_asset_metrics(P)
        for mth, g in M.groupby("method"):
            rows.append(dict(seed=cfg["seed"], method=mth, acc=g.acc.mean(), bacc=g.bacc.mean(), mcc=g.mcc.mean()))
        for mm in MODELS:
            d = (M[M.method == f"{mm}_tuned"].set_index("asset").bacc - M[M.method == f"{mm}_def"].set_index("asset").bacc) * 100
            rows.append(dict(seed=cfg["seed"], method=f"delta_{mm}", acc=np.nan, bacc=d.mean(), mcc=np.nan,
                             wilcoxon_p=stats.wilcoxon(d.values).pvalue))
    S = pd.DataFrame(rows); os.makedirs(out_dir, exist_ok=True); S.to_csv(os.path.join(out_dir, "seeds.csv"), index=False)
    return S

def regimes(run_dir, out_dir):
    """Extended sample: metrics by calendar year and by up/down market quarter (asset's own quarterly return)."""
    os.makedirs(out_dir, exist_ok=True)
    P = pd.read_csv(os.path.join(run_dir, "predictions.csv"), parse_dates=["date"])
    P["year"] = P.date.dt.year; P["q"] = P.date.dt.to_period("Q")
    qret = P.groupby(["asset", "q"]).r.sum().rename("qret")
    P = P.join(qret, on=["asset", "q"]); P["regime"] = np.where(P.qret < 0, "down", "up")
    rows = []
    for key in ["year", "regime"]:
        for (t, k), g in P.groupby(["asset", key]):
            if len(g) < 20: continue
            for m in RULES:
                p = g[f"pred_{m}"].values
                rows.append(dict(split=key, level=str(k), asset=t, method=m, n=len(g), acc=(p == g.y.values).mean(),
                                 bacc=balanced_accuracy_score(g.y, p) if g.y.nunique() > 1 else np.nan,
                                 mcc=mcc(g.y.values, p), up_actual=g.y.mean()))
    D = pd.DataFrame(rows); D.to_csv(os.path.join(out_dir, "regimes_by_asset.csv"), index=False)
    S = D.groupby(["split", "level", "method"])[["acc", "bacc", "mcc", "up_actual"]].mean().reset_index()
    S.to_csv(os.path.join(out_dir, "regimes_summary.csv"), index=False)
    return S

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "main": analyse(sys.argv[2], sys.argv[3], light="--light" in sys.argv)
    elif cmd == "seeds": seeds(sys.argv[2], sys.argv[3:])
    elif cmd == "regimes": regimes(sys.argv[2], sys.argv[3])
