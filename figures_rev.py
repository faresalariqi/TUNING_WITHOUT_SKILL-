"""New figures for the revision (style matched to the original figures)."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"KNN": "#0173b2", "RF": "#029e73", "GBM": "#d55e00", "ANN": "#cc78bc"}
MK = {"KNN": "o", "RF": "s", "GBM": "^", "ANN": "D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
                     "grid.color": "#d9d9d9", "grid.linewidth": 0.8, "axes.axisbelow": True})
RUNS = "/home/claude/r2/runs"; OUT = "/home/claude/r2/figs"

def fig_leaky():
    L = pd.concat([pd.read_csv(f"{RUNS}/leaky/leaky.csv"), pd.read_csv(f"{RUNS}/overlap/overlap.csv")])
    M = pd.read_csv("/home/claude/r2/stats_orig/metrics_by_asset.csv")
    honest = {m: M[M.method == f"{m}_def"].acc.mean() for m in COL}
    order = [("honest", "Walk-forward\n(main protocol)"),
             ("L1_shuffledKFold_stationary", "Shuffled 5-fold,\nscale-free features"),
             ("L2_shuffledKFold_rawprices", "Shuffled 5-fold,\nraw prices"),
             ("L3_sameday_features_walkforward", "Walk-forward,\nsame-day features"),
             ("L4_walkforward_h10", "Walk-forward,\n10-day target"),
             ("L4_shuffledKFold_h10", "Shuffled 5-fold,\n10-day target")]
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    w = 0.19
    for k, (p, lab) in enumerate(order):
        for j, m in enumerate(COL):
            v = honest[m] if p == "honest" else L[(L.protocol == p) & (L.model == m)].acc.mean()
            ax.bar(k + (j - 1.5) * w, 100 * v, w * 0.95, color=COL[m], label=m if k == 0 else None)
    ax.axhline(50, color="#444", ls=":", lw=1.2)
    up10 = 100 * L[L.protocol == "L4_walkforward_h10"].groupby("asset").up_share.first().mean()
    ax.hlines(53.02, -0.5, 3.5, color="#222", ls="--", lw=1.2)
    ax.text(1.5, 53.6, "always up, 1-day target", ha="center", va="bottom", fontsize=9.5, color="#222")
    ax.hlines(up10, 3.5, 5.5, color="#222", ls="--", lw=1.2)
    ax.text(4.0, up10 + 0.6, "always up, 10-day target", ha="center", va="bottom", fontsize=9.5, color="#222")
    ax.axvspan(3.5, 5.5, color="#f3f3f3", zorder=0)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=9.5)
    ax.set_ylabel("Mean accuracy across 19 assets (%)"); ax.set_ylim(40, 102)
    ax.set_xlim(-0.5, 5.5)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_leaky.png", dpi=300); plt.close(fig)

def fig_extended():
    D = pd.read_csv("/home/claude/r2/stats_ext/reg/regimes_by_asset.csv")
    Y = D[D.split == "year"]; Y = Y[Y.level.astype(int) >= 2016]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    years = sorted(Y.level.astype(int).unique())
    a_up = [100 * Y[(Y.method == "AlwaysUp") & (Y.level.astype(int) == y)].acc.mean() for y in years]
    ax.plot(years, a_up, color="#222", ls="--", lw=1.6, label="always up")
    for m in COL:
        a = [100 * Y[(Y.method == f"{m}_tuned") & (Y.level.astype(int) == y)].acc.mean() for y in years]
        ax.plot(years, a, color=COL[m], marker=MK[m], lw=1.3, ms=5, label=f"{m} tuned")
    ax.axhline(50, color="#444", ls=":", lw=1.2)
    ax.set_xticks(years); ax.set_xticklabels([str(y) for y in years], fontsize=9.5)
    ax.set_ylabel("Mean accuracy across assets (%)"); ax.set_title("(a) Accuracy by calendar year", fontsize=11, loc="left")
    ax = axes[1]
    R = D[D.split == "regime"]; w = 0.16
    for j, m in enumerate(["AlwaysUp"] + [f"{x}_tuned" for x in COL]):
        vals = [100 * R[(R.method == m) & (R.level == l)].acc.mean() for l in ["up", "down"]]
        c = "#222" if m == "AlwaysUp" else COL[m.split("_")[0]]
        ax.bar(np.arange(2) + (j - 2) * w, vals, w * 0.95, color=c)
    ax.axhline(50, color="#444", ls=":", lw=1.2)
    ax.set_ylim(45, 57); ax.set_xticks([0, 1]); ax.set_xticklabels(["up quarters", "down quarters"], fontsize=9.5)
    ax.set_ylabel("Mean accuracy (%)"); ax.set_title("(b) Accuracy by market regime", fontsize=11, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, frameon=False, loc="lower center", fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig6_extended.png", dpi=300); plt.close(fig)

if __name__ == "__main__":
    import os; os.makedirs(OUT, exist_ok=True)
    for f in sys.argv[1:]: globals()[f]()
