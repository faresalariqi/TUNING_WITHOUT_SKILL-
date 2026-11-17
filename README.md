# Tuning Without Skill? — code and data

Replication package for *“Tuning Without Skill? Leakage-Free Walk-Forward Evidence on Hyperparameter
Optimization for Daily Directional Forecasting across 19 Assets”* (BEU Fen Bilimleri Dergisi, revised version).

## Contents

| Path | What it is |
|---|---|
| `wf_pipeline.ipynb` | Original notebook that produced the main results (Colab, October 2026) |
| `study.py` | Revision pipeline: same features, models, search spaces and walk-forward logic as the notebook, plus the robustness runs (`run`, `leaky`, `overlap`, `stationarity`) |
| `stats_rev.py` | All statistics: Table 4–5 metrics, bootstrap intervals, paired tests vs always-up, Pesaran–Timmermann, circular-shift test, Diebold–Mariano, TOST, MDE, selection-optimism decomposition, regime breakdown |
| `figures_rev.py` | Figures 6 and 7 |
| `download_data.py` | Downloads the price data from Yahoo Finance |
| `data/raw/*.csv` | Price-data snapshot used in the paper (daily OHLCV, adjusted, 2015-01-01 to 2026-10-12) |
| `results/main/` | Original out-of-sample predictions and selected hyperparameters (main run) |
| `results/*/` | Prediction files of every robustness run |

## Reproducing the results

```bash
pip install -r requirements.txt
# main protocol (Oct 2023 - Oct 2026), seed 42, 12 candidates
python study.py run --data-dir data/raw --out results/rerun_s42 --seed 42 --end 2026-10-12
python stats_rev.py main results/main stats/main
# robustness
for s in 1 2 3 4; do python study.py run --data-dir data/raw --out results/seed_$s --seed $s --end 2026-10-12; done
python study.py run --data-dir data/raw --out results/budget48_def --n-iter 48 --with-default --end 2026-10-12
python study.py run --data-dir data/raw --out results/extended --start 2015-01-01 --end 2026-10-12 --test-len 252
python study.py leaky --data-dir data/raw --out results/leaky --end 2026-10-12
python study.py overlap --data-dir data/raw --out results/overlap --end 2026-10-12
python study.py stationarity --data-dir data/raw --out results/stationarity --end 2026-10-12
python stats_rev.py seeds stats/seeds results/rerun_s42 results/seed_1 results/seed_2 results/seed_3 results/seed_4
python stats_rev.py regimes results/extended stats/extended
```

The US-listed assets in the original download end on 2026-10-09 (the download was made on 2026-10-12, before
that day's close). To reproduce the main run exactly, truncate them at that date.

## Software

Original run: Google Colab, October 2026. Re-run and robustness analyses: Python 3.13, scikit-learn 1.9.1,
XGBoost 3.4.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, statsmodels 0.15.0. The re-run reproduces 100% of the
original KNN, persistence and always-up predictions and 95–99% of the RF, GBM and ANN predictions; the
differences come from library versions.
