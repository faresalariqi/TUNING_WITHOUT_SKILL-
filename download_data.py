# Run this in ONE Google Colab cell. It downloads the daily data used in the study and the
# longer history needed for the revision (2015-01-01 to 2026-10-12), then downloads a zip.
# pip install yfinance

import os, zipfile, pandas as pd, yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "JNJ", "TSLA",
           "SPY", "QQQ", "IWM", "EEM", "^GDAXI", "XU100.IS",
           "TLT", "GLD", "USO", "EURUSD=X", "BTC-USD", "ETH-USD"]

os.makedirs("raw", exist_ok=True)
for t in tickers:
    df = yf.download(t, start="2015-01-01", end="2026-10-13", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(f"raw/{t.replace('^', '').replace('=', '_')}.csv")
    print(f"{t:10s} {len(df):5d} rows  {df.index[0].date()} -> {df.index[-1].date()}")

with zipfile.ZipFile("price_data.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir("raw")):
        z.write(os.path.join("raw", f), f)



