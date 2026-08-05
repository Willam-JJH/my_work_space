"""Expand US stock universe from multiple sources"""
import yfinance as yf; import pandas as pd; import numpy as np; import time, requests, re

# Try multiple sources for ticker lists
tickers = set()

# Source 1: Wikipedia SP500 via raw text (bypass 403)
print("Source 1: Wikipedia SP500...")
try:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    # Extract tickers from HTML tables
    import io
    tables = pd.read_html(io.StringIO(r.text))
    for sym in tables[0]["Symbol"]:
        tickers.add(str(sym).replace(".", "-"))
    print(f"  SP500: {len(tables[0])} tickers")
except Exception as e: print(f"  SP500 failed: {e}")

# Source 2: NASDAQ 100
print("Source 2: NASDAQ 100...")
try:
    r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        if "Ticker" in t.columns:
            for sym in t["Ticker"]:
                tickers.add(str(sym).replace(".", "-"))
    print(f"  NASDAQ100 added")
except Exception as e: print(f"  NASDAQ100 failed: {e}")

# Source 3: Russell 1000 from web
print("Source 3: Russell components...")
try:
    r = requests.get("https://en.wikipedia.org/wiki/Russell_1000_Index",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        if "Ticker" in t.columns:
            for sym in t["Ticker"]:
                tickers.add(str(sym).replace(".", "-"))
    print(f"  Russell added")
except Exception as e: print(f"  Russell failed: {e}")

# Source 4: Dow Jones
print("Source 4: Dow Jones...")
try:
    r = requests.get("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        if "Symbol" in t.columns:
            for sym in t["Symbol"]:
                tickers.add(str(sym).replace(".", "-"))
    print(f"  Dow added")
except Exception as e: print(f"  Dow failed: {e}")

# Source 5: Hardcoded additional mid/small caps (~300 more)
print("Source 5: Hardcoded additional...")
extra = """
IWM SPY QQQ DIA IJR MDY IWM VTI VOO VEA VWO BND AGG TLT SHY LQD HYG
XLE XLF XLK XLV XLY XLI XLB XLU XLP XLC XBI XRT SMH SOXX IGV IBB ITA
ARKK ARKW ARKG ARKF ARKQ JETS TAN ICLN PBW QCLN LIT DRIV FIVG
SPY QQQ IWM DIA TLT GLD SLV USO UNG DBC DBA JO JO EEM EFA FXI EWZ
RSX INDA EWG EWJ EWU EWC EWA EWH EWT EWY EWP EWI EWD EWN EWM THD TUR
PLTR CRWD DDOG SNOW MDB NET ZS DASH SQ COIN HOOD RBLX U PATH CFLT
GTLB S DV AFRM SOFI HOOD RIVN LCID DNA ESTC CFI BSY IOT
""".split()
for t in extra: tickers.add(t.strip())

tickers = sorted(set(t for t in tickers if t and len(t) <= 5 and t[0].isalpha()))
print(f"\nTotal unique tickers: {len(tickers)}")

# Download
print("Downloading...")
data = {}
for i in range(0, len(tickers), 50):
    batch = tickers[i:i+50]
    try:
        df = yf.download(batch, start="2015-01-01", end="2024-12-31", auto_adjust=True, progress=False, timeout=30)
        if "Close" in df.columns:
            for t in batch:
                if t in df["Close"].columns:
                    data[t] = df["Close"][t]
        print(f"  Batch {i//50+1}/{(len(tickers)-1)//50+1} | ok: {len(data)}")
    except Exception as e: print(f"  Batch {i//50+1} err: {str(e)[:60]}")
    time.sleep(1)

close = pd.DataFrame(data).ffill()
close = close.dropna(axis=1, thresh=int(len(close)*0.3))
returns = np.log(close/close.shift(1)).dropna(how="all")
returns.to_parquet("D:/code/data/us_expanded_returns.parquet")
print(f"\nSaved: {returns.shape[1]} stocks x {returns.shape[0]} days")
print(f"Size: {returns.memory_usage(deep=True).sum()/1e6:.1f} MB")
