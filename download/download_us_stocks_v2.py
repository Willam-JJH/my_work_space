"""
Download as many US stocks as possible, targeting 3000+ tickers.
Saves price and volume data to parquet files.
Optimized: pre-filter tickers, batch download, suppress noise.
"""
import warnings
warnings.filterwarnings("ignore")
import os, time, sys, io, contextlib
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request

print("=" * 70)
print("US STOCK UNIVERSE EXPANDER - TARGET 3000+ TICKERS")
print("=" * 70)

OUTPUT_DIR = r"D:/code/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_PARQUET = os.path.join(OUTPUT_DIR, "us_price_expanded.parquet")
VOLUME_PARQUET = os.path.join(OUTPUT_DIR, "us_volume_expanded.parquet")

START = "2000-01-01"
END   = "2024-12-31"
BATCH_SIZE = 50
SLEEP_SEC = 0.5

# ---------- Wikipedia scraper ----------
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"

def wiki_tickers(url, idx=0, cols=None):
    try:
        tables = pd.read_html(url, storage_options={"headers": {"User-Agent": UA}})
        if idx < len(tables):
            df = tables[idx]
            for c in (cols or ["Symbol","Ticker","Ticker symbol","Symbols"]):
                if c in df.columns:
                    raw = df[c].dropna().astype(str).str.strip().str.upper()
                    return {t.split(".")[0] for t in raw if t != "nan"}
    except Exception as e:
        print(f"  [WARN] Wiki failed {url}: {e}")
    return set()

def get_sp500():
    print("\n[1] S&P 500 from Wikipedia...")
    t = wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    print(f"  -> {len(t)} tickers"); return t

def get_nasdaq100():
    print("\n[2] NASDAQ-100 from Wikipedia...")
    t = wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100")
    if len(t) < 50:
        t2 = wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100#Components")
        t |= t2
    print(f"  -> {len(t)} tickers"); return t

# ---------- NASDAQ Trader (filtered) ----------
def get_nasdaqtrader_filtered():
    """Fetch all listed stocks, filter out warrants/units/rights/ETFs."""
    print("\n[3] NASDAQ/NYSE listed stocks (filtered)...")
    tickers = set()
    for name, url in [("NASDAQ","ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"),
                       ("NYSE","ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt")]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            for line in text.splitlines():
                if "|" not in line or line.startswith("Symbol"): continue
                parts = line.split("|")
                sym = parts[0].strip().upper()
                # Skip if it has non-alpha chars
                if not sym.isalpha(): continue
                # Skip warrants (W), units (U), rights (R), preferred (P) suffixes
                if sym.endswith("W") or sym.endswith("U") or sym.endswith("R"):
                    if len(sym) >= 4:
                        continue
                # Skip test symbols
                if sym.startswith("TEST") or sym.startswith("Z"): continue
                tickers.add(sym)
            print(f"  -> {len(tickers)} from {name}")
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")
            try:
                http_url = url.replace("ftp://","https://")
                req = urllib.request.Request(http_url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8")
                for line in text.splitlines():
                    if "|" not in line or line.startswith("Symbol"): continue
                    parts = line.split("|")
                    sym = parts[0].strip().upper()
                    if not sym.isalpha(): continue
                    if sym.endswith("W") or sym.endswith("U") or sym.endswith("R"):
                        if len(sym) >= 4: continue
                    if sym.startswith("TEST") or sym.startswith("Z"): continue
                    tickers.add(sym)
                print(f"  -> {len(tickers)} from {name} (HTTP)")
            except Exception as e2:
                print(f"  [WARN] {name} HTTP also failed: {e2}")
    return tickers

# ---------- Hardcoded tickers from file ----------
def load_hardcoded_tickers(filepath):
    """Load tickers from a text file (one per line)."""
    print(f"\n[4] Loading hardcoded tickers from {filepath}...")
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return set()
    with open(filepath) as f:
        tickers = {line.strip().upper() for line in f if line.strip() and not line.startswith("#")}
    print(f"  -> {len(tickers)} tickers")
    return tickers

# ---------- Collect and clean ----------
def clean_tickers(raw_set):
    cleaned = set()
    exclude = {"", "A","AA","AAA","AAAA","ETF","FUND","INC","CORP","LTD","LLC",
               "LP","GP","TRUST","UNIT","NOTE","DUE","SHS","CL","COM","SER",
               "WTS","WT","RT","WI","PR","CV","TEST"}
    for t in raw_set:
        t = t.strip().upper()
        if "." in t: t = t.split(".")[0]
        if not t: continue
        if t in exclude: continue
        if len(t) > 5: continue
        if not t.isascii(): continue
        if not t.replace("-","").isalpha(): continue
        cleaned.add(t)
    return sorted(cleaned)

def collect_all():
    all_t = set()
    for name, fn in [("S&P 500", get_sp500), ("NASDAQ-100", get_nasdaq100),
                     ("NASDAQ Trader", get_nasdaqtrader_filtered)]:
        try:
            all_t |= fn()
            print(f"  Union: {len(all_t)}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    # Add hardcoded
    hard_path = os.path.join(OUTPUT_DIR, "us_stock_list.txt")
    all_t |= load_hardcoded_tickers(hard_path)
    return clean_tickers(all_t)

# ---------- Download ----------
class NullWriter(io.StringIO):
    def write(self, s): pass

def download_batch(tickers_batch):
    for attempt in range(2):
        try:
            with contextlib.redirect_stdout(NullWriter()):
                data = yf.download(
                    tickers_batch, start=START, end=END,
                    auto_adjust=True, progress=False, actions=False,
                    group_by="ticker", threads=True,
                )
            return data
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                return None
    return None

def download_all(tickers):
    total = len(tickers)
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {total} TICKERS IN {len(batches)} BATCHES OF {BATCH_SIZE}")
    print(f"{'='*70}")

    price_dfs, volume_dfs = [], []
    success, fail = 0, 0

    for idx, batch in enumerate(batches):
        print(f"\r  Batch {idx+1}/{len(batches)} ({idx*BATCH_SIZE+1}-{min((idx+1)*BATCH_SIZE,total)}/{total})...", end="")
        data = download_batch(list(batch))
        if data is None:
            fail += len(batch)
            time.sleep(SLEEP_SEC)
            continue

        for ticker in batch:
            ts = str(ticker)
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    l0 = data.columns.get_level_values(0)
                    l1 = data.columns.get_level_values(1)
                    if "Close" in l0 and "Volume" in l0:
                        # (Price, Ticker)
                        if (("Close", ts) in data.columns and ("Volume", ts) in data.columns):
                            p = data[("Close", ts)].dropna()
                            v = data[("Volume", ts)].dropna()
                        else:
                            fail += 1; continue
                    elif ts in l0:
                        if ((ts, "Close") in data.columns and (ts, "Volume") in data.columns):
                            p = data[(ts, "Close")].dropna()
                            v = data[(ts, "Volume")].dropna()
                        else:
                            fail += 1; continue
                    else:
                        fail += 1; continue
                else:
                    if "Close" in data.columns and "Volume" in data.columns:
                        p = data["Close"].dropna()
                        v = data["Volume"].dropna()
                    else:
                        fail += 1; continue
                if len(p) > 0:
                    p.name = ts; price_dfs.append(p)
                if len(v) > 0:
                    v.name = ts; volume_dfs.append(v)
                success += 1
            except Exception:
                fail += 1

        time.sleep(SLEEP_SEC)

    print()
    print(f"  Success: {success}, Failed: {fail}")

    print("\nMerging...")
    price_df = pd.concat(price_dfs, axis=1, join="outer") if price_dfs else pd.DataFrame()
    volume_df = pd.concat(volume_dfs, axis=1, join="outer") if volume_dfs else pd.DataFrame()

    if not price_df.empty:
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index().loc["2000-01-01":"2024-12-31"]
        price_df = price_df[sorted(price_df.columns)]
    if not volume_df.empty:
        volume_df.index = pd.to_datetime(volume_df.index)
        volume_df = volume_df.sort_index().loc["2000-01-01":"2024-12-31"]
        volume_df = volume_df[sorted(volume_df.columns)]

    return price_df, volume_df, success, fail

# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("COLLECTING TICKERS")
    print("="*70)

    tickers = collect_all()

    print(f"\n{'='*70}")
    print(f"TOTAL CLEAN TICKERS: {len(tickers)}")
    print(f"{'='*70}")

    if not tickers:
        print("ERROR: No tickers collected!")
        sys.exit(1)

    price_df, volume_df, success, fail = download_all(tickers)

    print(f"\n{'='*70}")
    print("SAVING")
    print(f"{'='*70}")
    print(f"Price shape: {price_df.shape}")
    print(f"Volume shape: {volume_df.shape}")

    if not price_df.empty:
        price_df.to_parquet(PRICE_PARQUET, index=True)
    if not volume_df.empty:
        volume_df.to_parquet(VOLUME_PARQUET, index=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Tickers attempted: {len(tickers)}")
    print(f"  Successfully downloaded: {success}")
    print(f"  Failed: {fail}")
    print(f"  Date range: {START} to {END}")
    for path in [PRICE_PARQUET, VOLUME_PARQUET]:
        if os.path.exists(path):
            print(f"  {os.path.basename(path)}: {os.path.getsize(path)/1024/1024:.2f} MB")
    print(f"\nDONE")
