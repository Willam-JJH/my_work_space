"""
Download US stocks - focused approach with rate limit handling.
Sources: Hardcoded list + Wikipedia S&P 500 + NASDAQ-100.
Saves price and volume to parquet.
"""
import warnings
warnings.filterwarnings("ignore")
import os, time, sys, io, contextlib, random
import pandas as pd
import numpy as np
import yfinance as yf

print("=" * 70)
print("US STOCK UNIVERSE EXPANDER v3")
print("=" * 70)

OUTPUT_DIR = r"D:/code/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PRICE_PARQUET = os.path.join(OUTPUT_DIR, "us_price_expanded.parquet")
VOLUME_PARQUET = os.path.join(OUTPUT_DIR, "us_volume_expanded.parquet")
TICKER_FILE = os.path.join(OUTPUT_DIR, "us_stock_list.txt")

START = "2000-01-01"
END   = "2024-12-31"
BATCH_SIZE = 30
BASE_DELAY  = 1.0  # seconds between batches
RATE_LIMIT_DELAY = 30.0  # seconds to wait if rate limited

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"

# ---------- Collect tickers ----------
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
        print(f"  [WARN] Wiki failed: {e}")
    return set()

def load_hardcoded(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {l.strip().upper() for l in f if l.strip() and not l.startswith("#")}

def clean(tickers):
    out = set()
    exclude = {"","A","AA","AAA","AAAA","ETF","FUND","INC","CORP","LTD","LLC","LP","GP"}
    for t in tickers:
        t = t.strip().upper()
        if "." in t: t = t.split(".")[0]
        if not t or t in exclude or len(t) > 5 or not t.isascii(): continue
        if not t.replace("-","").isalpha(): continue
        out.add(t)
    return sorted(out)

def collect():
    all_t = set()
    # Wikipedia S&P 500
    print("[1] S&P 500...")
    all_t |= wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    print(f"  -> {len(all_t)}")
    # Wikipedia NASDAQ-100
    print("[2] NASDAQ-100...")
    t2 = wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100")
    if len(t2) < 50:
        t2 |= wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100#Components")
    all_t |= t2
    print(f"  -> {len(all_t)}")
    # Hardcoded list
    print("[3] Hardcoded list...")
    all_t |= load_hardcoded(TICKER_FILE)
    print(f"  -> {len(all_t)}")
    return clean(all_t)

# ---------- Download ----------
def download_batch(tickers_batch, attempt=0):
    """Download a batch with rate limit handling."""
    if attempt >= 3:
        return None  # Give up after 3 retries
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            data = yf.download(
                tickers_batch, start=START, end=END,
                auto_adjust=True, progress=False, actions=False,
                group_by="ticker", threads=True,
            )
        return data
    except Exception as e:
        err = str(e)
        if "Rate limited" in err or "Too Many Requests" in err:
            wait = RATE_LIMIT_DELAY * (attempt + 1) * random.uniform(0.8, 1.2)
            print(f"\n  RATE LIMITED (attempt {attempt+1}), waiting {wait:.0f}s...")
            time.sleep(wait)
            return download_batch(tickers_batch, attempt + 1)
        elif attempt < 2:
            time.sleep(3)
            return download_batch(tickers_batch, attempt + 1)
        return None

def extract_data(data, ticker):
    """Extract Close and Volume from yfinance download output for one ticker."""
    ts = str(ticker)
    try:
        if isinstance(data.columns, pd.MultiIndex):
            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)
            if "Close" in l0:
                if ("Close", ts) in data.columns and ("Volume", ts) in data.columns:
                    return data[("Close", ts)].dropna(), data[("Volume", ts)].dropna()
            elif ts in l0:
                if (ts, "Close") in data.columns and (ts, "Volume") in data.columns:
                    return data[(ts, "Close")].dropna(), data[(ts, "Volume")].dropna()
        else:
            if "Close" in data.columns and "Volume" in data.columns:
                return data["Close"].dropna(), data["Volume"].dropna()
    except Exception:
        pass
    return None, None

def download_all(tickers):
    total = len(tickers)
    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {total} TICKERS IN {len(batches)} BATCHES")
    print(f"{'='*70}")

    price_dfs, volume_dfs = [], []
    success, fail = 0, 0
    start_time = time.time()

    for idx, batch in enumerate(batches):
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(batches) - idx - 1) / rate if rate > 0 else 0
        print(f"\r  [{idx+1}/{len(batches)}] {idx*BATCH_SIZE+1}-{min((idx+1)*BATCH_SIZE,total)}/{total} | "
              f"OK:{success} FAIL:{fail} | ETA:{remaining/60:.0f}m", end="")

        data = download_batch(list(batch))
        if data is None:
            fail += len(batch)
            time.sleep(BASE_DELAY)
            continue

        for tkr in batch:
            p, v = extract_data(data, tkr)
            if p is not None and len(p) > 0:
                p.name = str(tkr); price_dfs.append(p)
                v.name = str(tkr); volume_dfs.append(v)
                success += 1
            else:
                fail += 1

        # Avoid rate limiting - add jitter
        time.sleep(BASE_DELAY * random.uniform(0.8, 1.5))

    total_time = time.time() - start_time
    print(f"\n\n  Elapsed: {total_time/60:.1f}m | Success: {success} | Failed: {fail}")

    # Merge
    print("\nMerging price data...")
    price_df = pd.concat(price_dfs, axis=1, join="outer") if price_dfs else pd.DataFrame()
    print(f"  Price columns: {price_df.shape[1]}")
    print("Merging volume data...")
    volume_df = pd.concat(volume_dfs, axis=1, join="outer") if volume_dfs else pd.DataFrame()
    print(f"  Volume columns: {volume_df.shape[1]}")

    if not price_df.empty:
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        price_df = price_df.loc["2000-01-01":"2024-12-31"]
        price_df = price_df[sorted(price_df.columns)]
    if not volume_df.empty:
        volume_df.index = pd.to_datetime(volume_df.index)
        volume_df = volume_df.sort_index()
        volume_df = volume_df.loc["2000-01-01":"2024-12-31"]
        volume_df = volume_df[sorted(volume_df.columns)]

    return price_df, volume_df, success, fail

# ---------- MAIN ----------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("COLLECTING TICKERS")
    print("="*70)
    tickers = collect()
    print(f"\nTotal clean tickers: {len(tickers)}")

    if not tickers:
        print("ERROR: No tickers!"); sys.exit(1)

    price_df, volume_df, success, fail = download_all(tickers)

    print(f"\n{'='*70}")
    print("SAVING")
    print(f"{'='*70}")
    print(f"Price: {price_df.shape}")
    print(f"Volume: {volume_df.shape}")

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
