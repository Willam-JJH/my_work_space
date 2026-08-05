"""
Download ALL China A-share close price and volume data via direct HTTP API.
Saves to D:/code/data/cn_price.parquet and D:/code/data/cn_volume.parquet.
Resumable via D:/code/data/cn_ckpt.pkl checkpoint.
"""
import os
import time
import pickle
import pandas as pd
import requests
import akshare as ak

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = r"D:/code/data"
PRICE_PATH  = os.path.join(DATA_DIR, "cn_price.parquet")
VOLUME_PATH = os.path.join(DATA_DIR, "cn_volume.parquet")
CKPT_PATH   = os.path.join(DATA_DIR, "cn_ckpt.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────
START_DATE = "20000101"
END_DATE   = "20241231"
REQUIRED_ROWS = 200
RETRIES = 3
BACKOFFS = [3, 6, 10]
SESSION_REFRESH = 100  # new session every N requests

# ── helpers ────────────────────────────────────────────────────────────

def make_session():
    """Create a fresh requests Session."""
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Windows"})
    return s

def fetch_hist_direct(symbol, session):
    """Fetch daily hist for one stock via direct HTTP API.
    Returns (close_series, volume_series) or (None, None).
    """
    market_code = 1 if symbol.startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": f"{market_code}.{symbol}",
        "beg": START_DATE,
        "end": END_DATE,
    }
    for attempt in range(RETRIES):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data_json = r.json()
            if not (data_json.get("data") and data_json["data"].get("klines")):
                return None, None
            # Parse kline data
            klines = data_json["data"]["klines"]
            closes = []
            volumes = []
            dates = []
            for line in klines:
                parts = line.split(",")
                dates.append(parts[0])  # date string
                closes.append(float(parts[2]))  # index 2 = close
                volumes.append(float(parts[5]))  # index 5 = volume
            return pd.Series(closes, index=pd.to_datetime(dates)), pd.Series(volumes, index=pd.to_datetime(dates))
        except Exception:
            if attempt < RETRIES - 1:
                time.sleep(BACKOFFS[attempt])
            else:
                return None, None
    return None, None


def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        return ckpt.get("done", set()), ckpt.get("price", {}), ckpt.get("volume", {})
    return set(), {}, {}


def save_checkpoint(done, price_dict, volume_dict):
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"done": done, "price": price_dict, "volume": volume_dict}, f)
    os.replace(tmp, CKPT_PATH)


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("China A-share data downloader (direct API)", flush=True)
    print("=" * 60, flush=True)

    done_codes, price_dict, volume_dict = load_checkpoint()
    print(f"Checkpoint loaded: {len(done_codes)} stocks already downloaded.", flush=True)

    # Get stock list via akshare
    print("\nFetching A-share spot list ...", flush=True)
    spot_df = ak.stock_zh_a_spot()
    print(f"Total spot entries: {len(spot_df)}", flush=True)

    code_col = next((c for c in ["代码", "symbol", "code"] if c in spot_df.columns), None)
    if code_col is None:
        code_col = [c for c in spot_df.columns if spot_df[c].dtype == object][0]
    all_codes = spot_df[code_col].astype(str).str.strip().tolist()

    filtered = []
    for code in all_codes:
        lower = code.lower()
        if lower.startswith("sh") or lower.startswith("sz"):
            plain = code[2:]
            if plain.isdigit():
                filtered.append(plain)
    print(f"Filtered: {len(filtered)} main-board stocks.", flush=True)

    total = len(filtered)
    print(f"\nStarting download of {total} stocks ...\n", flush=True)

    session = make_session()
    req_count = 0

    for idx, symbol in enumerate(filtered):
        if symbol in done_codes:
            continue

        # Refresh session periodically to avoid stale connections
        if req_count > 0 and req_count % SESSION_REFRESH == 0:
            session = make_session()
            print(f"  Session refreshed at request {req_count}.", flush=True)

        closes, volumes = fetch_hist_direct(symbol, session)
        req_count += 1

        if closes is not None and len(closes) > REQUIRED_ROWS:
            price_dict[symbol] = closes
            volume_dict[symbol] = volumes
        done_codes.add(symbol)

        completed = len(done_codes)
        if completed % 500 == 0:
            saved = len(price_dict)
            print(f"  [{completed}/{total}] stocks processed, {saved} saved.", flush=True)
            save_checkpoint(done_codes, price_dict, volume_dict)
            print(f"  Checkpoint saved.", flush=True)

    # Final save
    print(f"\nFinalizing: {len(price_dict)} stocks.", flush=True)
    price_df = pd.DataFrame(price_dict)
    volume_df = pd.DataFrame(volume_dict)
    price_df.index = pd.to_datetime(price_df.index, errors="coerce") if not isinstance(price_df.index, pd.DatetimeIndex) else price_df.index
    volume_df.index = pd.to_datetime(volume_df.index, errors="coerce") if not isinstance(volume_df.index, pd.DatetimeIndex) else volume_df.index
    price_df.to_parquet(PRICE_PATH, index=True)
    volume_df.to_parquet(VOLUME_PATH, index=True)
    print(f"Saved price: {price_df.shape}, volume: {volume_df.shape}", flush=True)

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
        print("Checkpoint removed. Done.", flush=True)


if __name__ == "__main__":
    main()
