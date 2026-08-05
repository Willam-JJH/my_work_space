"""Download US stock price+volume data from yfinance."""
import sys
import os
import time
import pandas as pd
import yfinance as yf

OUT_DIR = "D:/code/data"
START = "2000-01-01"
END = "2024-12-31"

# ---- S&P 500 from Wikipedia ----
def get_sp500():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    dfs = pd.read_html(url, header=0, storage_options=headers)
    return dfs[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


# ---- NASDAQ 100 from Wikipedia ----
def get_nasdaq100():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    dfs = pd.read_html(url, header=0, storage_options=headers)
    for df in dfs:
        if "Ticker" in df.columns:
            return df["Ticker"].str.replace(".", "-", regex=False).tolist()
        if "Symbol" in df.columns:
            return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return dfs[4].iloc[:, 0].str.replace(".", "-", regex=False).tolist()


# ---- Hardcoded tickers ----
EXTRA = [
    "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "USO", "UNG",
    "XLE", "XLF", "XLK", "XLV", "XLY", "XLI", "XLB", "XLU", "XLP", "XLC",
    "XBI", "XRT", "SMH", "SOXX", "IGV", "IBB",
    "ARKK", "ARKW",
    "PLTR", "CRWD", "DDOG", "SNOW", "MDB", "NET", "ZS", "DASH",
    "SQ", "COIN", "HOOD", "RBLX", "U",
]


def extract_batch(data, batch, price_list, volume_list):
    """Extract Close/Volume from a yfinance download result."""
    if data is None or data.empty:
        return 0
    is_multi = isinstance(data.columns, pd.MultiIndex)
    count = 0
    for ticker in batch:
        try:
            if is_multi:
                if ("Close", ticker) not in data.columns:
                    continue
                close_ser = data[("Close", ticker)].dropna()
                vol_ser = data[("Volume", ticker)].dropna()
            else:
                if "Close" not in data.columns:
                    continue
                close_ser = data["Close"].dropna()
                vol_ser = data["Volume"].dropna()
            if close_ser.empty:
                continue
            price_list.append(pd.DataFrame({ticker: close_ser}))
            volume_list.append(pd.DataFrame({ticker: vol_ser}))
            count += 1
        except Exception as e:
            print(f"    Warning: {ticker} failed: {e}")
    return count


# ---- Main ----
def main():
    print("Fetching S&P 500 tickers from Wikipedia ...")
    sp500 = get_sp500()
    print(f"  Got {len(sp500)} tickers")

    print("Fetching NASDAQ-100 tickers from Wikipedia ...")
    ndx = get_nasdaq100()
    print(f"  Got {len(ndx)} tickers")

    all_tickers = sorted(set(sp500 + ndx + EXTRA))
    print(f"\nTotal unique tickers: {len(all_tickers)}")

    batch_size = 30
    price_list = []
    volume_list = []
    total_ok = 0

    n_batches = (len(all_tickers) + batch_size - 1) // batch_size

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} tickers) ...", end=" ")

        data = yf.download(
            tickers=batch,
            start=START,
            end=END,
            auto_adjust=True,
            actions=False,
            progress=False,
        )

        ok = extract_batch(data, batch, price_list, volume_list)
        total_ok += ok
        print(f"got {ok}")
        time.sleep(0.3)

    if not price_list:
        print("ERROR: no data downloaded.")
        sys.exit(1)

    print(f"\nTotal tickers with data: {total_ok}")

    # Merge
    price_df = pd.concat(price_list, axis=1).sort_index()
    volume_df = pd.concat(volume_list, axis=1).sort_index()

    os.makedirs(OUT_DIR, exist_ok=True)

    price_path = os.path.join(OUT_DIR, "us_price.parquet")
    volume_path = os.path.join(OUT_DIR, "us_volume.parquet")

    price_df.to_parquet(price_path)
    volume_df.to_parquet(volume_path)

    price_size_mb = os.path.getsize(price_path) / 1e6
    vol_size_mb = os.path.getsize(volume_path) / 1e6
    print(f"\n=== Summary ===")
    print(f"  Number of stocks:         {price_df.shape[1]}")
    print(f"  Date range:               {price_df.index[0].date()}  to  {price_df.index[-1].date()}")
    print(f"  Trading days:             {len(price_df)}")
    print(f"  Price file:               {price_path}  ({price_size_mb:.1f} MB)")
    print(f"  Volume file:              {volume_path}  ({vol_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
