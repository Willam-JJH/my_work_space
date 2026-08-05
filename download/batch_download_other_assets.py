"""
Download crypto, forex and commodities close prices from yfinance,
batch by batch, and save to a single parquet file.
"""
import time
import yfinance as yf
import pandas as pd

# ── Tickers ──────────────────────────────────────────────────────────────
CRYPTO = [
    "BTC-USD","ETH-USD","BNB-USD","XRP-USD","SOL-USD","ADA-USD",
    "DOGE-USD","AVAX-USD","DOT-USD","MATIC-USD","LINK-USD","UNI-USD",
    "SHIB-USD","LTC-USD","ATOM-USD","ETC-USD","XLM-USD","FIL-USD",
    "ALGO-USD","VET-USD","ICP-USD","SAND-USD","MANA-USD","EGLD-USD",
    "THETA-USD","AXS-USD","FLOW-USD","GALA-USD","QNT-USD","ENJ-USD",
    "AAVE-USD","MKR-USD","GRT-USD","SNX-USD","COMP-USD","ZEC-USD",
    "BAT-USD","LRC-USD","1INCH-USD","ANKR-USD","CELO-USD","DASH-USD",
    "DYDX-USD","ENS-USD","FTM-USD","IOTA-USD","KSM-USD","NEO-USD",
    "OKB-USD","ONE-USD","RUNE-USD","SUSHI-USD","WAVES-USD","XTZ-USD",
    "YFI-USD","ZIL-USD","CHZ-USD","HOT-USD","KAVA-USD",
]

FOREX = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X",
    "USDCHF=X","NZDUSD=X","EURGBP=X","EURJPY=X","GBPJPY=X",
    "AUDJPY=X","EURAUD=X","EURCHF=X","GBPCHF=X","USDMXN=X",
    "USDZAR=X","USDTRY=X","USDSGD=X","USDHKD=X","USDSEK=X",
    "USDNOK=X","USDDKK=X","USDPLN=X","EURSEK=X","EURNOK=X",
    "EURPLN=X","USDCNY=X","USDINR=X","USDBRL=X","USDKRW=X",
]

COMMODITIES = [
    "GC=F","SI=F","CL=F","NG=F","HG=F","ZC=F","ZS=F","KC=F",
    "CT=F","SB=F","PL=F","PA=F","ZW=F","KE=F","CC=F","OJ=F",
    "RB=F","HO=F","BZ=F","^GSPC","^DJI","^IXIC","^RUT","^VIX",
    "^FTSE","^GDAXI","^N225","^HSI","^STOXX50E","^FCHI",
    "^AXJO","^BVSP",
]

# ── Settings ─────────────────────────────────────────────────────────────
BATCH_SIZE = 30
SLEEP_SEC = 1
START = "2000-01-01"
END = "2024-12-31"
OUTPUT = "D:/code/data/other_assets.parquet"

all_tickers = CRYPTO + FOREX + COMMODITIES
print(f"Total tickers: {len(all_tickers)}")
print(f"Crypto: {len(CRYPTO)}, Forex: {len(FOREX)}, Commodities/Indices: {len(COMMODITIES)}")
print(f"Batch size: {BATCH_SIZE}, delay: {SLEEP_SEC}s")
print()

# ── Download batch by batch ──────────────────────────────────────────────
close_dfs = []
batch_count = 0

for i in range(0, len(all_tickers), BATCH_SIZE):
    batch = all_tickers[i : i + BATCH_SIZE]
    batch_count += 1
    print(f"[Batch {batch_count}] Downloading {len(batch)} tickers: {batch[0]} ... {batch[-1]}")

    try:
        data = yf.download(
            tickers=batch,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        # retry once
        time.sleep(2)
        try:
            data = yf.download(
                tickers=batch,
                start=START,
                end=END,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e2:
            print(f"  Retry also failed: {e2}")
            continue

    # Each ticker gets a column named after the batch with Close price
    for ticker in batch:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                # Check the order of levels
                level_names = data.columns.names
                if level_names == ["Price", "Ticker"]:
                    # single-ticker style: ("Close", "TICKER")
                    if ("Close", ticker) in data.columns:
                        series = data[("Close", ticker)]
                    else:
                        # maybe single ticker returned without second level
                        series = data["Close"] if "Close" in data.columns else None
                elif level_names == ["Ticker", "Price"]:
                    # multi-ticker with group_by='ticker': ("TICKER", "Close")
                    if (ticker, "Close") in data.columns:
                        series = data[(ticker, "Close")]
                    else:
                        series = None
                else:
                    # unknown structure, try a safe fallback
                    series = data.get("Close", None) if "Close" in data.columns else None
            else:
                series = data["Close"] if "Close" in data.columns else None

            if series is None or series.dropna().empty:
                print(f"  No Close data for {ticker}")
                continue

            series = series.dropna()
            df_ticker = series.to_frame(ticker)
            close_dfs.append(df_ticker)
            print(f"  {ticker}: {len(series)} rows")
        except Exception as e:
            print(f"  Failed to process {ticker}: {e}")

    if i + BATCH_SIZE < len(all_tickers):
        time.sleep(SLEEP_SEC)

# ── Merge and save ───────────────────────────────────────────────────────
if close_dfs:
    merged = pd.concat(close_dfs, axis=1, join="outer")
    merged.index = pd.to_datetime(merged.index)
    merged.sort_index(inplace=True)
    merged.to_parquet(OUTPUT)
    print(f"\nSaved to {OUTPUT}")
    print(f"Shape: {merged.shape}")
    print(f"Date range: {merged.index.min().date()} → {merged.index.max().date()}")
    print(f"Columns ({len(merged.columns)}):")
    for col in merged.columns:
        non_null = merged[col].notna().sum()
        print(f"  {col:20s}  {non_null:>6,} non-null rows")
else:
    print("No data downloaded.")
