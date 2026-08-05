"""Download US stock history back to 1990 for top 200 most liquid stocks."""
import yfinance as yf
import pandas as pd
from pathlib import Path

# Top ~200 most liquid US stocks (by volume/market cap)
tickers = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "BRK-B", "BRK-A",
    "TSLA", "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "CVX", "HD", "PFE",
    "ABBV", "MRK", "KO", "PEP", "BAC", "TMO", "COST", "DIS", "CSCO", "WMT",
    "AVGO", "MCD", "CRM", "ABT", "NFLX", "ADBE", "NKE", "DHR", "VZ", "TXN",
    "WFC", "LLY", "QCOM", "ORCL", "LIN", "PM", "NEE", "RTX", "AMGN", "T",
    "HON", "LOW", "UPS", "CAT", "SPGI", "C", "BA", "IBM", "GS", "CMCSA",
    "INTU", "AMAT", "AMD", "NOW", "TMUS", "ISRG", "SYK", "TJX", "SCHW",
    "DE", "BLK", "MDT", "LMT", "COP", "PLD", "ADP", "EL", "AXP", "GILD",
    "BKNG", "ZTS", "ADI", "MO", "MMC", "UNP", "CI", "SO", "DUK", "APD",
    "REGN", "EOG", "WM", "CB", "NSC", "ETN", "ITW", "TGT", "ATVI", "FIS",
    "ICE", "VRTX", "BDX", "PSA", "GM", "CARR", "CL", "AON", "MCO", "PGR",
    "FCX", "MET", "NOC", "USB", "FDX", "SRE", "APTV", "AIG", "ROP", "GE",
    "CHTR", "SHW", "ALL", "PRU", "HUM", "OXY", "MS", "TRV", "PNC", "EMR",
    "DOW", "BAX", "F", "MMM", "SBUX", "TEL", "KMB", "PSX", "AEP", "HES",
    "HPQ", "VLO", "DD", "NEM", "MPC", "KMI", "EXC", "PAYX", "LEN", "WMB",
    "CTVA", "ROST", "GIS", "HCA", "GM", "ED", "EIX", "PCG", "D", "PEG",
    "AWK", "WEC", "DTE", "AEE", "ETR", "CMS", "LNT", "AGR", "CNP", "ATO",
    "FE", "NI", "PNW", "AES", "DAR", "BG", "ADM", "CTVA", "CF", "IP",
    "WRK", "PKG", "AVY", "BLL", "EMN", "IFF", "PPG", "ECL", "XYL",
    "ROK", "ABB", "IR", "DOV", "NDSN", "PH", "GWW", "FAST", "ALGN",
    "DXCM", "IDXX", "MRNA", "BNTX", "ILMN", "WAT", "MTD", "KEYS",
    "FTV", "ZBRA", "CDNS", "SNPS", "ANSS", "TTWO", "EA", "RBLX",
]

# Remove duplicates while preserving order
seen = set()
tickers_clean = []
for t in tickers:
    if t not in seen:
        seen.add(t)
        tickers_clean.append(t)
tickers = tickers_clean

print(f"Downloading {len(tickers)} tickers...")

data = yf.download(
    tickers,
    start="1990-01-01",
    end="2024-12-31",
    auto_adjust=True,
    actions=False,
    group_by="ticker",
    threads=True,
    progress=True,
)

print(f"Download finished. Data shape: {data.shape}")

# Extract Close prices from MultiIndex
# yfinance can produce (Price, Ticker) or (Ticker, Price) layout depending on version
if isinstance(data.columns, pd.MultiIndex):
    # Find which level contains price field names (Open, High, Low, Close, Volume)
    price_keywords = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
    nlevels = data.columns.nlevels
    price_level = None
    ticker_level = None
    for i in range(nlevels):
        vals = set(data.columns.get_level_values(i).unique())
        overlap = vals & price_keywords
        if len(overlap) >= 3:
            price_level = i
        else:
            ticker_level = i
    print(f"Price level={price_level}, Ticker level={ticker_level}")
    if price_level is not None:
        # Use xs to select 'Close' from price level, dropping that level
        close_df = data.xs('Close', level=price_level, axis=1)
    else:
        close_df = data.iloc[:, 0::5].copy()  # fallback heuristic
    # Flatten to a simple Index of ticker names if still MultiIndex
    if isinstance(close_df.columns, pd.MultiIndex):
        close_df.columns = close_df.columns.droplevel(price_level if price_level == 0 else 1 - price_level)
elif isinstance(data.columns, pd.Index) and isinstance(data.columns[0], tuple):
    # Single-index with tuples — filter for Close
    close_df = data.loc[:, [c for c in data.columns if c[-1] == 'Close']].copy()
    close_df.columns = [c[0] for c in close_df.columns]
else:
    close_df = data.copy()

close_df = close_df.sort_index(axis=1)  # sort ticker columns
close_df = close_df.dropna(axis=1, how="all")

out_dir = Path("D:/code/data")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "us_deep_history.parquet"
close_df.to_parquet(out_path)

print(f"\nSaved {close_df.shape[1]} stocks x {close_df.shape[0]} days to {out_path}")
print(f"Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
print(f"Tickers: {sorted(close_df.columns.tolist())}")
