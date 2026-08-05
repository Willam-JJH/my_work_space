"""
Download comprehensive market data: US, CN, Crypto, Forex, Commodities
=====================================================================
Saves to D:/code/data/ for experiments.
"""
import numpy as np; import pandas as pd; import yfinance as yf
import akshare as ak; import time, os, warnings; warnings.filterwarnings('ignore')

SAVE_DIR = "D:/code/data"
os.makedirs(SAVE_DIR, exist_ok=True)
START, END = "2015-01-01", "2024-12-31"

def download_yf(tickers, name, batch_size=50):
    """Download from Yahoo Finance in batches."""
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False)
            if "Close" in df.columns:
                for t in batch:
                    if t in df["Close"].columns:
                        all_data[t] = df["Close"][t]
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}")
        time.sleep(1)
    return pd.DataFrame(all_data)

# ============================================================
# 1. US STOCKS — S&P 500 + NASDAQ 100
# ============================================================
print("[1/5] US Stocks...")

# S&P 500
try:
    sp500_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    sp500 = list(sp500_table["Symbol"].values)
except:
    sp500 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V","JNJ","WMT","PG","MA",
             "UNH","HD","DIS","BAC","NFLX","ADBE","CRM","XOM","CSCO","ABT","CVX","KO","PEP","TMO",
             "COST","MCD","WFC","DHR","ACN","NKE","LIN","QCOM","TXN","AMGN","HON","IBM","GE","CAT",
             "PM","MS","INTU","LOW","BMY","BA","AMAT","NOW","RTX","CMCSA","ORCL","AMD","UBER","SPGI",
             "GS","SBUX","BLK","UNP","AXP","PFE","DE","TJX","ISRG","SYK","PLD","COP","ETN","ADI",
             "BKNG","MDLZ","GILD","ADP","LRCX","VRTX","C","CI","CB","SCHW","ZTS","BSX","TMUS","MO",
             "EQIX","ICE","SO","DUK","MU","AON","KLAC","MCK","PYPL","CME","SNPS","CDNS","APH","ITW","CMG"]

# NASDAQ 100
try:
    nasdaq_table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
    nasdaq100 = list(nasdaq_table["Ticker"].values)
except:
    nasdaq100 = []

us_tickers = list(set(sp500 + nasdaq100))
us_tickers = [t.replace(".", "-") for t in us_tickers if isinstance(t, str)]
print(f"  {len(us_tickers)} US stocks")

us_close = download_yf(us_tickers, "US")
# Keep stocks with >60% data
us_close = us_close.dropna(axis=1, thresh=int(len(us_close)*0.4))
us_returns = np.log(us_close / us_close.shift(1)).dropna(how="all")
print(f"  Kept: {us_returns.shape[1]} stocks x {us_returns.shape[0]} days")

us_returns.to_parquet(f"{SAVE_DIR}/us_returns.parquet")
print(f"  Saved: us_returns.parquet")

# ============================================================
# 2. CRYPTO (via yfinance)
# ============================================================
print("[2/5] Crypto...")
crypto_tickers = [f"{c}-USD" for c in [
    "BTC","ETH","BNB","XRP","SOL","ADA","DOGE","AVAX","DOT","MATIC",
    "LINK","UNI","SHIB","LTC","ATOM","ETC","XLM","FIL","ALGO","VET",
    "ICP","SAND","MANA","EGLD","THETA","AXS","FLOW","GALA","QNT","ENJ"]]
crypto_close = download_yf(crypto_tickers, "Crypto", batch_size=10)
crypto_close = crypto_close.dropna(axis=1, thresh=int(len(crypto_close)*0.4))
crypto_returns = np.log(crypto_close / crypto_close.shift(1)).dropna(how="all")
print(f"  {crypto_returns.shape[1]} cryptos x {crypto_returns.shape[0]} days")
crypto_returns.to_parquet(f"{SAVE_DIR}/crypto_returns.parquet")

# ============================================================
# 3. FOREX
# ============================================================
print("[3/5] Forex...")
forex_tickers = [f"{c}=X" for c in [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD",
    "EURGBP","EURJPY","GBPJPY","AUDJPY","EURAUD","EURCHF","GBPCHF",
    "USDMXN","USDZAR","USDTRY","USDSGD","USDHKD","USDSEK",
    "USDNOK","USDDKK","USDPLN","EURSEK","EURNOK","EURPLN"]]
forex_close = download_yf(forex_tickers, "Forex", batch_size=15)
forex_close = forex_close.dropna(axis=1, thresh=int(len(forex_close)*0.4))
forex_returns = np.log(forex_close / forex_close.shift(1)).dropna(how="all")
print(f"  {forex_returns.shape[1]} forex x {forex_returns.shape[0]} days")
forex_returns.to_parquet(f"{SAVE_DIR}/forex_returns.parquet")

# ============================================================
# 4. COMMODITIES & INDICES
# ============================================================
print("[4/5] Commodities & Indices...")
comm_tickers = [
    "GC=F","SI=F","CL=F","NG=F","HG=F","ZC=F","ZS=F","KC=F","CT=F","SB=F",
    "PL=F","PA=F","ZW=F","KE=F","CC=F","OJ=F","RB=F","HO=F","BZ=F",
    "ES=F","NQ=F","YM=F","RTY=F","^GSPC","^DJI","^IXIC","^RUT","^VIX",
    "^FTSE","^GDAXI","^N225","^HSI","^STOXX50E","^FCHI","^AXJO"]
comm_close = download_yf(comm_tickers, "Comm/Idx", batch_size=15)
comm_close = comm_close.dropna(axis=1, thresh=int(len(comm_close)*0.4))
comm_returns = np.log(comm_close / comm_close.shift(1)).dropna(how="all")
print(f"  {comm_returns.shape[1]} commodities/indices x {comm_returns.shape[0]} days")
comm_returns.to_parquet(f"{SAVE_DIR}/comm_idx_returns.parquet")

# ============================================================
# 5. CHINA A-SHARES (via akshare)
# ============================================================
print("[5/5] China A-Shares...")

cn_returns_list = {}

# 5a. 沪深300 constituents
try:
    hs300 = ak.index_stock_cons_csindex(symbol="000300")
    hs300_codes = list(hs300["成分券代码"].values)
    print(f"  沪深300: {len(hs300_codes)} stocks")
except Exception as e:
    print(f"  沪深300 failed: {e}, using fallback")
    hs300_codes = []

# 5b. 中证500
try:
    zz500 = ak.index_stock_cons_csindex(symbol="000905")
    zz500_codes = list(zz500["成分券代码"].values)
    print(f"  中证500: {len(zz500_codes)} stocks")
except:
    zz500_codes = []

cn_codes = list(set(hs300_codes + zz500_codes))
print(f"  Total CN stocks to download: {len(cn_codes)}")

# Download daily data for each stock (akshare stock_zh_a_hist)
successful = 0
for i, code in enumerate(cn_codes):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=START.replace("-",""),
                                end_date=END.replace("-",""), adjust="qfq")
        if len(df) > 200:
            cn_returns_list[code] = pd.Series(
                df["收盘"].values.astype(float),
                index=pd.to_datetime(df["日期"])
            )
            successful += 1
    except:
        pass
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(cn_codes)} | {successful} ok")
    time.sleep(0.3)

print(f"  Downloaded: {successful}/{len(cn_codes)} CN stocks")

cn_close = pd.DataFrame(cn_returns_list).sort_index()
cn_close = cn_close.dropna(axis=1, thresh=int(len(cn_close)*0.4))
cn_returns = np.log(cn_close / cn_close.shift(1)).dropna(how="all")
print(f"  CN: {cn_returns.shape[1]} stocks x {cn_returns.shape[0]} days")

# Save as long format (like original data)
cn_long = cn_returns.reset_index().melt(id_vars="index", var_name="stkcd", value_name="dretwd")
cn_long.columns = ["trddt","stkcd","dretwd"]
cn_long.to_parquet(f"{SAVE_DIR}/cn_returns.parquet", index=False)
print(f"  Saved: cn_returns.parquet")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  DATA DOWNLOAD COMPLETE")
print("=" * 60)
print(f"  US Stocks:     {us_returns.shape[1]} x {us_returns.shape[0]} days")
print(f"  Crypto:        {crypto_returns.shape[1]} x {crypto_returns.shape[0]} days")
print(f"  Forex:         {forex_returns.shape[1]} x {forex_returns.shape[0]} days")
print(f"  Commodities:   {comm_returns.shape[1]} x {comm_returns.shape[0]} days")
print(f"  China A-Share: {cn_returns.shape[1]} x {cn_returns.shape[0]} days")
print(f"  Total assets:  {us_returns.shape[1]+crypto_returns.shape[1]+forex_returns.shape[1]+comm_returns.shape[1]+cn_returns.shape[1]}")
print(f"  Saved to: {SAVE_DIR}/")
print("=" * 60)
