"""
Massive US + CN Stock Download
===============================
US: S&P 500 + NASDAQ 100 + Russell 2000 + Dow Jones
CN: 沪深300 + 中证500 + 中证1000 + 创业板 + 科创板
No limits — download everything.
"""
import akshare as ak; import yfinance as yf; import pandas as pd; import numpy as np
import time, os, warnings; warnings.filterwarnings('ignore')

SAVE_DIR = "D:/code/data"
os.makedirs(SAVE_DIR, exist_ok=True)
START, END = "2015-01-01", "2024-12-31"

# ============================================================
# US STOCKS — all major indices
# ============================================================
print("=" * 60)
print("[1/6] US Stock Tickers from Wikipedia...")
us_symbols = set()

# S&P 500
try:
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    for s in sp500["Symbol"]:
        us_symbols.add(str(s).replace(".", "-"))
    print(f"  S&P 500: {len(sp500)} symbols")
except Exception as e: print(f"  S&P 500 failed: {e}")

# NASDAQ 100
try:
    nasdaq100 = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
    for s in nasdaq100["Ticker"]:
        us_symbols.add(str(s).replace(".", "-"))
    print(f"  NASDAQ 100: {len(nasdaq100)} symbols")
except Exception as e: print(f"  NASDAQ 100 failed: {e}")

# Dow Jones
try:
    dow = pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")[1]
    for s in dow["Symbol"]:
        us_symbols.add(str(s).replace(".", "-"))
    print(f"  Dow Jones: {len(dow)} symbols")
except Exception as e: print(f"  Dow failed: {e}")

# Russell 2000 (partial via Wikipedia)
try:
    rus = pd.read_html("https://en.wikipedia.org/wiki/Russell_2000_Index")[2]
    for s in rus["Ticker"]:
        us_symbols.add(str(s).replace(".", "-"))
    print(f"  Russell 2000: {len(rus)} symbols")
except Exception as e: print(f"  Russell 2000 failed: {e}")

us_symbols = list(us_symbols)
print(f"  Total unique US symbols: {len(us_symbols)}")

# Download US stocks
print("[2/6] Downloading US stocks...")
us_data = {}
batch_size = 100
for i in range(0, len(us_symbols), batch_size):
    batch = us_symbols[i:i+batch_size]
    try:
        df = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False, timeout=30)
        if "Close" in df.columns:
            for t in batch:
                if t in df["Close"].columns:
                    us_data[t] = df["Close"][t]
        print(f"  US batch {i//batch_size+1}/{(len(us_symbols)-1)//batch_size+1} | {len(batch)} tried | total ok: {len(us_data)}")
    except Exception as e:
        print(f"  US batch {i//batch_size+1} error: {str(e)[:80]}")
    time.sleep(1)

us_close = pd.DataFrame(us_data)
us_close = us_close.dropna(axis=1, thresh=int(len(us_close)*0.4))
us_returns = np.log(us_close / us_close.shift(1)).dropna(how="all")
us_returns.to_parquet(f"{SAVE_DIR}/us_returns.parquet")
print(f"  US saved: {us_returns.shape[1]} stocks x {us_returns.shape[0]} days")

# ============================================================
# CHINA A-SHARES
# ============================================================
print("[3/6] China A-Share Tickers...")
cn_codes = set()

for idx_name, idx_code in [("沪深300","000300"),("中证500","000905"),("中证1000","000852"),
                            ("创业板指","399006"),("科创50","000688"),("深证100","399330")]:
    try:
        df = ak.index_stock_cons_csindex(symbol=idx_code)
        for c in df["成分券代码"].values:
            cn_codes.add(str(c))
        print(f"  {idx_name}: {len(df)} stocks")
    except Exception as e: print(f"  {idx_name} failed: {e}")
    time.sleep(0.5)

cn_codes = list(cn_codes)
print(f"  Total unique CN codes: {len(cn_codes)}")

print("[4/6] Downloading China A-Shares (this will take a while)...")
cn_data = {}
for i, code in enumerate(cn_codes):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=START.replace("-",""),
                                end_date=END.replace("-",""), adjust="qfq")
        if len(df) > 200:
            cn_data[code] = pd.Series(df["收盘"].values.astype(float), index=pd.to_datetime(df["日期"]))
    except: pass
    if (i+1) % 200 == 0:
        print(f"  CN {i+1}/{len(cn_codes)} | ok: {len(cn_data)}")
    time.sleep(0.25)

cn_close = pd.DataFrame(cn_data).sort_index()
cn_close = cn_close.dropna(axis=1, thresh=int(len(cn_close)*0.4))
cn_returns = np.log(cn_close / cn_close.shift(1)).dropna(how="all")
cn_returns.to_parquet(f"{SAVE_DIR}/cn_returns.parquet")
print(f"  CN saved: {cn_returns.shape[1]} stocks x {cn_returns.shape[0]} days")

# ============================================================
# CRYPTO (more)
# ============================================================
print("[5/6] Crypto...")
crypto_tickers = [f"{c}-USD" for c in [
    "BTC","ETH","BNB","XRP","SOL","ADA","DOGE","AVAX","DOT","MATIC","LINK","UNI",
    "SHIB","LTC","ATOM","ETC","XLM","FIL","ALGO","VET","ICP","SAND","MANA","EGLD",
    "THETA","AXS","FLOW","GALA","QNT","ENJ","AAVE","MKR","GRT","SNX","COMP","ZEC",
    "BAT","LRC","1INCH","ANKR","CELO","DASH","DYDX","ENS","FTM","IOTA","KSM","NEO",
    "OKB","ONE","RUNE","SUSHI","WAVES","XTZ","YFI","ZIL","CHZ","HOT","KAVA"]]
crypto_close = pd.DataFrame()
for i in range(0, len(crypto_tickers), 30):
    batch = crypto_tickers[i:i+30]
    try:
        df = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False)
        if "Close" in df.columns:
            crypto_close = pd.concat([crypto_close, df["Close"]], axis=1)
    except: pass
    time.sleep(1)
crypto_close = crypto_close.dropna(axis=1, thresh=int(len(crypto_close)*0.4))
crypto_returns = np.log(crypto_close / crypto_close.shift(1)).dropna(how="all")
crypto_returns.to_parquet(f"{SAVE_DIR}/crypto_returns.parquet")
print(f"  Crypto saved: {crypto_returns.shape[1]} coins x {crypto_returns.shape[0]} days")

# ============================================================
# FOREX + COMMODITIES (more)
# ============================================================
print("[6/6] Forex + Commodities...")
all_pairs = []
# Forex
forex_list = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X",
              "EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X","EURCHF=X","GBPCHF=X",
              "USDMXN=X","USDZAR=X","USDTRY=X","USDSGD=X","USDHKD=X","USDSEK=X","USDNOK=X",
              "USDDKK=X","USDPLN=X","EURSEK=X","EURNOK=X","EURPLN=X","USDCNY=X","USDINR=X",
              "USDBRL=X","USDKRW=X","EURTRY=X","GBPSEK=X","NZDCAD=X","AUDNZD=X","GBPAUD=X"]
# Commodities
comm_list = ["GC=F","SI=F","CL=F","NG=F","HG=F","ZC=F","ZS=F","KC=F","CT=F","SB=F",
             "PL=F","PA=F","ZW=F","KE=F","CC=F","OJ=F","RB=F","HO=F","BZ=F","LB=F",
             "FC=F","QM=F","GF=F","HE=F","LE=F","DL=F","RR=F","ZC=F","ZL=F","ZM=F","ZR=F"]
# Indices
idx_list = ["^GSPC","^DJI","^IXIC","^RUT","^VIX","^FTSE","^GDAXI","^N225","^HSI",
            "^STOXX50E","^FCHI","^AXJO","^BVSP","^BSESN","^KS11","^MXX","^JKSE",
            "^NSEI","^TNX","^TYX"]
all_pairs = forex_list + comm_list + idx_list

all_close = pd.DataFrame()
for i in range(0, len(all_pairs), 30):
    batch = all_pairs[i:i+30]
    try:
        df = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False)
        if "Close" in df.columns:
            all_close = pd.concat([all_close, df["Close"]], axis=1)
    except: pass
    time.sleep(1)

all_close = all_close.dropna(axis=1, thresh=int(len(all_close)*0.4))
all_returns = np.log(all_close / all_close.shift(1)).dropna(how="all")
all_returns.to_parquet(f"{SAVE_DIR}/forex_comm_idx_returns.parquet")
print(f"  F/C/I saved: {all_returns.shape[1]} pairs x {all_returns.shape[0]} days")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  DOWNLOAD COMPLETE")
print("=" * 60)
total_assets = us_returns.shape[1] + cn_returns.shape[1] + crypto_returns.shape[1] + all_returns.shape[1]
total_size_mb = sum(os.path.getsize(f"{SAVE_DIR}/{f}") for f in os.listdir(SAVE_DIR) if f.endswith('.parquet')) / 1e6
print(f"  US Stocks:     {us_returns.shape[1]:>5d} x {us_returns.shape[0]:>5d} days")
print(f"  China A-Shares:{cn_returns.shape[1]:>5d} x {cn_returns.shape[0]:>5d} days")
print(f"  Crypto:        {crypto_returns.shape[1]:>5d} x {crypto_returns.shape[0]:>5d} days")
print(f"  Forex/Comm/Idx:{all_returns.shape[1]:>5d} x {all_returns.shape[0]:>5d} days")
print(f"  ---")
print(f"  TOTAL ASSETS:  {total_assets}")
print(f"  TOTAL SIZE:    {total_size_mb:.0f} MB")
print(f"  Saved to:      {SAVE_DIR}/")
print("=" * 60)
