"""
Download ALL data needed for Proposal: Price + Volume
======================================================
US: yfinance (Close + Volume)
CN: akshare (Close + Volume) — 主板全量
Save: us_price_vol.parquet, cn_price_vol.parquet
"""
import numpy as np; import pandas as pd; import yfinance as yf
import akshare as ak; import time, os, pickle, warnings; warnings.filterwarnings('ignore')

SAVE = "D:/code/data"; os.makedirs(SAVE, exist_ok=True)
START, END = "2000-01-01", "2024-12-31"  # 25 years for proposal

# ============================================================
# US: Price + Volume (all available)
# ============================================================
print("=" * 60)
print("[1/2] US Stocks: Price + Volume")
print("=" * 60)

# Get ALL tickers we can
us_tickers = set()
# Expanded list (~1000 tickers)
extra = """
AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B JPM V JNJ WMT PG MA UNH HD DIS BAC NFLX ADBE
CRM XOM CSCO ABT CVX KO PEP TMO COST MCD WFC DHR ACN NKE LIN QCOM TXN AMGN HON IBM GE CAT
PM MS INTU LOW BMY BA AMAT NOW RTX CMCSA ORCL AMD UBER SPGI GS SBUX BLK UNP AXP PFE DE TJX
ISRG SYK PLD COP ETN ADI BKNG MDLZ GILD ADP LRCX VRTX C CI CB SCHW ZTS BSX TMUS MO EQIX
ICE SO DUK MU AON KLAC MCK PYPL CME SNPS CDNS APH ITW CMG TGT USB PNC MMM APD NOC FDX BDX
EL EW GM A REGN ROP MAR HLT NSC PSA AFL AIG ALL BK MET PRU TRV ABNB AZO ORLY ROST YUM EA
CTAS ECL FAST GWW ODFL PAYX VRSK AME CPRT CTSH DAL DFS DLTR EFX EXC GIS GPN HCA HPQ HUM
KHC LVS MCHP NEM NUE PWR SRE STZ TEL TTWO URI VLO WELL WMB XYL ZBRA TEAM DDOG HUBS NET MDB
ZS OKTA PLTR SNOW COIN RBLX DASH TTD SQ ZM DOCU BILL SNAP PINS U LCID RIVN HOOD AFRM SOFI
DNA GTLB DV ESTC PATH GFS ARM MRVL ALAB DELL PLTR SPOT DKNG CVNA CAVA APP RDDT CART IOT
BROS DUOL QS VFS RKLB ASTS KVYO PCOR IONQ AI QBTS RGTI KULR SERV MSTR CLSK BITF MARA RIOT
CORZ WULF HUT IREN BTDR CIFR SDIG HIVE DGHI RKLB PL  DJT RBRK
SPY QQQ IWM DIA TLT GLD SLV USO UNG XLE XLF XLK XLV XLY XLI XLB XLU XLP XLC XBI XRT SMH
SOXX IGV IBB ITA ARKK ARKW ARKG ARKF IWM IJR MDY VTI VOO VEA VWO BND AGG TLT SHY LQD HYG
""".split()
for t in extra: us_tickers.add(t.strip())

# Try to get SP500 from Wikipedia
try:
    import requests, io
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    tables = pd.read_html(io.StringIO(r.text))
    for sym in tables[0]["Symbol"]: us_tickers.add(str(sym).replace(".", "-"))
    print(f"  Wikipedia SP500: {len(tables[0])} added")
except Exception as e: print(f"  Wikipedia: {e}")

us_tickers = sorted(set(t for t in us_tickers if t and len(t)<=5 and t[0].isalpha()))
print(f"  Total US tickers: {len(us_tickers)}")

us_price = {}; us_volume = {}
for i in range(0, len(us_tickers), 50):
    batch = us_tickers[i:i+50]
    try:
        df = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False, timeout=30)
        if "Close" in df.columns:
            for t in batch:
                if t in df["Close"].columns:
                    us_price[t] = df["Close"][t]
                    if "Volume" in df.columns and t in df["Volume"].columns:
                        us_volume[t] = df["Volume"][t]
        print(f"  Batch {i//50+1}/{(len(us_tickers)-1)//50+1} | ok: {len(us_price)}")
    except Exception as e: print(f"  err: {str(e)[:60]}")
    time.sleep(0.8)

# Save US
us_price_df = pd.DataFrame(us_price).ffill()
us_volume_df = pd.DataFrame(us_volume).ffill().fillna(0)
us_price_df.to_parquet(f"{SAVE}/us_price.parquet")
us_volume_df.to_parquet(f"{SAVE}/us_volume.parquet")
print(f"  US saved: {us_price_df.shape[1]} stocks x {us_price_df.shape[0]} days")

# ============================================================
# CN: Price + Volume (all main board A-shares)
# ============================================================
print("\n" + "=" * 60)
print("[2/2] China A-Shares: Price + Volume")
print("=" * 60)

spot = ak.stock_zh_a_spot()
all_raw = list(spot['代码'].values)
# Strip prefix: sh600xxx -> 600xxx
def strip(c):
    for p in ['sh','sz']:
        if c.startswith(p): return c[len(p):]
    return None
all_codes = [strip(c) for c in all_raw if strip(c) is not None]
print(f"  Main board: {len(all_codes)} stocks")

CKPT = f"{SAVE}/cn_download_ckpt.pkl"
cn_price = {}; cn_volume = {}
if os.path.exists(CKPT):
    with open(CKPT,'rb') as f: saved = pickle.load(f)
    cn_price = saved.get('price', {}); cn_volume = saved.get('volume', {})
    print(f"  Resumed: {len(cn_price)} price, {len(cn_volume)} volume")

for i, code in enumerate(all_codes):
    if code in cn_price: continue
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20150101", end_date="20241231", adjust="qfq")
            if len(df) > 200:
                idx = pd.to_datetime(df["日期"])
                cn_price[code] = pd.Series(df["收盘"].values.astype(float), index=idx)
                if "成交量" in df.columns:
                    cn_volume[code] = pd.Series(df["成交量"].values.astype(float), index=idx)
            break
        except:
            if attempt < 2: time.sleep(5 + attempt * 3)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(all_codes)} | price: {len(cn_price)} | volume: {len(cn_volume)}")
        with open(CKPT,'wb') as f: pickle.dump({'price':cn_price,'volume':cn_volume}, f)
    time.sleep(0.15)

# Save CN
cn_price_df = pd.DataFrame(cn_price).sort_index().ffill()
cn_volume_df = pd.DataFrame(cn_volume).sort_index().ffill().fillna(0)
cn_price_df.to_parquet(f"{SAVE}/cn_price.parquet")
cn_volume_df.to_parquet(f"{SAVE}/cn_volume.parquet")
print(f"  CN saved: {cn_price_df.shape[1]} stocks x {cn_price_df.shape[0]} days")
if os.path.exists(CKPT): os.remove(CKPT)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  DOWNLOAD COMPLETE: Price + Volume")
print("=" * 60)
total = us_price_df.shape[1] + cn_price_df.shape[1]
size_mb = sum(os.path.getsize(f"{SAVE}/{f}") for f in os.listdir(SAVE) if f.endswith('.parquet')) / 1e6
print(f"  US: {us_price_df.shape[1]} stocks")
print(f"  CN: {cn_price_df.shape[1]} stocks")
print(f"  Total assets: {total}")
print(f"  Total size: {size_mb:.0f} MB")
print("=" * 60)
