"""Download ALL China A-shares with retry + checkpoint"""
import akshare as ak; import pandas as pd; import numpy as np; import time, os, pickle
SAVE="D:/code/data/cn_all_returns.parquet"; CKPT="D:/code/data/cn_ckpt.pkl"
START="20150101"; END="20241231"

print("Getting all A-share codes...")
spot = ak.stock_zh_a_spot()
all_raw = list(spot['代码'].values)
def strip_prefix(c):
    for p in ['sh','sz']:
        if c.startswith(p): return c[len(p):]
    return None
all_codes = [strip_prefix(c) for c in all_raw if strip_prefix(c) is not None]
print(f"Main board: {len(all_codes)} stocks")

# Resume from checkpoint
cn_data = {}
if os.path.exists(CKPT):
    with open(CKPT,'rb') as f: cn_data = pickle.load(f)
    print(f"Resumed: {len(cn_data)} stocks already downloaded")

for i, code in enumerate(all_codes):
    if code in cn_data: continue
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START, end_date=END, adjust="qfq")
            if len(df) > 200:
                cn_data[code] = pd.Series(df["收盘"].values.astype(float), index=pd.to_datetime(df["日期"]))
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(5 + attempt * 3)
            else:
                pass  # skip after 3 failures
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(all_codes)} | ok: {len(cn_data)}")
        # Save checkpoint every 500
        if (i+1) % 500 == 0:
            with open(CKPT,'wb') as f: pickle.dump(cn_data, f)
            print(f"  Checkpoint: {len(cn_data)} stocks")
    time.sleep(0.15)

# Save
cn_close = pd.DataFrame(cn_data).sort_index()
cn_close = cn_close.dropna(axis=1, thresh=int(len(cn_close)*0.4))
cn_returns = np.log(cn_close / cn_close.shift(1)).dropna(how="all")
cn_returns.to_parquet(SAVE)
print(f"Saved: {cn_returns.shape[1]} stocks x {cn_returns.shape[0]} days → {SAVE}")
print(f"Size: {os.path.getsize(SAVE)/1e6:.1f} MB")
# Clean checkpoint
if os.path.exists(CKPT): os.remove(CKPT)
