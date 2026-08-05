import pandas as pd, os

# Crypto
btc_dir = 'D:/code/data/binance_crypto/spot/monthly/klines/BTCUSDT/1d/'
dfs = [pd.read_csv(os.path.join(btc_dir, f)) for f in os.listdir(btc_dir)]
btc = pd.concat(dfs)
print(f'BTCUSDT: {btc.shape[0]} daily rows, cols: {list(btc.columns)}')
print(f'  date range: {btc.iloc[:,0].min()} ~ {btc.iloc[:,0].max()}')

# Nonstock returns
df = pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
forex = [c for c in df.columns if '=X' in c]
commod = [c for c in df.columns if '_Spot' in c]
crypto = [c for c in df.columns if 'USDT' in c or 'USDC' in c]
index = [c for c in df.columns if c not in forex+commod+crypto]
print(f'\nlog_returns_nonstock: {df.shape[0]} days x {df.shape[1]} assets')
print(f'  Forex pairs: {len(forex)}, sample: {forex[:5]}')
print(f'  Commodities: {len(commod)}, sample: {commod[:5]}')
print(f'  Crypto pairs: {len(crypto)}, sample: {crypto[:5]}')
print(f'  Indices: {len(index)}, sample: {index[:5]}')

# Commodities spot
df = pd.read_parquet('D:/code/data/commodities_spot.parquet')
print(f'\ncommodities_spot: {df.shape[0]} days x {df.shape[1]} commodities')

# Factors
for name in ['STK_MKT_CARHARTFOURFACTORS', 'STK_MKT_MOMENTUM', 'FI_T10', 'adj_factor']:
    df = pd.read_csv(f'D:/code/data/factors/{name}.csv')
    print(f'{name}: {df.shape[0]} rows x {df.shape[1]} cols')

# CN daily_returns actual stats
df = pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
print(f'\nCN daily_returns (long format): {df.shape[0]} rows, {df["stkcd"].nunique()} unique stocks, {df["trddt"].nunique()} unique dates')

# Fundamentals
for f in ['balance_sheet', 'cashflow_direct', 'income_statement']:
    df = pd.read_parquet(f'D:/code/data/fundamentals/{f}.parquet')
    print(f'{f}: {df.shape[0]} rows x {df.shape[1]} cols')

# International
df = pd.read_csv('D:/code/data/international/IDX_Gidxtrd.csv')
print(f'\nInternational indices: {df.shape[0]} rows, {df["Indexcd"].nunique()} unique indices, {df["Trddt"].nunique()} unique dates')

# economic indicators
df = pd.read_parquet('D:/code/data/economic_indicators.parquet')
print(f'economic_indicators: {df.shape[0]} rows x {df.shape[1]} indicators')

# FRED
df = pd.read_parquet('D:/code/data/fred_all_indicators.parquet')
print(f'fred_all_indicators: {df.shape[0]} rows x {df.shape[1]} indicators')
