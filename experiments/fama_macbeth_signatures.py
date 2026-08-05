"""
Fama-MacBeth Cross-Sectional Pricing with Path Signatures
===========================================================
Tests: Can signature components predict next-month returns cross-sectionally?
Controls: size, momentum, volatility, turnover
Threshold: |t| > 3.0 (Harvey-Liu-Zhu 2016)
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import numpy as np; import pandas as pd; import torch; import signatory
from scipy import stats; from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm; from statsmodels.stats.multitest import multipletests
import warnings; warnings.filterwarnings('ignore')

CPU=torch.device("cpu"); torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA
# ============================================================
print("="*60)
print("  FAMA-MACBETH: Path Signatures as Pricing Factors")
print("="*60)

# US stocks only for this first test
print("[1/5] Loading US data...")
price=pd.read_parquet("/home/user2/meta_attn/data/us_price_expanded.parquet")
vol=pd.read_parquet("/home/user2/meta_attn/data/us_volume_expanded.parquet")
common=sorted(set(price.columns)&set(vol.columns))
idx=price.index.intersection(vol.index)
price=price.loc[idx,common].ffill().fillna(1e-8)
vol=vol.loc[idx,common].ffill().fillna(1)

# Take top N most complete stocks
comp=price.notna().sum()/len(price)
top_n=2000
top=comp.nlargest(top_n).index.tolist()
price=price[top]; vol=vol[top]
n_a=len(top); n_days=len(price)
print(f"  {n_a} stocks x {n_days} days | {price.index[0].date()} → {price.index[-1].date()}")

# ============================================================
# MONTHLY PATH SIGNATURES (Proposal d=3 embedding)
# ============================================================
print("[2/5] Computing monthly path signatures...")
logP=np.log(np.maximum(price.values.astype(np.float64),1e-8))
logV=np.log(np.maximum(vol.values.astype(np.float64),1e-8))
vi=logV-pd.DataFrame(logV).rolling(20,min_periods=1).mean().values

# Group by month, compute signatures for each stock-month
# Get monthly date boundaries
dates=price.index
months=pd.Series(dates).dt.to_period("M").unique()
sig_depth=3; sig_dim=signatory.signature_channels(3,sig_depth)  # 39
print(f"  {len(months)} months, sig dim={sig_dim}")

# Compute signature for each stock-month
monthly_sigs={}  # {month: array(n_a, sig_dim)}
monthly_rets={}  # {month: array(n_a,)} — next-month forward returns

for mi, month in enumerate(months[:-1]):  # exclude last month (no forward return)
    mask=(dates>=month.start_time)&(dates<=month.end_time)
    month_days=dates[mask]
    if len(month_days)<15: continue  # require at least 15 trading days

    next_month=months[mi+1]
    next_mask=(dates>=next_month.start_time)&(dates<=next_month.end_time)

    sigs_month=np.zeros((n_a,sig_dim),dtype=np.float32)
    rets_month=np.zeros(n_a,dtype=np.float32)

    for ai in range(n_a):
        # Get this stock's daily data in this month
        p=logP[mask,ai]; v=logV[mask,ai]; vi_m=vi[mask,ai]
        if len(p)<15: continue

        # Build d=3 path
        path=np.stack([p,v,vi_m],-1).astype(np.float32)  # (L, 3)
        path_t=torch.FloatTensor(path).unsqueeze(0).to(CPU)  # (1, L, 3)
        sig=signatory.signature(path_t,sig_depth,basepoint=True).cpu().numpy()[0]
        sigs_month[ai]=sig

        # Forward return: logP at end of next month - logP at end of this month
        if next_mask.sum()>0:
            p_next=logP[next_mask,ai]
            rets_month[ai]=p_next[-1]-p[-1] if len(p_next)>0 else 0

    monthly_sigs[month]=sigs_month
    monthly_rets[month]=rets_month
    if (mi+1)%50==0: print(f"  Month {mi+1}/{len(months)-1}")

n_months=len(monthly_sigs)
print(f"  Valid months with signatures: {n_months}")

# ============================================================
# CONTROL VARIABLES
# ============================================================
print("[3/5] Computing control variables...")
# Size proxy: log price at start of month
# Momentum: cumulative return months t-12 to t-2
# Volatility: std of daily returns in month t
# Turnover proxy: log volume / log price ratio

monthly_controls={}  # {month: array(n_a, n_controls)}

for mi, month in enumerate(months[1:-1]):  # need t-12 data
    if month not in monthly_sigs: continue

    mask=(dates>=month.start_time)&(dates<=month.end_time)
    n_controls=4
    controls=np.zeros((n_a,n_controls),dtype=np.float32)

    for ai in range(n_a):
        p_month=price.values[mask,ai]
        v_month=vol.values[mask,ai]
        if len(p_month)<5: continue

        # Size: log price at month start
        controls[ai,0]=np.log(max(p_month[0],1e-8))

        # Volatility
        rets_daily=np.diff(np.log(np.maximum(p_month,1e-8)))
        controls[ai,1]=np.std(rets_daily) if len(rets_daily)>0 else 0

        # Volume/price ratio
        controls[ai,2]=np.log(max(v_month.mean(),1))/(controls[ai,0]+1e-8)

    # Momentum: past 12-month return (excluding current month)
    if mi>=12:
        past_mask=(dates>=months[mi-12].start_time)&(dates<month.start_time)
        for ai in range(n_a):
            p_past=price.values[past_mask,ai]
            if len(p_past)>10:
                controls[ai,3]=np.log(max(p_past[-1],1e-8))-np.log(max(p_past[0],1e-8))

    monthly_controls[month]=controls
    if (mi+1)%50==0: print(f"  Control {mi+1}/{len(months)-1}")

# ============================================================
# FAMA-MACBETH REGRESSIONS
# ============================================================
print("[4/5] Running Fama-MacBeth regressions...")
# For each month, regress: next_month_return ~ sig_components + controls
# Collect coefficient time series for each sig component

aligned_months=sorted(set(monthly_sigs.keys())&set(monthly_rets.keys())&set(monthly_controls.keys()))
print(f"  Aligned months: {len(aligned_months)}")

# Stack data across months
from sklearn.preprocessing import StandardScaler

coef_ts={}  # {sig_idx: [coefs across months]}
tstats={}   # {sig_idx: t-statistic}

# Flatten all sigs and controls across months for scaling
all_sigs_flat=np.concatenate([monthly_sigs[m].reshape(-1,sig_dim) for m in aligned_months])
all_ctrl_flat=np.concatenate([monthly_controls[m] for m in aligned_months])
sig_scaler=RobustScaler().fit(all_sigs_flat)
ctrl_scaler=RobustScaler().fit(all_ctrl_flat)

fm_coefficients=np.zeros((len(aligned_months),sig_dim+4+1))  # sigs + 4 controls + intercept
fm_hac_se=np.zeros((len(aligned_months),sig_dim+4+1))
fm_r2_full=np.zeros(len(aligned_months))
fm_r2_restricted=np.zeros(len(aligned_months))

for i,month in enumerate(aligned_months):
    sigs=sig_scaler.transform(monthly_sigs[month])
    ctrls=ctrl_scaler.transform(monthly_controls[month])
    rets=monthly_rets[month]

    # Remove samples with NaN returns
    valid=np.isfinite(rets)&(np.abs(rets)<0.5)  # filter extreme returns
    if valid.sum()<30: continue

    X=np.concatenate([sigs[valid],ctrls[valid]],axis=1)
    X=sm.add_constant(X)
    y=rets[valid]

    try:
        model=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        fm_coefficients[i]=model.params
        fm_hac_se[i]=model.bse
        fm_r2_full[i]=model.rsquared
        X_restricted=sm.add_constant(ctrls[valid])
        model_restricted=sm.OLS(y,X_restricted).fit()
        fm_r2_restricted[i]=model_restricted.rsquared
    except: pass

    if (i+1)%50==0: print(f"  FM {i+1}/{len(aligned_months)}")

# ============================================================
# RESULTS
# ============================================================
print("[5/5] Results...")
# Remove months where regression failed
valid_mask=fm_coefficients[:,0]!=0
valid_months=valid_mask.sum()
fm_coefficients=fm_coefficients[valid_mask]
fm_hac_se=fm_hac_se[valid_mask]
fm_r2_full=fm_r2_full[valid_mask]
fm_r2_restricted=fm_r2_restricted[valid_mask]
print(f"  Successful months: {len(fm_coefficients)}/{len(aligned_months)}")

# Calculate t-stats for each coefficient
n_vars=sig_dim+4  # sig components + controls
sig_tstats=[]
sig_pvals=[]
n_months_valid=len(fm_coefficients)
for j in range(1,sig_dim+1):  # sig components (skip intercept)
    coefs=fm_coefficients[:,j]
    if len(coefs)>10 and coefs.std()>0:
        fm_mean=coefs.mean()
        fm_se=coefs.std(ddof=1)/np.sqrt(len(coefs))
        t=fm_mean/fm_se
        p=2*stats.t.sf(abs(t),df=len(coefs)-1)
        sig_tstats.append((j-1,t,fm_mean,fm_se))
        sig_pvals.append(p)
    else:
        sig_tstats.append((j-1,0,0,0))
        sig_pvals.append(1.0)

# Control t-stats
ctrl_names=["Size","Volatility","Vol/Price","Momentum"]
ctrl_tstats=[]
for j in range(sig_dim+1,sig_dim+5):
    coefs=fm_coefficients[:,j]
    if len(coefs)>10 and coefs.std()>0:
        fm_mean=coefs.mean()
        fm_se=coefs.std(ddof=1)/np.sqrt(len(coefs))
        t=fm_mean/fm_se
        ctrl_tstats.append((ctrl_names[j-sig_dim-1],t,fm_mean))

# Sort by absolute t-stat
sig_tstats.sort(key=lambda x:abs(x[1]),reverse=True)

print("\n"+"="*65)
print("  FAMA-MACBETH RESULTS: Signature Components as Pricing Factors")
print("="*65)
print(f"\n  Top 15 Signature Components (sorted by |t|):")
print(f"  {'Component':<15} {'|t|':>8} {'Coef':>10} {'Significant?':>12}")
hlz_threshold=3.0
significant=0
for idx,tstat,coef_mean,coef_std in sig_tstats[:15]:
    sig_mark="HLZ ✓" if abs(tstat)>hlz_threshold else ""
    if abs(tstat)>hlz_threshold: significant+=1
    print(f"  Sig_{idx:<11} {abs(tstat):>8.2f} {coef_mean:>10.6f} {sig_mark:>12}")

print(f"\n  Significant at |t|>3.0: {significant}/{sig_dim}")
print(f"\n  Control Variables:")
for name,tstat,coef_mean in ctrl_tstats:
    sig_mark="HLZ ✓" if abs(tstat)>hlz_threshold else ""
    print(f"  {name:<15} |t|={abs(tstat):.2f} coef={coef_mean:.6f} {sig_mark}")

# Economic significance
avg_returns=np.array([monthly_rets[m].mean() for m in aligned_months])
avg_ret=np.mean(avg_returns)*100  # monthly % return
# Incremental cross-sectional R² (full - restricted, across valid months)
r2_valid=(fm_r2_full!=0)&(fm_r2_restricted!=0)
if r2_valid.sum()>0:
    inc_r2_arr=fm_r2_full[r2_valid]-fm_r2_restricted[r2_valid]
    inc_r2_mean=np.mean(inc_r2_arr)
    full_r2_mean=np.mean(fm_r2_full[r2_valid])
else:
    inc_r2_mean=0; full_r2_mean=0
print(f"\n  Avg monthly return: {avg_ret:.4f}%")
print(f"  Mean full-model cross-sectional R²: {full_r2_mean:.6f}")
print(f"  Mean incremental R² (signatures only): {inc_r2_mean:.6f}")

# Economic significance for top sig components (bps = coef * 10000)
print(f"\n  Economic Significance — Top Components (monthly coef in bps):")
print(f"  {'Component':<15} {'Coef (bps)':>12} {'|t|':>8}")
for idx,tstat,coef_mean,coef_se in sig_tstats[:10]:
    bps=coef_mean*10000
    print(f"  Sig_{idx:<11} {bps:>12.2f} {abs(tstat):>8.2f}")

# Multiple testing correction across all sig components
pvals_array=np.array(sig_pvals)
_,pvals_holm,_,_=multipletests(pvals_array,method='holm')
_,pvals_bh,_,_=multipletests(pvals_array,method='fdr_bh')
n_holm_sig=(pvals_holm<0.05).sum()
n_bh_sig=(pvals_bh<0.05).sum()
print(f"\n  Multiple Testing Correction (alpha=0.05):")
print(f"  Holm-Bonferroni significant: {n_holm_sig}/{sig_dim}")
print(f"  Benjamini-Hochberg FDR significant: {n_bh_sig}/{sig_dim}")
print("="*65)
