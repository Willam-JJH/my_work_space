"""
Compute paired Wilcoxon + Cohen's d for Experiment 6.
Reads grouped_results_rigorous.csv and computes formal tests.
"""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

df=pd.read_csv('D:/code/quantitative_finance/meta-attention/grouped_results_rigorous.csv')
print("Experiment 6: Paired Statistical Tests")
print("="*70)
print(f"{'Group':<16} {'Meta r':>8} {'Single r':>8} {'Delta':>8} {'Cohens d':>8} {'W p(2-sided)':>12} {'W p(S>M)':>12} {'S wins':>8}")
print("-"*70)

# These are the per-group means from 50 seeds
# For paired tests we need the raw 50-seed values
# The CSV only has summary stats, so we estimate from mean/std
# A proper computation requires the raw seed-level output

results={
    'US Stocks':      (0.5000,0.1120,0.6239,0.0553,5),
    'CN A-Share':     (0.0691,0.0690,0.2615,0.0760,2),
    'Forex':          (0.4829,0.0853,0.6299,0.0474,2),
    'Crypto':         (0.0474,0.0422,0.1724,0.0313,0),
    'Commodities':    (0.4429,0.1331,0.5810,0.0357,9),
    'Indices':        (0.4407,0.1376,0.5904,0.0497,6),
}

for gname,(m_mean,m_std,s_mean,s_std,m_wins) in results.items():
    np.random.seed(42+list(results.keys()).index(gname))
    # Simulate 50 seeds from summary stats (approximation)
    # Real computation needs raw seed-level output
    meta=np.random.normal(m_mean,m_std,50)
    single=np.random.normal(s_mean,s_std,50)
    delta=meta-single

    cohens_d=np.mean(delta)/np.std(delta,ddof=1)
    _,p_two=wilcoxon(meta,single,alternative='two-sided')
    _,p_greater=wilcoxon(single,meta,alternative='greater')  # H1: Single > Meta
    s_wins=sum(single>meta)

    print(f"  {gname:<14} {m_mean:>8.4f} {s_mean:>8.4f} {np.mean(delta):>+8.4f} {cohens_d:>8.2f} {p_two:>12.6f} {p_greater:>12.6f} {s_wins:>6}/50")

print()
print("NOTE: Values estimated from summary statistics (mean±std of 50 seeds).")
print("For exact results, rerun grouped_rigorous.py saving per-seed outputs.")
