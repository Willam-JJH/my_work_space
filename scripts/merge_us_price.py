"""
Merge US price parquet files into a single clean, deduplicated file.

Strategy: For each ticker (column), keep data from the source file that
provides the most non-null observations (largest coverage).
"""

import pandas as pd
import os
import time

DATA_DIR = "D:/code/data"

# ---------------------------------------------------------------------------
# Step 1: List all parquet files and report metadata
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: All parquet files in data directory")
print("=" * 70)

parquet_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])
file_metadata = []

for fname in parquet_files:
    path = os.path.join(DATA_DIR, fname)
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)

    try:
        df = pd.read_parquet(path)
        n_cols = len(df.columns)
        n_rows = len(df)

        # Determine date range
        if isinstance(df.index, pd.DatetimeIndex):
            date_min = df.index.min().strftime("%Y-%m-%d")
            date_max = df.index.max().strftime("%Y-%m-%d")
        elif "date" in df.columns:
            date_min = str(df["date"].min())
            date_max = str(df["date"].max())
        else:
            date_min = "N/A"
            date_max = "N/A"

        file_metadata.append({
            "filename": fname,
            "columns": n_cols,
            "rows": n_rows,
            "date_min": date_min,
            "date_max": date_max,
            "size_mb": round(size_mb, 1),
        })
        print(f"  {fname:35s}  columns={n_cols:<5d}  rows={n_rows:<6d}"
              f"  dates={date_min} to {date_max}  size={size_mb:>8.1f} MB")
    except Exception as e:
        print(f"  {fname:35s}  ERROR: {e}")

# ---------------------------------------------------------------------------
# Step 2: Load the three US price files
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2: Loading US price files for merging")
print("=" * 70)

files_to_merge = {
    "us_price.parquet":         os.path.join(DATA_DIR, "us_price.parquet"),
    "us_price_expanded.parquet": os.path.join(DATA_DIR, "us_price_expanded.parquet"),
    "us_deep_history.parquet":  os.path.join(DATA_DIR, "us_deep_history.parquet"),
}

dataframes = {}
for label, path in files_to_merge.items():
    t0 = time.time()
    df = pd.read_parquet(path)
    elapsed = time.time() - t0
    print(f"  Loaded {label:35s}  shape={df.shape}  in {elapsed:.1f}s")
    dataframes[label] = df

# ---------------------------------------------------------------------------
# Step 3: Merge with largest-coverage-per-column strategy
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3: Merging — keeping largest coverage per ticker")
print("=" * 70)

# For each file, count non-null values per column
coverage = {}
for label, df in dataframes.items():
    nonnull_counts = df.notna().sum()
    coverage[label] = nonnull_counts
    print(f"  {label:35s}  tickers={len(nonnull_counts):<6d}  "
          f"median nonnull coverage={nonnull_counts.median():.0f}")

# Collect all tickers across all three files
all_tickers = sorted(set().union(*[set(df.columns) for df in dataframes.values()]))
print(f"\n  Total unique tickers across all files: {len(all_tickers)}")

# Build merged dataframe column by column
merged_parts = []
per_ticker_source = []  # tracking which source won

for ticker in all_tickers:
    best_label = None
    best_count = -1

    for label in files_to_merge:
        if ticker in dataframes[label].columns:
            cnt = coverage[label].get(ticker, 0)
            if cnt > best_count:
                best_count = cnt
                best_label = label

    if best_label is not None:
        series = dataframes[best_label][ticker].copy()
        # Rename the series so it becomes a column later
        series.name = ticker
        merged_parts.append(series)
        per_ticker_source.append((ticker, best_label, best_count))

# Concatenate all columns into the merged frame
print("  Concatenating columns...")
t0 = time.time()
merged = pd.concat(merged_parts, axis=1)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s — merged shape: {merged.shape}")

# Ensure index is datetime
if not isinstance(merged.index, pd.DatetimeIndex):
    merged.index = pd.to_datetime(merged.index)

# Sort index and columns
merged.sort_index(inplace=True)
merged = merged[sorted(merged.columns)]

# Count source contributions
from collections import Counter
src_counter = Counter(entry[1] for entry in per_ticker_source)
print("\n  Column source breakdown:")
for label, cnt in src_counter.most_common():
    print(f"    {label:35s}  {cnt:>5d} tickers")

# ---------------------------------------------------------------------------
# Step 4: Deduplicate rows (same index timestamp)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 4: Deduplication")
print("=" * 70)
before = len(merged)
# If there are duplicate index entries, average them (for safety)
if merged.index.duplicated().any():
    n_dup = merged.index.duplicated().sum()
    print(f"  Found {n_dup} duplicate index entries — aggregating with mean")
    merged = merged.groupby(level=0).mean()
else:
    print(f"  No duplicate index entries found")
after = len(merged)
print(f"  Rows before={before}, after={after}")

# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 5: Saving to us_price_full.parquet")
print("=" * 70)

output_path = os.path.join(DATA_DIR, "us_price_full.parquet")
t0 = time.time()
merged.to_parquet(output_path, index=True)
elapsed = time.time() - t0

out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"  Saved to {output_path}")
print(f"  Time: {elapsed:.1f}s  Size: {out_size_mb:.1f} MB")

# ---------------------------------------------------------------------------
# Step 6: Final summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Count unique stocks (columns excluding special ones)
skip_cols = {"date", "Date"}
ticker_cols = [c for c in merged.columns if c not in skip_cols]
unique_stocks = len(ticker_cols)

date_min = merged.index.min().strftime("%Y-%m-%d")
date_max = merged.index.max().strftime("%Y-%m-%d")
total_rows = len(merged)
total_cols = len(merged.columns)

print(f"  File:             {output_path}")
print(f"  Unique US stocks: {unique_stocks}")
print(f"  Date range:       {date_min} to {date_max}")
print(f"  Total rows:       {total_rows}")
print(f"  Total columns:    {total_cols}")
print(f"  Size:             {out_size_mb:.1f} MB")
print(f"  Non-null values:  {merged.notna().sum().sum():,}")
print(f"  Coverage:         {merged.notna().sum().sum() / (total_rows * total_cols) * 100:.1f}%")
print("=" * 70)
