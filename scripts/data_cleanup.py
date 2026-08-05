"""Data directory cleanup: inspect parquet files, remove subsets, delete stale pickles."""
import os
import sys
import glob

DATA_DIR = "D:/code/data"

def human_size(n_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"

def main():
    import pandas as pd  # lazy import so we can fail early if pandas is missing

    parquet_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))

    if not parquet_files:
        print("No parquet files found in", DATA_DIR)
        return

    # ------------------------------------------------------------------ #
    # 1 & 2. Read each file and collect metadata
    # ------------------------------------------------------------------ #
    summaries = []
    for fp in parquet_files:
        fname = os.path.basename(fp)
        try:
            df = pd.read_parquet(fp)
        except Exception as exc:
            print(f"  ERROR reading {fname}: {exc}")
            continue

        cols = list(df.columns)
        n_rows = len(df)

        # Try to infer a date range from a datetime-like index or column
        date_range = None
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            date_range = f"{idx.min().date()} to {idx.max().date()}"
        else:
            # look for a column named 'date' or 'Date'
            for date_col in ("date", "Date", "DATE", "datetime", "time", "timestamp"):
                if date_col in df.columns:
                    try:
                        dmin = pd.to_datetime(df[date_col]).min()
                        dmax = pd.to_datetime(df[date_col]).max()
                        date_range = f"{dmin.date()} to {dmax.date()}"
                    except Exception:
                        pass
                    break

        summary = {
            "file": fname,
            "path": fp,
            "cols": cols,
            "n_rows": n_rows,
            "size": os.path.getsize(fp),
            "date_range": date_range or "N/A",
            "ticker_cols": [c for c in cols if "ticker" in c.lower()],
        }
        summaries.append(summary)

        print(f"File  : {fname}")
        print(f"  Columns ({len(cols)}): {cols}")
        print(f"  Rows  : {n_rows:,}")
        print(f"  Size  : {human_size(summary['size'])}")
        print(f"  Dates : {summary['date_range']}")
        print()

    # ------------------------------------------------------------------ #
    # 3. Duplicate ticker columns across files
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Ticker columns per file:")
    for s in summaries:
        print(f"  {s['file']}: {s['ticker_cols']}")

    # Check if the same ticker column name appears in multiple files
    all_ticker_sets = {s["file"]: set(s["ticker_cols"]) for s in summaries}
    from collections import Counter
    ticker_counter: Counter = Counter()
    for ticker_set in all_ticker_sets.values():
        for tc in ticker_set:
            ticker_counter[tc] += 1
    dupes = {k: v for k, v in ticker_counter.items() if v > 1}
    if dupes:
        print("\nDuplicate ticker column names across files:")
        for tc, cnt in dupes.items():
            appearing = [f for f, ts in all_ticker_sets.items() if tc in ts]
            print(f"  '{tc}' appears in {cnt} files: {appearing}")
    else:
        print("\nNo duplicate ticker column names across files.")
    print()

    # ------------------------------------------------------------------ #
    # 4. Remove files that are strict subsets of another file
    #    (A is a strict subset of B if A's columns are a subset of B's
    #     AND A has fewer rows, same ticker coverage roughly)
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Checking for strict subset files ...")
    to_remove = set()
    for i, a in enumerate(summaries):
        for j, b in enumerate(summaries):
            if i == j or a["file"] in to_remove or b["file"] in to_remove:
                continue
            # a is subset of b: a's cols subset of b's cols, and a's rows <= b's rows
            if set(a["cols"]).issubset(set(b["cols"])) and a["n_rows"] <= b["n_rows"]:
                # Now check if the ticker columns actually have identical content
                # (i.e., the overlapping tickers hold identical data)
                shared_ticker_cols = [tc for tc in a["ticker_cols"] if tc in b["ticker_cols"]]
                is_subset = True
                if shared_ticker_cols:
                    try:
                        df_a = pd.read_parquet(a["path"])
                        df_b = pd.read_parquet(b["path"])
                        for tc in shared_ticker_cols:
                            # Compare sorted unique tickers
                            tickers_a = set(df_a[tc].dropna().unique())
                            tickers_b = set(df_b[tc].dropna().unique())
                            if not tickers_a.issubset(tickers_b):
                                is_subset = False
                                break
                    except Exception as exc:
                        print(f"  Warning: could not compare {a['file']} vs {b['file']}: {exc}")
                        is_subset = False

                if is_subset and set(a["cols"]).issubset(set(b["cols"])):
                    print(f"  SUBSET: {a['file']} ({a['n_rows']} rows, {len(a['cols'])} cols)"
                          f" is subset of {b['file']} ({b['n_rows']} rows, {len(b['cols'])} cols)")
                    to_remove.add(a["file"])

    for f in sorted(to_remove):
        fp = os.path.join(DATA_DIR, f)
        os.remove(fp)
        print(f"  Removed: {f}")
    if not to_remove:
        print("  No subset files found.")
    print()

    # ------------------------------------------------------------------ #
    # 5. Delete old partial / checkpoint pickle files
    # ------------------------------------------------------------------ #
    print("=" * 60)
    pkl_patterns = [
        "cn_ckpt.pkl", "cn_vol_ckpt.pkl", "us_ckpt.pkl", "us_vol_ckpt.pkl",
        "cn_price_ckpt.pkl", "cn_volume_ckpt.pkl",
        "us_price_ckpt.pkl", "us_volume_ckpt.pkl",
        "crypto_ckpt.pkl", "other_ckpt.pkl",
        "*.partial.pkl", "*.checkpoint.pkl", "*.tmp.pkl",
    ]
    deleted_pkl = 0
    for pattern in pkl_patterns:
        for fp in glob.glob(os.path.join(DATA_DIR, pattern)):
            os.remove(fp)
            print(f"  Deleted: {os.path.basename(fp)}")
            deleted_pkl += 1
    if deleted_pkl == 0:
        print("  No stale pickle files found.")
    print()

    # ------------------------------------------------------------------ #
    # 6. Final clean file list with sizes
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Final clean file list:")
    remaining = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    for fp in remaining:
        sz = os.path.getsize(fp)
        print(f"  {os.path.basename(fp):45s} {human_size(sz):>8s}")
    total_size = sum(os.path.getsize(fp) for fp in remaining)
    print(f"  {'':45s} {'-----':>8s}")
    print(f"  {'Total':45s} {human_size(total_size):>8s}")
    print()

if __name__ == "__main__":
    main()
