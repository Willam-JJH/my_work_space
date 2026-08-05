"""
Self-contained clean experiment — price floor + completeness filter.
Compares Baseline (SlimMetaTrans) vs P0 (BlockAttnRes) on filtered data.
Runs only baseline + P0 (P2, P1b have known issues; skipped).
"""

import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, time, os, gc, warnings, sys, json, argparse, traceback
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
N_STOCKS_US  = 1500       # reduced from 3000 — clean survivors
N_STOCKS_CN  = 1000       # reduced from 3000 — clean survivors
PRICE_FLOOR_US = 5.0
PRICE_FLOOR_CN = 5.0
COMPLETENESS_FLOOR = 0.80
WINDOW       = 250
PRED_HORIZON = 21
SIG_DEPTH    = 3
D_MODEL      = 128
N_LAYERS     = 4
N_HEADS      = 4
D_FF         = 256
DROPOUT      = 0.1
BATCH        = 2
EPOCHS       = 40
DATA_DIR     = "/home/user2/meta_attn/data"

# ---- GPU setup ----
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

if GPU.type == "cuda":
    dev_props = torch.cuda.get_device_properties(0)
    vram_gb = dev_props.total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"TF32: enabled  |  AMP: available={torch.cuda.amp.autocast is not None}")

torch.manual_seed(42); np.random.seed(42)

# ============================================================
# ORIGINAL (unfiltered) results for comparison
# ============================================================
ORIGINAL_RESULTS = {
    "US": {
        "baseline": {"mse": 0.0197, "model_r": 0.0814, "pred_r": 0.3384, "delta": -0.2570},
        "p0":       {"mse": 0.0182, "model_r": 0.3356, "pred_r": 0.3191, "delta": +0.0165},
    },
    "CN": {
        "baseline": {"mse": 0.0465, "model_r": -0.2356, "pred_r": 0.6474, "delta": -0.8830},
        "p0":       {"mse": 0.0305, "model_r": 0.0760,  "pred_r": 0.3907, "delta": -0.3147},
    },
}

# ============================================================
# DATA LOADING — with price floor + completeness filter
# ============================================================

def compute_signatures_batched(logP, logV, vi, n_a, window, depth, split_idx, val_idx, device="cpu"):
    """Compute signatures WITHOUT storing (n_samp, n_a, window, 3) tensor.
    Process all assets in one batch via signatory — 50x faster than per-asset loop."""
    C = 3
    sig_dim = signatory.signature_channels(C, depth)
    n_days = logP.shape[0]
    n_samp = n_days - window - PRED_HORIZON + 1

    chunk_size = min(64, n_samp)
    n_chunks = (n_samp + chunk_size - 1) // chunk_size

    all_sigs = []
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, n_samp)
        paths_chunk = np.zeros((end - start, n_a, window, C), dtype=np.float32)
        for i, si in enumerate(range(start, end)):
            paths_chunk[i] = np.stack([
                logP[si:si + window], logV[si:si + window], vi[si:si + window]
            ], -1).transpose(1, 0, 2)
        flat = paths_chunk.reshape(-1, window, C)
        sigs_flat = np.zeros((flat.shape[0], sig_dim), dtype=np.float32)
        pt = torch.FloatTensor(flat).to(device)
        for bi in range(0, flat.shape[0], 256):
            sigs_flat[bi:bi + 256] = signatory.signature(pt[bi:bi + 256], depth, basepoint=True).cpu().numpy()
        all_sigs.append(sigs_flat.reshape(end - start, n_a, sig_dim))
        if ci % 10 == 0:
            print(f"    sig chunk {ci + 1}/{n_chunks}")
        del paths_chunk, flat, pt; gc.collect()

    sigs = np.concatenate(all_sigs); del all_sigs; gc.collect()

    from sklearn.preprocessing import RobustScaler
    sigs = np.nan_to_num(sigs, nan=0, posinf=0, neginf=0)
    flat = sigs.reshape(-1, sig_dim)
    p1, p99 = np.percentile(flat, [1, 99], axis=0)
    sigs = np.clip(sigs, p1, p99)
    sc = RobustScaler().fit(sigs[:split_idx].reshape(-1, sig_dim))
    sigs_tr = sc.transform(sigs[:split_idx].reshape(-1, sig_dim)).reshape(split_idx, n_a, sig_dim)
    sigs_val = sc.transform(sigs[split_idx:val_idx].reshape(-1, sig_dim)).reshape(val_idx - split_idx, n_a, sig_dim)
    sigs_te = sc.transform(sigs[val_idx:].reshape(-1, sig_dim)).reshape(n_samp - val_idx, n_a, sig_dim)
    return sigs_tr, sigs_val, sigs_te, sig_dim


def load_market(name, n_stocks_target, price_path, vol_path, price_floor):
    """Load market data with price floor + completeness filter, then split and compute returns.

    Returns:
        (logP, logV, vi, n_a, split, val_split, n_samp)   — path data
        (X_ret_tr, X_ret_val, X_ret_te)                   — return data
        (y_fwd[:split], y_fwd[split:val_split], y_fwd[val_split:])  — labels
    """
    # ---- Load raw data ----
    print(f"\n  --- Loading {name} from parquet ---")
    price = pd.read_parquet(price_path)
    vol = pd.read_parquet(vol_path)
    common = sorted(set(price.columns) & set(vol.columns))
    idx = price.index.intersection(vol.index)
    price = price.loc[idx, common]
    vol = vol.loc[idx, common]
    n_raw = price.shape[1]
    print(f"  Raw: {n_raw} stocks x {len(idx)} days")

    # ---- Filter A: Price floor (remove penny stocks) ----
    medians = price.median()
    valid_price = medians[medians >= price_floor].index.tolist()
    n_removed_price = n_raw - len(valid_price)
    price = price[valid_price]
    vol = vol[valid_price]
    print(f"  PRICE_FLOOR >= ${price_floor:.1f}:  removed {n_removed_price} stocks, {len(valid_price)} survive")

    # ---- Filter B: Completeness floor ----
    comp = price.notna().sum() / len(price)
    valid_comp = comp[comp >= COMPLETENESS_FLOOR].index.tolist()
    n_removed_comp = len(valid_price) - len(valid_comp)
    price = price[valid_comp]
    vol = vol[valid_comp]
    print(f"  COMPLETENESS >= {COMPLETENESS_FLOOR:.0%}:  removed {n_removed_comp} stocks, {len(valid_comp)} survive")

    # ---- Select top N from survivors (cleanest) ----
    # Recompute completeness on surviving pool for ranking
    comp_final = price.notna().sum() / len(price)
    top_n = min(n_stocks_target, len(comp_final))
    top_ids = comp_final.nlargest(top_n).index.tolist()
    price = price[top_ids].ffill().fillna(1e-8)
    vol = vol[top_ids].ffill().fillna(1)
    n_final = len(top_ids)
    print(f"  Final selection: top {n_final} / {len(comp_final)} survivors (target={n_stocks_target})")

    # ---- Build log-price / log-volume / vol-innovation ----
    logP = np.log(np.maximum(price.values.astype(np.float64), 1e-8))
    logV = np.log(np.maximum(vol.values.astype(np.float64), 1e-8))
    vi = logV - pd.DataFrame(logV).rolling(20, min_periods=1).mean().values
    n_a = n_final
    n_days = logP.shape[0]
    n_samp = n_days - WINDOW - PRED_HORIZON + 1

    train_end = np.searchsorted(price.index, pd.Timestamp('2015-01-01'))
    val_end = np.searchsorted(price.index, pd.Timestamp('2020-01-01'))
    split = max(WINDOW + PRED_HORIZON, train_end) - WINDOW - PRED_HORIZON + 1
    val_split = max(split + WINDOW, val_end - WINDOW - PRED_HORIZON + 1)

    y_fwd = logP[WINDOW + PRED_HORIZON - 1:WINDOW + PRED_HORIZON - 1 + n_samp] - logP[WINDOW - 1:WINDOW - 1 + n_samp]
    y_fwd = np.nan_to_num(y_fwd, nan=0, posinf=0, neginf=0)
    y_fwd = np.clip(y_fwd, -np.percentile(np.abs(y_fwd), 99), np.percentile(np.abs(y_fwd), 99))

    ret_vals = np.diff(logP, axis=0)
    ret_vals = np.vstack([np.zeros((1, n_a)), ret_vals])
    X_ret_tr = np.zeros((split, n_a, 30), dtype=np.float32)
    X_ret_val = np.zeros((val_split - split, n_a, 30), dtype=np.float32)
    X_ret_te = np.zeros((n_samp - val_split, n_a, 30), dtype=np.float32)
    for i in range(split):
        X_ret_tr[i] = ret_vals[i + WINDOW - 30:i + WINDOW].T
    for i, si in enumerate(range(split, val_split)):
        X_ret_val[i] = ret_vals[si + WINDOW - 30:si + WINDOW].T
    for i, si in enumerate(range(val_split, n_samp)):
        X_ret_te[i] = ret_vals[si + WINDOW - 30:si + WINDOW].T
    for x in [X_ret_tr, X_ret_val, X_ret_te]:
        np.clip(x, -np.percentile(np.abs(x), 99), np.percentile(np.abs(x), 99), out=x)

    print(f"  {n_a} stocks x {n_days}d | Train:{split} Val:{val_split - split} Test:{n_samp - val_split}")
    return (logP, logV, vi, n_a, split, val_split, n_samp), \
           (X_ret_tr, X_ret_val, X_ret_te), \
           (y_fwd[:split], y_fwd[split:val_split], y_fwd[val_split:])


def classical_factors(X_ret):
    """Compute classical factors: volatility, momentum, RSI."""
    v5 = X_ret[:, :, -5:].std(-1, keepdims=True)
    m5 = X_ret[:, :, -5:].mean(-1, keepdims=True)
    m10 = X_ret[:, :, -10:].mean(-1, keepdims=True)
    rsi = (X_ret[:, :, -5:] > 0).mean(-1, keepdims=True)
    return np.concatenate([v5, m5, m10, rsi], -1)


# ============================================================
# BASELINE MODEL — SlimMetaTrans (inlined)
# ============================================================
class BaselineSlimMetaTrans(nn.Module):
    """Inline copy of SlimMetaTrans — the baseline model."""
    def __init__(self, n_a, sig_dim, cl_dim):
        super().__init__()
        self.na = n_a
        self.d = D_MODEL
        self.sig_proj = nn.Sequential(nn.Linear(sig_dim, self.d), nn.LayerNorm(self.d))
        self.sig_pos = nn.Parameter(torch.randn(1, 5000, self.d) * 0.02)
        self.cl_proj = nn.Sequential(nn.Linear(cl_dim, self.d), nn.LayerNorm(self.d))
        self.cl_tok = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(self.d, N_HEADS, D_FF, DROPOUT, 'gelu', True, True)
            for _ in range(N_LAYERS)
        ])
        self.ret_head = nn.Linear(self.d, n_a)
        self.unc_head = nn.Sequential(nn.Linear(self.d, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, sigs, cl):
        B = sigs.shape[0]
        st = self.sig_proj(sigs) + self.sig_pos[:, :self.na, :]
        cp = cl.mean(1)
        ct = self.cl_proj(cp).unsqueeze(1) + self.cl_tok
        x = torch.cat([self.cls.expand(B, -1, -1), st, ct], 1)
        for l in self.layers:
            x = l(x)
        return self.ret_head(x[:, 0]), self.unc_head(x[:, 0]).squeeze(-1)


# ============================================================
# MODEL FACTORY — baseline + P0 only
# ============================================================
_MODELS = {}

def baseline_factory(n_a, sig_dim, cl_dim):
    return BaselineSlimMetaTrans(n_a, sig_dim, cl_dim)

def _register_baseline():
    _MODELS["baseline"] = baseline_factory

def _maybe_import_p0():
    try:
        from p0_attn_res import BlockAttnResTransformer
        def p0_factory(n_a, sig_dim, cl_dim):
            return BlockAttnResTransformer(n_a, sig_dim, cl_dim, d_model=D_MODEL,
                                           n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT)
        _MODELS["p0"] = p0_factory
        print("  P0  (BlockAttnRes)      — imported OK")
    except Exception as e:
        print(f"  P0  (BlockAttnRes)      — SKIP: {e}")
        _MODELS["p0"] = None


# ============================================================
# SHARED TRAINING LOOP
# ============================================================
def compute_loss(ret_pred, unc_pred, y):
    """Huber(ret_pred, y) + 0.1 * MSE(unc_pred, |error|)"""
    loss_r = F.huber_loss(ret_pred, y, delta=1.0)
    with torch.no_grad():
        err = (ret_pred - y).abs().mean(1)
    loss_u = 0.1 * F.mse_loss(unc_pred, err)
    return loss_r + loss_u


def train_model(model, tr_loader, sigs_val_n, cl_val_n, y_val, epochs, device,
                use_amp=True, label="model"):
    """Shared training loop. Returns best_val_mse, total_time_secs."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    best_val = float("inf")
    t_start = time.time()

    for ep in range(epochs):
        model.train()
        tl = 0.0
        for s, c, y_b in tr_loader:
            s_g = s.to(device, non_blocking=True)
            c_g = c.to(device, non_blocking=True)
            y_g = y_b.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                out = model(s_g, c_g)
                ret_pred, unc_pred = out[0], out[1]
                loss = compute_loss(ret_pred, unc_pred, y_g)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        sch.step()

        # Validation (batched)
        model.eval()
        val_mse = 0.0; nb = 0
        with torch.no_grad():
            for i in range(0, len(sigs_val_n), BATCH * 8):
                sv = torch.FloatTensor(sigs_val_n[i:i + BATCH * 8]).to(device)
                cv = torch.FloatTensor(cl_val_n[i:i + BATCH * 8]).to(device)
                yv = torch.FloatTensor(y_val[i:i + BATCH * 8]).to(device)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    out = model(sv, cv)
                    rv = out[0]
                val_mse += F.mse_loss(rv, yv).item() * sv.shape[0]
                nb += sv.shape[0]
        val_mse /= nb
        if val_mse < best_val:
            best_val = val_mse

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (ep + 1) % max(1, epochs // 4) == 0:
            print(f"    {label}  Epoch {ep + 1:3d}/{epochs} | Loss: {tl / max(len(tr_loader), 1):.6f} | Val MSE: {val_mse:.6f}")

    t_train = time.time() - t_start
    return best_val, t_train


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, sigs_te_n, cl_te_n, y_te, device, use_amp=True):
    """Return dict: mse, model_r, pred_r, delta."""
    model.eval()
    rps, ups = [], []
    with torch.no_grad():
        for i in range(0, len(sigs_te_n), BATCH * 8):
            sv = torch.FloatTensor(sigs_te_n[i:i + BATCH * 8]).to(device)
            cv = torch.FloatTensor(cl_te_n[i:i + BATCH * 8]).to(device)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                out = model(sv, cv)
                rp, up = out[0], out[1]
            rps.append(rp.cpu().numpy())
            ups.append(up.cpu().numpy())
    rp = np.concatenate(rps)
    up = np.concatenate(ups)
    mse = float(np.mean((rp - y_te) ** 2))
    err = np.abs(rp - y_te).mean(1)
    bl = np.abs(rp).mean(1)
    model_r = float(spearmanr(up, err)[0]) if len(err) > 2 else 0.0
    pred_r = float(spearmanr(bl, err)[0]) if len(err) > 2 else 0.0
    return {"mse": mse, "model_r": model_r, "pred_r": pred_r, "delta": model_r - pred_r}


# ============================================================
# RUN ONE MODEL ON ONE MARKET
# ============================================================
def run_model_on_market(model_name, factory_fn, market_name, path_data, ret_data, y_data, epochs, device, use_amp=True):
    """Full pipeline for one model on one market. Returns result dict or None on failure."""
    (logP, logV, vi, n_a, split, val_split, n_samp) = path_data
    (Xr_tr, Xr_val, Xr_te) = ret_data
    (y_tr, y_val, y_te) = y_data

    print(f"\n  {'=' * 45}")
    print(f"  [{model_name}] on {market_name}  (n_a={n_a}, epochs={epochs})")
    print(f"  {'=' * 45}")

    # Compute signatures
    t0 = time.time()
    sigs_tr, sigs_val, sigs_te, sig_dim = compute_signatures_batched(
        logP, logV, vi, n_a, WINDOW, SIG_DEPTH, split, val_split,
        device=(device if device.type == "cuda" else "cpu")
    )
    print(f"  Signatures: {sig_dim}d | {time.time() - t0:.0f}s")

    # Classical factors
    cl_tr = classical_factors(Xr_tr)
    cl_val = classical_factors(Xr_val)
    cl_te = classical_factors(Xr_te)
    cl_dim = cl_tr.shape[-1]

    # Scale inputs
    from sklearn.preprocessing import RobustScaler as RS
    sc_s = RS().fit(sigs_tr.reshape(-1, sig_dim))
    sc_c = RS().fit(cl_tr.reshape(-1, cl_dim))
    sigs_tr_n = sc_s.transform(sigs_tr.reshape(-1, sig_dim)).reshape(sigs_tr.shape)
    sigs_val_n = sc_s.transform(sigs_val.reshape(-1, sig_dim)).reshape(sigs_val.shape)
    sigs_te_n = sc_s.transform(sigs_te.reshape(-1, sig_dim)).reshape(sigs_te.shape)
    cl_tr_n = sc_c.transform(cl_tr.reshape(-1, cl_dim)).reshape(cl_tr.shape)
    cl_val_n = sc_c.transform(cl_val.reshape(-1, cl_dim)).reshape(cl_val.shape)
    cl_te_n = sc_c.transform(cl_te.reshape(-1, cl_dim)).reshape(cl_te.shape)

    # Build model
    try:
        model = factory_fn(n_a, sig_dim, cl_dim).to(device)
    except Exception as e:
        print(f"  ERROR building model: {e}")
        return None

    n_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_p:,}")

    # DataLoader
    tr_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(sigs_tr_n), torch.FloatTensor(cl_tr_n), torch.FloatTensor(y_tr)
    )
    tr_ld = DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True)

    # Train
    try:
        best_val, t_train = train_model(model, tr_ld, sigs_val_n, cl_val_n, y_val,
                                        epochs, device, use_amp, label=model_name)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM during training — skipping {model_name} on {market_name}")
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM during training — skipping {model_name} on {market_name}")
            torch.cuda.empty_cache()
            return None
        raise

    # Evaluate
    ev = evaluate_model(model, sigs_te_n, cl_te_n, y_te, device, use_amp)
    ev["market"] = market_name
    ev["model"] = model_name
    ev["params"] = n_p
    ev["best_val_mse"] = float(best_val)
    ev["train_time_s"] = round(t_train, 1)
    ev["n_a"] = n_a
    ev["sig_dim"] = sig_dim

    # Inline comparison with original
    if market_name in ORIGINAL_RESULTS and model_name in ORIGINAL_RESULTS[market_name]:
        orig = ORIGINAL_RESULTS[market_name][model_name]
        print(f"  Test MSE: {ev['mse']:.6f} (orig={orig['mse']:.4f})  |  "
              f"Model r: {ev['model_r']:.4f} (orig={orig['model_r']:.4f})  |  "
              f"Pred r: {ev['pred_r']:.4f} (orig={orig['pred_r']:.4f})  |  "
              f"Delta: {ev['delta']:+.4f} (orig={orig['delta']:+.4f})  |  "
              f"Time: {t_train:.0f}s")
    else:
        print(f"  Test MSE: {ev['mse']:.6f} | Model r: {ev['model_r']:.4f} | Pred r: {ev['pred_r']:.4f} | Delta: {ev['delta']:+.4f} | Time: {t_train:.0f}s")

    del model; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ev


# ============================================================
# COMPARISON TABLE
# ============================================================
def print_final_comparison(results):
    """Print side-by-side comparison: clean vs original."""
    print("\n" + "=" * 80)
    print("  === CLEAN vs ORIGINAL ===")
    print("=" * 80)

    # Build lookup: (market, model) -> clean_result
    clean_lookup = {}
    for r in results:
        clean_lookup[(r["market"], r["model"])] = r

    all_markets = ["US", "CN"]
    all_models = ["baseline", "p0"]
    metrics = [("MSE", "mse"), ("Model r", "model_r"), ("Pred r", "pred_r"), ("Delta", "delta")]

    for metric_label, metric_key in metrics:
        header_parts = [f"{metric_label:8s}"]
        for mkt in all_markets:
            for mdl in all_models:
                header_parts.append(f"{mkt}-{mdl:8s}")
        print("".join(header_parts))

        # Row: orig values
        orig_parts = [f"{'orig':8s}"]
        for mkt in all_markets:
            for mdl in all_models:
                val = ORIGINAL_RESULTS[mkt][mdl][metric_key]
                orig_parts.append(f"{val:<12.4f}")
        print("".join(orig_parts))

        # Row: clean values
        clean_parts = [f"{'clean':8s}"]
        for mkt in all_markets:
            for mdl in all_models:
                key = (mkt, mdl)
                if key in clean_lookup:
                    val = clean_lookup[key][metric_key]
                    clean_parts.append(f"{val:<12.4f}")
                else:
                    clean_parts.append(f"{'N/A':<12}")
        print("".join(clean_parts))

        # Row: difference
        diff_parts = [f"{'diff':8s}"]
        for mkt in all_markets:
            for mdl in all_models:
                key = (mkt, mdl)
                if key in clean_lookup:
                    diff = clean_lookup[key][metric_key] - ORIGINAL_RESULTS[mkt][mdl][metric_key]
                    diff_parts.append(f"{diff:<+12.4f}")
                else:
                    diff_parts.append(f"{'N/A':<12}")
        print("".join(diff_parts))
        print()

    print("=" * 80)

    # Also print the original comparison table format
    print("\n" + "=" * 120)
    print("  DETAILED RESULTS TABLE")
    print("=" * 120)
    header = f"  {'Market':<6} {'Model':<8} {'Clean?':<7} {'n_a':<5} {'MSE':<12} {'Model r':<9} {'Pred r':<9} {'Delta':<9} {'#Params':<9} {'Time(s)':<8} {'ValMSE':<12}"
    print(header)
    print("  " + "-" * 115)

    # Print original reference rows
    for mkt in all_markets:
        for mdl in all_models:
            o = ORIGINAL_RESULTS[mkt][mdl]
            print(f"  {mkt:<6} {mdl:<8} {'ORIG':<7} {'3000':<5} {o['mse']:<12.4f} {o['model_r']:<9.4f} {o['pred_r']:<9.4f} {o['delta']:<+9.4f} {'---':<9} {'---':<8} {'---':<12}")

    # Print clean rows
    for r in sorted(results, key=lambda x: (x["market"], x["model"])):
        print(f"  {r['market']:<6} {r['model']:<8} {'CLEAN':<7} {r['n_a']:<5} {r['mse']:<12.6f} {r['model_r']:<9.4f} {r['pred_r']:<9.4f} {r['delta']:<+9.4f} {r['params']:<9,} {r['train_time_s']:<8.0f} {r['best_val_mse']:<12.6f}")

    print("=" * 120)


# ============================================================
# MAIN
# ============================================================
def run(parsed_args):
    """Run clean experiment: baseline + P0 on US + CN."""
    global EPOCHS

    # ---- Quick mode ----
    if parsed_args.quick:
        global N_STOCKS_US, N_STOCKS_CN
        N_STOCKS_US = N_STOCKS_CN = 100
        EPOCHS = 5
        print(f"\nQUICK MODE: n_stocks=100, epochs=5")

    # ---- Import models (baseline + P0 only) ----
    print("\nImporting models (baseline + P0 only)...")
    _register_baseline()
    _maybe_import_p0()

    # Resolve which models to run
    if parsed_args.models == "all":
        model_keys = [k for k in _MODELS if _MODELS[k] is not None]
    else:
        model_keys = [k for k in parsed_args.models.split(",") if k in _MODELS and _MODELS[k] is not None]

    if not model_keys:
        print("ERROR: No valid models selected.")
        return

    # Resolve which markets
    market_list = []
    if parsed_args.market in ("US", "both"):
        market_list.append(("US", N_STOCKS_US, f"{DATA_DIR}/us_price_expanded.parquet",
                            f"{DATA_DIR}/us_volume_expanded.parquet", PRICE_FLOOR_US))
    if parsed_args.market in ("CN", "both"):
        market_list.append(("CN", N_STOCKS_CN, f"{DATA_DIR}/cn_price.parquet",
                            f"{DATA_DIR}/cn_volume.parquet", PRICE_FLOOR_CN))

    use_amp = not parsed_args.no_amp
    print(f"\nModels: {model_keys}")
    print(f"Markets: {[m[0] for m in market_list]}")
    print(f"AMP: {use_amp} | Batch: {BATCH} | Epochs: {EPOCHS}")
    print(f"Filters: PRICE_FLOOR=US${PRICE_FLOOR_US}/CN${PRICE_FLOOR_CN} | COMPLETENESS_FLOOR={COMPLETENESS_FLOOR:.0%}")
    print(f"N_STOCKS: US={N_STOCKS_US}  CN={N_STOCKS_CN}")
    print(f"\n{'#' * 60}")
    print(f"  CLEAN EXPERIMENT — Price Floor + Completeness Filter")
    print(f"  Comparing: baseline (SlimMetaTrans) vs P0 (BlockAttnRes)")
    print(f"{'#' * 60}")

    all_results = []
    for name, n_stocks, price_f, vol_f, pf in market_list:
        print(f"\n{'=' * 60}\n  Loading {name} ({n_stocks} stocks target, price_floor=${pf:.1f})\n{'=' * 60}")
        pd_data, ret_data, y_data = load_market(name, n_stocks, price_f, vol_f, pf)
        for mk in model_keys:
            factory = _MODELS[mk]
            if factory is None:
                continue
            try:
                res = run_model_on_market(mk, factory, name, pd_data, ret_data, y_data,
                                          EPOCHS, GPU, use_amp)
                if res is not None:
                    all_results.append(res)
            except Exception as e:
                print(f"  [{mk}] on {name} FAILED: {e}")
                traceback.print_exc()
                gc.collect()
                if GPU.type == "cuda":
                    torch.cuda.empty_cache()

    # ---- Final comparison table ----
    print_final_comparison(all_results)

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    # Add script's directory to sys.path so imports work
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    parser = argparse.ArgumentParser(description="Clean Experiment — Price Floor + Completeness Filter")
    parser.add_argument("--market", type=str, default="both",
                        choices=["US", "CN", "both"],
                        help="Which market(s) to train on")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated models: baseline,p0,all")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: n_stocks=100, epochs=5")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override EPOCHS (overrides --quick)")
    args = parser.parse_args()

    if args.epochs is not None:
        EPOCHS = args.epochs

    run(args)
