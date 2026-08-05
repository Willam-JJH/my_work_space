"""
Self-contained training launcher for RTX 4090.
Compares Baseline (SlimMetaTrans) vs P0 (BlockAttnRes) vs P2 (KDA) vs P1a (MultiPattern).
"""

import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, time, os, gc, warnings, sys, json, argparse, traceback
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
N_STOCKS_US  = 3000
N_STOCKS_CN  = 3000
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

# ---- 4090-specific ----
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

if GPU.type == "cuda":
    dev_props = torch.cuda.get_device_properties(0)
    vram_gb = dev_props.total_memory / 1e9
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    # Enable TF32 for RTX 4090
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"TF32: enabled  |  AMP: available={torch.cuda.amp.autocast is not None}")

torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA LOADING (inlined from meta_transformer_optimized.py)
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


def load_market(name, n_stocks, price_path, vol_path):
    """Load market data, split, and compute returns."""
    price = pd.read_parquet(price_path)
    vol = pd.read_parquet(vol_path)
    common = sorted(set(price.columns) & set(vol.columns))
    idx = price.index.intersection(vol.index)
    price = price.loc[idx, common]
    vol = vol.loc[idx, common]
    comp = price.notna().sum() / len(price)
    top = comp.nlargest(min(n_stocks, len(comp))).index.tolist()
    price = price[top].ffill().fillna(1e-8)
    vol = vol[top].ffill().fillna(1)
    logP = np.log(np.maximum(price.values.astype(np.float64), 1e-8))
    logV = np.log(np.maximum(vol.values.astype(np.float64), 1e-8))
    vi = logV - pd.DataFrame(logV).rolling(20, min_periods=1).mean().values
    n_a = len(top)
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
# BASELINE MODEL (inlined SlimMetaTrans)
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
# MODEL FACTORY — import improvement models with graceful fallback
# ============================================================
_MODELS = {}

def _register_baseline():
    _MODELS["baseline"] = baseline

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

def _maybe_import_p2():
    try:
        from p2_kda_attention import KDATransformer
        def p2_factory(n_a, sig_dim, cl_dim):
            return KDATransformer(n_a, sig_dim, cl_dim, d_model=D_MODEL, n_layers=N_LAYERS,
                                  n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT)
        _MODELS["p2"] = p2_factory
        print("  P2  (KDA Attention)     — imported OK")
    except Exception as e:
        print(f"  P2  (KDA Attention)     — SKIP: {e}")
        _MODELS["p2"] = None

def _maybe_import_p1a():
    try:
        from p1a_multi_pattern import PatternExtractingTransformer
        def p1a_factory(n_a, sig_dim, cl_dim):
            return PatternExtractingTransformer(n_a, sig_dim, cl_dim, d_model=D_MODEL,
                                                n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
                                                n_layers=N_LAYERS)
        _MODELS["p1a"] = p1a_factory
        print("  P1a (MultiPattern)      — imported OK")
    except Exception as e:
        print(f"  P1a (MultiPattern)      — SKIP: {e}")
        _MODELS["p1a"] = None

def _maybe_import_p1b():
    # P1b depends on P0 — if P0 import failed, P1b should also be skipped
    if _MODELS.get("p0") is None:
        print("  P1b (JointFusion)       — SKIP: requires P0")
        _MODELS["p1b"] = None
        return
    try:
        from p1b_fusion import JointFusionTransformer
        def p1b_factory(n_a, sig_dim, cl_dim):
            return JointFusionTransformer(n_a, sig_dim, cl_dim, d_model=D_MODEL,
                                          n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT)
        _MODELS["p1b"] = p1b_factory
        print("  P1b (JointFusion)       — imported OK")
    except Exception as e:
        print(f"  P1b (JointFusion)       — SKIP: {e}")
        _MODELS["p1b"] = None

def baseline(n_a, sig_dim, cl_dim):
    return BaselineSlimMetaTrans(n_a, sig_dim, cl_dim)


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
                use_amp=True, label="model", model_name=None):
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
                if model_name == 'p1b':
                    ret_pred, unc_pred, _ = model(s_g, c_g)
                else:
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
                                        epochs, device, use_amp, label=model_name,
                                        model_name=model_name)
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

    print(f"  Test MSE: {ev['mse']:.6f} | Model r: {ev['model_r']:.4f} | Pred r: {ev['pred_r']:.4f} | Delta: {ev['delta']:+.4f} | Time: {t_train:.0f}s")

    del model; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ev


# ============================================================
# MAIN
# ============================================================
def print_table(results):
    """Print comparison table."""
    if not results:
        print("\n  No results to display.")
        return
    print("\n" + "=" * 100)
    print("  COMPARISON TABLE")
    print("=" * 100)
    header = f"  {'Market':<6} {'Model':<8} {'n_a':<5} {'MSE':<12} {'Model r':<9} {'Pred r':<9} {'Delta':<9} {'#Params':<9} {'Time(s)':<8} {'ValMSE':<12}"
    print(header)
    print("  " + "-" * 95)
    for r in sorted(results, key=lambda x: (x["market"], x["model"])):
        print(f"  {r['market']:<6} {r['model']:<8} {r['n_a']:<5} {r['mse']:<12.6f} {r['model_r']:<9.4f} {r['pred_r']:<9.4f} {r['delta']:<+9.4f} {r['params']:<9,} {r['train_time_s']:<8.0f} {r['best_val_mse']:<12.6f}")
    print("=" * 100)

    # Per-market best delta
    markets = sorted(set(r["market"] for r in results))
    for mkt in markets:
        mkt_results = [r for r in results if r["market"] == mkt]
        best = max(mkt_results, key=lambda x: x["delta"])
        print(f"  Best {mkt} model by Delta: {best['model']}  ({best['delta']:+.4f})")


def run(parsed_args):
    """Run all experiments per CLI args."""
    global N_STOCKS_US, N_STOCKS_CN, EPOCHS

    # ---- Quick mode ----
    if parsed_args.quick:
        N_STOCKS_US = N_STOCKS_CN = 100
        EPOCHS = 5
        print(f"\nQUICK MODE: n_stocks=100, epochs=5")

    # ---- Import improvement models ----
    print("\nImporting improvement models...")
    _register_baseline()
    _maybe_import_p0()
    _maybe_import_p2()
    _maybe_import_p1a()
    _maybe_import_p1b()

    # Resolve which models
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
                            f"{DATA_DIR}/us_volume_expanded.parquet"))
    if parsed_args.market in ("CN", "both"):
        market_list.append(("CN", N_STOCKS_CN, f"{DATA_DIR}/cn_price.parquet",
                            f"{DATA_DIR}/cn_volume.parquet"))

    use_amp = not parsed_args.no_amp
    print(f"\nModels: {model_keys}")
    print(f"Markets: {[m[0] for m in market_list]}")
    print(f"AMP: {use_amp} | Batch: {BATCH} | Epochs: {EPOCHS}")

    all_results = []
    for name, n_stocks, price_f, vol_f in market_list:
        print(f"\n{'=' * 60}\n  Loading {name} ({n_stocks} stocks)\n{'=' * 60}")
        pd_data, ret_data, y_data = load_market(name, n_stocks, price_f, vol_f)
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

    print_table(all_results)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return all_results


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    # Add k3_improvements dir to sys.path so imports work
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    parser = argparse.ArgumentParser(description="RTX 4090 Training Launcher")
    parser.add_argument("--market", type=str, default="both",
                        choices=["US", "CN", "both"],
                        help="Which market(s) to train on")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated models: baseline,p0,p1a,p1b,p2,all")
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
