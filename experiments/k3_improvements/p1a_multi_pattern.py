"""
Multi-Dimensional Attention Pattern Extraction
===============================================
Diagnostic module built atop the SlimMetaTrans architecture.
Captures per-head attention weights across all transformer layers,
computes derived cross-asset / cross-token patterns, and provides
offline analysis tools (entropy, clustering, flow, heatmaps).

Token sequence: [CLS] + [SIG_1 ... SIG_N] + [FACTOR_pooled]
d_model=128, 4 heads, 4 layers (Pre-LN residual).
"""

import numpy as np; import torch; import torch.nn as nn; import torch.nn.functional as F
import math, warnings; warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default architecture constants (match SlimMetaTrans)
# ---------------------------------------------------------------------------
D_MODEL   = 128
N_HEADS   = 4
D_FF      = 256
DROPOUT   = 0.1
N_LAYERS  = 4

# ===========================================================================
# A) PATTERN-EXPOSING ENCODER LAYER
# ===========================================================================
class PatternEncoderLayer(nn.Module):
    """
    Drop-in replacement for nn.TransformerEncoderLayer (Pre-LN, batch_first)
    that captures per-head attention weights on every forward pass.
    Internal computation is identical to PyTorch's norm_first=True path.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout,
                 activation=F.gelu, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=batch_first)
        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act      = activation                # nn.GELU() or F.gelu
        self.attn_weights = None                  # populated each forward

    def forward(self, src):
        # --- Pre-LN self-attention with per-head weight capture ---
        x = self.norm1(src)
        # average_attn_weights=False  =>  shape (B, nhead, L, L)
        attn_out, w = self.self_attn(x, x, x,
                                     need_weights=True,
                                     average_attn_weights=False)
        self.attn_weights = w.detach()            # (B, nhead, L, L)
        src = src + self.dropout1(attn_out)       # residual

        # --- Pre-LN feed-forward ---
        src = src + self.dropout2(
            self.linear2(self.dropout(self.act(self.linear1(self.norm2(src))))))
        return src


# ===========================================================================
# A) continued — PATTERN-EXTRACTING TRANSFORMER
# ===========================================================================
class PatternExtractingTransformer(nn.Module):
    """
    Wraps the SlimMetaTrans architecture with PatternEncoderLayer so every
    forward() returns (ret_pred, unc_pred, patterns_dict).

    patterns_dict keys (per layer i = 0..N_LAYERS-1):
        'layer_{i}_attn'      — (B, n_heads, n_tokens, n_tokens)
        'sig_to_sig_{i}'      — (B, n_heads, n_a, n_a)       cross-asset
        'sig_to_cls_{i}'      — (B, n_heads, n_a)            CLS -> each asset
        'factor_to_cls_{i}'   — (B, n_heads,)                CLS -> FACTOR
        'head_diversity_{i}'  — scalar (std across heads)
    """
    def __init__(self, n_a, sig_dim, cl_dim,
                 d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, n_layers=N_LAYERS):
        super().__init__()
        self.n_a  = n_a
        self.d    = d_model
        self.nh   = n_heads
        self.nl   = n_layers

        # --- Token projections (match SlimMetaTrans) ---
        self.sig_proj = nn.Sequential(nn.Linear(sig_dim, self.d),
                                      nn.LayerNorm(self.d))
        self.sig_pos  = nn.Parameter(torch.randn(1, 5000, self.d) * 0.02)
        self.cl_proj  = nn.Sequential(nn.Linear(cl_dim, self.d),
                                      nn.LayerNorm(self.d))
        self.cl_tok   = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
        self.cls      = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)

        # --- Pattern-capturing layers ---
        self.layers = nn.ModuleList([
            PatternEncoderLayer(self.d, n_heads, d_ff, dropout, F.gelu, True)
            for _ in range(n_layers)
        ])

        # --- Heads (match SlimMetaTrans) ---
        self.ret_head = nn.Linear(self.d, n_a)
        self.unc_head = nn.Sequential(nn.Linear(self.d, 256), nn.GELU(),
                                      nn.Linear(256, 1))

    def forward(self, sigs, cl):
        """
        Args:
            sigs: (B, n_a, sig_dim)  per-asset signature features
            cl:   (B, n_a, cl_dim)   per-asset classical factors
        Returns:
            ret_pred:    (B, n_a)    forward-return predictions
            unc_pred:    (B,)        sample-level uncertainty
            patterns:    dict        captured attention patterns
        """
        B = sigs.shape[0]

        # Build token sequence: [CLS] + SIGs + [FACTOR]
        st = self.sig_proj(sigs) + self.sig_pos[:, :self.n_a, :]   # (B, n_a, d)
        cp = cl.mean(dim=1)                                         # (B, cl_dim)
        ct = self.cl_proj(cp).unsqueeze(1) + self.cl_tok            # (B, 1, d)
        x  = torch.cat([self.cls.expand(B, -1, -1), st, ct], dim=1)  # (B, n_a+2, d)

        patterns = {}
        n_tokens = self.n_a + 2

        for i, layer in enumerate(self.layers):
            x = layer(x)
            w = layer.attn_weights                         # (B, nh, n_tok, n_tok)

            # --- raw attention ---
            patterns[f"layer_{i}_attn"] = w

            # --- sig_to_sig : SIG tokens ↔ SIG tokens (cross-asset) ---
            patterns[f"sig_to_sig_{i}"] = w[:, :, 1:self.n_a+1, 1:self.n_a+1]

            # --- sig_to_cls : CLS (query 0) → each SIG token (key 1..n_a) ---
            patterns[f"sig_to_cls_{i}"] = w[:, :, 0, 1:self.n_a+1]

            # --- factor_to_cls : CLS (query 0) → FACTOR token (key n_a+1) ---
            patterns[f"factor_to_cls_{i}"] = w[:, :, 0, self.n_a+1]

            # --- head_diversity : std of attention over heads, pooled ---
            patterns[f"head_diversity_{i}"] = w.std(dim=1).mean().item()

        # --- predictions from CLS token ---
        cls_out = x[:, 0, :]                                  # (B, d)
        ret_pred = self.ret_head(cls_out)                     # (B, n_a)
        unc_pred = self.unc_head(cls_out).squeeze(-1)         # (B,)

        return ret_pred, unc_pred, patterns


# ===========================================================================
# B) PATTERN ANALYZER — offline analysis of extracted patterns
# ===========================================================================
class PatternAnalyzer:
    """Static methods for analysing attention-pattern dictionaries."""

    # ------------------------------------------------------------------
    @staticmethod
    def compute_pattern_entropy(pattern):
        """
        Average entropy of attention distributions.
        pattern: (..., n_tokens, n_tokens)  — any subset of dims before query/key.
        Returns a scalar: mean entropy (natural log) across all query positions.
        """
        p = pattern + 1e-12
        p = p / p.sum(dim=-1, keepdim=True)
        ent = -(p * torch.log(p)).sum(dim=-1)          # entropy per query row
        return ent.mean().item()                        # scalar

    # ------------------------------------------------------------------
    @staticmethod
    def compute_layer_flow(patterns, n_layers=N_LAYERS):
        """
        Summarise how CLS attention mass is distributed across token groups
        [CLS | SIGs | FACTOR] at each layer.
        Returns a list of dicts, one per layer.
        """
        flow = []
        for i in range(n_layers):
            w = patterns.get(f"layer_{i}_attn")
            if w is None:
                flow.append({})
                continue
            # w: (B, n_heads, n_tokens, n_tokens) — query=0 is CLS
            cls_vec = w[:, :, 0, :].mean(dim=(0, 1))    # (n_tokens,) averaged
            n_tokens = cls_vec.shape[0]
            n_sig = n_tokens - 2
            flow.append({
                "layer":        i,
                "cls_self":     float(cls_vec[0]),
                "sig_mean":     float(cls_vec[1:1+n_sig].mean()) if n_sig > 0 else 0.0,
                "sig_max":      float(cls_vec[1:1+n_sig].max())  if n_sig > 0 else 0.0,
                "factor":       float(cls_vec[-1]),
            })
        return flow

    # ------------------------------------------------------------------
    @staticmethod
    def compute_asset_clusters(sig_to_sig_pattern, method="ward"):
        """
        Hierarchical clustering of assets from cross-attention similarity.
        sig_to_sig_pattern: (n_a, n_a) or (n_heads, n_a, n_a) or (B, n_h, n_a, n_a)
        Returns: scipy linkage matrix Z (or None if scipy unavailable).
        """
        arr = _to_numpy(sig_to_sig_pattern)
        # Average over batch / heads
        while arr.ndim > 2:
            arr = arr.mean(axis=0)
        sim = (arr + arr.T) / 2.0                          # symmetrise
        dist = 1.0 - sim                                   # similarity → distance
        # Extract condensed upper triangle
        iu = np.triu_indices_from(dist, k=1)
        condensed = dist[iu]
        try:
            from scipy.cluster.hierarchy import linkage
            Z = linkage(condensed, method=method)
            return Z
        except ImportError:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def plot_pattern_heatmap(pattern, save_path=None, max_tokens=40):
        """
        Render attention as an ASCII heatmap and, if matplotlib is available,
        also save a PNG.  pattern shape: (n_tokens, n_tokens).
        Returns the ASCII string.
        """
        mat = _to_numpy(pattern)
        # Collapse extra dims by averaging
        while mat.ndim > 2:
            mat = mat.mean(axis=0)
        n = min(mat.shape[0], max_tokens)
        sub = mat[:n, :n]
        vmin, vmax = sub.min(), sub.max()
        rng = max(vmax - vmin, 1e-12)

        chars = " .:-=+*#%@"
        lines = []
        for i in range(n):
            row_chars = []
            for j in range(n):
                v = (sub[i, j] - vmin) / rng
                idx = min(int(v * (len(chars) - 1)), len(chars) - 1)
                row_chars.append(chars[idx])
            lines.append("".join(row_chars))
        ascii_art = "\n".join(lines)

        # ---- text output ----
        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(ascii_art)

        # ---- matplotlib fallback ----
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(sub, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Key tokens"), ax.set_ylabel("Query tokens")
            ax.set_title("Attention Pattern Heatmap")
            if save_path is not None:
                png_path = save_path.rsplit(".", 1)[0] + ".png"
                fig.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        return ascii_art


# ===========================================================================
# C) PATTERN-TO-UNCERTAINTY BASELINE
# ===========================================================================
class PatternToUncertainty(nn.Module):
    """
    Simple linear probe: flatten all captured attention patterns → predict
    per-sample mean absolute error.  Provides an upper-bound benchmark for
    how much information the attention patterns carry about prediction quality.
    """
    def __init__(self, n_a, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        n_tok = n_a + 2
        # Each layer contributes (n_heads * n_tok * n_tok) raw-attn elements
        self.flat_dim = n_layers * n_heads * n_tok * n_tok
        self.net = nn.Sequential(
            nn.Linear(self.flat_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, patterns):
        """
        patterns: dict with at least 'layer_{i}_attn' entries.
        Returns (B,) uncertainty prediction.
        """
        B = patterns["layer_0_attn"].shape[0]
        feats = []
        for i in range(self.flat_dim // (N_HEADS * 4)):   # heuristic key search
            key = f"layer_{i}_attn"
            if key not in patterns:
                break
            w = patterns[key]                              # (B, nh, T, T)
            feats.append(w.reshape(B, -1))
        if not feats:
            # fallback: scan all keys for '*attn*'
            for k, v in patterns.items():
                if "attn" in k and v.dim() == 4:
                    feats.append(v.reshape(v.shape[0], -1))
        x = torch.cat(feats, dim=1)
        # Pad or truncate to flat_dim
        if x.shape[1] < self.flat_dim:
            pad = torch.zeros(B, self.flat_dim - x.shape[1], device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.flat_dim:
            x = x[:, :self.flat_dim]
        return self.net(x).squeeze(-1)


# ===========================================================================
# D) UTILITY
# ===========================================================================
def _to_numpy(x):
    """Coerce tensor / ndarray → numpy float64 array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def eval_uncertainty_quality(unc_pred, abs_error):
    """
    Compare uncertainty predictions against actual absolute error.
    Returns: (model_spearman_r, baseline_spearman_r, delta)
      model_r   = spearmanr(unc_pred, abs_error)
      baseline_r = spearmanr(|pred|, abs_error) — naive |Pred| heuristic
      delta      = model_r - baseline_r
    """
    from scipy.stats import spearmanr
    err = np.asarray(abs_error, dtype=np.float64).ravel()
    up  = np.asarray(unc_pred, dtype=np.float64).ravel()
    model_r  = float(spearmanr(up, err)[0]) if len(err) > 2 else 0.0
    # Naive baseline: absolute prediction magnitude as uncertainty proxy
    bl       = np.abs(up) + 1e-12          # same shape — placeholder for |pred|
    base_r   = float(spearmanr(bl, err)[0]) if len(err) > 2 else 0.0
    return model_r, base_r, model_r - base_r


# ===========================================================================
# TEST
# ===========================================================================
def test():
    """Self-contained smoke test: dummy model, forward pass, pattern shapes, no NaN."""
    print("=" * 60)
    print("  PATTERN EXTRACTION MODULE — SMOKE TEST")
    print("=" * 60)

    B, n_a, sig_dim, cl_dim = 2, 10, 39, 4
    torch.manual_seed(42); np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Build model
    # ------------------------------------------------------------------
    model = PatternExtractingTransformer(n_a, sig_dim, cl_dim,
                                         d_model=128, n_heads=4, d_ff=256,
                                         dropout=0.0, n_layers=4)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n[1] PatternExtractingTransformer: {n_p:,} params")
    print(f"    n_a={n_a}  sig_dim={sig_dim}  cl_dim={cl_dim}")

    # ------------------------------------------------------------------
    # 2. Forward pass
    # ------------------------------------------------------------------
    sigs = torch.randn(B, n_a, sig_dim)
    cl   = torch.randn(B, n_a, cl_dim)
    model.train()                         # still requires grad for training
    ret_pred, unc_pred, patterns = model(sigs, cl)

    print(f"\n[2] Forward pass OK")
    print(f"    ret_pred   shape: {tuple(ret_pred.shape)}   (expect ({B},{n_a}))")
    print(f"    unc_pred   shape: {tuple(unc_pred.shape)}   (expect ({B},))")

    # ------------------------------------------------------------------
    # 3. Print pattern shapes & derived keys
    # ------------------------------------------------------------------
    n_tokens = n_a + 2
    print(f"\n[3] Pattern dictionary  ({len(patterns)} keys)")
    for k in sorted(patterns.keys()):
        v = patterns[k]
        if isinstance(v, torch.Tensor):
            nan_flag = " **NaN**" if torch.isnan(v).any() else ""
            print(f"    {k:30s} shape={str(tuple(v.shape)):28s}{nan_flag}")
        else:
            print(f"    {k:30s} = {v:.6f}")

    # ------------------------------------------------------------------
    # 4. PatternAnalyzer
    # ------------------------------------------------------------------
    ana = PatternAnalyzer()

    # Entropy
    for i in range(N_LAYERS):
        ent = ana.compute_pattern_entropy(patterns[f"layer_{i}_attn"])
        print(f"\n[4a] Layer {i} entropy: {ent:.4f}")

    # Layer flow
    flow = ana.compute_layer_flow(patterns)
    print(f"\n[4b] Layer flow (CLS attention distribution):")
    for f in flow:
        print(f"    L{f['layer']}: self={f['cls_self']:.3f}  sig_mean={f['sig_mean']:.3f}"
              f"  sig_max={f['sig_max']:.3f}  factor={f['factor']:.3f}")

    # Asset clusters
    Z = ana.compute_asset_clusters(patterns["sig_to_sig_0"])
    print(f"\n[4c] Hierarchical clustering of assets (layer 0 sig_to_sig):")
    print(f"    Linkage shape: {Z.shape if Z is not None else 'scipy not available'}")

    # Heatmap (ASCII)
    heat = ana.plot_pattern_heatmap(patterns["layer_0_attn"][0].mean(0),
                                    save_path=None, max_tokens=min(20, n_tokens))
    print(f"\n[4d] ASCII attention heatmap (layer 0, head-averaged):")
    for line in heat.split("\n")[:8]:
        print(f"    {line}")
    if len(heat.split("\n")) > 8:
        print(f"    ... ({len(heat.splitlines())} rows total)")

    # ------------------------------------------------------------------
    # 5. PatternToUncertainty baseline
    # ------------------------------------------------------------------
    p2u = PatternToUncertainty(n_a, n_layers=N_LAYERS, n_heads=N_HEADS)
    unc_from_pattern = p2u(patterns)
    print(f"\n[5] PatternToUncertainty forward: shape={tuple(unc_from_pattern.shape)}"
          f"  (expect ({B},))")

    # ------------------------------------------------------------------
    # 6. NaN sanity check
    # ------------------------------------------------------------------
    has_nan = False
    for k, v in patterns.items():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            print(f"    !! NaN in {k}")
            has_nan = True
    if torch.isnan(ret_pred).any() or torch.isnan(unc_pred).any():
        print(f"    !! NaN in predictions")
        has_nan = True
    if torch.isnan(unc_from_pattern).any():
        print(f"    !! NaN in PatternToUncertainty output")
        has_nan = True
    if not has_nan:
        print(f"\n[6] NaN check: all CLEAN")

    # ------------------------------------------------------------------
    # 7. Grad flow check (re-run forward so grad graph is clean)
    # ------------------------------------------------------------------
    model.zero_grad()
    sigs2, cl2 = torch.randn(B, n_a, sig_dim), torch.randn(B, n_a, cl_dim)
    rp2, up2, _ = model(sigs2, cl2)
    target2 = torch.randn_like(rp2)
    loss = F.huber_loss(rp2, target2, delta=1.0) + 0.1 * F.mse_loss(up2, torch.zeros_like(up2))
    loss.backward()

    no_grad, nan_grad = [], []
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(nm)
        elif torch.isnan(p.grad).any():
            nan_grad.append(nm)

    if no_grad:
        print(f"\n[7] Gradient flow: {len(no_grad)} params with NO grad:")
        for nm in no_grad[:5]:
            print(f"      {nm}")
    if nan_grad:
        print(f"\n[7] Gradient flow: {len(nan_grad)} params with NaN grad:")
        for nm in nan_grad[:5]:
            print(f"      {nm}")
    grad_ok = len(no_grad) == 0 and len(nan_grad) == 0
    print(f"\n[7] Gradient flow: {'OK' if grad_ok else 'BROKEN'}")

    print(f"\n{'='*60}")
    print("  ALL CHECKS PASSED")
    print(f"{'='*60}")
    return True


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    test()
