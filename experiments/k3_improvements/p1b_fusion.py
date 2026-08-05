"""
P1b Joint Fusion Module
========================

Combines AttnRes cross-block weights + per-block attention patterns + |Pred|
baseline into a multi-source uncertainty head.  The core hypothesis: AttnRes
weights serve as a meta-signal for "which block to trust" and this
meta-information can break the fusion bottleneck.

Architecture
------------
  JointFusionTransformer
    ├── 2 x BlockAttnResLayer  (from p0_attn_res, with hooked attention)
    ├── ret_head: Linear(d_model, n_a)
    ├── MultiSourceUncertaintyHead
    │     Inputs: |Pred| + pattern_emb(32) + attnres_emb(16) + CLS_hidden(d)
    │     Fusion: concat(177) → Linear(177,128) → GELU → Drop
    │              → Linear(128,64) → GELU → Linear(64,1)
    └── Ablation modes: pred_only / pred_patterns / pred_attnres /
                        pred_patterns_attnres / full

Forward returns (ret_pred, unc_pred, fusion_info_dict).

fusion_info keys:
    cb_weights      list[Tensor(B, 3, 3)]           cross-block attention
    block_patterns  list[Tensor(B, nhead, L, L)]    per-block self-attn patterns
    pred_signal     Tensor(B, 1)                    |Pred| magnitude
    cls_hidden      Tensor(B, d_model)              final CLS token embedding

References:
    p0_attn_res.py            BlockAttnResLayer, RMSNorm
    p1a_multi_pattern.py      per-head pattern capture approach
    meta_transformer_optimized.py  SlimMetaTrans (API baseline)

Author: K3 Roadmap P1b
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# ---------------------------------------------------------------------------
# Import BlockAttnResLayer from sibling module
# ---------------------------------------------------------------------------
try:
    from .p0_attn_res import BlockAttnResLayer, RMSNorm
except ImportError:
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from p0_attn_res import BlockAttnResLayer, RMSNorm  # type: ignore[no-redef]


# ===========================================================================
# HookedMHA — transparent hook for capturing per-head self-attention weights
# ===========================================================================
class HookedMHA(nn.Module):
    """
    Thin wrapper around nn.MultiheadAttention that always captures per-head
    attention weights on every forward call, regardless of the caller's
    need_weights setting.

    This acts as a "hook" in the architectural sense: it intercepts the
    self-attention computation in each BlockAttnResLayer block and records
    the full (B, n_heads, L, L) pattern for downstream fusion.

    Interface-compatible with nn.MultiheadAttention when need_weights=False
    (returns (attn_output, None)), so BlockAttnResLayer._block_forward sees
    exactly what it expects.

    Attributes:
        mha               wrapped nn.MultiheadAttention (owns the parameters)
        captured_weights  (B, n_heads, L, L)  — overwritten each forward
    """

    def __init__(self, mha_module: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha_module
        self.captured_weights: torch.Tensor | None = None

    def forward(self, query, key, value, need_weights=False,
                average_attn_weights=True, **kwargs):
        # Always compute un-averaged per-head weights
        out, w = self.mha(query, key, value,
                          need_weights=True,
                          average_attn_weights=False,
                          **kwargs)
        self.captured_weights = w.detach()               # (B, n_heads, L, L)

        # Return interface matches what the caller expects
        if need_weights:
            if average_attn_weights and w is not None:
                w = w.mean(dim=1)
            return out, w
        return out, None


# ===========================================================================
# MultiSourceUncertaintyHead
# ===========================================================================
class MultiSourceUncertaintyHead(nn.Module):
    """
    Uncertainty estimator that fuses multiple information sources.

    Four input channels (see forward() for tensor shapes):
        pred_signal    |Pred| magnitude  (the "naive" heuristic)
        pattern_emb    attention-pattern embedding (projected to 32d)
        attnres_emb    AttnRes cross-block weight embedding (projected to 16d)
        cls_hidden     final CLS token hidden state (d_model)

    Ablation modes control which subsets are concatenated and fed through
    a mode-specific MLP:

        pred_only              → only |Pred|
        pred_patterns          → |Pred| + patterns
        pred_attnres           → |Pred| + AttnRes weights
        pred_patterns_attnres  → |Pred| + patterns + AttnRes
        full                   → |Pred| + patterns + AttnRes + CLS (177-dim)

    All variants are housed in the same class; mode is selected at init time.
    """

    def __init__(self, pattern_dim: int, d_model: int = 128,
                 dropout: float = 0.1, mode: str = 'full'):
        """
        Args:
            pattern_dim: flattened dimension of all per-block attention
                         patterns (depends on n_a, n_heads, n_layers).
            d_model:     model dimension (128).
            dropout:     dropout rate inside fusion MLP.
            mode:        one of {'pred_only', 'pred_patterns', 'pred_attnres',
                                 'pred_patterns_attnres', 'full'}.
        """
        super().__init__()
        assert mode in ('pred_only', 'pred_patterns', 'pred_attnres',
                        'pred_patterns_attnres', 'full'), f"Unknown mode: {mode}"
        self.mode = mode
        self.d_model = d_model

        # ---- Channel projections (created when needed) ----
        needs_patterns = ('patterns' in mode) or (mode == 'full')
        needs_attnres  = ('attnres' in mode)  or (mode == 'full')
        if needs_patterns:
            self.pattern_proj = nn.Sequential(
                nn.Linear(pattern_dim, 32),
                nn.GELU(),
            )
        if needs_attnres:
            # 2 layers x 3 query blocks x 3 key blocks = 18
            self.attnres_proj = nn.Sequential(
                nn.Linear(18, 16),
                nn.GELU(),
            )

        # ---- Build mode-specific fusion MLP ----
        in_dim = {
            'pred_only': 1,
            'pred_patterns': 1 + 32,
            'pred_attnres': 1 + 16,
            'pred_patterns_attnres': 1 + 32 + 16,
            'full': 1 + 32 + 16 + d_model,          # 177
        }[mode]

        if mode == 'full':
            # As specified: 177 → 128 → 64 → 1
            self.fusion = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
        else:
            # Compact head for ablated variants
            hidden = max(in_dim // 2, 16)
            self.fusion = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, pred_signal, patterns_flat, cb_flat, cls_hidden):
        """
        Args:
            pred_signal:   (B, 1)            |Pred| magnitude (mean over assets)
            patterns_flat: (B, pattern_dim)  flattened attention patterns
            cb_flat:       (B, 18)           flattened cross-block weights
            cls_hidden:    (B, d_model)      final CLS token embedding
        Returns:
            unc_pred: (B,)  per-sample uncertainty estimate
        """
        components = [pred_signal]

        if self.mode in ('pred_patterns', 'pred_patterns_attnres', 'full'):
            p_emb = self.pattern_proj(patterns_flat)          # (B, 32)
            components.append(p_emb)

        if self.mode in ('pred_attnres', 'pred_patterns_attnres', 'full'):
            a_emb = self.attnres_proj(cb_flat)                # (B, 16)
            components.append(a_emb)

        if self.mode == 'full':
            components.append(cls_hidden)                     # (B, d_model)

        x = torch.cat(components, dim=-1)
        return self.fusion(x).squeeze(-1)                     # (B,)


# ===========================================================================
# JointFusionTransformer
# ===========================================================================
class JointFusionTransformer(nn.Module):
    """
    P1b Joint Fusion Transformer — combines BlockAttnRes (P0) with per-block
    attention pattern capture and a multi-source uncertainty head.

    Token sequence (same as BlockAttnResTransformer):
        [CLS(1)]  [SIG_1 … SIG_n_a]  [FACTOR(1)]
        Block 0    Block 1            Block 2

    Hook mechanism:
        After creating each BlockAttnResLayer, its internal nn.MultiheadAttention
        modules (attn_0, attn_1, attn_2) are wrapped in HookedMHA.  These
        transparently capture per-head self-attention weights on every forward
        pass and expose them via self._hooked_attns for downstream fusion.

    Forward returns:
        ret_pred    (B, n_a)   per-asset return prediction
        unc_pred    (B,)       per-sample uncertainty estimate
        fusion_info dict       diagnostic meta-information
    """

    def __init__(self, n_a: int, sig_dim: int, cl_dim: int,
                 d_model: int = 128, n_heads: int = 4, d_ff: int = 256,
                 dropout: float = 0.1, unc_mode: str = 'full',
                 max_stocks: int = 5000):
        """
        Args:
            n_a:        number of assets.
            sig_dim:    signature feature dimension per asset.
            cl_dim:     classical factor dimension per asset.
            d_model:    model hidden dimension (128).
            n_heads:    number of attention heads (4).
            d_ff:       feed-forward hidden dimension (256).
            dropout:    dropout probability.
            unc_mode:   ablation mode for MultiSourceUncertaintyHead.
            max_stocks: max assets for positional encoding buffer.
        """
        super().__init__()
        self.n_a = n_a
        self.d = d_model
        self.n_heads = n_heads
        self.unc_mode = unc_mode

        # ---- Input projections (match BlockAttnResTransformer / SlimMetaTrans) ----
        self.sig_proj = nn.Sequential(
            nn.Linear(sig_dim, d_model),
            RMSNorm(d_model),
        )
        self.sig_pos = nn.Parameter(torch.randn(1, max_stocks, d_model) * 0.02)

        self.cl_proj = nn.Sequential(
            nn.Linear(cl_dim, d_model),
            RMSNorm(d_model),
        )
        self.cl_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- Block Attention Residual layers ----
        self.layer1 = BlockAttnResLayer(d_model, n_heads, d_ff, dropout)
        self.layer2 = BlockAttnResLayer(d_model, n_heads, d_ff, dropout)

        # ---- Hook attention modules for pattern capture ----
        # After wrapping, layer{N}.attn_{i} is a HookedMHA whose .mha is the
        # original nn.MultiheadAttention.  Module hierarchy is preserved, so
        # parameters, state_dict, and gradients all work normally.
        self._hooked_attns: list[tuple[int, int, HookedMHA]] = []
        for layer_idx, layer in enumerate([self.layer1, self.layer2]):
            for blk in range(3):
                old_attn = getattr(layer, f'attn_{blk}')
                hooked = HookedMHA(old_attn)
                setattr(layer, f'attn_{blk}', hooked)
                self._hooked_attns.append((layer_idx, blk, hooked))

        # ---- Return prediction head (same as BlockAttnResTransformer) ----
        self.ret_head = nn.Linear(d_model, n_a)

        # ---- Multi-source uncertainty head ----
        # pattern_dim  =  2 layers
        #               * n_heads
        #               * (1*1  +  n_a*n_a  +  1*1)
        #                 CLS     SIG          FACTOR  per-block self-attn
        pattern_dim = 2 * n_heads * (1 + n_a * n_a + 1)
        self.unc_head = MultiSourceUncertaintyHead(
            pattern_dim=pattern_dim,
            d_model=d_model,
            dropout=dropout,
            mode=unc_mode,
        )

        # Diagnostic storage — overwritten each forward call
        self.cb_weights: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    def forward(self, sigs: torch.Tensor, cl: torch.Tensor):
        """
        Args:
            sigs: (B, n_a, sig_dim)  per-asset signature embeddings
            cl:   (B, n_a, cl_dim)   per-asset classical factors
        Returns:
            ret_pred:    (B, n_a)    per-asset return prediction
            unc_pred:    (B,)        per-sample uncertainty estimate
            fusion_info: dict        cb_weights, block_patterns, pred_signal,
                                     cls_hidden (detached for diagnostics)
        """
        B = sigs.shape[0]

        # ---- 1. Build token sequence ----
        st = self.sig_proj(sigs) + self.sig_pos[:, :self.n_a, :]      # (B, n_a, D)
        cp = cl.mean(dim=1)                                            # (B, cl_dim)
        ct = self.cl_proj(cp).unsqueeze(1) + self.cl_tok               # (B, 1, D)
        x = torch.cat([self.cls.expand(B, -1, -1), st, ct], dim=1)    # (B, T, D)

        # ---- 2. Block Attention Residual layers ----
        self.cb_weights = []
        x, w1 = self.layer1(x)
        self.cb_weights.append(w1)                                     # (B, 3, 3)
        x, w2 = self.layer2(x)
        self.cb_weights.append(w2)                                     # (B, 3, 3)

        # ---- 3. Predictions from CLS token ----
        cls_hidden = x[:, 0, :]                                        # (B, d_model)
        ret_pred = self.ret_head(cls_hidden)                           # (B, n_a)
        pred_signal = ret_pred.abs().mean(dim=1, keepdim=True)         # (B, 1)

        # ---- 4. Collect captured attention patterns ----
        # Six entries: (L0,B0), (L0,B1), (L0,B2), (L1,B0), (L1,B1), (L1,B2)
        block_patterns = [h.captured_weights for _, _, h in self._hooked_attns]

        # ---- 5. Build fusion info dict (detached for safe diagnostic use) ----
        fusion_info = {
            'cb_weights':     [w.detach() for w in self.cb_weights],
            'block_patterns': block_patterns,   # already detached in HookedMHA
            'pred_signal':    pred_signal.detach(),
            'cls_hidden':     cls_hidden.detach(),
        }

        # ---- 6. Multi-source uncertainty (non-detached signals for training) ----
        unc_pred = self._compute_uncertainty(
            pred_signal=pred_signal,
            cb_weights_list=self.cb_weights,
            block_patterns_list=block_patterns,
            cls_hidden=cls_hidden,
        )

        return ret_pred, unc_pred, fusion_info

    # ------------------------------------------------------------------
    def _compute_uncertainty(self, pred_signal, cb_weights_list,
                             block_patterns_list, cls_hidden):
        """
        Flatten captured signals and pass through MultiSourceUncertaintyHead.

        All inputs are in the autograd graph where possible (block_patterns are
        detached in HookedMHA to avoid second-order gradients through the
        per-block self-attention softmax).
        """
        # Flatten attention patterns: each (B, n_heads, L, L) → (B, n_heads*L*L)
        patterns_flat = torch.cat(
            [p.reshape(p.shape[0], -1) for p in block_patterns_list],
            dim=-1,
        )  # (B, pattern_dim)

        # Flatten cross-block weights: each (B, 3, 3) → (B, 9)
        cb_flat = torch.cat(
            [w.reshape(w.shape[0], -1) for w in cb_weights_list],
            dim=-1,
        )  # (B, 18)

        return self.unc_head(pred_signal, patterns_flat, cb_flat, cls_hidden)


# ===========================================================================
# UNIT TEST
# ===========================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("  P1b Joint Fusion Module — Unit Test")
    print("=" * 64)

    # ---- Configuration ----
    B = 2
    n_a = 20
    sig_dim = 39
    cl_dim = 4
    d_model = 128
    n_heads = 4
    d_ff = 256
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  n_a={n_a}  sig_dim={sig_dim}  cl_dim={cl_dim}  B={B}")
    print(f"  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff}")

    torch.manual_seed(42)
    sigs = torch.randn(B, n_a, sig_dim)
    cl = torch.randn(B, n_a, cl_dim)

    # =====================================================================
    # Test 1 — Full mode: shapes, fusion info, gradients
    # =====================================================================
    print(f"\n{'─' * 50}")
    print("  [1] Full mode — forward shapes")
    print(f"{'─' * 50}")

    model_full = JointFusionTransformer(
        n_a=n_a, sig_dim=sig_dim, cl_dim=cl_dim,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        unc_mode='full',
    )
    n_total = sum(p.numel() for p in model_full.parameters())
    n_train = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"  Total params:     {n_total:>10,}")
    print(f"  Trainable params: {n_train:>10,}")

    model_full.eval()
    with torch.no_grad():
        ret_pred, unc_pred, fusion_info = model_full(sigs, cl)

    print(f"\n  ret_pred       shape: {tuple(ret_pred.shape)}")
    print(f"  unc_pred       shape: {tuple(unc_pred.shape)}")
    print(f"\n  fusion_info keys: {list(fusion_info.keys())}")
    print(f"  cb_weights:        list of {len(fusion_info['cb_weights'])}")
    for i, w in enumerate(fusion_info['cb_weights']):
        print(f"    [{i}] shape={tuple(w.shape)}")
    print(f"  block_patterns:    list of {len(fusion_info['block_patterns'])}")
    for i, p in enumerate(fusion_info['block_patterns']):
        print(f"    [{i}] shape={tuple(p.shape)}  "
              f"(L{i//3} block{i%3})")
    print(f"  pred_signal:       shape={tuple(fusion_info['pred_signal'].shape)}")
    print(f"  cls_hidden:        shape={tuple(fusion_info['cls_hidden'].shape)}")

    # ---- Gradient check (full mode) ----
    print(f"\n  Gradient flow check (full mode) ...")
    model_full.train()
    rp, up, _ = model_full(sigs, cl)
    y_dummy = torch.randn_like(rp)
    loss = F.huber_loss(rp, y_dummy, delta=1.0)
    with torch.no_grad():
        err = (rp - y_dummy).abs().mean(dim=1)
    loss = loss + 0.1 * F.mse_loss(up, err)
    model_full.zero_grad(set_to_none=True)
    loss.backward()

    nil_grads = [n for n, p in model_full.named_parameters()
                 if p.requires_grad and p.grad is None]
    zero_grads = [n for n, p in model_full.named_parameters()
                  if p.requires_grad and p.grad is not None
                  and p.grad.norm().item() == 0.0]

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Parameters with None grad: {len(nil_grads)}")
    for n in nil_grads[:8]:
        print(f"    NIL: {n}")
    print(f"  Parameters with zero grad: {len(zero_grads)}")
    for n in zero_grads[:8]:
        print(f"    ZERO: {n}")

    if nil_grads or zero_grads:
        print(f"  *** GRADIENT ISSUES DETECTED ***")
    else:
        print(f"  All gradients flow correctly.")

    # =====================================================================
    # Test 2 — All ablation variants
    # =====================================================================
    print(f"\n{'─' * 50}")
    print("  [2] Ablation variants")
    print(f"{'─' * 50}")

    for mode in ['pred_only', 'pred_patterns', 'pred_attnres',
                 'pred_patterns_attnres', 'full']:
        m = JointFusionTransformer(
            n_a=n_a, sig_dim=sig_dim, cl_dim=cl_dim,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            unc_mode=mode,
        )
        m.eval()
        with torch.no_grad():
            rp, up, fi = m(sigs, cl)

        n_p = sum(p.numel() for p in m.unc_head.parameters())
        print(f"  {mode:24s}  unc_pred={up[0].item():+.4f}  "
              f"unc_head_params={n_p:>7,}  ret_pred_norm={rp.norm().item():.2f}")

    # =====================================================================
    # Test 3 — Gradient flow per ablation mode
    # =====================================================================
    print(f"\n{'─' * 50}")
    print("  [3] Gradient flow per mode")
    print(f"{'─' * 50}")

    all_ok = True
    for mode in ['pred_only', 'pred_patterns', 'pred_attnres',
                 'pred_patterns_attnres', 'full']:
        m = JointFusionTransformer(
            n_a=n_a, sig_dim=sig_dim, cl_dim=cl_dim,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            unc_mode=mode,
        )
        m.train()
        rp, up, _ = m(sigs, cl)
        y_dummy = torch.randn_like(rp)
        loss = F.huber_loss(rp, y_dummy, delta=1.0)
        with torch.no_grad():
            err = (rp - y_dummy).abs().mean(dim=1)
        loss = loss + 0.1 * F.mse_loss(up, err)
        m.zero_grad(set_to_none=True)
        loss.backward()

        nil_count = sum(1 for _, p in m.named_parameters()
                        if p.requires_grad and p.grad is None)
        zero_count = sum(1 for _, p in m.named_parameters()
                         if p.requires_grad and p.grad is not None
                         and p.grad.norm().item() == 0.0)
        status = "OK" if (nil_count == 0 and zero_count == 0) else "FAIL"
        if status != "OK":
            all_ok = False
        print(f"  {mode:24s}  nil_grads={nil_count}  zero_grads={zero_count}  "
              f"loss={loss.item():.6f}  [{status}]")

    # =====================================================================
    # Test 4 — Different modes produce different uncertainty
    # =====================================================================
    print(f"\n{'─' * 50}")
    print("  [4] Mode diversity check")
    print(f"{'─' * 50}")

    unc_values = {}
    for mode in ['pred_only', 'pred_patterns', 'pred_attnres',
                 'pred_patterns_attnres', 'full']:
        m = JointFusionTransformer(
            n_a=n_a, sig_dim=sig_dim, cl_dim=cl_dim,
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            unc_mode=mode,
        )
        m.eval()
        with torch.no_grad():
            _, up, _ = m(sigs, cl)
        unc_values[mode] = up.clone()

    # Verify that different modes give different outputs (not all identical)
    vals = torch.stack(list(unc_values.values()), dim=0)  # (5, B)
    unique_pairs = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if not torch.allclose(vals[i], vals[j], atol=1e-4):
                unique_pairs += 1
    total_pairs = 5 * 4 // 2  # 10
    print(f"  Unique pairwise comparisons: {unique_pairs}/{total_pairs}")
    print(f"  (Different modes should produce different uncertainty values)")

    # =====================================================================
    # Test 5 — NaN and infinity sanity
    # =====================================================================
    print(f"\n{'─' * 50}")
    print("  [5] NaN / Inf sanity")
    print(f"{'─' * 50}")

    model_full.eval()
    with torch.no_grad():
        rp, up, fi = model_full(sigs, cl)

    def _check_nan_inf(name, tensor):
        if isinstance(tensor, torch.Tensor):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan or has_inf:
                return f"  !! {name}: nan={has_nan} inf={has_inf}"
        elif isinstance(tensor, list):
            for i, t in enumerate(tensor):
                if isinstance(t, torch.Tensor):
                    has_nan = torch.isnan(t).any().item()
                    has_inf = torch.isinf(t).any().item()
                    if has_nan or has_inf:
                        return f"  !! {name}[{i}]: nan={has_nan} inf={has_inf}"
        return None

    issues = []
    issues.append(_check_nan_inf("ret_pred", rp))
    issues.append(_check_nan_inf("unc_pred", up))
    for k, v in fi.items():
        issues.append(_check_nan_inf(k, v))
    issues = [x for x in issues if x is not None]

    if issues:
        for msg in issues:
            print(msg)
    else:
        print("  All tensors clean (no NaN, no Inf).")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 64}")
    if all_ok and not issues:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — see details above")
    print(f"{'=' * 64}")
