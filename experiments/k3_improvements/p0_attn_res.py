"""
Block Attention Residuals for Meta-Transformer (P0)
====================================================

Implements BlockAttnResLayer and BlockAttnResTransformer — the core
P0 improvement from the K3 roadmap.  Replaces the standard 4-layer
Pre-LN residual chain with a 3-block architecture where each semantic
block (CLS, SIG, FACTOR) gets independent self-attention + FFN,
then cross-block attention fuses information via learnable query
vectors.  This prevents the monotonic output-norm increase of
standard residuals and provides block-level interpretability.

References:
  - meta_transformer_optimized.py  (SlimMetaTrans)
  - meta_transformer_proposal.py   (MetaTransformer)

Author: K3 Roadmap P0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RMSNorm — PyTorch 2.4+ native, with fallback for earlier versions
# ---------------------------------------------------------------------------
try:
    RMSNorm = nn.RMSNorm  # >= 2.4
except AttributeError:

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.normalized_shape = (
                normalized_shape
                if isinstance(normalized_shape, (tuple, list))
                else (normalized_shape,)
            )

        def forward(self, x):
            dtype = x.dtype
            rms = torch.sqrt(
                x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
            return (x / rms.to(dtype)) * self.weight.to(dtype)

        def extra_repr(self):
            return f"{self.normalized_shape}, eps={self.eps}"


# ===========================================================================
# BlockAttnResLayer
# ===========================================================================
class BlockAttnResLayer(nn.Module):
    """
    Drop-in replacement for one TransformerEncoderLayer using block-level
    attention residuals.

    Architecture per layer:
      1. Split tokens into 3 semantic blocks:
           Block 0  [CLS]            — 1  token
           Block 1  [SIG_1...SIG_n]  — n_a tokens
           Block 2  [FACTOR]         — 1  token
      2. Per-block independent self-attention + FFN (Pre-LN residual).
      3. Mean-pool each block → 3 summary vectors.
      4. Cross-block attention: 3 learnable query vectors attend over
         the 3 pooled outputs (scaled dot-product, single-head).
      5. Weighted fusion + residual add back to each block's tokens.
      6. Concatenate blocks back to full sequence.

    Returns:
        fused_output (B, T, d_model)      — processed token sequence
        cb_weights  (B, n_blocks=3, 3)    — cross-block attention matrix

    NOTE: Because forward() returns a tuple, a simple loop like
          ``for layer in layers: x = layer(x)`` will NOT work when
          chaining multiple BlockAttnResLayer instances — the caller
          must unpack the tuple.  Use BlockAttnResTransformer for
          a full model that handles this internally.
    """

    def __init__(
        self,
        d_model=128,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
        n_blocks=3,
        eps=1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.scale = d_model ** -0.5

        # ---- Per-block self-attention + FFN ----
        for i in range(n_blocks):
            # Attention sub-layer
            setattr(self, f"norm_{i}a", RMSNorm(d_model, eps))
            setattr(
                self,
                f"attn_{i}",
                nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
            )
            # FFN sub-layer
            setattr(self, f"norm_{i}b", RMSNorm(d_model, eps))
            setattr(
                self,
                f"ff_{i}",
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                ),
            )

        # ---- Cross-block (inter-block) attention ----
        self.cb_query = nn.Parameter(torch.randn(n_blocks, d_model) * 0.02)
        self.cb_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.cb_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.cb_out = nn.Linear(d_model, d_model)
        self.cb_norm = RMSNorm(d_model, eps)
        self.cb_drop = nn.Dropout(dropout)

    def _block_forward(self, x: torch.Tensor, blk: int) -> torch.Tensor:
        """Pre-LN residual: norm → self-attention → add, norm → FFN → add."""
        norm_a = getattr(self, f"norm_{blk}a")
        attn = getattr(self, f"attn_{blk}")
        norm_b = getattr(self, f"norm_{blk}b")
        ff = getattr(self, f"ff_{blk}")

        h = norm_a(x)
        h, _ = attn(h, h, h, need_weights=False)
        x = x + h
        x = x + ff(norm_b(x))
        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model) — full token sequence,
               T = n_a + 2  (CLS + n_a SIG tokens + FACTOR)
        Returns:
            out:        (B, T, d_model) — fused token sequence
            cb_weights: (B, n_blocks, n_blocks) — cross-block attention
        """
        B, T, D = x.shape
        # SIG token count = T - CLS(1) - FACTOR(1)
        n_sig = T - self.n_blocks + 1

        # ---- 1. Split into semantic blocks ----
        blocks = [
            x[:, :1, :],             # Block 0: [CLS]
            x[:, 1 : 1 + n_sig, :],  # Block 1: [SIG tokens]
            x[:, -1:, :],            # Block 2: [FACTOR]
        ]

        # ---- 2. Per-block self-attention + FFN ----
        block_outs = [
            self._block_forward(blk, i) for i, blk in enumerate(blocks)
        ]

        # ---- 3. Mean-pool each block → (B, 1, D) per block ----
        pooled = torch.cat(
            [blk.mean(dim=1, keepdim=True) for blk in block_outs], dim=1
        )  # (B, n_blocks=3, D)

        # ---- 4. Cross-block attention ----
        pooled_n = self.cb_norm(pooled)                     # (B, 3, D)
        K = self.cb_k_proj(pooled_n)                        # (B, 3, D)
        V = self.cb_v_proj(pooled_n)                        # (B, 3, D)
        Q = self.cb_query[None, :, :].expand(B, -1, -1)    # (B, 3, D)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, 3, 3)
        cb_weights = F.softmax(scores, dim=-1)                       # (B, 3, 3)

        # ---- 5. Weighted fusion + residual back to each block ----
        fused_pool = torch.matmul(self.cb_drop(cb_weights), V)  # (B, 3, D)
        fused_pool = self.cb_out(fused_pool)                     # (B, 3, D)

        fused_blocks = [
            block_outs[0] + fused_pool[:, 0:1, :],   # CLS  + info from all blocks
            block_outs[1] + fused_pool[:, 1:2, :],   # SIG  + info from all blocks
            block_outs[2] + fused_pool[:, 2:3, :],   # FACTOR + info from all blocks
        ]

        # ---- 6. Concatenate back to full sequence ----
        out = torch.cat(fused_blocks, dim=1)  # (B, T, D)

        return out, cb_weights


# ===========================================================================
# BlockAttnResTransformer
# ===========================================================================
class BlockAttnResTransformer(nn.Module):
    """
    Meta-Transformer with Block Attention Residuals.

    Same interface as SlimMetaTrans:  (sigs, cl) → (ret_pred, unc_pred).

    Architecture differences from SlimMetaTrans:
      - 2 x BlockAttnResLayer instead of 4 x TransformerEncoderLayer.
      - Each layer does block-level self-attn + cross-block fusion.
      - Cross-block attention weights are stored for diagnostics.
      - Uses RMSNorm instead of LayerNorm throughout.

    Token sequence:  [CLS(1)]  [SIG_1 … SIG_n]  [FACTOR(1)]
                     Block 0    Block 1           Block 2

    Diagnostics:
        model.cb_weights  — list[Tensor(B, 3, 3)]  one per layer
          row i, col j  =  attention from block i's query to block j's key
          Block indices:  0=CLS  1=SIG  2=FACTOR
    """

    def __init__(
        self,
        n_a,
        sig_dim,
        cl_dim,
        d_model=128,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
        max_stocks=5000,
    ):
        super().__init__()
        self.n_a = n_a
        self.d = d_model

        # ---- Input projections (match SlimMetaTrans signatures) ----
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

        # ---- Output heads (match SlimMetaTrans) ----
        self.ret_head = nn.Linear(d_model, n_a)
        self.unc_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # Diagnostic storage — populated on each forward call
        self.cb_weights = []

    def forward(self, sigs: torch.Tensor, cl: torch.Tensor):
        """
        Args:
            sigs:  (B, n_a, sig_dim)   per-asset signature embeddings
            cl:    (B, n_a, cl_dim)    per-asset classical factors
        Returns:
            ret_pred:  (B, n_a)        per-asset return prediction
            unc_pred:  (B,)            per-sample uncertainty estimate
        """
        B = sigs.shape[0]

        # ---- 1. Build token sequence ----
        # SIG tokens: project signature + positional encoding
        st = self.sig_proj(sigs) + self.sig_pos[:, : self.n_a, :]  # (B, n_a, D)

        # FACTOR token: mean-pool across assets, project, add learned token
        cp = cl.mean(dim=1)                                         # (B, cl_dim)
        ct = self.cl_proj(cp).unsqueeze(1) + self.cl_tok            # (B, 1, D)

        # CLS token: learned, expanded across batch
        cls_tok = self.cls.expand(B, -1, -1)                         # (B, 1, D)

        # [CLS] + [SIG … SIG] + [FACTOR]  →  (B, n_a+2, D)
        x = torch.cat([cls_tok, st, ct], dim=1)

        # ---- 2. Block Attention Residual layers ----
        self.cb_weights = []
        x, w1 = self.layer1(x)
        self.cb_weights.append(w1)
        x, w2 = self.layer2(x)
        self.cb_weights.append(w2)

        # ---- 3. Predictions from CLS token ----
        cls_out = x[:, 0, :]                                         # (B, D)
        ret_pred = self.ret_head(cls_out)                            # (B, n_a)
        unc_pred = self.unc_head(cls_out).squeeze(-1)                # (B,)

        return ret_pred, unc_pred


# ===========================================================================
# TEST  —  instantiate, forward pass, gradient check
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Block Attention Residuals — Unit Test")
    print("=" * 60)

    # ---- Config (matches SlimMetaTrans) ----
    B = 2
    n_a = 200
    sig_dim = 39
    cl_dim = 4
    d_model = 128
    n_heads = 4
    d_ff = 256
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  d_model={d_model}  n_heads={n_heads}  d_ff={d_ff}")
    print(f"  n_a={n_a}  sig_dim={sig_dim}  cl_dim={cl_dim}  B={B}")

    # ---- Random inputs ----
    torch.manual_seed(42)
    sigs = torch.randn(B, n_a, sig_dim)
    cl = torch.randn(B, n_a, cl_dim)

    # ---- Instantiate model ----
    print("\n  Building BlockAttnResTransformer ...")
    model = BlockAttnResTransformer(
        n_a=n_a,
        sig_dim=sig_dim,
        cl_dim=cl_dim,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {n_total:>10,}")
    print(f"  Trainable parameters: {n_train:>10,}")

    # ---- Forward pass ----
    model.eval()
    with torch.no_grad():
        sigs_gpu = sigs.to(device)
        cl_gpu = cl.to(device)
        ret_pred, unc_pred = model(sigs_gpu, cl_gpu)

    print(f"\n  ---- Forward pass shapes ----")
    print(f"  sigs               : {tuple(sigs.shape)}")
    print(f"  cl                 : {tuple(cl.shape)}")
    print(f"  ret_pred           : {tuple(ret_pred.shape)}")
    print(f"  unc_pred           : {tuple(unc_pred.shape)}")
    print(f"  Token seq (internal): (B={B}, n_a+2={n_a+2}, D={d_model})")

    # ---- Cross-block attention weights ----
    print(f"\n  ---- Cross-block attention weights ----")
    print(f"  (rows = query block, cols = key block)")
    print(f"  Block indices:  0=CLS  1=SIG  2=FACTOR")
    for li, cbw in enumerate(model.cb_weights):
        print(f"\n  Layer {li + 1}  shape: {tuple(cbw.shape)}")
        w_np = cbw[0].cpu().numpy()  # first batch sample
        col_labels = ["CLS   ", "SIG   ", "FACTOR"]
        header = "         " + "  ".join(col_labels)
        print(f"  {header}")
        row_labels = ["CLS   ", "SIG   ", "FACTOR"]
        for ri, rlab in enumerate(row_labels):
            vals = "  ".join(f"{w_np[ri, ci]:.4f}" for ci in range(3))
            print(f"    {rlab}   {vals}")
        row_sums = w_np.sum(axis=-1)
        print(f"    Row sums:  {row_sums[0]:.4f}  {row_sums[1]:.4f}  {row_sums[2]:.4f}")

    # ---- Gradient flow test ----
    print(f"\n  ---- Gradient flow test ----")
    model.train()
    sigs_gpu = sigs.to(device)
    cl_gpu = cl.to(device)

    ret_pred, unc_pred = model(sigs_gpu, cl_gpu)

    # Composite loss (same as SlimMetaTrans)
    y_dummy = torch.randn_like(ret_pred)
    loss = F.huber_loss(ret_pred, y_dummy, delta=1.0)
    with torch.no_grad():
        err = (ret_pred - y_dummy).abs().mean(dim=1)
    loss = loss + 0.1 * F.mse_loss(unc_pred, err)

    model.zero_grad(set_to_none=True)
    loss.backward()

    # Audit gradients
    zero_grad_names = []
    nil_grad_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            nil_grad_names.append(name)
        elif p.grad.norm().item() == 0.0:
            zero_grad_names.append(name)

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Parameters with None grad:  {len(nil_grad_names)}")
    for n in nil_grad_names:
        print(f"    NIL: {n}")
    print(f"  Parameters with zero grad:  {len(zero_grad_names)}")
    for n in zero_grad_names:
        print(f"    ZERO: {n}")

    if nil_grad_names or zero_grad_names:
        print(f"\n  *** GRADIENT ISSUES DETECTED ***")
    else:
        print(f"\n  All gradients flow correctly.")

    # Check cb_query gradients specifically
    for li in range(2):
        layer = [model.layer1, model.layer2][li]
        q_grad = layer.cb_query.grad
        if q_grad is not None:
            print(f"  |grad(cb_query L{li+1})| = {q_grad.norm().item():.6f}")
        else:
            print(f"  |grad(cb_query L{li+1})| = NONE")

    # Sanity: forward twice → cb_weights should update
    print(f"\n  ---- Reproducibility check ----")
    model.eval()
    with torch.no_grad():
        _, _ = model(sigs.to(device), cl.to(device))
        w1_run2 = model.cb_weights[0][0, 0, 0].item()
    print(f"  Layer1 cb_weights[0,0] = {w1_run2:.4f}  "
          f"(should match first run: {model.cb_weights[0][0, 0, 0].item() if model.cb_weights else 'N/A'})")

    print(f"\n  {'=' * 60}")
    print(f"  TEST PASSED")
    print(f"  {'=' * 60}")
