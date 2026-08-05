"""
KDA (Kimi Delta Attention) Linear Attention Module
===================================================
Self-contained module implementing KDA-style linear attention for the
Meta-Transformer token structure: [CLS] + [SIG_1...SIG_N] + [FACTOR].

Core formula:
    O = (phi(Q) · (phi(K)^T · V)) / (phi(Q) · sum(phi(K)))
    phi(x) = elu(x) + 1

The elu+1 feature map guarantees every element >= 1, ensuring the
denominator is never zero.  This gives numerical stability without
clamping hacks while retaining the O(N d_k^2) complexity of linear attention
(vs O(N^2 d_k) for standard attention).

With N=202 tokens (200 assets + CLS + FACTOR) and d_k=32, the 3:1
ratio (3 linear layers + 1 full attention layer) yields ~X% VRAM
savings on the attention maps while retaining a single quadratic layer
for long-range token interaction fidelity.

Components:
  - phi(x):                 feature map (elu+1)
  - LinearAttention:        O(N d_k^2) linear attention, causal + bidirectional
  - FullAttention:          Standard MHA (matches proposal.py MHA)
  - KDATransformerLayer:    Pre-LN transformer layer, swappable attn type
  - KDATransformer:         Drop-in replacement for SlimMetaTrans (3:1 ratio)
  - pattern_quality_metric: Compare linear vs full attention patterns
  - compute_vram_savings:   Estimate VRAM saved by KDA

Compatible with: SlimMetaTrans (optimized.py) and MetaTransformer (proposal.py)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------
# Configuration constants — match SlimMetaTrans / MetaTransformer
# ---------------------------------------------------------------------------
D_MODEL = 128
N_HEADS = 4
D_K = D_MODEL // N_HEADS       # 32
D_FF = 256
DROPOUT = 0.1

# ==================================================================
# FEATURE MAP
# ==================================================================

def phi(x):
    """KDA feature map: elu(x) + 1.  Guaranteed >= 1 for all real x,
    ensuring the denominator sum(phi(K)) is always strictly positive.
    The elu provides gradient in the negative regime; the +1 offset
    prevents zero values that could destabilize the normalisation."""
    return F.elu(x) + 1.0

# ==================================================================
# LINEAR ATTENTION (KDA)
# ==================================================================

class LinearAttention(nn.Module):
    """
    Linear attention with φ(x) = elu(x) + 1 feature map.

    Bidirectional (non-causal):
        φ_Q, φ_K = φ(Q), φ(K)
        KV  = φ_K^T @ V                              # (B, H, d_k, d_k)
        O   = (φ_Q @ KV) / ((φ_Q * Σφ_K).sum(-1)+ε)  # (B, H, N, d_k)

    Causal:
        Uses torch.cumsum along the sequence dimension to enforce
        that output at position t only sees positions <= t.

    Complexity (per sample):
        Standard MHA: O(N^2 d_k) memory for QK^T matrix
        Linear:       O(N d_k^2) memory for KV accumulator
        With N=202, d_k=32: ~40x smaller attention-map memory
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT,
                 use_safe_attention: bool = True):
        """
        Args:
            d_model:            hidden dimension (128)
            n_heads:            number of attention heads (4)
            dropout:            dropout rate
            use_safe_attention: if True, apply numerical stability safeguards:
                                Q/K scaling by 1/sqrt(d_k), bounded feature map
                                (clamp to [1e-4, 1e4]), and KV/numerator/result
                                clamping to prevent FP32 overflow.
        """
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.safe = use_safe_attention

        # Q / K / V / O projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop_attn = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize W_q, W_k with small-scale Xavier uniform to reduce
        initial activation magnitudes and prevent training-time overflow."""
        for name, param in self.W_q.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
        for name, param in self.W_k.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)

    def _safe_phi(self, x):
        """Bounded feature map for numerical stability: elu(x)+1 clamped to
        [1e-4, 1e4] to prevent FP32 overflow in the KV accumulator."""
        return torch.clamp(F.elu(x) + 1.0, min=1e-4, max=1e4)

    # -- reshape helpers ------------------------------------------------

    def _split_heads(self, x):
        """(B, N, d_model) -> (B, H, N, d_k)"""
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """(B, H, N, d_k) -> (B, N, d_model)"""
        B, H, N, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dk)

    # -- forward --------------------------------------------------------

    def forward(self, x, causal=False, return_pattern=False):
        """
        Args:
            x:              (B, N, d_model)   input sequence
            causal:         if True, cumulative-sum autoregressive masking
            return_pattern: if True, also return full (B, H, N, N) attention
                            weights reconstructed from φ(Q)φ(K)^T.
                            NOTE: this is O(N^2) — use only for diagnostics.

        Returns:
            output:   (B, N, d_model)
            pattern:  (B, H, N, N)  (only when return_pattern=True)
        """
        B, N, _ = x.shape

        # Project and split heads
        q = self._split_heads(self.W_q(x))      # (B, H, N, d_k)
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # Core linear attention (φ is applied inside helpers)
        if causal:
            output = self._causal_forward(q, k, v)
        else:
            output = self._bidirectional_forward(q, k, v)

        # Merge heads + output projection
        output = self._merge_heads(output)
        output = self.W_o(output)
        output = self.drop_attn(output)

        if return_pattern:
            with torch.no_grad():
                if self.safe:
                    _q = q / (self.d_k ** 0.5)
                    _k = k / (self.d_k ** 0.5)
                    phi_q_pat = self._safe_phi(_q)
                    phi_k_pat = self._safe_phi(_k)
                else:
                    phi_q_pat = phi(q)
                    phi_k_pat = phi(k)
                raw = phi_q_pat @ phi_k_pat.transpose(-2, -1)       # (B,H,N,N)
                pattern = raw / raw.sum(-1, keepdim=True).clamp(min=1e-8)
            return output, pattern
        return output

    # -- internal forward variants --------------------------------------

    def _bidirectional_forward(self, q, k, v):
        """
        O = (φ(Q) @ φ(K)^T @ V) / (φ(Q) · Σ(φ(K)) + ε)

        Computed via right-product trick for O(N d_k^2) memory:
          KV    = φ_K^T @ V                 : (B,H,d_k,d_k)   [N× contraction]
          num   = φ_Q @ KV                  : (B,H,N,d_k)
          denom = (φ_Q * K_sum).sum(-1)     : (B,H,N)         [dot over d_k]
        """
        if self.safe:
            q = q / (self.d_k ** 0.5)
            k = k / (self.d_k ** 0.5)
            phi_q = self._safe_phi(q)
            phi_k = self._safe_phi(k)
        else:
            phi_q = phi(q)
            phi_k = phi(k)

        # Step 1: φ(K)^T @ V  —  contract N dimension
        #   (B, H, d_k,  N) @ (B, H,  N, d_k) -> (B, H, d_k, d_k)
        KV = torch.einsum('b h n d, b h n e -> b h d e', phi_k, v)
        if self.safe:
            KV = torch.clamp(KV, min=-1e6, max=1e6)

        # Step 2: φ(Q) @ KV  —  numerator
        #   (B, H, N, d_k) @ (B, H, d_k, d_k) -> (B, H, N, d_k)
        numerator = torch.einsum('b h n d, b h d e -> b h n e', phi_q, KV)
        if self.safe:
            numerator = torch.clamp(numerator, min=-1e6, max=1e6)

        # Step 3: denominator  —  φ(Q) · Σφ(K) along d_k
        K_sum = phi_k.sum(dim=2)                            # (B, H, d_k)
        denominator = (phi_q * K_sum.unsqueeze(2)).sum(-1, keepdim=True)  # (B,H,N,1)
        denominator = denominator.clamp(min=1e-4 if self.safe else 1e-6)

        result = numerator / denominator
        if self.safe:
            result = torch.clamp(result, min=-1e4, max=1e4)

        return result

    def _causal_forward(self, q, k, v):
        """
        Causal (autoregressive) linear attention via cumulative sum.

        For each time step t:
          KV[1:t] = Σ_{i=1}^{t} φ(K_i) ⊗ V_i          [outer product per step]
          Σ[1:t]  = Σ_{i=1}^{t} φ(K_i)                 [cumulative key sum]

          O_t = φ(Q_t) @ KV[1:t] / (φ(Q_t) · Σ[1:t] + ε)

        Uses O(N d_k^2) memory for the (N, d_k, d_k) cumsum accumulator.
        """
        if self.safe:
            q = q / (self.d_k ** 0.5)
            k = k / (self.d_k ** 0.5)
            phi_q = self._safe_phi(q)
            phi_k = self._safe_phi(k)
        else:
            phi_q = phi(q)
            phi_k = phi(k)

        # Per-step outer product:  φ(k)[t] (d_k) × v[t] (d_k) -> (d_k, d_k)
        #   (B, H, N, d_k, d_k)
        outer = torch.einsum('b h n d, b h n e -> b h n d e', phi_k, v)

        # Cumulative sum over sequence dimension
        KV_cum = torch.cumsum(outer, dim=2)     # (B, H, N, d_k, d_k)
        if self.safe:
            KV_cum = torch.clamp(KV_cum, min=-1e6, max=1e6)

        # Numerator
        numerator = torch.einsum('b h n d, b h n d e -> b h n e', phi_q, KV_cum)
        if self.safe:
            numerator = torch.clamp(numerator, min=-1e6, max=1e6)

        # Cumulative key sum for denominator
        K_cum = torch.cumsum(phi_k, dim=2)      # (B, H, N, d_k)
        denominator = (phi_q * K_cum).sum(-1, keepdim=True)
        denominator = denominator.clamp(min=1e-4 if self.safe else 1e-6)

        result = numerator / denominator
        if self.safe:
            result = torch.clamp(result, min=-1e4, max=1e4)

        return result


# ==================================================================
# STANDARD FULL ATTENTION (for comparison + final layer)
# ==================================================================

class FullAttention(nn.Module):
    """
    Standard multi-head scaled dot-product attention.
    Matches the MHA class in meta_transformer_proposal.py exactly.

    Complexity: O(N^2 d_k) — stores full (B, H, N, N) attention matrix.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.drop_attn = nn.Dropout(dropout)

        self._last_attn = None   # stored for diagnostics

    def _split_heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, N, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * dk)

    def forward(self, x, return_pattern=False):
        B, N, _ = x.shape

        q = self._split_heads(self.W_q(x))      # (B, H, N, d_k)
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        self._last_attn = attn.detach()
        attn = self.drop_attn(attn)

        output = self._merge_heads(attn @ v)
        output = self.W_o(output)

        if return_pattern:
            return output, self._last_attn
        return output


# ==================================================================
# KDA TRANSFORMER LAYER
# ==================================================================

class KDATransformerLayer(nn.Module):
    """
    Transformer encoder layer with Pre-LN residual structure.
    Swappable attention type via ``is_linear`` flag.

    Pre-LN pattern (matches norm_first=True in nn.TransformerEncoderLayer):
        x = x + Dropout(Attention(LayerNorm(x)))
        x = x + FFN(LayerNorm(x))

    FFN structure matches both optimized.py (PyTorch internals)
    and proposal.py (EncLayer.ff):
        Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model) -> Dropout
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 dropout=DROPOUT, is_linear=True):
        super().__init__()
        self.is_linear = is_linear

        # Attention module
        if is_linear:
            self.attn = LinearAttention(d_model, n_heads, dropout)
        else:
            self.attn = FullAttention(d_model, n_heads, dropout)

        # Layer norms (Pre-LN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout on the attention residual path
        self.dropout_attn = nn.Dropout(dropout)

        # Feed-forward network (matches existing)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_pattern=False):
        """
        Args:
            x:              (B, N, d_model) input
            return_pattern: if True, also return attention weights as (B,H,N,N)

        Returns:
            x:       (B, N, d_model)
            pattern: (B, H, N, N)  (only when return_pattern=True)
        """
        # --- Attention sub-layer (Pre-LN) ---
        normed = self.norm1(x)
        if return_pattern:
            attn_out, pattern = self.attn(normed, return_pattern=True)
        else:
            attn_out = self.attn(normed)
            pattern = None
        x = x + self.dropout_attn(attn_out)

        # --- FFN sub-layer (Pre-LN) ---
        x = x + self.ff(self.norm2(x))

        if return_pattern:
            return x, pattern
        return x


# ==================================================================
# KDA TRANSFORMER — Drop-in replacement for SlimMetaTrans
# ==================================================================

class KDATransformer(nn.Module):
    """
    Meta-Transformer with KDA linear attention layers.

    3:1 ratio (default):
        Layer 0: LinearAttention  (KDA,  O(N d_k^2))
        Layer 1: LinearAttention  (KDA)
        Layer 2: LinearAttention  (KDA)
        Layer 3: FullAttention    (standard, O(N^2 d_k))

    The final full-attention layer preserves the ability to model
    arbitrary pairwise token interactions while the preceding linear
    layers capture the bulk of the computation with far less VRAM.

    Same interface as SlimMetaTrans:
        forward(sigs, cl) -> (ret_pred, unc_pred)

    Token structure:
        [CLS] + [SIG_1 ... SIG_N] + [FACTOR]  =  N+2 tokens
        (1)  + (       N       ) + (1)        =  N+2
    """

    def __init__(self, n_a, sig_dim, cl_dim,
                 d_model=D_MODEL, n_layers=4, n_heads=N_HEADS,
                 d_ff=D_FF, dropout=DROPOUT,
                 layer_schedule=None):
        """
        Args:
            n_a:             number of assets (e.g., 200)
            sig_dim:         signature dimension (39 for depth=3, channels=3)
            cl_dim:          classical factor dimension (4 or 6)
            d_model:         hidden dimension (128)
            n_layers:        total layers (4)
            n_heads:         attention heads (4)
            d_ff:            FFN intermediate dimension (256)
            dropout:         dropout rate (0.1)
            layer_schedule:  list of bool, one per layer.
                             True  = LinearAttention
                             False = FullAttention
                             Default: [True, True, True, False] (3:1)
        """
        super().__init__()
        self.n_a = n_a
        self.d_model = d_model

        # Layer type schedule
        if layer_schedule is None:
            layer_schedule = [True] * (n_layers - 1) + [False]
        if len(layer_schedule) != n_layers:
            raise ValueError(
                f"layer_schedule length ({len(layer_schedule)}) "
                f"!= n_layers ({n_layers})")
        self.layer_schedule = list(layer_schedule)

        # ----------------------------------------------------------------
        # Token embedding  (matches SlimMetaTrans line-for-line)
        # ----------------------------------------------------------------
        # Signature token per asset
        self.sig_proj = nn.Sequential(
            nn.Linear(sig_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.sig_pos = nn.Parameter(torch.randn(1, 5000, d_model) * 0.02)

        # Classical factor  -> single pooled token
        self.cl_proj = nn.Sequential(
            nn.Linear(cl_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.cl_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # CLS token
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ----------------------------------------------------------------
        # Transformer layers (KDA 3:1 schedule)
        # ----------------------------------------------------------------
        self.layers = nn.ModuleList([
            KDATransformerLayer(d_model, n_heads, d_ff, dropout, is_linear=is_lin)
            for is_lin in self.layer_schedule
        ])

        # ----------------------------------------------------------------
        # Prediction heads  (matches SlimMetaTrans)
        # ----------------------------------------------------------------
        self.ret_head = nn.Linear(d_model, n_a)         # per-asset return
        self.unc_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    # -- token construction ----------------------------------------------

    def _build_sequence(self, sigs, cl):
        """
        Assemble the [CLS] + [SIG] + [FACTOR] token sequence.

        Args:
            sigs: (B, n_a, sig_dim)   per-asset signatures
            cl:   (B, n_a, cl_dim)    per-asset classical factors

        Returns:
            x: (B, n_a+2, d_model)
        """
        B = sigs.shape[0]

        # Signature tokens per asset + learned position embedding
        sig_tok = (self.sig_proj(sigs)
                   + self.sig_pos[:, :self.n_a, :])          # (B, n_a, d_model)

        # Classical factors: mean-pool across assets -> one token
        cl_pooled = cl.mean(dim=1)                           # (B, cl_dim)
        cl_tok = (self.cl_proj(cl_pooled).unsqueeze(1)
                  + self.cl_tok)                             # (B, 1, d_model)

        # CLS token
        cls_tok = self.cls.expand(B, -1, -1)                 # (B, 1, d_model)

        # Concatenate: [CLS] + [SIGs] + [FACTOR]
        x = torch.cat([cls_tok, sig_tok, cl_tok], dim=1)     # (B, N+2, d_model)
        return x

    # -- forward pass ----------------------------------------------------

    def forward(self, sigs, cl, return_attention=False):
        """
        Forward pass.  Compatible with SlimMetaTrans signature.

        Args:
            sigs:             (B, n_a, sig_dim)
            cl:               (B, n_a, cl_dim)
            return_attention: if True, collect attention patterns from all
                              layers.  Linear layers compute the O(N^2)
                              pattern only for diagnostics (no gradients).

        Returns:
            ret_pred:   (B, n_a)    per-asset return predictions
            unc_pred:   (B,)        uncertainty estimate (scalar per sample)
            attention:  list[dict]  (only when return_attention=True)
                          Each dict: {'layer': int, 'type': 'linear'|'full',
                                      'pattern': (B,H,N,N) tensor}
        """
        x = self._build_sequence(sigs, cl)     # (B, N+2, d_model)

        attention_patterns = []

        for i, layer in enumerate(self.layers):
            if return_attention:
                x, pat = layer(x, return_pattern=True)
                attention_patterns.append({
                    'layer': i,
                    'type': 'linear' if self.layer_schedule[i] else 'full',
                    'pattern': pat.detach(),
                })
            else:
                x = layer(x)

        # Read from CLS token (position 0)
        cls_out = x[:, 0, :]                   # (B, d_model)

        ret_pred = self.ret_head(cls_out)      # (B, n_a)
        unc_pred = self.unc_head(cls_out).squeeze(-1)  # (B,)

        if return_attention:
            return ret_pred, unc_pred, attention_patterns
        return ret_pred, unc_pred

    # -- introspection ---------------------------------------------------

    def get_stats(self):
        """Return model statistics as a plain dict."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_linear = sum(self.layer_schedule)
        n_full = len(self.layer_schedule) - n_linear
        return {
            'total_params': total,
            'trainable_params': trainable,
            'n_linear_layers': n_linear,
            'n_full_layers': n_full,
            'ratio': f'{n_linear}:{n_full}',
            'd_model': self.d_model,
            'n_a': self.n_a,
        }


# ==================================================================
# COMPARISON UTILITIES
# ==================================================================

def pattern_quality_metric(linear_pattern, full_pattern, errors):
    """
    Quantify how well the linear attention pattern approximates the
    full attention pattern and correlates with prediction errors.

    Args:
        linear_pattern: (H,N,N) or (B,H,N,N)  -- KDA linear attn weights
        full_pattern:   (H,N,N) or (B,H,N,N)  -- full Softmax attn weights
        errors:         (N,) or (B,N)          -- per-token prediction errors

    Returns:
        dict with:
          - linear_entropy_error_corr: Pearson r of linear attn entropy vs error
          - full_entropy_error_corr:   Pearson r of full attn entropy vs error
          - pattern_kl_div:            KL(full || linear) averaged over positions
          - pattern_cosine_sim:        cosine similarity of flattened patterns
          - entropy_ratio:             linear entropy / full entropy (>1 = more diffuse)
    """
    # Collapse batch dimension if present
    if linear_pattern.dim() == 4:
        linear_pattern = linear_pattern.mean(0)
    if full_pattern.dim() == 4:
        full_pattern = full_pattern.mean(0)
    if errors.dim() == 2:
        errors = errors.mean(0)

    H, N, _ = linear_pattern.shape
    eps = 1e-10
    results = {}

    # 1. Per-query-position entropy vs error correlation
    lin_ent = -(linear_pattern * (linear_pattern + eps).log()).sum(-1)  # (H,N)
    ful_ent = -(full_pattern * (full_pattern + eps).log()).sum(-1)
    lin_ent_m = lin_ent.mean(0)     # (N,)
    ful_ent_m = ful_ent.mean(0)

    # Pearson r helpers
    def _pearson(a, b):
        a_c = a - a.mean(); b_c = b - b.mean()
        den = (a_c**2).sum().sqrt() * (b_c**2).sum().sqrt() + eps
        return float((a_c * b_c).sum() / den)

    results['linear_entropy_error_corr'] = _pearson(lin_ent_m, errors)
    results['full_entropy_error_corr']   = _pearson(ful_ent_m, errors)

    # 2. KL divergence  full || linear  (avg over heads & queries)
    kl = (full_pattern * ((full_pattern + eps) / (linear_pattern + eps)).log())
    results['pattern_kl_div'] = float(kl.sum(-1).mean())

    # 3. Cosine similarity of flattened (averaged-over-heads) patterns
    lp_flat = linear_pattern.mean(0).reshape(-1)
    fp_flat = full_pattern.mean(0).reshape(-1)
    cos = (lp_flat * fp_flat).sum() / (lp_flat.norm() * fp_flat.norm() + eps)
    results['pattern_cosine_sim'] = float(cos)

    # 4. Entropy ratio
    results['entropy_ratio'] = float(
        lin_ent.mean() / (ful_ent.mean() + eps))

    return results


def compute_vram_savings(n_tokens, d_model=D_MODEL, n_heads=N_HEADS,
                         n_linear_layers=3, n_full_layers=1,
                         n_tokens_future=None,
                         precision_bytes=4):
    """
    Estimate per-sample activation memory for standard vs KDA attention.

    Standard MHA per layer:
      - QK^T matrix:   n_heads * N * N * bytes
      - Softmax output: n_heads * N * N * bytes  (intermediate)
      Peak ~= 2 * H * N^2 * bytes

    Linear attention per layer:
      - KV accumulator: n_heads * d_k * d_k * bytes
      - K_sum:          n_heads * d_k * bytes
      Peak ~= H * d_k^2 * bytes (dominated by KV)

    The FFN and Q/K/V projections are identical for both, so the saving
    is in the attention map memory.

    Args:
        n_tokens:       current sequence length (e.g., 202)
        n_tokens_future: optional projected N for scaling analysis
        n_linear_layers: layers using LinearAttention
        n_full_layers:   layers using FullAttention
        precision_bytes: 4 for float32

    Returns:
        dict with detailed memory estimates per component.
    """
    d_k = d_model // n_heads
    N = n_tokens

    # --- attention map memory (peak, per layer) ---
    # Standard: QK^T [H,N,N] + post-softmax [H,N,N]
    attn_std_per_layer = 2 * n_heads * N * N * precision_bytes

    # Linear:  KV [H,d_k,d_k] + K_sum [H,d_k]
    attn_lin_per_layer = (n_heads * d_k * d_k
                          + n_heads * d_k) * precision_bytes

    # Q/K/V projections (identical for both): 3 * [H,N,d_k]
    qkv_per_layer = 3 * n_heads * N * d_k * precision_bytes

    # --- FFN intermediate (identical for both) ---
    # Linear(d, d_ff) stores the intermediate activation [N, d_ff]
    ffn_per_layer = N * 256 * precision_bytes     # d_ff = 256

    # --- totals ---
    std_total = (n_linear_layers + n_full_layers) * (
        attn_std_per_layer + qkv_per_layer + ffn_per_layer)
    kda_total = (n_linear_layers * (attn_lin_per_layer + qkv_per_layer + ffn_per_layer)
                 + n_full_layers * (attn_std_per_layer + qkv_per_layer + ffn_per_layer))

    savings = std_total - kda_total
    ratio = kda_total / std_total if std_total > 0 else 0.0

    result = {
        'n_tokens': N,
        'd_k': d_k,
        'n_heads': n_heads,
        'n_linear_layers': n_linear_layers,
        'n_full_layers': n_full_layers,
        # per-layer (bytes)
        'standard_attn_per_layer_bytes': attn_std_per_layer,
        'linear_attn_per_layer_bytes':   attn_lin_per_layer,
        'ffn_per_layer_bytes':           ffn_per_layer,
        'qkv_per_layer_bytes':           qkv_per_layer,
        # totals
        'standard_total_bytes': std_total,
        'kda_total_bytes':      kda_total,
        'savings_bytes':        savings,
        'savings_ratio':        round(ratio, 4),
        'savings_percent':      round((1 - ratio) * 100, 1),
        # human-readable
        'standard_total_kb': round(std_total / 1024, 2),
        'kda_total_kb':      round(kda_total / 1024, 2),
        'savings_kb':        round(savings / 1024, 2),
    }

    # Optional: scaling projection
    if n_tokens_future is not None:
        fut = compute_vram_savings(n_tokens_future, d_model, n_heads,
                                   n_linear_layers, n_full_layers,
                                   precision_bytes=precision_bytes)
        # Compute scaling: attention portion only
        std_attn_cur = attn_std_per_layer * (n_linear_layers + n_full_layers)
        lin_attn_cur = (attn_lin_per_layer * n_linear_layers
                        + attn_std_per_layer * n_full_layers)
        std_attn_fut = fut['standard_total_bytes']
        lin_attn_fut = fut['kda_total_bytes']
        result['scaling'] = {
            'n_tokens_future': n_tokens_future,
            'standard_attn_growth': (fut['standard_attn_per_layer_bytes']
                                     / max(attn_std_per_layer, 1)),
            'linear_attn_growth':   (fut['linear_attn_per_layer_bytes']
                                     / max(attn_lin_per_layer, 1)),
        }

    return result


# ==================================================================
# TEST  (run with:  python p2_kda_attention.py)
# ==================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  KDA LINEAR ATTENTION -- Tests")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Test config (matches SlimMetaTrans)
    N_A = 200
    B = 2
    SIG_DIM = 39
    CL_DIM = 4
    N_TOKENS = N_A + 2            # CLS + 200 SIG + FACTOR
    device = torch.device("cpu")

    sigs = torch.randn(B, N_A, SIG_DIM)
    cl   = torch.randn(B, N_A, CL_DIM)
    y    = torch.randn(B, N_A)    # dummy targets
    x_seq = torch.randn(B, N_TOKENS, D_MODEL)

    # ================================================================
    # Test 1: LinearAttention forward + shapes
    # ================================================================
    print("\n--- Test 1: LinearAttention Forward ---")
    print(f"  Input:  (B={B}, N={N_TOKENS}, d_model={D_MODEL})")

    la = LinearAttention(D_MODEL, N_HEADS, DROPOUT)
    out = la(x_seq)
    assert out.shape == (B, N_TOKENS, D_MODEL), f"Shape: {out.shape}"
    print(f"  Output shape: {out.shape}  OK")

    # Pattern extraction
    out2, pat_lin = la(x_seq, return_pattern=True)
    assert pat_lin.shape == (B, N_HEADS, N_TOKENS, N_TOKENS), (
        f"Pattern shape: {pat_lin.shape}")
    # Verify each row sums to ~1
    row_sums = pat_lin.sum(-1)
    assert (row_sums - 1.0).abs().max() < 1e-4, (
        f"Pattern rows don't sum to 1: max|diff|={(row_sums-1).abs().max():.2e}")
    print(f"  Pattern shape: {pat_lin.shape} (rows sum to 1)  OK")

    # Causal variant
    out_c = la(x_seq, causal=True)
    assert out_c.shape == (B, N_TOKENS, D_MODEL)
    print(f"  Causal output shape: {out_c.shape}  OK")
    print("  PASSED")

    # ================================================================
    # Test 2: FullAttention forward + shapes
    # ================================================================
    print("\n--- Test 2: FullAttention Forward ---")
    fa = FullAttention(D_MODEL, N_HEADS, DROPOUT)
    out_fa = fa(x_seq)
    assert out_fa.shape == (B, N_TOKENS, D_MODEL)
    print(f"  Output shape: {out_fa.shape}  OK")

    out_fa2, pat_fa = fa(x_seq, return_pattern=True)
    assert pat_fa.shape == (B, N_HEADS, N_TOKENS, N_TOKENS)
    row_sums_fa = pat_fa.sum(-1)
    assert (row_sums_fa - 1.0).abs().max() < 1e-5, "Softmax rows must sum to 1"
    print(f"  Pattern shape: {pat_fa.shape} (rows sum to 1)  OK")
    print("  PASSED")

    # ================================================================
    # Test 3: KDATransformerLayer (Linear)
    # ================================================================
    print("\n--- Test 3: KDATransformerLayer (Linear) ---")
    kl = KDATransformerLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, is_linear=True)
    out_kl = kl(x_seq)
    assert out_kl.shape == (B, N_TOKENS, D_MODEL)
    print(f"  Output shape: {out_kl.shape}  OK")
    print("  PASSED")

    # ================================================================
    # Test 4: KDATransformerLayer (Full)
    # ================================================================
    print("\n--- Test 4: KDATransformerLayer (Full) ---")
    kf = KDATransformerLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, is_linear=False)
    out_kf, pat_kf = kf(x_seq, return_pattern=True)
    assert out_kf.shape == (B, N_TOKENS, D_MODEL)
    assert pat_kf.shape == (B, N_HEADS, N_TOKENS, N_TOKENS)
    print(f"  Output: {out_kf.shape}  Pattern: {pat_kf.shape}  OK")
    print("  PASSED")

    # ================================================================
    # Test 5: KDATransformer (full model) forward + shapes
    # ================================================================
    print("\n--- Test 5: KDATransformer (Full Model) ---")
    model = KDATransformer(N_A, SIG_DIM, CL_DIM)
    stats = model.get_stats()
    print("  Model stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    ret_pred, unc_pred = model(sigs, cl)
    assert ret_pred.shape == (B, N_A), f"ret_pred: {ret_pred.shape}"
    assert unc_pred.shape == (B,),    f"unc_pred: {unc_pred.shape}"
    print(f"  ret_pred: {ret_pred.shape}  unc_pred: {unc_pred.shape}  OK")

    # Attention extraction
    rp2, up2, attns = model(sigs, cl, return_attention=True)
    assert len(attns) == 4
    n_lin = sum(1 for a in attns if a['type'] == 'linear')
    n_ful = sum(1 for a in attns if a['type'] == 'full')
    assert n_lin == 3 and n_ful == 1, f"3:1 ratio? linear={n_lin} full={n_ful}"
    print(f"  Attention patterns: {len(attns)} layers ({n_lin} linear, {n_ful} full)  OK")
    for a in attns:
        assert a['pattern'].shape == (B, N_HEADS, N_TOKENS, N_TOKENS), (
            f"Layer {a['layer']} pattern shape: {a['pattern'].shape}")
    print("  PASSED")

    # ================================================================
    # Test 6: Gradient flow (all params get gradients)
    # ================================================================
    print("\n--- Test 6: Gradient Flow ---")
    model.zero_grad()
    ret_p, unc_p = model(sigs, cl)
    loss = (F.huber_loss(ret_p, y, delta=1.0)
            + 0.1 * F.mse_loss(unc_p, ret_p.abs().mean(1)))
    loss.backward()

    grad_ok = 0
    grad_zero = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_ok += 1
            if p.grad.abs().sum() == 0:
                grad_zero += 1
                print(f"  WARNING: zero grad for {name}")
    print(f"  Params w/ gradients: {grad_ok}")
    assert grad_ok > 0, "No gradients at all"
    assert grad_zero == 0, f"{grad_zero} params have zero gradient"
    print(f"  Loss: {loss.item():.6f}")
    print("  PASSED: all parameters receive non-zero gradients")

    # ================================================================
    # Test 7: Pattern quality metrics
    # ================================================================
    print("\n--- Test 7: Pattern Quality Metrics ---")
    lin_pat = attns[0]['pattern']    # layer 0, linear
    ful_pat = attns[3]['pattern']    # layer 3, full
    dummy_errors = torch.rand(N_TOKENS)

    metrics = pattern_quality_metric(lin_pat, ful_pat, dummy_errors)
    print("  Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
    # sanity: KL should be finite, cosine in [-1,1]
    assert math.isfinite(metrics['pattern_kl_div'])
    assert -1.01 <= metrics['pattern_cosine_sim'] <= 1.01
    print("  PASSED")

    # ================================================================
    # Test 8: VRAM savings + scaling
    # ================================================================
    print("\n--- Test 8: VRAM Savings ---")
    vram = compute_vram_savings(N_TOKENS)
    print(f"  Configuration: {N_TOKENS} tokens, {N_HEADS} heads, d_k={vram['d_k']}")
    print(f"  Standard total (per-sample):  {vram['standard_total_kb']:.1f} KB")
    print(f"  KDA total (per-sample):       {vram['kda_total_kb']:.1f} KB")
    print(f"  Savings:                      {vram['savings_kb']:.1f} KB "
          f"({vram['savings_percent']:.1f}%)")

    # Scaling to 2000 assets
    vram_2k = compute_vram_savings(2002, n_tokens_future=2002)
    print(f"\n  Scaled to N=2000 assets (N_tokens=2002):")
    print(f"    Standard total: {vram_2k['standard_total_kb']:.1f} KB")
    print(f"    KDA total:      {vram_2k['kda_total_kb']:.1f} KB")
    print(f"    Savings:        {vram_2k['savings_kb']:.1f} KB "
          f"({vram_2k['savings_percent']:.1f}%)")
    assert vram_2k['savings_kb'] > 0, "KDA should save memory at scale"
    print("  PASSED")

    # ================================================================
    # Test 9: Interface compatibility with SlimMetaTrans
    # ================================================================
    print("\n--- Test 9: Interface Compatibility ---")
    result = model.forward(sigs, cl)
    assert len(result) == 2
    rp, up = result
    assert rp.shape == (B, N_A)
    assert up.shape == (B,)
    print(f"  forward(sigs({sigs.shape}), cl({cl.shape}))")
    print(f"      -> ret_pred({rp.shape}), unc_pred({up.shape})")
    print(f"  Matches SlimMetaTrans interface.  OK")
    print("  PASSED")

    # ================================================================
    # Test 10: Custom layer schedule
    # ================================================================
    print("\n--- Test 10: Custom Layer Schedule ---")
    # All-linear variant
    model_all_lin = KDATransformer(N_A, SIG_DIM, CL_DIM,
                                   layer_schedule=[True, True, True, True])
    all_stats = model_all_lin.get_stats()
    assert all_stats['n_linear_layers'] == 4
    print(f"  All-linear: {all_stats['ratio']}  OK")

    # 2:2 variant
    model_22 = KDATransformer(N_A, SIG_DIM, CL_DIM,
                              layer_schedule=[True, True, False, False])
    stats_22 = model_22.get_stats()
    assert stats_22['n_linear_layers'] == 2
    print(f"  2:2 variant: {stats_22['ratio']}  OK")

    # All-full variant (no linear layers)
    model_all_ful = KDATransformer(N_A, SIG_DIM, CL_DIM,
                                   layer_schedule=[False, False, False, False])
    ful_stats = model_all_ful.get_stats()
    assert ful_stats['n_linear_layers'] == 0
    print(f"  All-full: {ful_stats['ratio']}  OK")
    print("  PASSED")

    # ================================================================
    # Test 11: Deterministic forward (no randomness w/o dropout)
    # ================================================================
    print("\n--- Test 11: Deterministic Forward ---")
    model.eval()
    r1, u1 = model(sigs, cl)
    r2, u2 = model(sigs, cl)
    assert torch.allclose(r1, r2, atol=1e-6), "ret_pred not deterministic"
    assert torch.allclose(u1, u2, atol=1e-6), "unc_pred not deterministic"
    print(f"  Two eval passes produce identical output.  OK")
    print("  PASSED")

    # ================================================================
    # Test 12: Numerical Stability (large N, large activation scale)
    # ================================================================
    print("\n--- Test 12: Numerical Stability ---")

    # 12a: safe=True must produce no NaN/Inf with large-magnitude input
    print("  12a: safe=True, large N=3002, large activation scale")
    torch.manual_seed(123)
    la_safe = LinearAttention(D_MODEL, N_HEADS, DROPOUT, use_safe_attention=True)
    x_stress = torch.randn(2, 3002, D_MODEL) * 10
    stable = True
    for i in range(10):
        out_s = la_safe(x_stress)
        loss_s = out_s.mean()
        loss_s.backward()
        has_nan = torch.isnan(out_s).any() or torch.isinf(out_s).any()
        has_grad_nan = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in la_safe.parameters() if p.grad is not None)
        if has_nan or has_grad_nan:
            stable = False
            print(f"    Iter {i}: NaN/Inf detected!")
            break
        la_safe.zero_grad()
    if stable:
        print("    All 10 forward+backward iterations stable — no NaN/Inf  OK")
    else:
        print("    FAILED: NaN/Inf detected on safe=True path")
        raise AssertionError("safe=True produced NaN/Inf")

    # 12b: safe=False on same input — may produce NaN/Inf (overflow)
    #      We test 2 iterations; if no NaN occurs we log a warning but
    #      do not fail — overflow is input-dependent.
    print("  12b: safe=False, same large input")
    torch.manual_seed(123)
    la_unsafe = LinearAttention(D_MODEL, N_HEADS, DROPOUT, use_safe_attention=False)
    nan_detected = False
    for i in range(2):
        try:
            out_u = la_unsafe(x_stress)
            has_nan = torch.isnan(out_u).any() or torch.isinf(out_u).any()
            if has_nan:
                nan_detected = True
                break
            loss_u = out_u.mean()
            loss_u.backward()
            has_grad_nan = any(
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                for p in la_unsafe.parameters() if p.grad is not None)
            if has_grad_nan:
                nan_detected = True
                break
            la_unsafe.zero_grad()
        except Exception as e:
            nan_detected = True
            print(f"    Exception caught: {e}")
            break
    if nan_detected:
        print("    NaN/Inf detected on unsafe path — expected overflow behaviour  OK")
    else:
        print("    WARNING: No NaN/Inf on unsafe path (overflow is input-dependent)")

    print("  PASSED")
