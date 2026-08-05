"""
Improved Transformer with real attention pattern capture.
Captures attention weights from all layers/heads for the Upper Transformer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttentionWithPatterns(nn.Module):
    """Multi-head attention that stores attention weights for later retrieval."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # Most recent attention weights: (batch, n_heads, seq, seq)
        self.register_buffer('_last_weights', torch.zeros(1, 1, 1, 1))

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        B, N, E = query.shape
        q = self.q_proj(query).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scale = math.sqrt(self.d_k)
        attn_weights = (q @ k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # STORE for later retrieval
        self._last_weights = attn_weights.detach()  # (B, n_heads, N, N)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, E)
        return self.out_proj(attn_output)

    def get_weights(self):
        return self._last_weights


class TransformerEncoderLayerWithPatterns(nn.TransformerEncoderLayer):
    """Transformer encoder layer that captures attention weights."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                        batch_first=True, norm_first=True)
        # Replace the standard MHA with our custom one
        self.self_attn = MultiheadAttentionWithPatterns(d_model, nhead, dropout)

    def get_attention_weights(self):
        return self.self_attn.get_weights()


class BaseTransformerV2(nn.Module):
    """Improved base transformer with proper attention pattern capture.
    Architecture: cross-asset attention — each asset is a token, time steps are features.
    """
    def __init__(self, d_model=128, n_layers=4, n_heads=8, d_ff=256, dropout=0.1, lookback=30):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Input: (batch, n_assets, lookback) → (batch, n_assets, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(lookback, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )
        # Learnable position encoding for assets
        self.pos_emb = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithPatterns(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, capture_patterns=False):
        """x: (batch, n_assets, lookback) → pred: (batch, n_assets)"""
        B, N, L = x.shape
        h = self.input_proj(x)  # (B, N, d_model)
        h = h + self.pos_emb[:, :N, :]
        h = self.dropout(h)

        patterns = {}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if capture_patterns:
                patterns[i] = layer.get_attention_weights()  # (B, n_heads, N, N)

        pred = self.pred_head(h).squeeze(-1)  # (B, N)
        return pred, patterns if capture_patterns else pred

    def collect_all_patterns(self, x):
        """Return predictions + attention patterns from all layers.
        Returns: pred (B,N), patterns (B, n_layers*n_heads, N, N)"""
        B, N, L = x.shape
        h = self.input_proj(x)
        h = h + self.pos_emb[:, :N, :]
        h = self.dropout(h)

        all_patterns = []
        for layer in self.layers:
            h = layer(h)
            w = layer.get_attention_weights()  # (B, n_heads, N, N)
            all_patterns.append(w)
        all_patterns = torch.cat(all_patterns, dim=1)  # (B, n_layers*n_heads, N, N)
        pred = self.pred_head(h).squeeze(-1)
        return pred, all_patterns


class UpperTransformerV2(nn.Module):
    """Transformer that reads REAL attention patterns as tokens.
    Input: (batch, n_total_heads, N_assets, N_assets) attention matrices
    Each head's N×N matrix is flattened to a token, then processed by transformer.
    """
    def __init__(self, n_total_heads=32, max_assets=300, d_model=256, n_layers=4,
                 n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.max_assets = max_assets
        # Flatten N×N attention matrix to d_model
        self.token_proj = nn.Linear(max_assets * max_assets, d_model)
        # Layer-wise positional encoding
        self.head_pos_emb = nn.Parameter(torch.randn(1, n_total_heads, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Error prediction head
        self.error_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1), nn.Softplus()
        )

    def forward(self, patterns):
        """patterns: (batch, n_heads, N, N) → error_pred: (batch,)"""
        B, H, N, _ = patterns.shape
        # Pad or truncate attention matrices to max_assets
        if N < self.max_assets:
            pad = torch.zeros(B, H, self.max_assets, self.max_assets, device=patterns.device)
            pad[:, :, :N, :N] = patterns
            patterns = pad
        elif N > self.max_assets:
            patterns = patterns[:, :, :self.max_assets, :self.max_assets]
        # Flatten each head's matrix
        flat = patterns.reshape(B, H, -1)  # (B, H, N²)
        tokens = self.token_proj(flat) + self.head_pos_emb[:, :H, :]  # (B, H, d_model)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+H, d_model)

        # Transformer
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]  # (B, d_model)
        return self.error_head(cls_out).squeeze(-1)  # (B,)

    def train_step(self, patterns, errors, optimizer):
        self.train()
        pred = self.forward(patterns)
        loss = F.mse_loss(pred, errors)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0)
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def predict(self, patterns):
        self.eval()
        if isinstance(patterns, torch.Tensor):
            patterns = patterns.to(next(self.parameters()).device)
        else:
            patterns = torch.FloatTensor(patterns).to(next(self.parameters()).device)
        return self.forward(patterns).cpu().numpy()
