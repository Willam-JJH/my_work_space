"""
Final Experiment: LLM + Upper Transformer for Financial Prediction
==================================================================
Exp A: Upper Transformer — real attention patterns → error prediction
Exp B: GPT-2 + multimodal embeddings → return prediction + uncertainty
Exp C: Combined — Upper Transformer output → GPT-2 fusion

python3 final_experiment.py --exp A --market us
"""
import os, sys, math, json, warnings, argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
@dataclass
class C:
    data_dir: str = "/home/user2/meta_attn/data"
    lookback: int = 30
    batch_size: int = 32
    n_sample: int = 200

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 256
    dropout: float = 0.1
    base_epochs: int = 60
    base_lr: float = 1e-3

    upper_d_model: int = 256
    upper_n_layers: int = 4
    upper_n_heads: int = 8
    upper_d_ff: int = 512
    upper_epochs: int = 40
    upper_lr: float = 1e-3

    llm_model: str = "gpt2"
    llm_epochs: int = 40
    llm_lr: float = 5e-4

    combined_epochs: int = 30
    combined_lr: float = 5e-4

    seed: int = 42
    save_dir: str = "/home/user2/meta_attn/experiments_output"

c = C()
torch.manual_seed(c.seed); np.random.seed(c.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(c.seed)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEV} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ============================================================
# DATA
# ============================================================
def load_market(market="us"):
    if market == "us":
        df = pd.read_parquet(f"{c.data_dir}/us_market/log_returns.parquet")
        ret = df.fillna(method='ffill').fillna(0).values.astype(np.float32)
    elif market == "cn":
        df = pd.read_parquet(f"{c.data_dir}/cn_market/daily_returns.parquet")
        pivot = df.pivot(index='trddt', columns='stkcd', values='dretwd')
        ret = pivot.fillna(method='ffill').fillna(0).values.astype(np.float32)
    else:
        raise ValueError(f"Unknown market: {market}")

    mask = ret.std(axis=0) > 1e-8
    ret = ret[:, mask]
    print(f"  {market}: {ret.shape[0]}d × {ret.shape[1]} assets")

    L = c.lookback
    n_samp, n_a = ret.shape[0] - L, ret.shape[1]
    X = np.zeros((n_samp, n_a, L), dtype=np.float32)
    y = np.zeros((n_samp, n_a), dtype=np.float32)
    for i in range(n_samp):
        X[i] = ret[i:i+L].T; y[i] = ret[i+L]
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
    clip = np.percentile(np.abs(y), 99); y = np.clip(y, -clip, clip)
    split = int(n_samp * 0.7)
    return (X[:split], y[:split]), (X[split:], y[split:]), n_a

class FinDS(Dataset):
    def __init__(self, X, y, n_sample=None):
        self.X, self.y = X, y
        self.n_sample = n_sample or X.shape[1]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        if self.X.shape[1] > self.n_sample:
            idx = np.random.choice(self.X.shape[1], self.n_sample, replace=False)
        else:
            idx = np.arange(self.X.shape[1])
        return torch.FloatTensor(self.X[i][idx]), torch.FloatTensor(self.y[i][idx])

# ============================================================
# MULTI-HEAD ATTENTION WITH PATTERN CAPTURE
# ============================================================
class MHA(nn.Module):
    """Multi-head attention that stores attention weights."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads; self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self._weights = None  # (B, n_heads, N, N)

    def forward(self, x, attn_mask=None):
        B, N, E = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            w = w.masked_fill(attn_mask == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.drop(w)
        self._weights = w.detach()
        out = (w @ v).transpose(1, 2).contiguous().view(B, N, E)
        return self.out_proj(out)

class EncoderLayer(nn.Module):
    """Transformer encoder layer with pattern capture."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ============================================================
# BASE TRANSFORMER — Bottom Model
# ============================================================
class BaseTransformer(nn.Module):
    """Predicts returns. Captures attention patterns from all layers."""
    def __init__(self, d_model=128, n_layers=4, n_heads=8, d_ff=256, dropout=0.1, lookback=30):
        super().__init__()
        self.d_model = d_model; self.n_layers = n_layers; self.n_heads = n_heads
        self.proj = nn.Sequential(nn.Linear(lookback, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
        self.pos = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

    def forward(self, x, get_patterns=False):
        B, N, L = x.shape
        h = self.drop(self.proj(x) + self.pos[:, :N, :])
        patterns = {}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if get_patterns:
                patterns[i] = layer.attn._weights  # (B, n_heads, N, N)
        pred = self.head(h).squeeze(-1)
        return (pred, patterns) if get_patterns else pred

    def train_step(self, x, y, opt):
        self.train(); pred = self.forward(x)
        loss = F.huber_loss(pred, y, delta=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0)
        opt.step(); return loss.item()

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval(); ps, ys = [], []
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            ps.append(self.forward(x).cpu().numpy())
            ys.append(y.cpu().numpy())
        return np.concatenate(ps), np.concatenate(ys)

# ============================================================
# EXPERIMENT A: UPPER TRANSFORMER
# ============================================================
class UpperTransformer(nn.Module):
    """Reads attention patterns → predicts prediction error."""
    def __init__(self, n_total_heads=32, d_model=256, n_layers=4, n_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        # n_total_heads = n_base_layers * n_base_heads
        self.proj = nn.Linear(c.n_sample * c.n_sample, d_model)  # N² → d_model (will resize)
        self.pos_emb = nn.Parameter(torch.randn(1, n_total_heads, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.error_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model//2, 1), nn.Softplus()
        )

    def forward(self, patterns):
        """patterns: (B, total_heads, N, N) → error: (B,)"""
        B, H, N, _ = patterns.shape
        flat = patterns.reshape(B, H, N * N)
        tokens = self.proj(flat) + self.pos_emb[:, :H, :]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for layer in self.enc_layers:
            x = layer(x)
        return self.error_head(x[:, 0]).squeeze(-1)

    def train_step(self, patterns, errors, opt):
        self.train(); pred = self.forward(patterns)
        loss = F.mse_loss(pred, errors)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0)
        opt.step(); return loss.item()

    @torch.no_grad()
    def predict(self, patterns):
        self.eval()
        if not isinstance(patterns, torch.Tensor):
            patterns = torch.FloatTensor(patterns).to(DEV)
        return self.forward(patterns).cpu().numpy()

# ============================================================
# EXPERIMENT B: GPT-2 PREDICTOR
# ============================================================
class LLMPredictor(nn.Module):
    """GPT-2 predicts returns from price+technical embeddings. LoRA-like: freeze most, train last 2 layers."""
    def __init__(self, d_embed=128, llm_name="gpt2"):
        super().__init__()
        self.price_enc = nn.Sequential(nn.Linear(c.lookback, d_embed*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_embed*2, d_embed))
        self.tech_enc = nn.Sequential(nn.Linear(6, d_embed), nn.GELU(), nn.Linear(d_embed, d_embed))
        self.proj = nn.Linear(d_embed * 2, 768)  # → gpt2 hidden size

        from transformers import GPT2Model
        self.llm = GPT2Model.from_pretrained(llm_name)
        for p in self.llm.parameters(): p.requires_grad = False
        for layer in self.llm.h[-2:]:
            for p in layer.parameters(): p.requires_grad = True

        self.mu_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 1))
        self.sigma_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 1), nn.Softplus())

    def _tech_features(self, x):
        B, N, L = x.shape
        v5 = x[:,:,-5:].std(-1,keepdim=True); v10 = x[:,:,-10:].std(-1,keepdim=True)
        v20 = x[:,:,-min(L,20):].std(-1,keepdim=True); m5 = x[:,:,-5:].mean(-1,keepdim=True)
        m10 = x[:,:,-10:].mean(-1,keepdim=True); rsi = (x[:,:,-5:]>0).float().mean(-1,keepdim=True)
        return torch.cat([v5,v10,v20,m5,m10,rsi], dim=-1)

    def forward(self, x):
        B, N, L = x.shape
        p = self.price_enc(x)
        t = self.tech_enc(self._tech_features(x))
        h = self.proj(torch.cat([p, t], dim=-1))
        out = self.llm(inputs_embeds=h).last_hidden_state
        return self.mu_head(out).squeeze(-1), self.sigma_head(out).squeeze(-1)

    def train_step(self, x, y, opt):
        self.train(); mu, sigma = self.forward(x)
        mse = F.mse_loss(mu, y)
        nll = ((mu-y)**2/(2*sigma**2+1e-8) + torch.log(sigma+1e-8)).mean()
        loss = mse + 0.1 * nll
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0)
        opt.step(); return loss.item()

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval(); mus, sigs, ys = [], [], []
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            mu, sigma = self.forward(x)
            mus.append(mu.cpu().numpy()); sigs.append(sigma.cpu().numpy()); ys.append(y.cpu().numpy())
        return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)

# ============================================================
# METRICS
# ============================================================
def sr(pred: np.ndarray, true: np.ndarray) -> float:
    return float(spearmanr(pred, true)[0])

def abs_pred_baseline(preds: np.ndarray) -> np.ndarray:
    return np.abs(preds).mean(axis=1)

# ============================================================
# TRAINERS
# ============================================================
def train_base(model, tr_loader, te_loader, epochs, lr, label=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"\n  Training Base Transformer [{label}]")
    for ep in range(epochs):
        model.train(); tl = 0
        for x, y in tr_loader:
            tl += model.train_step(x.to(DEV), y.to(DEV), opt)
        sch.step()
        if (ep+1) % 15 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/len(tr_loader):.6f}")
    preds, targs = model.evaluate(te_loader)
    mse = float(np.mean((preds - targs)**2))
    print(f"    Test MSE: {mse:.6f} | Baseline |Pred| Spearman: {sr(abs_pred_baseline(preds), np.abs(preds-targs).mean(1)):.4f}")
    return model

def run_exp_a(train_data, test_data, n_assets):
    print(f"\n{'#'*50}\n# EXP A: Upper Transformer\n{'#'*50}")
    (X_tr, y_tr), (X_te, y_te) = train_data, test_data
    ns = min(n_assets, c.n_sample)
    tr_ds = FinDS(X_tr, y_tr, ns); te_ds = FinDS(X_te, y_te, ns)
    tr_ld = DataLoader(tr_ds, c.batch_size, shuffle=True, drop_last=True)
    te_ld = DataLoader(te_ds, c.batch_size, shuffle=False)

    # Train base
    base = BaseTransformer(c.d_model, c.n_layers, c.n_heads, c.d_ff, c.dropout, c.lookback).to(DEV)
    base = train_base(base, tr_ld, te_ld, c.base_epochs, c.base_lr, "Base-A")
    base.eval()

    # Extract patterns & errors from training data
    print("  Extracting patterns from training data...")
    all_patterns, all_errors = [], []
    for x, y in tr_ld:
        x, y = x.to(DEV), y.to(DEV)
        pred, pats = base(x, get_patterns=True)
        # Stack patterns: dict{layer: (B, h, N, N)} → (B, total_h, N, N)
        stacked = torch.cat([pats[i] for i in range(c.n_layers)], dim=1)
        all_patterns.append(stacked.cpu().numpy())
        all_errors.append((pred - y).abs().mean(1).detach().cpu().numpy())
    p_train = np.concatenate(all_patterns)
    e_train = np.concatenate(all_errors)

    # Train upper transformer
    total_heads = c.n_layers * c.n_heads
    upper = UpperTransformer(total_heads, c.upper_d_model, c.upper_n_layers, c.upper_n_heads, c.upper_d_ff, c.dropout).to(DEV)
    opt = torch.optim.AdamW(upper.parameters(), lr=c.upper_lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.upper_epochs)

    print(f"  Training Upper Transformer ({total_heads} head tokens)...")
    for ep in range(c.upper_epochs):
        upper.train(); tl = 0
        for i in range(0, len(p_train), c.batch_size):
            pb = torch.FloatTensor(p_train[i:i+c.batch_size]).to(DEV)
            eb = torch.FloatTensor(e_train[i:i+c.batch_size]).to(DEV)
            tl += upper.train_step(pb, eb, opt)
        sch.step()
        if (ep+1) % 10 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/(len(p_train)//c.batch_size+1):.6f}")

    # Evaluate
    print("  Evaluating...")
    all_pe, all_te = [], []
    for x, y in te_ld:
        x, y = x.to(DEV), y.to(DEV)
        pred, pats = base(x, get_patterns=True)
        stacked = torch.cat([pats[i] for i in range(c.n_layers)], dim=1)
        pe = upper.predict(stacked)
        te = (pred - y).abs().mean(1).detach().cpu().numpy()
        all_pe.append(pe); all_te.append(te)
    pe_all = np.concatenate(all_pe); te_all = np.concatenate(all_te)
    _, targs = base.evaluate(te_ld)
    r_up = sr(pe_all, te_all)
    r_bl = sr(abs_pred_baseline(base.evaluate(te_ld)[0]), te_all[:len(pe_all)])
    print(f"\n  EXP A Results: Upper r={r_up:.4f} | |Pred| Baseline r={r_bl:.4f} | Δ={r_up-r_bl:+.4f}")
    return {'exp':'A', 'upper_spearman': r_up, 'baseline_spearman': r_bl, 'delta': r_up-r_bl}

def run_exp_b(train_data, test_data, n_assets):
    print(f"\n{'#'*50}\n# EXP B: LLM + Multimodal (GPT-2)\n{'#'*50}")
    (X_tr, y_tr), (X_te, y_te) = train_data, test_data
    ns = min(n_assets, c.n_sample)
    bs = min(c.batch_size, 8)  # Smaller batch for GPT-2
    tr_ds = FinDS(X_tr, y_tr, ns); te_ds = FinDS(X_te, y_te, ns)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True, drop_last=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False)

    print("  Loading GPT-2...")
    model = LLMPredictor(d_embed=c.d_model, llm_name=c.llm_model).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,} | Trainable: {n_train:,} ({100*n_train/n_params:.1f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c.llm_lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.llm_epochs)

    print(f"  Training...")
    for ep in range(c.llm_epochs):
        model.train(); tl = 0
        for x, y in tr_ld:
            tl += model.train_step(x.to(DEV), y.to(DEV), opt)
        sch.step()
        if (ep+1) % 10 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

    mu, sigma, targs = model.evaluate(te_ld)
    mse = float(np.mean((mu-targs)**2))
    err_pred = sigma.mean(axis=1)  # mean uncertainty as error proxy
    err_true = np.abs(mu-targs).mean(axis=1)
    r_llm = sr(err_pred, err_true)
    r_bl = sr(abs_pred_baseline(mu), err_true)
    print(f"\n  EXP B Results: MSE={mse:.6f} | LLM r={r_llm:.4f} | Baseline r={r_bl:.4f} | Δ={r_llm-r_bl:+.4f}")
    return {'exp':'B', 'test_mse':mse, 'llm_spearman':r_llm, 'baseline_spearman':r_bl, 'delta':r_llm-r_bl}

# ============================================================
# EXPERIMENT C: COMBINED (Upper Transformer → GPT-2 Fusion)
# ============================================================
class CombinedModel(nn.Module):
    """Upper Transformer error signal gated into GPT-2 predictions."""
    def __init__(self, base_attn_d_model=128, llm_name="gpt2", n_total_heads=32):
        super().__init__()
        # GPT-2 predictor (reuse EXP B architecture)
        self.price_enc = nn.Sequential(nn.Linear(c.lookback, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 128))
        self.tech_enc = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 128))
        self.proj = nn.Linear(128 * 2 + 1, 768)  # +1 for upper transformer error signal

        from transformers import GPT2Model
        self.llm = GPT2Model.from_pretrained(llm_name)
        for p in self.llm.parameters(): p.requires_grad = False
        for layer in self.llm.h[-3:]:  # Unfreeze last 3 layers for fusion
            for p in layer.parameters(): p.requires_grad = True

        self.mu_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 1))
        self.sigma_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 1), nn.Softplus())

    def _tech_features(self, x):
        B, N, L = x.shape
        v5 = x[:,:,-5:].std(-1,keepdim=True); v10 = x[:,:,-10:].std(-1,keepdim=True)
        v20 = x[:,:,-min(L,20):].std(-1,keepdim=True); m5 = x[:,:,-5:].mean(-1,keepdim=True)
        m10 = x[:,:,-10:].mean(-1,keepdim=True); rsi = (x[:,:,-5:]>0).float().mean(-1,keepdim=True)
        return torch.cat([v5,v10,v20,m5,m10,rsi], dim=-1)

    def forward(self, returns, error_signal):
        """returns: (B,N,L), error_signal: (B,) from Upper Transformer"""
        B, N, L = returns.shape
        p = self.price_enc(returns)
        t = self.tech_enc(self._tech_features(returns))
        # Broadcast error signal to all assets
        err = error_signal.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1)
        h = self.proj(torch.cat([p, t, err], dim=-1))
        out = self.llm(inputs_embeds=h).last_hidden_state
        return self.mu_head(out).squeeze(-1), self.sigma_head(out).squeeze(-1)

    def train_step(self, returns, targets, error_signal, opt):
        self.train(); mu, sigma = self.forward(returns, error_signal)
        mse = F.mse_loss(mu, targets)
        nll = ((mu-targets)**2/(2*sigma**2+1e-8) + torch.log(sigma+1e-8)).mean()
        loss = mse + 0.1 * nll
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0)
        opt.step(); return loss.item()

    @torch.no_grad()
    def evaluate(self, loader, error_predictor, base_model):
        self.eval(); mus, sigs, ys = [], [], []
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            # Get error signal from Upper Transformer via base model patterns
            pred, pats = base_model(x, get_patterns=True)
            stacked = torch.cat([pats[i] for i in range(c.n_layers)], dim=1)
            err_signal = error_predictor(stacked)  # (B,)
            mu, sigma = self.forward(x, err_signal)
            mus.append(mu.cpu().numpy()); sigs.append(sigma.cpu().numpy()); ys.append(y.cpu().numpy())
        return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)


def run_exp_c(train_data, test_data, n_assets, base_model=None, upper_model=None):
    """Combined: Upper Transformer error signal → GPT-2 fusion."""
    print(f"\n{'#'*50}\n# EXP C: Combined (Upper Transformer → GPT-2)\n{'#'*50}")
    (X_tr, y_tr), (X_te, y_te) = train_data, test_data
    ns = min(n_assets, c.n_sample)
    bs = min(c.batch_size, 8)
    tr_ds = FinDS(X_tr, y_tr, ns); te_ds = FinDS(X_te, y_te, ns)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True, drop_last=True)
    te_ld = DataLoader(te_ds, bs, shuffle=False)

    # Train or reuse base model
    if base_model is None:
        print("  Training base transformer...")
        base = BaseTransformer(c.d_model, c.n_layers, c.n_heads, c.d_ff, c.dropout, c.lookback).to(DEV)
        base = train_base(base, tr_ld, te_ld, c.base_epochs, c.base_lr, "Base-C")
    else:
        base = base_model
    base.eval()

    # Extract patterns
    print("  Extracting patterns...")
    all_patterns, all_errors = [], []
    for x, y in tr_ld:
        x, y = x.to(DEV), y.to(DEV)
        pred, pats = base(x, get_patterns=True)
        stacked = torch.cat([pats[i] for i in range(c.n_layers)], dim=1)
        all_patterns.append(stacked.cpu().numpy())
        all_errors.append((pred - y).abs().mean(1).detach().cpu().numpy())
    p_train, e_train = np.concatenate(all_patterns), np.concatenate(all_errors)

    # Train or reuse upper
    total_heads = c.n_layers * c.n_heads
    if upper_model is None:
        upper = UpperTransformer(total_heads, c.upper_d_model, c.upper_n_layers, c.upper_n_heads, c.upper_d_ff, c.dropout).to(DEV)
        opt_u = torch.optim.AdamW(upper.parameters(), lr=c.upper_lr)
        sch_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, c.upper_epochs)
        print(f"  Training Upper Transformer...")
        for ep in range(c.upper_epochs):
            upper.train(); tl = 0
            for i in range(0, len(p_train), c.batch_size):
                pb = torch.FloatTensor(p_train[i:i+c.batch_size]).to(DEV)
                eb = torch.FloatTensor(e_train[i:i+c.batch_size]).to(DEV)
                tl += upper.train_step(pb, eb, opt_u)
            sch_u.step()
            if (ep+1) % 10 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/(len(p_train)//c.batch_size+1):.6f}")
    else:
        upper = upper_model
    upper.eval()

    # Train combined model
    print("  Loading GPT-2 for combined model...")
    combined = CombinedModel().to(DEV)
    n_t = sum(p.numel() for p in combined.parameters() if p.requires_grad)
    print(f"  Combined trainable: {n_t:,}")

    opt = torch.optim.AdamW([p for p in combined.parameters() if p.requires_grad], lr=c.combined_lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.combined_epochs)

    print(f"  Training Combined ({c.combined_epochs} epochs)...")
    for ep in range(c.combined_epochs):
        combined.train(); tl = 0
        for x, y in tr_ld:
            x, y = x.to(DEV), y.to(DEV)
            with torch.no_grad():
                pred, pats = base(x, get_patterns=True)
                stacked = torch.cat([pats[i] for i in range(c.n_layers)], dim=1)
                err_signal = upper(stacked)  # (B,)
            tl += combined.train_step(x, y, err_signal.detach(), opt)
        sch.step()
        if (ep+1) % 10 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

    # Evaluate
    mu, sigma, targs = combined.evaluate(te_ld, upper, base)
    mse = float(np.mean((mu-targs)**2))
    err_pred = sigma.mean(axis=1)
    err_true = np.abs(mu-targs).mean(axis=1)
    r_c = sr(err_pred, err_true)
    r_bl = sr(abs_pred_baseline(mu), err_true)
    print(f"\n  EXP C Results: MSE={mse:.6f} | Combined r={r_c:.4f} | Baseline r={r_bl:.4f} | Δ={r_c-r_bl:+.4f}")
    return {'exp':'C', 'test_mse':mse, 'combined_spearman':r_c, 'baseline_spearman':r_bl, 'delta':r_c-r_bl}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='all', choices=['A','B','C','all'])
    parser.add_argument('--market', default='us')
    args = parser.parse_args()

    train, test, n_assets = load_market(args.market)
    print(f"  Train: {train[0].shape} | Test: {test[0].shape}\n")

    # Train shared base model once for all experiments
    ns = min(n_assets, c.n_sample)
    tr_ds = FinDS(train[0], train[1], ns)
    te_ds = FinDS(test[0], test[1], ns)
    tr_ld = DataLoader(tr_ds, c.batch_size, shuffle=True, drop_last=True)
    te_ld = DataLoader(te_ds, c.batch_size, shuffle=False)

    shared_base = BaseTransformer(c.d_model, c.n_layers, c.n_heads, c.d_ff, c.dropout, c.lookback).to(DEV)
    shared_base = train_base(shared_base, tr_ld, te_ld, c.base_epochs, c.base_lr, "Shared")
    shared_base.eval()

    # Extract patterns for shared upper
    print("\n  Extracting shared patterns...")
    all_patterns, all_errors = [], []
    for x, y in tr_ld:
        x, y = x.to(DEV), y.to(DEV)
        pred, pats = shared_base(x, get_patterns=True)
        stacked = torch.cat([pats[i] for i in range(c.n_layers)], dim=1)
        all_patterns.append(stacked.cpu().numpy())
        all_errors.append((pred - y).abs().mean(1).detach().cpu().numpy())
    shared_p = np.concatenate(all_patterns); shared_e = np.concatenate(all_errors)

    results = {}

    if args.exp in ('A', 'all'):
        results['A'] = run_exp_a(train, test, n_assets)
    if args.exp in ('B', 'all'):
        results['B'] = run_exp_b(train, test, n_assets)
    if args.exp in ('C', 'all'):
        # Train shared upper transformer for C
        total_heads = c.n_layers * c.n_heads
        shared_upper = UpperTransformer(total_heads, c.upper_d_model, c.upper_n_layers, c.upper_n_heads, c.upper_d_ff, c.dropout).to(DEV)
        opt_u = torch.optim.AdamW(shared_upper.parameters(), lr=c.upper_lr)
        sch_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, c.upper_epochs)
        print("\n  Training shared Upper Transformer for EXP C...")
        for ep in range(c.upper_epochs):
            shared_upper.train(); tl = 0
            for i in range(0, len(shared_p), c.batch_size):
                pb = torch.FloatTensor(shared_p[i:i+c.batch_size]).to(DEV)
                eb = torch.FloatTensor(shared_e[i:i+c.batch_size]).to(DEV)
                tl += shared_upper.train_step(pb, eb, opt_u)
            sch_u.step()
            if (ep+1) % 10 == 0: print(f"    Epoch {ep+1:3d} | Loss: {tl/(len(shared_p)//c.batch_size+1):.6f}")
        shared_upper.eval()
        results['C'] = run_exp_c(train, test, n_assets, shared_base, shared_upper)

    os.makedirs(c.save_dir, exist_ok=True)
    with open(f"{c.save_dir}/results_{args.market}.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n{'='*50}\n  RESULTS SAVED → {c.save_dir}/results_{args.market}.json\n{'='*50}")
    for k,v in results.items():
        print(f"  {k}: {json.dumps({kk:round(vv,4) for kk,vv in v.items() if isinstance(vv,float)})}")

if __name__ == "__main__":
    main()
