"""
Full Pipeline: Multimodal Pretraining + Upper Transformer
==========================================================
Phase 1: Contrastive pretrain price/sig/tech encoders
Phase 2: Base Transformer → attention patterns
Phase 3: Upper Transformer fuses patterns + pretrained multimodal embeddings → error
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn
import torch.nn.functional as F; from torch.utils.data import Dataset, DataLoader
import signatory; import yfinance as yf; from scipy.stats import spearmanr
import math, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'}")

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","JNJ",
           "WMT","PG","MA","UNH","HD","BAC","NFLX","ADBE","CRM","XOM"]
LOOKBACK, SIG_DEPTH = 30, 3
D_EMBED, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT = 128, 64, 4, 4, 128, 0.1
BATCH, PRETRAIN_EP, BASE_EP, UPPER_EP = 64, 60, 50, 80
TEMP = 0.07
torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA
# ============================================================
print("[1/5] Loading data...")
data = yf.download(TICKERS, start="2015-01-01", end="2024-12-31", auto_adjust=True, progress=False)
close = data["Close"].dropna(axis=1, thresh=int(len(data)*0.8)).dropna(axis=0)
returns = np.log(close / close.shift(1)).dropna().values.astype(np.float32)
tickers = list(close.columns); n_assets = len(tickers)
print(f"  {n_assets} stocks x {returns.shape[0]} days")

n_samp = returns.shape[0] - LOOKBACK
X_raw = np.zeros((n_samp, n_assets, LOOKBACK), dtype=np.float32)
y_raw = np.zeros((n_samp, n_assets), dtype=np.float32)
for i in range(n_samp):
    X_raw[i] = returns[i:i+LOOKBACK].T; y_raw[i] = returns[i+LOOKBACK]
mu = X_raw.mean(axis=-1, keepdims=True); st = X_raw.std(axis=-1, keepdims=True) + 1e-8
X_raw = (X_raw - mu) / st
clip = np.percentile(np.abs(y_raw), 99); y_raw = np.clip(y_raw, -clip, clip)
split = int(n_samp * 0.7)

# Signatures + PCA
print("  Computing signatures...")
sig_dim_full = signatory.signature_channels(n_assets, SIG_DEPTH)
X_sig = torch.FloatTensor(X_raw).transpose(1,2).to(CPU)
signatures_full = np.zeros((n_samp, sig_dim_full), dtype=np.float32)
for i in range(0, n_samp, 128):
    signatures_full[i:i+128] = signatory.signature(X_sig[i:i+128], SIG_DEPTH, basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
signatures_full = StandardScaler().fit_transform(signatures_full)
sig_dim = 128
signatures = PCA(n_components=sig_dim).fit_transform(signatures_full)
print(f"  PCA: {sig_dim_full:,} → {sig_dim}")

# Technical features
def tech_feat(r):
    v5=r[:,:,-5:].std(-1,keepdims=True); v10=r[:,:,-10:].std(-1,keepdims=True)
    v20=r[:,:,-20:].std(-1,keepdims=True); m5=r[:,:,-5:].mean(-1,keepdims=True)
    m10=r[:,:,-10:].mean(-1,keepdims=True); rsi=(r[:,:,-5:]>0).mean(-1,keepdims=True)
    return np.concatenate([v5,v10,v20,m5,m10,rsi],-1)
X_tech = tech_feat(X_raw)

X_tr, X_te = X_raw[:split], X_raw[split:]
y_tr, y_te = y_raw[:split], y_raw[split:]
sig_tr, sig_te = signatures[:split], signatures[split:]
tech_tr, tech_te = X_tech[:split], X_tech[split:]
print(f"  Train: {split} | Test: {n_samp - split} | Sig dim: {sig_dim:,}")

# ============================================================
# MODALITY ENCODERS
# ============================================================
class PriceEnc(nn.Module):
    """EXACT copy from working multimodal_pretrain.py"""
    def __init__(self, n_a, d=D_EMBED):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(LOOKBACK, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.asset_pos = nn.Parameter(torch.randn(1, n_a, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
    def forward(self, x):
        B, N, L = x.shape
        h = self.proj(x) + self.asset_pos[:, :N, :]
        h, _ = self.attn(h, h, h)
        return self.out(h.mean(dim=1))

class SigEnc(nn.Module):
    """EXACT copy from working multimodal_pretrain.py"""
    def __init__(self, sd, d=D_EMBED):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd, d*4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*4, d*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d*2, d), nn.LayerNorm(d))
    def forward(self, x): return self.net(x)

class TechEnc(nn.Module):
    """EXACT copy from working multimodal_pretrain.py"""
    def __init__(self, n_a, d=D_EMBED):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.asset_pos = nn.Parameter(torch.randn(1, n_a, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
    def forward(self, x):
        B, N, _ = x.shape
        h = self.proj(x) + self.asset_pos[:, :N, :]
        h, _ = self.attn(h, h, h)
        return self.out(h.mean(dim=1))

# ============================================================
# PHASE 1: PRETRAIN
# ============================================================
print("[2/5] Contrastive pretraining...")
p_enc, s_enc, t_enc = PriceEnc(n_assets).to(GPU), SigEnc(sig_dim).to(GPU), TechEnc(n_assets).to(GPU)
opt_pt = torch.optim.AdamW(list(p_enc.parameters())+list(s_enc.parameters())+list(t_enc.parameters()), lr=5e-4, weight_decay=1e-4)
pt_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(sig_tr), torch.FloatTensor(tech_tr))
pt_ld = DataLoader(pt_ds, BATCH, shuffle=True, drop_last=True)

for ep in range(PRETRAIN_EP):
    tl, nb = 0, 0
    for xb, sb, tb in pt_ld:
        xb, sb, tb = xb.to(GPU), sb.to(GPU), tb.to(GPU)
        zp, zs, zt = F.normalize(p_enc(xb),-1), F.normalize(s_enc(sb),-1), F.normalize(t_enc(tb),-1)
        B = xb.shape[0]; loss = 0
        for za, zb in [(zp,zs),(zp,zt),(zs,zt)]:
            sim = (za@zb.T)/TEMP; labels = torch.arange(B,device=GPU)
            loss += (F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))/2
        loss /= 3
        opt_pt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p_enc.parameters())+list(s_enc.parameters())+list(t_enc.parameters()), 1.0)
        opt_pt.step()
        tl += loss.item(); nb += 1
    if (ep+1)%20==0: print(f"  Pretrain {ep+1:3d} | Loss: {tl/nb:.4f}")

p_enc.eval(); s_enc.eval(); t_enc.eval()

# Compute embeddings
with torch.no_grad():
    zp_tr = p_enc(torch.FloatTensor(X_tr).to(GPU)).cpu()
    zs_tr = s_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu()
    zt_tr = t_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu()
    zp_te = p_enc(torch.FloatTensor(X_te).to(GPU)).cpu()
    zs_te = s_enc(torch.FloatTensor(sig_te).to(GPU)).cpu()
    zt_te = t_enc(torch.FloatTensor(tech_te).to(GPU)).cpu()

# ============================================================
# BASE TRANSFORMER
# ============================================================
class MHA(nn.Module):
    def __init__(self,dm,nh,dp=0.1):
        super().__init__(); self.nh=nh; self.dk=dm//nh
        self.q=nn.Linear(dm,dm); self.k=nn.Linear(dm,dm)
        self.v=nn.Linear(dm,dm); self.out=nn.Linear(dm,dm)
        self.drop=nn.Dropout(dp); self._w=None
    def forward(self,x):
        B,N,E=x.shape
        q=self.q(x).view(B,N,self.nh,self.dk).transpose(1,2)
        k=self.k(x).view(B,N,self.nh,self.dk).transpose(1,2)
        v=self.v(x).view(B,N,self.nh,self.dk).transpose(1,2)
        w=(q@k.transpose(-2,-1))/math.sqrt(self.dk); w=F.softmax(w,-1)
        self._w=w.detach(); w=self.drop(w)
        return self.out((w@v).transpose(1,2).contiguous().view(B,N,E))

class EncLayer(nn.Module):
    def __init__(self,d,nh,df,dp):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=MHA(d,nh,dp)
        self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,df),nn.GELU(),nn.Dropout(dp),nn.Linear(df,d),nn.Dropout(dp))
    def forward(self,x): return x+self.ff(self.n2(x+self.attn(self.n1(x))))

class BaseTrans(nn.Module):
    def __init__(self):
        super().__init__(); self.nl=N_LAYERS
        self.proj=nn.Linear(LOOKBACK,D_MODEL); self.pos=nn.Parameter(torch.randn(1,500,D_MODEL)*0.02)
        self.drop=nn.Dropout(DROPOUT); self.layers=nn.ModuleList([EncLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(N_LAYERS)])
        self.head=nn.Sequential(nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,1))
    def forward(self,x,get_pat=False):
        B,N,L=x.shape; h=self.drop(self.proj(x)+self.pos[:,:N,:]); pats={}
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if get_pat: pats[i]=layer.attn._w
        pred=self.head(h).squeeze(-1)
        return (pred,pats) if get_pat else pred

# ============================================================
# UPPER TRANSFORMER (with multimodal embeddings)
# ============================================================
class FullUpper(nn.Module):
    def __init__(self, n_ht, attn_dim, d=128, nl=4, nh=4, dff=256, do=0.15):
        super().__init__()
        # Attention pattern projection
        self.attn_proj = nn.Sequential(nn.Linear(attn_dim, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, n_ht, d) * 0.02)
        # Multimodal projection (3 × D_EMBED = 384)
        self.mm_proj = nn.Sequential(nn.Linear(D_EMBED*3, d), nn.LayerNorm(d))
        self.mm_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers = nn.ModuleList([EncLayer(d, nh, dff, do) for _ in range(nl)])
        self.err_head = nn.Sequential(nn.Linear(d, d*2), nn.GELU(), nn.Dropout(do), nn.Linear(d*2, d), nn.GELU(), nn.Dropout(do), nn.Linear(d, 1))
    def forward(self, patterns, mm_emb):
        B, H, N, _ = patterns.shape
        pooled = patterns.mean(dim=-1).reshape(B, H, N)
        flat = pooled.reshape(B, H*N)
        a_tok = self.attn_proj(flat).unsqueeze(1).expand(B, H, -1) + self.pos[:, :H, :]
        m_tok = self.mm_proj(mm_emb).unsqueeze(1) + self.mm_tok
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, m_tok, a_tok], dim=1)
        for layer in self.layers: x = layer(x)
        return self.err_head(x[:, 0]).squeeze(-1)

# ============================================================
# TRAIN BASE
# ============================================================
print("[3/5] Training Base Transformer...")
class FinDS(Dataset):
    def __init__(self,X,y): self.X=torch.FloatTensor(X); self.y=torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

tr_ld=DataLoader(FinDS(X_tr,y_tr),BATCH,shuffle=True,drop_last=True)
te_ld=DataLoader(FinDS(X_te,y_te),BATCH,shuffle=False)

base=BaseTrans().to(GPU); opt_b=torch.optim.AdamW(base.parameters(),lr=1e-3,weight_decay=1e-4)
sch_b=torch.optim.lr_scheduler.CosineAnnealingLR(opt_b,BASE_EP)
for ep in range(BASE_EP):
    base.train(); tl=0
    for x,y in tr_ld:
        pred=base(x.to(GPU)); loss=F.huber_loss(pred,y.to(GPU),delta=1.0)
        opt_b.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt_b.step(); tl+=loss.item()
    sch_b.step()
    if (ep+1)%15==0: print(f"  Epoch {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

base.eval()
with torch.no_grad():
    preds, ys = [], []
    for x,y in te_ld: preds.append(base(x.to(GPU)).cpu().numpy()); ys.append(y.numpy())
    preds=np.concatenate(preds); ys=np.concatenate(ys)
base_mse=float(np.mean((preds-ys)**2)); base_err=np.abs(preds-ys).mean(1); base_abs=np.abs(preds).mean(1)
print(f"  Base MSE: {base_mse:.6f} | |Pred| r: {spearmanr(base_abs,base_err)[0]:.4f}")

# ============================================================
# TRAIN UPPER
# ============================================================
print("[4/5] Extracting patterns + training Full Upper Transformer...")
base.eval()
all_pats, all_errs, all_mm = [], [], []
with torch.no_grad():
    for i in range(0, split, BATCH):
        xb = torch.FloatTensor(X_tr[i:i+BATCH]).to(GPU)
        yb = torch.FloatTensor(y_tr[i:i+BATCH]).to(GPU)
        pred, pats = base(xb, get_pat=True)
        stacked = torch.cat([pats[j].cpu() for j in range(N_LAYERS)], dim=1)
        all_pats.append(stacked.numpy()); all_errs.append((pred-yb).abs().mean(1).cpu().numpy())
        mm = torch.cat([zp_tr[i:i+BATCH], zs_tr[i:i+BATCH], zt_tr[i:i+BATCH]], dim=-1)
        all_mm.append(mm.numpy())
p_train=np.concatenate(all_pats); e_train=np.concatenate(all_errs); m_train=np.concatenate(all_mm)

total_heads=N_LAYERS*N_HEADS; attn_dim=total_heads*n_assets

upper=FullUpper(total_heads,attn_dim).to(GPU)
opt_u=torch.optim.AdamW(upper.parameters(),lr=3e-3,weight_decay=1e-5)
sch_u=torch.optim.lr_scheduler.CosineAnnealingLR(opt_u,UPPER_EP)

for ep in range(UPPER_EP):
    upper.train(); tl=0; nb=0
    for i in range(0,len(p_train),BATCH):
        pb=torch.FloatTensor(p_train[i:i+BATCH]).to(GPU)
        mb=torch.FloatTensor(m_train[i:i+BATCH]).to(GPU)
        eb=torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        pe=upper(pb,mb); loss=F.mse_loss(pe,eb)
        opt_u.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(upper.parameters(),2.0); opt_u.step(); tl+=loss.item(); nb+=1
    sch_u.step()
    if (ep+1)%10==0:
        with torch.no_grad():
            std=upper(torch.FloatTensor(p_train[:256]).to(GPU), torch.FloatTensor(m_train[:256]).to(GPU)).std().item()
        print(f"  Epoch {ep+1:3d} | Loss: {tl/nb:.6f} | std: {std:.6f}")

# ============================================================
# EVALUATION
# ============================================================
print("[5/5] Evaluation...")
base.eval(); upper.eval()
all_up, all_true, all_abs, all_mm_te = [], [], [], []
with torch.no_grad():
    for i in range(0, n_samp-split, BATCH):
        xb=torch.FloatTensor(X_te[i:i+BATCH]).to(GPU)
        yb=torch.FloatTensor(y_te[i:i+BATCH]).to(GPU)
        pred,pats=base(xb,get_pat=True)
        stacked=torch.cat([pats[j].cpu() for j in range(N_LAYERS)],dim=1)
        mm=torch.cat([zp_te[i:i+BATCH],zs_te[i:i+BATCH],zt_te[i:i+BATCH]],dim=-1)
        up_err=upper(stacked.to(GPU),mm.to(GPU)).cpu().numpy()
        true_err=(pred-yb).abs().mean(1).cpu().numpy()
        all_up.append(up_err); all_true.append(true_err)
        all_abs.append(np.abs(pred.cpu().numpy()).mean(1))
up_err=np.concatenate(all_up); true_err=np.concatenate(all_true); abs_bl=np.concatenate(all_abs)

# Ablation: just multimodal (no attention)
class MMOnly(nn.Module):
    def __init__(self,in_d=D_EMBED*3,d=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_d,d*4),nn.GELU(),nn.Dropout(0.2),nn.Linear(d*4,d*2),nn.GELU(),nn.Linear(d*2,1))
    def forward(self,x): return self.net(x).squeeze(-1)

mm_model=MMOnly().to(GPU); opt_mm=torch.optim.AdamW(mm_model.parameters(),lr=1e-3)
for ep in range(50):
    mm_model.train()
    for i in range(0,len(m_train),BATCH):
        mb=torch.FloatTensor(m_train[i:i+BATCH]).to(GPU); eb=torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss=F.mse_loss(mm_model(mb),eb); opt_mm.zero_grad(); loss.backward(); opt_mm.step()

mm_model.eval(); mm_only=[]
with torch.no_grad():
    for i in range(0,n_samp-split,BATCH):
        mm=torch.cat([zp_te[i:i+BATCH],zs_te[i:i+BATCH],zt_te[i:i+BATCH]],dim=-1)
        mm_only.append(mm_model(mm.to(GPU)).cpu().numpy())
mm_only=np.concatenate(mm_only)

# Attn only
attn_train=p_train.mean(axis=-1).reshape(len(p_train),-1); attn_te=[]
with torch.no_grad():
    for i in range(0,n_samp-split,BATCH):
        end=min(i+BATCH,n_samp-split); xb=torch.FloatTensor(X_te[i:end]).to(GPU)
        _,pats=base(xb,get_pat=True)
        stacked=torch.cat([pats[j].cpu() for j in range(N_LAYERS)],dim=1)
        attn_te.append(stacked.mean(-1).reshape(end-i,-1).numpy())
attn_te=np.concatenate(attn_te)

class AttnOnlyMLP(nn.Module):
    def __init__(self,in_d,d=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_d,d*2),nn.GELU(),nn.Dropout(0.15),nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,1))
    def forward(self,x): return self.net(x).squeeze(-1)

attn_model=AttnOnlyMLP(attn_train.shape[1]).to(GPU); opt_a=torch.optim.AdamW(attn_model.parameters(),lr=1e-3)
for ep in range(50):
    attn_model.train()
    for i in range(0,len(attn_train),BATCH):
        ab=torch.FloatTensor(attn_train[i:i+BATCH]).to(GPU); eb=torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss=F.mse_loss(attn_model(ab),eb); opt_a.zero_grad(); loss.backward(); opt_a.step()
attn_model.eval()
with torch.no_grad(): attn_only=attn_model(torch.FloatTensor(attn_te).to(GPU)).cpu().numpy()

def sr(a,b): return float(spearmanr(a,b)[0])

print("=" * 65)
print("  FULL PIPELINE RESULTS")
print("=" * 65)
print(f"  {'Method':<35} {'Spearman r':>10}")
print(f"  {'-'*45}")
print(f"  {'|Pred| Baseline':<35} {sr(abs_bl, true_err):>10.4f}")
print(f"  {'Attention-only MLP':<35} {sr(attn_only, true_err):>10.4f}")
print(f"  {'Multimodal-only MLP':<35} {sr(mm_only, true_err):>10.4f}")
print(f"  {'Full Upper + MM + Attn':<35} {sr(up_err, true_err):>10.4f}")
print(f"  {'-'*45}")
best = max(sr(abs_bl,true_err), sr(attn_only,true_err), sr(mm_only,true_err))
delta = sr(up_err, true_err) - best
print(f"  Full vs best baseline: {delta:+.4f}")
print(f"  Base MSE: {base_mse:.6f}")
print("=" * 65)
