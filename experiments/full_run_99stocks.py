"""
Full Pipeline on 99 US Stocks
==============================
Phase 1: Contrastive pretrain price/sig/tech encoders
Phase 2: Base Transformer → attention patterns
Phase 3: Upper Transformer fuses patterns + pretrained multimodal → error prediction
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, pickle, os, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu" if torch.xpu.device_count() > 0 else "cpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0) if GPU.type=='xpu' else 'CPU'}")

LOOKBACK, SIG_DEPTH = 30, 2  # depth=2 for 99 stocks: 1+99+9801=9901 dim
D_EMBED, D_MODEL = 128, 64
N_LAYERS, N_HEADS, D_FF, DROPOUT = 4, 4, 128, 0.1
BATCH = 64; PRETRAIN_EP = 60; BASE_EP = 50; UPPER_EP = 80; TEMP = 0.07
torch.manual_seed(42); np.random.seed(42)

# ============================================================
# DATA
# ============================================================
print("[1/5] Loading data...")
returns = pd.read_parquet("D:/code/data/us_returns.parquet")
returns = returns.dropna(axis=1, thresh=int(len(returns)*0.4)).ffill().fillna(0)
ret_vals = returns.values.astype(np.float32)
tickers = list(returns.columns); n_assets = len(tickers)
print(f"  {n_assets} stocks x {ret_vals.shape[0]} days")

n_samp = ret_vals.shape[0] - LOOKBACK
X_raw = np.zeros((n_samp, n_assets, LOOKBACK), dtype=np.float32)
y_raw = np.zeros((n_samp, n_assets), dtype=np.float32)
for i in range(n_samp):
    X_raw[i] = ret_vals[i:i+LOOKBACK].T; y_raw[i] = ret_vals[i+LOOKBACK]
mu = X_raw.mean(axis=-1, keepdims=True); st = X_raw.std(axis=-1, keepdims=True) + 1e-8
X_raw = (X_raw - mu) / st
y_raw = np.clip(y_raw, -np.percentile(np.abs(y_raw), 99), np.percentile(np.abs(y_raw), 99))
split = int(n_samp * 0.7)

# Signatures + PCA
print("  Computing signatures (depth=2)...")
sig_dim_full = signatory.signature_channels(n_assets, SIG_DEPTH)
X_sig = torch.FloatTensor(X_raw).transpose(1,2).to(CPU)
sig_full = np.zeros((n_samp, sig_dim_full), dtype=np.float32)
for i in range(0, n_samp, 64):
    sig_full[i:i+64] = signatory.signature(X_sig[i:i+64], SIG_DEPTH, basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sig_full = StandardScaler().fit_transform(sig_full)
sig_dim = 128
sig = PCA(n_components=sig_dim).fit_transform(sig_full)
print(f"  Sig: {sig_dim_full:,} → PCA → {sig_dim}")

# Technical features
def tech_feat(r):
    v5=r[:,:,-5:].std(-1,keepdims=1); v10=r[:,:,-10:].std(-1,keepdims=1)
    v20=r[:,:,-20:].std(-1,keepdims=1); m5=r[:,:,-5:].mean(-1,keepdims=1)
    m10=r[:,:,-10:].mean(-1,keepdims=1); rsi=(r[:,:,-5:]>0).mean(-1,keepdims=1)
    return np.concatenate([v5,v10,v20,m5,m10,rsi],-1)
tech = tech_feat(X_raw)

X_tr, X_te = X_raw[:split], X_raw[split:]
y_tr, y_te = y_raw[:split], y_raw[split:]
sig_tr, sig_te = sig[:split], sig[split:]
tech_tr, tech_te = tech[:split], tech[split:]
print(f"  Train: {split} | Test: {n_samp-split}")

# ============================================================
# PRETRAINING
# ============================================================
print("[2/5] Contrastive pretraining...")

class PriceEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(LOOKBACK, D_EMBED*2), nn.GELU(), nn.Linear(D_EMBED*2, D_EMBED))
        self.pos = nn.Parameter(torch.randn(1, n_assets, D_EMBED) * 0.02)
        self.attn = nn.MultiheadAttention(D_EMBED, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(D_EMBED, D_EMBED), nn.LayerNorm(D_EMBED))
    def forward(self, x):
        B, N, L = x.shape; h = self.proj(x) + self.pos[:, :N, :]
        h, _ = self.attn(h, h, h); return self.out(h.mean(1))

class SigEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sig_dim, D_EMBED*4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D_EMBED*4, D_EMBED*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D_EMBED*2, D_EMBED), nn.LayerNorm(D_EMBED))
    def forward(self, x): return self.net(x)

class TechEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, D_EMBED*2), nn.GELU(), nn.Linear(D_EMBED*2, D_EMBED))
        self.pos = nn.Parameter(torch.randn(1, n_assets, D_EMBED) * 0.02)
        self.attn = nn.MultiheadAttention(D_EMBED, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(D_EMBED, D_EMBED), nn.LayerNorm(D_EMBED))
    def forward(self, x):
        B, N, _ = x.shape; h = self.proj(x) + self.pos[:, :N, :]
        h, _ = self.attn(h, h, h); return self.out(h.mean(1))

p_enc = PriceEnc().to(GPU); s_enc = SigEnc().to(GPU); t_enc = TechEnc().to(GPU)
all_pretrain_params = list(p_enc.parameters()) + list(s_enc.parameters()) + list(t_enc.parameters())
opt_pt = torch.optim.AdamW(all_pretrain_params, lr=5e-4, weight_decay=1e-4)
pt_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(sig_tr), torch.FloatTensor(tech_tr))
pt_ld = DataLoader(pt_ds, BATCH, shuffle=True, drop_last=True)

for ep in range(PRETRAIN_EP):
    tl = 0; nb = 0
    for xb, sb, tb in pt_ld:
        xb = xb.to(GPU); sb = sb.to(GPU); tb = tb.to(GPU)
        zp = F.normalize(p_enc(xb), -1); zs = F.normalize(s_enc(sb), -1); zt = F.normalize(t_enc(tb), -1)
        B = xb.shape[0]; loss = 0
        for za, zb in [(zp, zs), (zp, zt), (zs, zt)]:
            sim = (za @ zb.T) / TEMP; labels = torch.arange(B, device=GPU)
            loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        loss /= 3
        opt_pt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_pretrain_params, 1.0)
        opt_pt.step(); tl += loss.item(); nb += 1
    if (ep+1) % 15 == 0:
        with torch.no_grad():
            zpa = F.normalize(p_enc(torch.FloatTensor(X_tr[:128]).to(GPU)), -1)
            zsa = F.normalize(s_enc(torch.FloatTensor(sig_tr[:128]).to(GPU)), -1)
            align = (zpa * zsa).sum(-1).mean().item()
        print(f"  Pretrain {ep+1:3d} | Loss: {tl/nb:.4f} | Align: {align:.4f}")

p_enc.eval(); s_enc.eval(); t_enc.eval()
with torch.no_grad():
    zp_tr = p_enc(torch.FloatTensor(X_tr).to(GPU)).cpu().numpy()
    zs_tr = s_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu().numpy()
    zt_tr = t_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu().numpy()
    zp_te = p_enc(torch.FloatTensor(X_te).to(GPU)).cpu().numpy()
    zs_te = s_enc(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy()
    zt_te = t_enc(torch.FloatTensor(tech_te).to(GPU)).cpu().numpy()

mm_dim = D_EMBED * 3
mm_tr = np.concatenate([zp_tr, zs_tr, zt_tr], axis=-1)
mm_te = np.concatenate([zp_te, zs_te, zt_te], axis=-1)
print(f"  Multimodal embeddings: {mm_dim}-dim | Align: ready")

# ============================================================
# BASE TRANSFORMER
# ============================================================
print("[3/5] Base Transformer...")

class MHA(nn.Module):
    def __init__(self, dm, nh, dp=0.1):
        super().__init__(); self.nh=nh; self.dk=dm//nh
        self.q=nn.Linear(dm,dm); self.k=nn.Linear(dm,dm)
        self.v=nn.Linear(dm,dm); self.out=nn.Linear(dm,dm)
        self.drop=nn.Dropout(dp); self._w=None
    def forward(self, x):
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
        self.drop=nn.Dropout(DROPOUT)
        self.layers=nn.ModuleList([EncLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(N_LAYERS)])
        self.head=nn.Sequential(nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,1))
    def forward(self,x,get_pat=False):
        B,N,L=x.shape; h=self.drop(self.proj(x)+self.pos[:,:N,:]); pats={}
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if get_pat: pats[i]=layer.attn._w
        pred=self.head(h).squeeze(-1)
        return (pred,pats) if get_pat else pred

class FinDS(Dataset):
    def __init__(self,X,y,mm): self.X,self.y,self.mm=torch.FloatTensor(X),torch.FloatTensor(y),torch.FloatTensor(mm)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i],self.mm[i]

tr_ds = FinDS(X_tr,y_tr,mm_tr); te_ds = FinDS(X_te,y_te,mm_te)
tr_ld = DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True)
te_ld = DataLoader(te_ds,BATCH,shuffle=False)  # 注意: 99 stocks might be > GPU batch capacity, reduce if OOM

base=BaseTrans().to(GPU); opt_b=torch.optim.AdamW(base.parameters(),lr=1e-3,weight_decay=1e-4)
sch_b=torch.optim.lr_scheduler.CosineAnnealingLR(opt_b,BASE_EP)
for ep in range(BASE_EP):
    base.train(); tl=0
    for x,y,_ in tr_ld:
        pred=base(x.to(GPU)); loss=F.huber_loss(pred,y.to(GPU),delta=1.0)
        opt_b.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt_b.step(); tl+=loss.item()
    sch_b.step()
    if (ep+1)%15==0: print(f"  Base {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

base.eval()
with torch.no_grad():
    preds,ys=[],[]
    for x,y,_ in te_ld: preds.append(base(x.to(GPU)).cpu().numpy()); ys.append(y.numpy())
    preds=np.concatenate(preds); ys=np.concatenate(ys)
base_mse=float(np.mean((preds-ys)**2)); base_err=np.abs(preds-ys).mean(1); base_abs=np.abs(preds).mean(1)
print(f"  Base MSE: {base_mse:.6f} | |Pred| r: {spearmanr(base_abs,base_err)[0]:.4f}")

# ============================================================
# UPPER TRANSFORMER
# ============================================================
print("[4/5] Upper Transformer...")
base.eval()
all_pats,all_errs,all_mm=[],[],[]
with torch.no_grad():
    for x,y,mm in tr_ld:
        pred,pats=base(x.to(GPU),get_pat=True)
        stacked=torch.cat([pats[i].cpu() for i in range(N_LAYERS)],dim=1)
        all_pats.append(stacked.numpy()); all_mm.append(mm.numpy())
        all_errs.append((pred-y.to(GPU)).abs().mean(1).cpu().numpy())
p_train=np.concatenate(all_pats); e_train=np.concatenate(all_errs); m_train=np.concatenate(all_mm)

class UpperTrans(nn.Module):
    def __init__(self,n_ht,attn_dim,mm_d,d=128,nl=4,nh=4,dff=256,do=0.15):
        super().__init__()
        self.attn_proj=nn.Sequential(nn.Linear(attn_dim,d),nn.LayerNorm(d))
        self.pos=nn.Parameter(torch.randn(1,n_ht,d)*0.02)
        self.mm_proj=nn.Sequential(nn.Linear(mm_d,d*2),nn.GELU(),nn.Dropout(0.1),nn.Linear(d*2,d),nn.LayerNorm(d))
        self.mm_tok=nn.Parameter(torch.randn(1,1,d)*0.02); self.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
        self.layers=nn.ModuleList([EncLayer(d,nh,dff,do) for _ in range(nl)])
        self.err_head=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(do),nn.Linear(d*2,d),nn.GELU(),nn.Dropout(do),nn.Linear(d,1))
    def forward(self,patterns,mm_emb):
        B,H,N,_=patterns.shape; pooled=patterns.mean(-1).reshape(B,H,N); flat=pooled.reshape(B,H*N)
        a_tok=self.attn_proj(flat).unsqueeze(1).expand(B,H,-1)+self.pos[:,:H,:]
        m_tok=self.mm_proj(mm_emb).unsqueeze(1)+self.mm_tok; cls=self.cls.expand(B,-1,-1)
        x=torch.cat([cls,m_tok,a_tok],dim=1)
        for layer in self.layers: x=layer(x)
        return self.err_head(x[:,0]).squeeze(-1)

total_heads=N_LAYERS*N_HEADS; attn_dim=total_heads*n_assets
upper=UpperTrans(total_heads,attn_dim,mm_dim).to(GPU)
opt_u=torch.optim.AdamW(upper.parameters(),lr=3e-3,weight_decay=1e-5)
sch_u=torch.optim.lr_scheduler.CosineAnnealingLR(opt_u,UPPER_EP)

for ep in range(UPPER_EP):
    upper.train(); tl=0; nb=0
    for i in range(0,len(p_train),BATCH):
        pb=torch.FloatTensor(p_train[i:i+BATCH]).to(GPU)
        mb=torch.FloatTensor(m_train[i:i+BATCH]).to(GPU)
        eb=torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        pe=upper(pb,mb); loss=F.mse_loss(pe,eb)
        opt_u.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(upper.parameters(),2.0)
        opt_u.step(); tl+=loss.item(); nb+=1
    sch_u.step()
    if (ep+1)%10==0:
        with torch.no_grad():
            s=upper(torch.FloatTensor(p_train[:256]).to(GPU),torch.FloatTensor(m_train[:256]).to(GPU)).std().item()
        print(f"  Upper {ep+1:3d} | Loss: {tl/nb:.6f} | std: {s:.6f}")

# ============================================================
# EVALUATION
# ============================================================
print("[5/5] Evaluation...")
base.eval(); upper.eval()
all_up,all_true,all_abs=[],[],[]
with torch.no_grad():
    for x,y,mm in te_ld:
        pred,pats=base(x.to(GPU),get_pat=True)
        stacked=torch.cat([pats[i].cpu() for i in range(N_LAYERS)],dim=1)
        up_err=upper(stacked.to(GPU),mm.to(GPU)).cpu().numpy()
        true_err=(pred-y.to(GPU)).abs().mean(1).cpu().numpy()
        all_up.append(up_err); all_true.append(true_err)
        all_abs.append(np.abs(pred.cpu().numpy()).mean(1))
up_err=np.concatenate(all_up); true_err=np.concatenate(all_true); abs_bl=np.concatenate(all_abs)

# Baselines
class MMOnly(nn.Module):
    def __init__(self,in_d,d=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_d,d*4),nn.GELU(),nn.Dropout(0.2),nn.Linear(d*4,d*2),nn.GELU(),nn.Linear(d*2,1))
    def forward(self,x): return self.net(x).squeeze(-1)

mm_model=MMOnly(mm_dim).to(GPU); opt_mm=torch.optim.AdamW(mm_model.parameters(),lr=1e-3)
for ep in range(50):
    mm_model.train()
    for i in range(0,len(m_train),BATCH):
        mb=torch.FloatTensor(m_train[i:i+BATCH]).to(GPU); eb=torch.FloatTensor(e_train[i:i+BATCH]).to(GPU)
        loss=F.mse_loss(mm_model(mb),eb); opt_mm.zero_grad(); loss.backward(); opt_mm.step()
mm_model.eval()
with torch.no_grad():
    mm_only=np.concatenate([mm_model(torch.FloatTensor(mm_te[i:i+BATCH]).to(GPU)).cpu().numpy() for i in range(0,len(mm_te),BATCH)])

def sr(a,b): return float(spearmanr(a,b)[0])

print("=" * 65)
print(f"  FULL PIPELINE on {n_assets} US Stocks")
print("=" * 65)
print(f"  |Pred| Baseline:            {sr(abs_bl, true_err):.4f}")
print(f"  Multimodal-only (pretrain): {sr(mm_only, true_err):.4f}")
print(f"  Upper + MM + Attn:          {sr(up_err, true_err):.4f}")
print(f"  ---")
delta = sr(up_err, true_err) - max(sr(abs_bl, true_err), sr(mm_only, true_err))
print(f"  Δ vs best baseline:         {delta:+.4f}")
print(f"  Base MSE:                   {base_mse:.6f}")
print("=" * 65)
