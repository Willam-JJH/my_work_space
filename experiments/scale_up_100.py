"""
Scaled Experiment: 100 US Stocks + Deeper Model
=================================================
Keep per-asset features, deeper encoder, train longer.
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader; import signatory; from scipy.stats import spearmanr
import math, warnings; warnings.filterwarnings('ignore')

GPU=torch.device('xpu'); CPU=torch.device('cpu')
torch.manual_seed(42); np.random.seed(42)
print(f'GPU: {torch.xpu.get_device_name(0)}')

# Load US, take top 100 by data completeness
ret=pd.read_parquet('D:/code/data/us_returns.parquet')
completeness=ret.notna().sum()/len(ret)
top100=completeness.nlargest(100).index.tolist()
ret=ret[top100].ffill().fillna(0)
ret_vals=ret.values.astype(np.float32); n_assets=len(top100)
n_samp=ret_vals.shape[0]-30; print(f'{n_assets} stocks x {ret_vals.shape[0]}d')

X=np.zeros((n_samp,n_assets,30),dtype=np.float32); y=np.zeros((n_samp,n_assets),dtype=np.float32)
for i in range(n_samp): X[i]=ret_vals[i:i+30].T; y[i]=ret_vals[i+30]
mu=X.mean(-1,keepdims=1); st=X.std(-1,keepdims=1)+1e-8; X=(X-mu)/st
y=np.clip(y,-np.percentile(np.abs(y),99),np.percentile(np.abs(y),99))
split=int(n_samp*0.7)

# Signatures depth=1 + PCA
sig_full_dim=signatory.signature_channels(n_assets,1)
X_sig=torch.FloatTensor(X).transpose(1,2).to(CPU)
sig_full=np.zeros((n_samp,sig_full_dim),dtype=np.float32)
for i in range(0,n_samp,64): sig_full[i:i+64]=signatory.signature(X_sig[i:i+64],1,basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sig_full=StandardScaler().fit_transform(sig_full); sig=PCA(64).fit_transform(sig_full)

# Tech (per-asset, 5 features)
v5=X[:,:,-5:].std(-1,keepdims=1); v10=X[:,:,-10:].std(-1,keepdims=1)
m5=X[:,:,-5:].mean(-1,keepdims=1); m10=X[:,:,-10:].mean(-1,keepdims=1)
rsi=(X[:,:,-5:]>0).mean(-1,keepdims=1); tech=np.concatenate([v5,v10,m5,m10,rsi],-1)

X_tr,X_te=X[:split],X[split:]; y_tr,y_te=y[:split],y[split:]
sig_tr,sig_te=sig[:split],sig[split:]; tech_tr,tech_te=tech[:split],tech[split:]

# ============================================================
# PER-ASSET FEATURE EXTRACTION (small transformer per asset)
# ============================================================
class AssetEncoder(nn.Module):
    """Encode single asset (30d returns + 5 tech) → 32d embedding."""
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(35,64),nn.GELU(),nn.Linear(64,32),nn.LayerNorm(32))
    def forward(self,x): return self.proj(x)  # (batch*n_assets, 35) → (batch*n_assets, 32)

# ============================================================
# CROSS-ASSET TRANSFORMER
# ============================================================
class CrossAssetTransformer(nn.Module):
    """Attend across asset embeddings + signature."""
    def __init__(self, n_a, d=128, nh=4, nl=3, dff=256):
        super().__init__()
        self.n_a=n_a; self.d=d
        self.asset_enc=AssetEncoder()
        self.asset_pos=nn.Parameter(torch.randn(1,n_a,32)*0.02)
        self.sig_proj=nn.Sequential(nn.Linear(64,d),nn.LayerNorm(d))
        self.agg_proj=nn.Sequential(nn.Linear(n_a*32+d,d),nn.GELU(),nn.LayerNorm(d))
        self.trans=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d,nh,dff,dropout=0.1,activation='gelu',batch_first=True,norm_first=True),
            num_layers=nl)
        self.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
    def forward(self,returns,tech,sig):
        B,N,L=returns.shape
        # Per-asset encode
        r_flat=returns.reshape(B*N,L); t_flat=tech.reshape(B*N,5)
        feat=torch.cat([r_flat,t_flat],-1)  # (B*N, 35)
        a_emb=self.asset_enc(feat).view(B,N,32)+self.asset_pos[:,:N,:]  # (B,N,32)
        a_agg=a_emb.reshape(B,N*32)
        # Signature encode
        s_emb=self.sig_proj(sig)  # (B,d)
        # Aggregate
        agg=self.agg_proj(torch.cat([a_agg,s_emb],-1))  # (B,d)
        agg=agg.unsqueeze(1)  # (B,1,d)
        # Additional learnable tokens
        extra=torch.randn(B,4,self.d,device=agg.device)*0.02
        tokens=torch.cat([self.cls.expand(B,-1,-1),agg,extra],-2)  # (B,6,d)
        out=self.trans(tokens)
        return out[:,0]  # CLS token

# ============================================================
# FULL MODEL
# ============================================================
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=CrossAssetTransformer(n_assets)
        self.ret_head=nn.Linear(128,n_assets)
        self.unc_head=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Dropout(0.1),
                                    nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,returns,tech,sig):
        h=self.encoder(returns,tech,sig)
        return self.ret_head(h), self.unc_head(h).squeeze(-1)

class FinDS(Dataset):
    def __init__(self,X,tech,sig,y): self.X,self.tech,self.sig,self.y=torch.FloatTensor(X),torch.FloatTensor(tech),torch.FloatTensor(sig),torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.tech[i],self.sig[i],self.y[i]

tr_ds=FinDS(X_tr,tech_tr,sig_tr,y_tr); te_ds=FinDS(X_te,tech_te,sig_te,y_te)
BATCH=32
tr_ld=DataLoader(tr_ds,BATCH,shuffle=True,drop_last=True); te_ld=DataLoader(te_ds,BATCH,shuffle=False)

model=FullModel().to(GPU)
params=sum(p.numel() for p in model.parameters()); print(f'Params: {params:,}')
opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100)

print(f'Train: {split} | Test: {n_samp-split}')
for ep in range(100):
    model.train(); tl=0
    for x,tech,sig,y in tr_ld:
        pred_ret,pred_unc=model(x.to(GPU),tech.to(GPU),sig.to(GPU))
        loss_ret=F.huber_loss(pred_ret,y.to(GPU),delta=1.0)
        err=((pred_ret-y.to(GPU)).abs().mean(1)).detach()
        loss_unc=F.mse_loss(pred_unc,err)
        loss=loss_ret+0.1*loss_unc
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
        opt.step(); tl+=loss.item()
    sch.step()
    if (ep+1)%20==0: print(f'  {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}')

# Eval
model.eval()
all_ret,all_unc,all_y=[],[],[]
with torch.no_grad():
    for x,tech,sig,y in te_ld:
        pr,pu=model(x.to(GPU),tech.to(GPU),sig.to(GPU))
        all_ret.append(pr.detach().cpu().numpy()); all_unc.append(pu.detach().cpu().numpy())
        all_y.append(y.numpy())
pred_r=np.concatenate(all_ret); pred_u=np.concatenate(all_unc); true_y=np.concatenate(all_y)
mse=float(np.mean((pred_r-true_y)**2))
err_true=np.abs(pred_r-true_y).mean(1); abs_bl=np.abs(pred_r).mean(1)

def sr(a,b): return float(spearmanr(a,b)[0])
print('='*55)
print(f'  SCALED MODEL — {n_assets} Stocks, {params//1000}K params')
print('='*55)
print(f'  Return MSE:       {mse:.6f}')
print(f'  |Pred| r:         {sr(abs_bl,err_true):.4f}')
print(f'  Model Unc r:      {sr(pred_u,err_true):.4f}')
print(f'  Delta:            {sr(pred_u,err_true)-sr(abs_bl,err_true):+.4f}')
print('='*55)
