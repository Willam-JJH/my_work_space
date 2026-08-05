"""Standalone pretraining — save embeddings for full pipeline."""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import DataLoader; import yfinance as yf; import signatory; import pickle; import warnings
warnings.filterwarnings('ignore')

GPU=torch.device('xpu'); torch.manual_seed(42); np.random.seed(42)

TICKERS=['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM','V','JNJ','WMT','PG','MA','UNH','HD','BAC','NFLX','ADBE','CRM','XOM']
L=30; SD=3; D=128; BATCH=64; TEMP=0.07

data=yf.download(TICKERS,start='2015-01-01',end='2024-12-31',auto_adjust=True,progress=False)
close=data['Close'].dropna(axis=1,thresh=int(len(data)*0.8)).dropna(axis=0)
ret=np.log(close/close.shift(1)).dropna().values.astype(np.float32)
n_assets=len(close.columns); n_samp=ret.shape[0]-L
print(f'{n_assets} stocks x {ret.shape[0]} days')

X=np.zeros((n_samp,n_assets,L),dtype=np.float32)
y=np.zeros((n_samp,n_assets),dtype=np.float32)
for i in range(n_samp): X[i]=ret[i:i+L].T; y[i]=ret[i+L]
mu=X.mean(-1,keepdims=1); st=X.std(-1,keepdims=1)+1e-8; X=(X-mu)/st
y=np.clip(y,-np.percentile(np.abs(y),99),np.percentile(np.abs(y),99))

sig_dim=signatory.signature_channels(n_assets,SD)
X_sig=torch.FloatTensor(X).transpose(1,2).cpu()
sig=np.zeros((n_samp,sig_dim),dtype=np.float32)
for i in range(0,n_samp,128): sig[i:i+128]=signatory.signature(X_sig[i:i+128],SD,basepoint=True).cpu().numpy()
from sklearn.preprocessing import StandardScaler
sig=StandardScaler().fit_transform(sig)

def tech_feat(r):
    v5=r[:,:,-5:].std(-1,keepdims=1); v10=r[:,:,-10:].std(-1,keepdims=1)
    v20=r[:,:,-20:].std(-1,keepdims=1); m5=r[:,:,-5:].mean(-1,keepdims=1)
    m10=r[:,:,-10:].mean(-1,keepdims=1); rsi=(r[:,:,-5:]>0).mean(-1,keepdims=1)
    return np.concatenate([v5,v10,v20,m5,m10,rsi],-1)
tech=tech_feat(X)

split=int(n_samp*0.7)
X_tr,X_te=X[:split],X[split:]; y_tr,y_te=y[:split],y[split:]
sig_tr,sig_te=sig[:split],sig[split:]; tech_tr,tech_te=tech[:split],tech[split:]
print(f'Train: {split} | Test: {n_samp-split} | Sig dim: {sig_dim:,}')

class PriceEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(L,D*2),nn.GELU(),nn.Linear(D*2,D))
        self.pos=nn.Parameter(torch.randn(1,n_assets,D)*0.02)
        self.attn=nn.MultiheadAttention(D,4,batch_first=True)
        self.out=nn.Sequential(nn.Linear(D,D),nn.LayerNorm(D))
    def forward(self,x):
        B,N,L=x.shape; h=self.proj(x)+self.pos[:,:N,:]; h,_=self.attn(h,h,h)
        return self.out(h.mean(1))

class SigEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(sig_dim,D*4),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(D*4,D*2),nn.GELU(),nn.Dropout(0.1),
            nn.Linear(D*2,D),nn.LayerNorm(D))
    def forward(self,x): return self.net(x)

class TechEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(6,D*2),nn.GELU(),nn.Linear(D*2,D))
        self.pos=nn.Parameter(torch.randn(1,n_assets,D)*0.02)
        self.attn=nn.MultiheadAttention(D,4,batch_first=True)
        self.out=nn.Sequential(nn.Linear(D,D),nn.LayerNorm(D))
    def forward(self,x):
        B,N,_=x.shape; h=self.proj(x)+self.pos[:,:N,:]; h,_=self.attn(h,h,h)
        return self.out(h.mean(1))

p_enc=PriceEnc().to(GPU); s_enc=SigEnc().to(GPU); t_enc=TechEnc().to(GPU)
opt=torch.optim.AdamW(list(p_enc.parameters())+list(s_enc.parameters())+list(t_enc.parameters()),lr=5e-4,weight_decay=1e-4)
pt_ds=torch.utils.data.TensorDataset(torch.FloatTensor(X_tr),torch.FloatTensor(sig_tr),torch.FloatTensor(tech_tr))
pt_ld=DataLoader(pt_ds,BATCH,shuffle=True,drop_last=True)

print('Pretraining...')
for ep in range(60):
    tl=0; nb=0
    for xb,sb,tb in pt_ld:
        xb=xb.to(GPU); sb=sb.to(GPU); tb=tb.to(GPU)
        zp=F.normalize(p_enc(xb),-1); zs=F.normalize(s_enc(sb),-1); zt=F.normalize(t_enc(tb),-1)
        B=xb.shape[0]; loss=0
        for za,zb in [(zp,zs),(zp,zt),(zs,zt)]:
            sim=(za@zb.T)/TEMP; labels=torch.arange(B,device=GPU)
            loss+=(F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))/2
        loss/=3; opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(p_enc.parameters())+list(s_enc.parameters())+list(t_enc.parameters()),1.0)
        opt.step(); tl+=loss.item(); nb+=1
    if (ep+1)%15==0:
        with torch.no_grad():
            zpa=F.normalize(p_enc(torch.FloatTensor(X_tr[:128]).to(GPU)),-1)
            zsa=F.normalize(s_enc(torch.FloatTensor(sig_tr[:128]).to(GPU)),-1)
            align=(zpa*zsa).sum(-1).mean().item()
        print(f'  Epoch {ep+1:3d} | Loss: {tl/nb:.4f} | Align: {align:.4f}')

p_enc.eval(); s_enc.eval(); t_enc.eval()
with torch.no_grad():
    zp_tr=p_enc(torch.FloatTensor(X_tr).to(GPU)).cpu().numpy()
    zs_tr=s_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu().numpy()
    zt_tr=t_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu().numpy()
    zp_te=p_enc(torch.FloatTensor(X_te).to(GPU)).cpu().numpy()
    zs_te=s_enc(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy()
    zt_te=t_enc(torch.FloatTensor(tech_te).to(GPU)).cpu().numpy()

import pickle as pk
with open('/d/code/experiments/pretrained_embeddings.pkl','wb') as f:
    pk.dump({'zp_tr':zp_tr,'zs_tr':zs_tr,'zt_tr':zt_tr,'zp_te':zp_te,'zs_te':zs_te,'zt_te':zt_te,
             'X_tr':X_tr,'X_te':X_te,'y_tr':y_tr,'y_te':y_te},f)
torch.save({'price':p_enc.state_dict(),'sig':s_enc.state_dict(),'tech':t_enc.state_dict()},'/d/code/experiments/pretrained_models.pt')
print(f'Saved embeddings: zp_tr {zp_tr.shape}, zs_tr {zs_tr.shape}, zt_tr {zt_tr.shape}')
