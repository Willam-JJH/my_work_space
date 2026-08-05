import numpy as np, pandas as pd, time, sys, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

DEV='xpu'; B=128; L=30; SEED=42
def log(*a): print(' '.join(str(x) for x in a), flush=True)
log(f"GPU: {torch.xpu.get_device_name(0)}")

class MHA(nn.Module):
    def __init__(self,d=128,h=8,dp=.1):
        super().__init__(); self.h,self.dk=h,d//h
        self.Wqkv=nn.Linear(d,3*d,0); self.Wo=nn.Linear(d,d,0); self.drop=nn.Dropout(dp); self.pats={}
    def forward(self,x,store=True,nm='0'):
        B,S,D=x.shape; H,K=self.h,self.dk; qkv=self.Wqkv(x).view(B,S,3,H,K).permute(2,0,3,1,4)
        w=self.drop(F.softmax((qkv[0]@qkv[1].transpose(-2,-1))/K**0.5,dim=-1))
        if store: self.pats[nm]=w.detach()
        return self.Wo((w@qkv[2]).transpose(1,2).contiguous().view(B,S,D))
class Block(nn.Module):
    def __init__(self,d,h,ff,dp):
        super().__init__(); self.attn=MHA(d,h,dp); self.n1=nn.LayerNorm(d); self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(dp),nn.Linear(ff,d),nn.Dropout(dp))
    def forward(self,x,store=True,nm='0'): x=self.n1(x+self.attn(x,store,nm)); return self.n2(x+self.ffn(x))
class Base(nn.Module):
    def __init__(self,ns,d=128,h=8,nl=4,ff=256,dp=.1):
        super().__init__(); self.h,self.nl=h,nl
        self.proj=nn.Linear(ns,d); self.pe=nn.Parameter(torch.randn(1,L,d)*0.02)
        self.blocks=nn.ModuleList([Block(d,h,ff,dp) for _ in range(nl)])
        self.head=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Linear(d//2,ns))
    def forward(self,x,store=False):
        x=self.proj(x)+self.pe[:,:x.shape[1],:]
        for i,b in enumerate(self.blocks): x=b(x,store=store,nm=str(i))
        return self.head(x[:,-1,:]),x
    def get_pats(self): return {k:v for blk in self.blocks for k,v in blk.attn.pats.items()}

class PatternTransformer(nn.Module):
    def __init__(self,in_dim,d_model=128,nhead=4,n_enc=3,out_h=8,seq=30):
        super().__init__()
        self.input_proj=nn.Linear(in_dim,d_model); self.out_h,self.seq=out_h,seq
        self.pe=nn.Parameter(torch.randn(1,seq,d_model)*0.02)
        el=nn.TransformerEncoderLayer(d_model,nhead,256,0.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(el,n_enc)
        self.head_q=nn.Parameter(torch.randn(1,out_h,d_model)*0.1)
        self.pos_q=nn.Parameter(torch.randn(1,seq,d_model)*0.1)
        dl=nn.TransformerDecoderLayer(d_model,nhead,256,0.1,batch_first=True)
        self.decoder=nn.TransformerDecoder(dl,2)
        self.out=nn.Sequential(nn.Linear(d_model,128),nn.GELU(),nn.Linear(128,seq))
    def forward(self,x):
        B=x.shape[0]; x=self.input_proj(x)+self.pe[:,:x.shape[1],:]; mem=self.encoder(x)
        patterns=[]
        for h in range(self.out_h):
            q=self.head_q[:,h:h+1,:].expand(B,self.seq,-1)+self.pos_q
            d=self.decoder(q,mem); patterns.append(self.out(d).unsqueeze(1))
        return torch.cat(patterns,1)

us=pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns=pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_raw=pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn=cn_raw.pivot(index='trddt',columns='stkcd',values='dretwd')
cn=cn.dropna(axis=1,thresh=int(len(cn)*0.6)).ffill().dropna(axis=0)

groups={
    'US Stocks':(us,list(us.columns[:200])),
    'CN A-Share':(cn,list(cn.columns[:200])),
    'Forex':(ns,[c for c in ns.columns if '=X' in c][:50]),
    'Crypto':(ns,[c for c in ns.columns if '-USD' in c][:30]),
    'Commodities':(ns,[c for c in ns.columns if '=F' in c][:30]),
    'Indices':(ns,[c for c in ns.columns if c.startswith('^')][:30]),
}

all_r2s={}
for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    if len(tickers)<5: continue
    log(f"\n[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)")
    sub=src[tickers].ffill().dropna(axis=0)
    R=sub.values.astype(np.float32); N_stocks=R.shape[1]
    R=np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
    n=len(R)-L-1
    X=np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y=R[L+1:][:n].astype(np.float32); tr=int(n*0.7)
    log(f"  Train:{tr} Test:{n-tr}")

    np.random.seed(SEED); torch.manual_seed(SEED)
    Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
    Xe=torch.FloatTensor(X[tr:]).to(DEV)
    ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)

    base=Base(N_stocks).to(DEV); opt=torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
    crit=nn.HuberLoss(delta=1.0)
    for ep in range(60):
        base.train()
        for bx,by in ld: opt.zero_grad(); l=crit(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt.step()

    base.eval()
    with torch.no_grad():
        _,ht=base(Xt,True); p_tr=base.get_pats()['3'].cpu().numpy()
        _,he=base(Xe,True); p_te=base.get_pats()['3'].cpu().numpy()

    model=PatternTransformer(in_dim=N_stocks).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    loss_fn=nn.MSELoss()
    sc=StandardScaler(); p_flat=p_tr.reshape(len(p_tr),-1)
    p_sc=sc.fit_transform(p_flat).reshape(p_tr.shape)
    p_tr_t=torch.FloatTensor(p_sc).to(DEV)
    Xt2=torch.FloatTensor(X[:tr]).to(DEV)
    ld2=DataLoader(TensorDataset(Xt2,p_tr_t),batch_size=B,shuffle=True)

    t0=time.time()
    for ep in range(100):
        model.train(); el=0
        for bx,bp in ld2:
            opt.zero_grad()
            l=loss_fn(model(bx).reshape(len(bx),-1),bp.reshape(len(bx),-1))
            l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); el+=l.item()
        if (ep+1)%50==0: log(f"    ep{ep+1:3d} loss={el/len(ld2):.6f} {time.time()-t0:.0f}s")

    model.eval()
    with torch.no_grad():
        pred=model(torch.FloatTensor(X[tr:]).to(DEV)).cpu().numpy()
    pred_flat=sc.inverse_transform(pred.reshape(len(pred),-1)).reshape(p_te.shape)
    head_r2s=[r2_score(p_te[:,h].reshape(len(p_te),-1),pred_flat[:,h].reshape(len(pred),-1)) for h in range(8)]
    avg=np.mean(head_r2s)
    heads_str=', '.join([f'{r:.3f}' for r in head_r2s])
    log(f"  Head R2: [{heads_str}]")
    tag='RECONSTRUCTIBLE' if avg>0.3 else ('NOT' if avg<0.1 else 'Marginal')
    log(f"  Avg R2 = {avg:.4f} -> {tag}")
    all_r2s[gname]=avg

log(f"\n{'='*50}")
log("TRANSFORMER RECONSTRUCTION — ALL GROUPS")
log(f"{'='*50}")
for g,r in all_r2s.items():
    tag='WARNING' if r>0.3 else 'OK'
    log(f"  {g:<20} R2 = {r:+.4f} [{tag}]")
avg_all=np.mean(list(all_r2s.values()))
log(f"\n  Overall: {avg_all:.4f}")
if avg_all>0.3: log("  >>> WARNING: Patterns can be reconstructed!")
else: log("  >>> Patterns cannot be reconstructed -> independence supported")
