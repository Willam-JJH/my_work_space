"""
Grouped test — RIGOROUS: fresh base per seed. 50 seeds × 6 groups.
Real-time progress on GPU (Arc B390).
"""
import numpy as np, pandas as pd, time, sys, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr, wilcoxon

DEV = 'xpu'; B = 512; L = 30; N_SEEDS = 50
ENS_SEEDS = [42, 123, 456, 789, 1024]

def log(*a):
    print(' '.join(str(x) for x in a), flush=True)

log(f"GPU: {torch.xpu.get_device_name(0)} | Seeds: {N_SEEDS} | Batch: {B}")
log("RIGOROUS: fresh base model per seed (no sharing)")

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

class SinglePred(nn.Module):
    def __init__(self):
        super().__init__()
        self.tp=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        self.pred=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1),nn.Softplus())
    def forward(self,h):
        B,S,D=h.shape; tw=F.softmax(self.tp(h).squeeze(-1),-1)
        return self.pred(torch.cat([(h*tw.unsqueeze(-1)).sum(1),h[:,-1]],-1)).squeeze(-1)

class MetaPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(900,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,32))
        self.ha=nn.MultiheadAttention(32,1,batch_first=True,dropout=.1)
        self.pred=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,1),nn.Softplus())
    def forward(self,p):
        B,H,S,_=p.shape; e=self.enc(p.reshape(B,H,S*S).view(B*H,S*S)).view(B,H,-1)
        ao,aw=self.ha(e,e,e); return self.pred((ao*aw.mean(1).unsqueeze(-1)).sum(1)).squeeze(-1)

# Load
us = pd.read_parquet('D:/code/data/us_market/log_returns.parquet')
ns = pd.read_parquet('D:/code/data/log_returns_nonstock.parquet')
cn_df = pd.read_parquet('D:/code/data/cn_market/daily_returns.parquet')
cn = cn_df.pivot(index='trddt',columns='stkcd',values='dretwd').dropna(axis=1,thresh=int(len(cn_df['trddt'].unique())*0.6)).ffill()

fx=[c for c in ns.columns if '=X' in c]; crypto=[c for c in ns.columns if '-USD' in c]
fut=[c for c in ns.columns if '=F' in c]; idx=[c for c in ns.columns if c.startswith('^')]

groups = {
    'US Stocks':   (us, list(us.columns[:300])),
    'CN A-Share':  (cn, list(cn.columns[:300])),
    'Forex':       (ns, fx[:61]),
    'Crypto':      (ns, crypto[:33]),
    'Commodities': (ns, fut[:31]),
    'Indices':     (ns, idx[:38]),
}
groups = {k:v for k,v in groups.items() if len(v[1])>=10}
for n,(_,t) in groups.items(): log(f"  {n}: {len(t)}")

all_results = {}; global_t0 = time.time()

for gi,(gname,(src,tickers)) in enumerate(groups.items()):
    log(f"\n{'='*55}\n[{gi+1}/{len(groups)}] {gname} ({len(tickers)} assets)\n{'='*55}")

    sub = src[tickers].ffill().dropna(axis=0)
    R = sub.values.astype(np.float32); N = R.shape[1]
    R = np.nan_to_num((R-np.nanmean(R,0,keepdims=True))/(np.nanstd(R,0,keepdims=True)+1e-8),0)
    n = len(R)-L-1
    X = np.lib.stride_tricks.sliding_window_view(R,L,axis=0)[:n].transpose(0,2,1).astype(np.float32)
    y = R[L+1:][:n].astype(np.float32); tr = int(n*0.7)
    log(f"  Train:{tr} Test:{n-tr} | {N} stocks")

    meta_rs, single_rs = [], []; t1 = time.time()
    crit = nn.HuberLoss(delta=1.0); ec = nn.HuberLoss(delta=0.5)

    # 50 seeds: FRESH base + FRESH predictors per seed
    for si in range(N_SEEDS):
        seed = 42 + si
        np.random.seed(seed); torch.manual_seed(seed)

        # Prepare data on GPU
        Xt=torch.FloatTensor(X[:tr]).to(DEV); yt=torch.FloatTensor(y[:tr]).to(DEV)
        Xe=torch.FloatTensor(X[tr:]).to(DEV); ye=torch.FloatTensor(y[tr:]).to(DEV)
        ld=DataLoader(TensorDataset(Xt,yt),batch_size=B,shuffle=True)

        # Train base (30 epochs — sufficient for convergence check)
        base = Base(N).to(DEV); opt_b = torch.optim.AdamW(base.parameters(),lr=3e-4,weight_decay=1e-5)
        for ep in range(30):
            base.train()
            for bx,by in ld: opt_b.zero_grad(); l=crit(base(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt_b.step()

        # Extract patterns
        base.eval()
        with torch.no_grad():
            _,ht=base(Xt,True); pt=base.get_pats()['3'].clone()
            _,he=base(Xe,True); pp=base.get_pats()['3'].clone()
            rt,_=base(Xt,False); re,_=base(Xe,False)
        at=torch.abs(rt-yt).mean(-1); ae=torch.abs(re-ye).mean(-1)

        # Meta predictor (50 epochs)
        vn=len(pt)//5; p_tr,p_v=pt[:-vn],pt[-vn:]; pe_tr,pe_v=at[:-vn],at[-vn:]
        mm=MetaPred().to(DEV); om=torch.optim.AdamW(mm.parameters(),lr=1e-4,weight_decay=1e-3)
        br,sb=-1,None
        for ep in range(50):
            mm.train(); om.zero_grad(); l=ec(mm(p_tr),pe_tr); l.backward(); torch.nn.utils.clip_grad_norm_(mm.parameters(),2.0); om.step()
            if (ep+1)%10==0:
                mm.eval()
                with torch.no_grad(): rv,_=spearmanr(mm(p_v).cpu().numpy(),pe_v.cpu().numpy())
                if rv>br: br=rv; sb={k:v.clone() for k,v in mm.state_dict().items()}
                if rv<br-0.2: break
        if sb: mm.load_state_dict(sb)
        mm.eval()
        with torch.no_grad(): meta_rs.append(spearmanr(mm(pp).cpu().numpy(),ae.cpu().numpy())[0])

        # Single predictor (50 epochs)
        h_vn=len(ht)//5; h_tr,h_v=ht[:-h_vn],ht[-h_vn:]; he_tr,he_v=at[:-h_vn],at[-h_vn:]
        sm=SinglePred().to(DEV); os_=torch.optim.AdamW(sm.parameters(),lr=3e-4,weight_decay=1e-2)
        bs,ss=-1,None
        for ep in range(50):
            sm.train(); os_.zero_grad(); l=ec(sm(h_tr),he_tr); l.backward(); torch.nn.utils.clip_grad_norm_(sm.parameters(),2.0); os_.step()
            if (ep+1)%10==0:
                sm.eval()
                with torch.no_grad(): rv,_=spearmanr(sm(h_v).cpu().numpy(),he_v.cpu().numpy())
                if rv>bs: bs=rv; ss={k:v.clone() for k,v in sm.state_dict().items()}
                if rv<bs-0.2: break
        if ss: sm.load_state_dict(ss)
        sm.eval()
        with torch.no_grad(): single_rs.append(spearmanr(sm(he).cpu().numpy(),ae.cpu().numpy())[0])

        if (si+1) % 10 == 0:
            dt = time.time()-t1
            mr=np.mean(meta_rs); sr=np.mean(single_rs); w=sum(np.array(meta_rs)>np.array(single_rs))
            log(f"    [{si+1:2d}/{N_SEEDS}] Meta r={mr:+.4f} Single r={sr:+.4f} | M wins {w}/{si+1} | {dt:.0f}s")

    marr=np.array(meta_rs); sarr=np.array(single_rs)
    w,pw=wilcoxon(marr,sarr,alternative='greater')
    log(f"  >>> Meta: {marr.mean():+.4f}±{marr.std():.3f} | Single: {sarr.mean():+.4f}±{sarr.std():.3f}")
    log(f"  >>> M wins {sum(marr>sarr)}/{N_SEEDS} | Wilcoxon p={pw:.6f}")

    # Deep Ensemble (N=5, fresh base each)
    log(f"\n  --- Deep Ensemble (N=5) ---")
    ens_preds=[]; t2=time.time()
    for i,seed in enumerate(ENS_SEEDS):
        np.random.seed(seed); torch.manual_seed(seed)
        Xt2=torch.FloatTensor(X[:tr]).to(DEV); yt2=torch.FloatTensor(y[:tr]).to(DEV)
        Xe2=torch.FloatTensor(X[tr:]).to(DEV); ld2=DataLoader(TensorDataset(Xt2,yt2),batch_size=B,shuffle=True)
        be=Base(N).to(DEV); oe=torch.optim.AdamW(be.parameters(),lr=3e-4,weight_decay=1e-5)
        for ep in range(30):
            be.train()
            for bx,by in ld2: oe.zero_grad(); l=crit(be(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(be.parameters(),2.0); oe.step()
        be.eval()
        with torch.no_grad(): ens_preds.append(be(Xe2,False)[0].cpu().numpy())
        log(f"    Member {i+1}/5 OK ({time.time()-t2:.0f}s)")
    es=np.stack(ens_preds,axis=0); es_std=es.std(axis=0).mean(axis=1)
    em=es.mean(axis=0); y_np = y[tr:].astype(np.float32)  # aligned with Xe
    ae_de=np.abs(em-y_np[:len(em)]).mean(axis=1)
    r_de,_=spearmanr(es_std,ae_de)

    # MC Dropout
    np.random.seed(42); torch.manual_seed(42)
    Xt3=torch.FloatTensor(X[:tr]).to(DEV); ld3=DataLoader(TensorDataset(Xt3,torch.FloatTensor(y[:tr]).to(DEV)),batch_size=B,shuffle=True)
    bm=Base(N).to(DEV); om_=torch.optim.AdamW(bm.parameters(),lr=3e-4,weight_decay=1e-5)
    for ep in range(30):
        bm.train()
        for bx,by in ld3: om_.zero_grad(); l=crit(bm(bx,False)[0],by); l.backward(); torch.nn.utils.clip_grad_norm_(bm.parameters(),2.0); om_.step()
    bm.train(); mps=[]
    Xe3=torch.FloatTensor(X[tr:]).to(DEV)
    with torch.no_grad():
        for _ in range(10): mps.append(bm(Xe3,False)[0].cpu().numpy())
    mc_std=np.stack(mps,axis=0).std(axis=0).mean(axis=1)
    r_mc,_=spearmanr(mc_std[:len(ae_de)],ae_de)  # align lengths
    r_heu,_=spearmanr(np.abs(em).mean(axis=1)[:len(ae_de)],ae_de)
    log(f"    DE: r={r_de:+.4f} | MC: r={r_mc:+.4f} | |Pred|: r={r_heu:+.4f}")

    all_results[gname] = {
        'N':N,'train':tr,'test':n-tr,
        'meta_r':marr.mean(),'meta_std':marr.std(),
        'single_r':sarr.mean(),'single_std':sarr.std(),
        'meta_wins':int(sum(marr>sarr)),'wilcoxon_p':pw,
        'deep_ens_r':r_de,'mc_dropout_r':r_mc,'pred_heuristic_r':r_heu,
    }

# FINAL TABLE
log(f"\n{'='*90}")
log(f"FINAL RESULTS BY ASSET CLASS (rigorous: fresh base per seed)")
log(f"{'='*90}")
log(f"  {'Group':<16} {'N':>5} {'Meta r':>8} {'Single r':>8} {'M wins':>7} {'W p':>8} {'DE r':>8} {'MC r':>8} {'Heu r':>8}")
log(f"  {'-'*85}")
for name,r in all_results.items():
    log(f"  {name:<16} {r['N']:>5} {r['meta_r']:>+8.4f} {r['single_r']:>+8.4f} {r['meta_wins']:>4}/{N_SEEDS} {r['wilcoxon_p']:>8.6f} {r['deep_ens_r']:>+8.4f} {r['mc_dropout_r']:>+8.4f} {r['pred_heuristic_r']:>+8.4f}")

log(f"\n  Best method per group:")
for name,r in all_results.items():
    methods={'Meta':r['meta_r'],'DeepEns':r['deep_ens_r'],'MCDrop':r['mc_dropout_r'],'|Pred|':r['pred_heuristic_r']}
    best=max(methods,key=methods.get); second=sorted(methods.values(),reverse=True)[1]
    log(f"    {name:<16} → {best} (r={methods[best]:.4f}, gap to 2nd: {methods[best]-second:+.4f})")

dt_total = time.time()-global_t0
log(f"\n  Total time: {dt_total:.0f}s ({dt_total/60:.1f}min)")
pd.DataFrame(all_results).T.to_csv('grouped_results_rigorous.csv')
log(f"  Saved: grouped_results_rigorous.csv\n{'='*90}\nDONE")
