"""
Grouped Experiments by Asset Class
====================================
US (400 stocks) | CN (900 stocks) — full pipeline per group
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
from torch.utils.data import DataLoader; import signatory; from scipy.stats import spearmanr
import math, time, warnings; warnings.filterwarnings('ignore')

GPU = torch.device("xpu"); CPU = torch.device("cpu")
print(f"GPU: {torch.xpu.get_device_name(0)} | {torch.xpu.get_device_properties(0).total_memory/1e9:.1f}GB VRAM")
torch.manual_seed(42); np.random.seed(42)

LOOKBACK = 30
D_EMBED, D_MODEL = 128, 64; N_LAYERS, N_HEADS = 4, 4; D_FF, DROPOUT = 128, 0.1
TEMP = 0.07; SIG_DEPTH, SIG_PCA = 1, 128

# Building blocks
class MHA(nn.Module):
    def __init__(self,dm,nh,dp=0.1):
        super().__init__(); self.nh=nh; self.dk=dm//nh
        self.q=nn.Linear(dm,dm); self.k=nn.Linear(dm,dm); self.v=nn.Linear(dm,dm)
        self.out=nn.Linear(dm,dm); self.drop=nn.Dropout(dp); self._w=None
    def forward(self,x):
        B,N,E=x.shape; q=self.q(x).view(B,N,self.nh,self.dk).transpose(1,2)
        k=self.k(x).view(B,N,self.nh,self.dk).transpose(1,2); v=self.v(x).view(B,N,self.nh,self.dk).transpose(1,2)
        w=(q@k.transpose(-2,-1))/math.sqrt(self.dk); w=F.softmax(w,-1); self._w=w.detach()
        return self.out((self.drop(w)@v).transpose(1,2).contiguous().view(B,N,E))
class EncLayer(nn.Module):
    def __init__(self,d,nh,df,dp):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=MHA(d,nh,dp)
        self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,df),nn.GELU(),nn.Dropout(dp),nn.Linear(df,d),nn.Dropout(dp))
    def forward(self,x): return x+self.ff(self.n2(x+self.attn(self.n1(x))))
class BaseTrans(nn.Module):
    def __init__(self):
        super().__init__(); self.nl=N_LAYERS
        self.proj=nn.Linear(LOOKBACK,D_MODEL); self.pos=nn.Parameter(torch.randn(1,1200,D_MODEL)*0.02)
        self.drop=nn.Dropout(DROPOUT); self.layers=nn.ModuleList([EncLayer(D_MODEL,N_HEADS,D_FF,DROPOUT) for _ in range(N_LAYERS)])
        self.head=nn.Sequential(nn.Linear(D_MODEL,D_MODEL//2),nn.GELU(),nn.Linear(D_MODEL//2,1))
    def forward(self,x,get_pat=False):
        B,N,L=x.shape; h=self.drop(self.proj(x)+self.pos[:,:N,:]); pats={}
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if get_pat: pats[i]=layer.attn._w
        pred=self.head(h).squeeze(-1); return (pred,pats) if get_pat else pred

# ============================================================
def run_group(name, ret_df, n_sample, batch_sz, pretrain_ep, base_ep, upper_ep):
    """Run full pipeline on one asset group."""
    print(f"\n{'='*60}")
    print(f"  {name}: {n_sample} stocks")
    print(f"{'='*60}")

    # Select top stocks
    comp = ret_df.notna().sum()/len(ret_df)
    top = comp.nlargest(n_sample).index.tolist()
    ret = ret_df[top].ffill().fillna(0)
    ret_v = ret.values.astype(np.float32); n_assets = len(ret.columns)
    n_samp = ret_v.shape[0] - LOOKBACK
    print(f"  {n_assets} stocks x {ret_v.shape[0]}d")

    # Sliding windows
    X = np.zeros((n_samp, n_assets, LOOKBACK), dtype=np.float32)
    y = np.zeros((n_samp, n_assets), dtype=np.float32)
    for i in range(n_samp): X[i] = ret_v[i:i+LOOKBACK].T; y[i] = ret_v[i+LOOKBACK]
    mu = X.mean(axis=-1, keepdims=True); st = X.std(axis=-1, keepdims=True) + 1e-8
    X = (X - mu) / st; y = np.clip(y, -np.percentile(np.abs(y), 99), np.percentile(np.abs(y), 99))
    split = int(n_samp * 0.7)

    # Signatures
    t0 = time.time()
    print(f"  Signatures depth={SIG_DEPTH}...")
    sig_full_dim = signatory.signature_channels(n_assets, SIG_DEPTH)
    X_sig_t = torch.FloatTensor(X).transpose(1,2).to(CPU)
    sig_full = np.zeros((n_samp, sig_full_dim), dtype=np.float32)
    for i in range(0, n_samp, 32): sig_full[i:i+32] = signatory.signature(X_sig_t[i:i+32], SIG_DEPTH, basepoint=True).cpu().numpy()
    from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA
    sig_full = np.nan_to_num(sig_full, nan=0.0, posinf=0.0, neginf=0.0)
    sig_full = StandardScaler().fit_transform(sig_full)
    sig = PCA(min(SIG_PCA, sig_full_dim)).fit_transform(sig_full)
    sig_dim = sig.shape[1]; print(f"  Sig: {sig_full_dim:,}→{sig_dim} [{time.time()-t0:.0f}s]")

    # Technical
    v5 = X[:,:,-5:].std(-1,keepdims=1); m5 = X[:,:,-5:].mean(-1,keepdims=1)
    m10 = X[:,:,-10:].mean(-1,keepdims=1); rsi = (X[:,:,-5:] > 0).mean(-1,keepdims=1)
    tech = np.concatenate([v5, m5, m10, rsi], axis=-1)

    X_tr, X_te = X[:split], X[split:]; y_tr, y_te = y[:split], y[split:]
    sig_tr, sig_te = sig[:split], sig[split:]; tech_tr, tech_te = tech[:split], tech[split:]
    print(f"  Train: {split} | Test: {n_samp-split}")

    # --- Pretraining ---
    print(f"  Pretraining ({pretrain_ep} ep)...")
    class PriceEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_assets*LOOKBACK,D_EMBED*2),nn.GELU(),nn.Linear(D_EMBED*2,D_EMBED),nn.LayerNorm(D_EMBED))
        def forward(self,x): return self.net(x.reshape(x.shape[0],-1))
    class SigEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(sig_dim,D_EMBED*2),nn.GELU(),nn.Linear(D_EMBED*2,D_EMBED),nn.LayerNorm(D_EMBED))
        def forward(self,x): return self.net(x)
    class TechEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_assets*4,D_EMBED*2),nn.GELU(),nn.Linear(D_EMBED*2,D_EMBED),nn.LayerNorm(D_EMBED))
        def forward(self,x): return self.net(x.reshape(x.shape[0],-1))

    p_enc=PriceEnc().to(GPU); s_enc=SigEnc().to(GPU); t_enc=TechEnc().to(GPU)
    aparams=list(p_enc.parameters())+list(s_enc.parameters())+list(t_enc.parameters())
    opt_pt=torch.optim.AdamW(aparams,lr=3e-4,weight_decay=1e-4)
    pt_ds=torch.utils.data.TensorDataset(torch.FloatTensor(X_tr),torch.FloatTensor(sig_tr),torch.FloatTensor(tech_tr))
    pt_ld=DataLoader(pt_ds,batch_sz,shuffle=True,drop_last=True)

    for ep in range(pretrain_ep):
        tl=0; nb=0
        for xb,sb,tb in pt_ld:
            xb,sb,tb=xb.to(GPU),sb.to(GPU),tb.to(GPU)
            zp=F.normalize(p_enc(xb),-1); zs=F.normalize(s_enc(sb),-1); zt=F.normalize(t_enc(tb),-1)
            B=xb.shape[0]; loss=0
            for za,zb in [(zp,zs),(zp,zt),(zs,zt)]:
                sim=(za@zb.T)/TEMP; labels=torch.arange(B,device=GPU)
                loss+=(F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))/2
            loss/=3; opt_pt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(aparams,1.0); opt_pt.step(); tl+=loss.item(); nb+=1
        if (ep+1)%15==0: print(f"    Pretrain {ep+1:3d} | Loss: {tl/nb:.4f}")

    p_enc.eval(); s_enc.eval(); t_enc.eval()
    with torch.no_grad():
        zp_tr=p_enc(torch.FloatTensor(X_tr).to(GPU)).cpu().numpy()
        zs_tr=s_enc(torch.FloatTensor(sig_tr).to(GPU)).cpu().numpy(); zt_tr=t_enc(torch.FloatTensor(tech_tr).to(GPU)).cpu().numpy()
        zp_te=p_enc(torch.FloatTensor(X_te).to(GPU)).cpu().numpy()
        zs_te=s_enc(torch.FloatTensor(sig_te).to(GPU)).cpu().numpy(); zt_te=t_enc(torch.FloatTensor(tech_te).to(GPU)).cpu().numpy()
    mm_tr=np.concatenate([zp_tr,zs_tr,zt_tr],-1); mm_te=np.concatenate([zp_te,zs_te,zt_te],-1)
    mm_dim=mm_tr.shape[1]

    # --- Base Transformer ---
    print(f"  Base Transformer ({base_ep} ep)...")
    class FinDS(torch.utils.data.Dataset):
        def __init__(self,X,y,mm):
            self.X = torch.FloatTensor(X); self.y = torch.FloatTensor(y); self.mm = torch.FloatTensor(mm)
        def __len__(self): return len(self.X)
        def __getitem__(self,i): return self.X[i],self.y[i],self.mm[i]
    tr_ds=FinDS(X_tr,y_tr,mm_tr); te_ds=FinDS(X_te,y_te,mm_te)
    tr_ld=DataLoader(tr_ds,batch_sz,shuffle=True,drop_last=True); te_ld=DataLoader(te_ds,batch_sz,shuffle=False)

    base=BaseTrans().to(GPU); opt_b=torch.optim.AdamW(base.parameters(),lr=1e-3,weight_decay=1e-4)
    sch_b=torch.optim.lr_scheduler.CosineAnnealingLR(opt_b,base_ep)
    for ep in range(base_ep):
        base.train(); tl=0
        for x,y,_ in tr_ld:
            pred=base(x.to(GPU)); loss=F.huber_loss(pred,y.to(GPU),delta=1.0)
            opt_b.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),2.0); opt_b.step(); tl+=loss.item()
        sch_b.step()
        if (ep+1)%15==0: print(f"    Base {ep+1:3d} | Loss: {tl/len(tr_ld):.6f}")

    base.eval()
    with torch.no_grad():
        ps,ys=[],[]
        for x,y,_ in te_ld: ps.append(base(x.to(GPU)).cpu().numpy()); ys.append(y.numpy())
        ps=np.concatenate(ps); ys=np.concatenate(ys)
    base_mse=float(np.mean((ps-ys)**2)); base_err=np.abs(ps-ys).mean(1); base_abs=np.abs(ps).mean(1)
    print(f"    Base MSE: {base_mse:.6f} | |Pred| r: {spearmanr(base_abs,base_err)[0]:.4f}")

    # --- Upper Transformer ---
    print(f"  Upper Transformer ({upper_ep} ep)...")
    base.eval(); all_pats,all_errs,all_mm=[],[],[]
    with torch.no_grad():
        for x,y,mm in tr_ld:
            pred,pats=base(x.to(GPU),get_pat=True)
            stacked=torch.cat([pats[i].cpu() for i in range(N_LAYERS)],dim=1)
            all_pats.append(stacked.numpy()); all_mm.append(mm.numpy())
            all_errs.append((pred-y.to(GPU)).abs().mean(1).cpu().numpy())
    p_train=np.concatenate(all_pats); e_train=np.concatenate(all_errs); m_train=np.concatenate(all_mm)

    total_heads=N_LAYERS*N_HEADS; attn_dim=total_heads*n_assets
    class UpperTrans(nn.Module):
        def __init__(self,n_ht,attn_d,mm_d,d=128,nl=4,nh=4,dff=256,do=0.15):
            super().__init__()
            self.attn_proj=nn.Sequential(nn.Linear(attn_d,d),nn.LayerNorm(d))
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

    upper=UpperTrans(total_heads,attn_dim,mm_dim).to(GPU)
    opt_u=torch.optim.AdamW(upper.parameters(),lr=3e-3,weight_decay=1e-5)
    sch_u=torch.optim.lr_scheduler.CosineAnnealingLR(opt_u,upper_ep)
    for ep in range(upper_ep):
        upper.train(); tl=0; nb=0
        for i in range(0,len(p_train),batch_sz):
            pb=torch.FloatTensor(p_train[i:i+batch_sz]).to(GPU)
            mb=torch.FloatTensor(m_train[i:i+batch_sz]).to(GPU)
            eb=torch.FloatTensor(e_train[i:i+batch_sz]).to(GPU)
            pe=upper(pb,mb); loss=F.mse_loss(pe,eb)
            opt_u.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(upper.parameters(),2.0); opt_u.step(); tl+=loss.item(); nb+=1
        sch_u.step()
        if (ep+1)%20==0: print(f"    Upper {ep+1:3d} | Loss: {tl/nb:.6f}")

    # --- Evaluate ---
    base.eval(); upper.eval()
    all_up,all_true,all_abs=[],[],[]
    with torch.no_grad():
        for x,y,mm in te_ld:
            pred,pats=base(x.to(GPU),get_pat=True)
            stacked=torch.cat([pats[i].cpu() for i in range(N_LAYERS)],dim=1)
            up_err=upper(stacked.to(GPU),mm.to(GPU)).cpu().numpy()
            true_err=(pred-y.to(GPU)).abs().mean(1).cpu().numpy()
            all_up.append(up_err); all_true.append(true_err); all_abs.append(np.abs(pred.cpu().numpy()).mean(1))
    up_err=np.concatenate(all_up); true_err=np.concatenate(all_true); abs_bl=np.concatenate(all_abs)

    # MM-only baseline
    class MLP(nn.Module):
        def __init__(self,in_d,d=256):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(in_d,d*4),nn.GELU(),nn.Dropout(0.2),nn.Linear(d*4,d*2),nn.GELU(),nn.Dropout(0.2),nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,1))
        def forward(self,x): return self.net(x).squeeze(-1)
    mm_m=MLP(mm_dim).to(GPU); opt_m=torch.optim.AdamW(mm_m.parameters(),lr=1e-3)
    for ep in range(40):
        mm_m.train()
        for i in range(0,len(m_train)-batch_sz,batch_sz):
            mb=torch.FloatTensor(m_train[i:i+batch_sz]).to(GPU); eb=torch.FloatTensor(e_train[i:i+batch_sz]).to(GPU)
            loss=F.mse_loss(mm_m(mb),eb); opt_m.zero_grad(); loss.backward(); opt_m.step()
    mm_m.eval()
    with torch.no_grad(): mm_only=np.concatenate([mm_m(torch.FloatTensor(mm_te[i:i+batch_sz]).to(GPU)).cpu().numpy() for i in range(0,len(mm_te),batch_sz)])

    # Sig-only
    sig_m=MLP(sig_dim).to(GPU); opt_s=torch.optim.AdamW(sig_m.parameters(),lr=1e-3)
    for ep in range(40):
        sig_m.train()
        for i in range(0,len(sig_tr)-batch_sz,batch_sz):
            sb=torch.FloatTensor(sig_tr[i:i+batch_sz]).to(GPU); eb=torch.FloatTensor(e_train[i:i+batch_sz]).to(GPU)
            loss=F.mse_loss(sig_m(sb),eb); opt_s.zero_grad(); loss.backward(); opt_s.step()
    sig_m.eval()
    with torch.no_grad(): sig_only=np.concatenate([sig_m(torch.FloatTensor(sig_te[i:i+batch_sz]).to(GPU)).cpu().numpy() for i in range(0,len(sig_te),batch_sz)])

    def sr(a,b): return float(spearmanr(a,b)[0])
    results = {
        'group': name, 'n_stocks': n_assets, 'base_mse': base_mse,
        'pred_baseline': sr(abs_bl, true_err),
        'sig_only': sr(sig_only, true_err),
        'mm_only': sr(mm_only, true_err),
        'full_upper': sr(up_err, true_err),
    }
    results['delta'] = results['full_upper'] - max(results['pred_baseline'], results['sig_only'], results['mm_only'])
    return results

# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("  GROUPED EXPERIMENTS")
print("=" * 60)

all_results = []

# Check available data sizes
us_df = pd.read_parquet("D:/code/data/us_all_returns.parquet")
cn_df = pd.read_parquet("D:/code/data/cn_all_returns.parquet")
crypto_df = pd.read_parquet("D:/code/data/crypto_returns.parquet")
forex_df = pd.read_parquet("D:/code/data/forex_returns.parquet")
comm_df = pd.read_parquet("D:/code/data/comm_idx_returns.parquet")

print(f"Data: US={us_df.shape[1]} CN={cn_df.shape[1]} Crypto={crypto_df.shape[1]} Forex={forex_df.shape[1]} Comm={comm_df.shape[1]}")

# US: 424 stocks → run 400
all_results.append(run_group("US-400", us_df, 400, batch_sz=8, pretrain_ep=40, base_ep=40, upper_ep=60))

# CN: 3933 stocks → run 900
all_results.append(run_group("CN-900", cn_df, 900, batch_sz=4, pretrain_ep=40, base_ep=40, upper_ep=60))

# Print summary
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
for r in all_results:
    print(f"\n  {r['group']} ({r['n_stocks']} stocks)  |  Base MSE: {r['base_mse']:.6f}")
    print(f"  {'Method':<25} {'Spearman r':>10} {'vs Best':>10}")
    print(f"  {'-'*45}")
    methods = [('|Pred| Baseline', r['pred_baseline']), ('Signature-only', r['sig_only']),
               ('Multimodal-only', r['mm_only']), ('FULL Upper+MM+Attn', r['full_upper'])]
    best = max(v for _,v in methods)
    for name, val in methods:
        mark = " ← BEST" if val == best else ""
        print(f"  {name:<25} {val:>10.4f} {val-best:>+10.4f}{mark}")
    print(f"  Delta vs best baseline: {r['delta']:+.4f}")
print("=" * 65)
