"""
Factor Engine: Signature + Classical + Learned Factors
=======================================================
1. Path signature → named interpretable factors
2. Classical factors (momentum, volatility, etc.)
3. Autoencoder → ML-discovered latent factors
"""
import numpy as np; import pandas as pd; import torch; import torch.nn as nn; import torch.nn.functional as F
import signatory; from sklearn.decomposition import PCA

class FactorEngine:
    """Extract multiple factor types from price data."""

    def __init__(self, returns, lookback=30, sig_depth=2, device="cpu"):
        """
        returns: (n_days, n_assets) log return matrix
        lookback: sliding window size
        sig_depth: path signature depth
        """
        self.returns = returns
        self.lookback = lookback
        self.sig_depth = sig_depth
        self.device = device
        self.n_assets = returns.shape[1]

        # Create sliding windows
        self.n_samples = returns.shape[0] - lookback
        self.X = np.zeros((self.n_samples, self.n_assets, lookback), dtype=np.float32)
        for i in range(self.n_samples):
            self.X[i] = returns[i:i+lookback].T
        # Standardize
        mu = self.X.mean(axis=-1, keepdims=True)
        st = self.X.std(axis=-1, keepdims=True) + 1e-8
        self.X = (self.X - mu) / st

        # Cache
        self._signatures = None
        self._sig_names = None
        self._classical = None
        self._classical_names = None

    # ============================================================
    # 1. PATH SIGNATURE FACTORS (interpretable)
    # ============================================================
    def signature_factors(self):
        """Compute path signatures with named terms."""
        if self._signatures is not None:
            return self._signatures, self._sig_names

        # Compute signatures on CPU
        X_t = torch.FloatTensor(self.X).transpose(1, 2)
        sig_full = np.zeros((self.n_samples, signatory.signature_channels(self.n_assets, self.sig_depth)),
                           dtype=np.float32)
        for i in range(0, self.n_samples, 32):
            sig_full[i:i+32] = signatory.signature(X_t[i:i+32], self.sig_depth, basepoint=True).cpu().numpy()

        # Create named factor dictionary
        names = self._name_signature_terms()
        # Subset to manageable size via PCA for high-dim signatures
        if sig_full.shape[1] > 500:
            from sklearn.preprocessing import StandardScaler
            sig_full = StandardScaler().fit_transform(sig_full)
            sig_full = PCA(256).fit_transform(sig_full)
            names = [f"SIG_PCA_{i}" for i in range(256)]

        self._signatures = sig_full
        self._sig_names = names
        return sig_full, names

    def _name_signature_terms(self):
        """Generate interpretable names for signature terms."""
        names = ["S_const"]  # depth 0
        for i in range(self.n_assets):
            names.append(f"S_ret_{i}")  # total return of asset i
        if self.sig_depth >= 2:
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    if i == j:
                        names.append(f"S_vol_{i}")  # quadratic variation (volatility proxy)
                    else:
                        names.append(f"S_lead_{i}_{j}")  # cross-area (lead-lag)
        # Truncate to actual sig dim
        sig_dim = signatory.signature_channels(self.n_assets, self.sig_depth)
        return names[:sig_dim]

    # ============================================================
    # 2. CLASSICAL FACTORS
    # ============================================================
    def classical_factors(self):
        """Compute standard quantitative factors per asset per window."""
        if self._classical is not None:
            return self._classical, self._classical_names

        X = self.X  # (n_samples, n_assets, lookback)
        factors = {}
        names = []

        # Momentum factors
        for horizon in [5, 10, 20]:
            mom = X[:, :, -horizon:].mean(axis=-1)  # mean return over horizon
            factors[f"mom_{horizon}d"] = mom
            names.append(f"mom_{horizon}d")

        # Volatility factors
        for horizon in [5, 10, 20]:
            vol = X[:, :, -horizon:].std(axis=-1)
            factors[f"vol_{horizon}d"] = vol
            names.append(f"vol_{horizon}d")

        # RSI (relative strength)
        for horizon in [5, 10]:
            rsi = (X[:, :, -horizon:] > 0).mean(axis=-1)
            factors[f"rsi_{horizon}d"] = rsi
            names.append(f"rsi_{horizon}d")

        # Skewness & Kurtosis
        skew = np.mean((X - X.mean(axis=-1, keepdims=True))**3, axis=-1) / (X.std(axis=-1) + 1e-8)**3
        factors["skew"] = skew; names.append("skew")
        kurt = np.mean((X - X.mean(axis=-1, keepdims=True))**4, axis=-1) / (X.std(axis=-1) + 1e-8)**4
        factors["kurt"] = kurt; names.append("kurt")

        # Max drawdown proxy (min return in window)
        dd = X.min(axis=-1); factors["max_dd"] = dd; names.append("max_dd")

        # Range (max - min)
        rng = X.max(axis=-1) - X.min(axis=-1); factors["range"] = rng; names.append("range")

        # Volume proxy: sum of absolute returns
        vol_sum = np.abs(X).sum(axis=-1); factors["vol_sum"] = vol_sum; names.append("vol_sum")

        # Cross-asset correlation (market-wide, one per sample)
        n_corr = min(self.n_assets, 50)
        avg_corr_list = []
        for s in range(self.n_samples):
            c = np.corrcoef(X[s, :n_corr])[:n_corr, :n_corr]
            triu = c[np.triu_indices(n_corr, k=1)]
            avg_corr_list.append(triu.mean())
        avg_corr = np.array(avg_corr_list)  # (n_samples,)
        # Broadcast to per-asset: just repeat
        avg_corr_2d = np.tile(avg_corr[:, None], (1, self.n_assets))  # (n_samples, n_assets)
        factors["avg_corr"] = avg_corr_2d; names.append("avg_corr")

        # Stack: (n_samples, n_assets, n_factors)
        stacked = np.stack([factors[n] for n in names], axis=-1)

        self._classical = stacked
        self._classical_names = names
        return stacked, names

    # ============================================================
    # 3. AUTOENCODER → LEARNED FACTORS
    # ============================================================
    def learned_factors(self, n_factors=32, epochs=50, batch_size=64):
        """
        Train an autoencoder on the combined feature space,
        extract bottleneck as learned latent factors.
        """
        device = torch.device(self.device)

        # Combine signature + classical features
        sig, _ = self.signature_factors()
        classical, _ = self.classical_factors()
        # Pool classical: (n_samples, n_assets, n_factors) → (n_samples, n_factors) via mean
        classical_pooled = classical.mean(axis=1)

        from sklearn.preprocessing import StandardScaler
        features = np.concatenate([sig, classical_pooled], axis=-1)
        features = StandardScaler().fit_transform(features)
        feat_dim = features.shape[1]

        # Autoencoder
        class FactorAE(nn.Module):
            def __init__(self, in_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, latent_dim))
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128), nn.GELU(),
                    nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(512, in_dim))
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z

        model = FactorAE(feat_dim, n_factors).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        X_t = torch.FloatTensor(features).to(device)
        split = int(len(features) * 0.7)
        X_tr, X_val = X_t[:split], X_t[split:]

        for ep in range(epochs):
            model.train()
            idx = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), batch_size):
                batch = X_tr[idx[i:i+batch_size]]
                recon, _ = model(batch)
                loss = F.mse_loss(recon, batch)
                opt.zero_grad(); loss.backward(); opt.step()

        # Extract latent factors
        model.eval()
        with torch.no_grad():
            _, latent = model(X_t)
        latent = latent.cpu().numpy()

        # Name learned factors by their importance (PCA of latent)
        pca = PCA(n_components=min(n_factors, 10))
        pca.fit(latent)
        learned_names = [f"ML_F{i+1}" for i in range(n_factors)]

        return latent, learned_names

    # ============================================================
    # 4. COMBINED FACTOR MATRIX
    # ============================================================
    def combined_factors(self, use_learned=True, n_learned=32):
        """Return (n_samples, total_dim) combined factor matrix."""
        sig, sig_names = self.signature_factors()
        classical, cl_names = self.classical_factors()
        classical_pooled = classical.mean(axis=1)  # (n_samples, n_cl_factors)

        from sklearn.preprocessing import StandardScaler
        sig = StandardScaler().fit_transform(sig)

        combined = [sig, classical_pooled]
        all_names = list(sig_names) + cl_names

        if use_learned:
            learned, lrn_names = self.learned_factors(n_learned)
            combined.append(StandardScaler().fit_transform(learned))
            all_names += lrn_names

        result = np.concatenate(combined, axis=-1)
        return result, all_names


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    # Generate synthetic data
    n_stocks = 20; n_days = 1000
    fake_returns = np.random.randn(n_days, n_stocks).astype(np.float32) * 0.01

    engine = FactorEngine(fake_returns, lookback=20, sig_depth=1, device="cpu")

    # Signature factors
    sig, sig_names = engine.signature_factors()
    print(f"Signature: {sig.shape} | Names: {sig_names[:5]}...")

    # Classical factors
    cl, cl_names = engine.classical_factors()
    print(f"Classical: {cl.shape} | Names: {cl_names}")

    # Combined
    combined, all_names = engine.combined_factors(use_learned=False)
    print(f"Combined (no ML): {combined.shape} | Total factors: {len(all_names)}")

    print("\nFactor Engine ready!")
