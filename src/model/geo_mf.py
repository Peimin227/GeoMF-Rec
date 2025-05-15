# src/model/geo_mf.py

import torch

class GeoMFPTStrict:
    """
    GeoMF alternating optimization implemented in PyTorch.
    Updates P, Q via weighted ALS and X via projected gradient.
    """
    def __init__(self, K=50, max_iter=20, gamma=0.01, lam=0.1, eta=1e-3, device='cpu'):
        self.K = K
        self.max_iter = max_iter
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.device = torch.device(device)

    def fit(self, R, W, Y):
        """
        R, W: torch.Tensor of shape (M, N)
        Y:       torch.Tensor of shape (N, L)
        """
        M, N = R.shape
        L = Y.shape[1]
        # Initialize factors
        P = torch.randn(M, self.K, device=self.device) * 0.01
        Q = torch.randn(N, self.K, device=self.device) * 0.01
        X = torch.zeros(M, L, device=self.device)
        I_K = torch.eye(self.K, device=self.device)

        for it in range(self.max_iter):
            # Effective residual
            R_eff = R - X @ Y.t()

            # 1) Update P via weighted ALS
            for u in range(M):
                w_u = W[u]                             # (N,)
                idx = torch.nonzero(w_u, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                r_eff_u = R_eff[u, idx]               # (nu,)
                w_u_idx = w_u[idx]                    # (nu,)
                Q_idx = Q[idx]                        # (nu, K)
                # A = Q_idx^T diag(w) Q_idx + gamma I
                A = Q_idx.t() @ (w_u_idx.unsqueeze(1) * Q_idx) + self.gamma * I_K
                b = (r_eff_u * w_u_idx) @ Q_idx       # (K,)
                P[u] = torch.linalg.solve(A, b)

            # 2) Update Q via weighted ALS
            for i in range(N):
                w_i = W[:, i]                         # (M,)
                idx = torch.nonzero(w_i, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                r_eff_i = R_eff[idx, i]               # (mi,)
                w_i_idx = w_i[idx]                    # (mi,)
                P_idx = P[idx]                        # (mi, K)
                A = P_idx.t() @ (w_i_idx.unsqueeze(1) * P_idx) + self.gamma * I_K
                b = (r_eff_i * w_i_idx) @ P_idx       # (K,)
                Q[i] = torch.linalg.solve(A, b)

            # 3) Update X via projected gradient
            for u in range(M):
                w_u = W[u]                             # (N,)
                idx = torch.nonzero(w_u, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                r_u = R[u, idx]                       # (nu,)
                w_u_idx = w_u[idx]                    # (nu,)
                # latent part
                latent = (P[u].unsqueeze(0) @ Q[idx].t()).squeeze(0)  # (nu,)
                # geographic part
                Y_idx = Y[idx]                        # (nu, L)
                geo = Y_idx @ X[u]                    # (nu,)
                diff = geo - (r_u - latent)           # (nu,)
                grad = Y_idx.t() @ (w_u_idx * diff) + self.lam
                X[u] = torch.clamp(X[u] - self.eta * grad, min=0)

        self.P, self.Q, self.X = P, Q, X
        return self

    def save(self, path):
        torch.save({'P': self.P, 'Q': self.Q, 'X': self.X}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.P, self.Q, self.X = data['P'], data['Q'], data['X']