#!/usr/bin/env python3
import argparse
import scipy.sparse as sp
import torch
import numpy as np
from src.model.geo_mf import GeoMFPTStrict
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader
import random

class BPRDataset(Dataset):
    """Dataset for BPR sampling: yields (u, i_pos, j_neg) tuples."""
    def __init__(self, R, num_items, num_neg):
        """
        R: dense torch.Tensor (M x N), 1 if user interacted, else 0
        num_items: total number of items N
        num_neg: number of negatives per positive
        """
        self.R = R
        self.M, self.N = R.shape
        self.num_neg = num_neg
        # precompute positive item list per user
        self.user_pos = [torch.where(self.R[u] > 0)[0].tolist() for u in range(self.M)]
        # users that have at least one positive
        self.users = [u for u, pos in enumerate(self.user_pos) if len(pos) > 0]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # idx is index into self.users
        u = self.users[idx]
        pos_list = self.user_pos[u]
        i = random.choice(pos_list)
        # sample negatives
        negs = []
        while len(negs) < self.num_neg:
            j = random.randrange(self.N)
            if self.R[u, j] == 0:
                negs.append(j)
        return u, i, torch.tensor(negs, dtype=torch.long)

def bpr_fine_tune(R, W, Y, P, Q, X, epochs=5, lr=1e-3, num_neg=5, batch_size=256, num_workers=4):
    device = P.device
    P = P.detach().requires_grad_(True)
    Q = Q.detach().requires_grad_(True)
    X = X.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([P, Q, X], lr=lr)
    sigmoid = torch.sigmoid

    dataset = BPRDataset(R, R.shape[1], num_neg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    for epoch in range(epochs):
        total_loss = 0.0
        for u_batch, pos_batch, neg_batch in loader:
            u_batch = u_batch.to(device)
            pos_batch = pos_batch.to(device)
            neg_batch = neg_batch.to(device)  # shape (batch, num_neg)

            # compute positive and negative scores in batch
            P_batch = P[u_batch]            # (batch, K)
            X_batch = X[u_batch]            # (batch, L)
            Q_pos = Q[pos_batch]            # (batch, K)
            Y_pos = Y[pos_batch]            # (batch, L)
            pos_scores = (P_batch * Q_pos).sum(dim=1) + (X_batch * Y_pos).sum(dim=1)

            # for negatives, expand P,X to match negs
            P_expand = P_batch.unsqueeze(1).expand(-1, num_neg, -1)  # (batch, num_neg, K)
            X_expand = X_batch.unsqueeze(1).expand(-1, num_neg, -1)  # (batch, num_neg, L)
            Q_neg = Q[neg_batch]            # (batch, num_neg, K)
            Y_neg = Y[neg_batch]            # (batch, num_neg, L)
            neg_scores = (P_expand * Q_neg).sum(dim=2) + (X_expand * Y_neg).sum(dim=2)  # (batch, num_neg)

            # BPR loss
            diff = pos_scores.unsqueeze(1) - neg_scores
            loss = -torch.log(sigmoid(diff)).sum()
            # regularization
            loss = loss + 1e-4 * (P_batch.pow(2).sum() + Q_pos.pow(2).sum() + Q_neg.pow(2).sum())
            loss = loss + 1e-5 * X_batch.abs().sum()

            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                X[u_batch].clamp_(min=0)
            optimizer.step()
            total_loss += loss.item()
        print(f"BPR Epoch {epoch+1}/{epochs}, Loss={total_loss:.4f}")
    return P, Q, X

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_users', type=int, default=None)
    parser.add_argument('--sample_items', type=int, default=None)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--bpr_epochs', type=int, default=5, help="BPR 微调轮数")
    parser.add_argument('--bpr_lr', type=float, default=1e-3, help="BPR 学习率")
    parser.add_argument('--bpr_neg', type=int, default=5, help="每用户负采样数")
    parser.add_argument('--bpr_batch', type=int, default=256, help="BPR batch size")
    parser.add_argument('--bpr_workers', type=int, default=4, help="Number of DataLoader workers")
    args = parser.parse_args()

    # 加载 & 子采样 & 转 tensor
    R = sp.load_npz('data/processed/R_train.npz')
    W = sp.load_npz('data/processed/W_train.npz')
    Y = sp.load_npz('data/processed/Y.npz')

    if args.sample_users:
        R = R[:args.sample_users, :]
        W = W[:args.sample_users, :]
        print(f"Subsampled to first {args.sample_users} users.")
    if args.sample_items:
        R = R[:, :args.sample_items]
        W = W[:, :args.sample_items]
        Y = Y[:args.sample_items, :]
        print(f"Subsampled to first {args.sample_items} items.")

    device = torch.device('cpu')
    R_t = torch.from_numpy(R.toarray()).float().to(device)
    W_t = torch.from_numpy(W.toarray()).float().to(device)
    Y_t = torch.from_numpy(Y.toarray()).float().to(device)

    # 1) 严格论文版训练
    model = GeoMFPTStrict(K=args.K, gamma=args.gamma, lam=args.lam, eta=args.eta, max_iter=args.max_iter)
    model.fit(R_t, W_t, Y_t)
    P, Q, X = model.P, model.Q, model.X

    # 2) BPR 微调
    print("Starting BPR fine-tuning …")
    P, Q, X = bpr_fine_tune(R_t, W_t, Y_t, P, Q, X,
                            epochs=args.bpr_epochs,
                            lr=args.bpr_lr,
                            num_neg=args.bpr_neg,
                            batch_size=args.bpr_batch,
                            num_workers=args.bpr_workers)

    # 保存
    np.savez('data/processed/geomf_model_hybrid.npz',
             P=P.detach().cpu().numpy(),
             Q=Q.detach().cpu().numpy(),
             X=X.detach().cpu().numpy())
    print("Hybrid model saved.")