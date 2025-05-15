#!/usr/bin/env python3
# train.py

import argparse
import numpy as np
import scipy.sparse as sp
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from scipy.sparse import csr_matrix

def train_pytorch_mb(R, W, Y, M, N, L,
                     K, gamma, lam, lr,
                     epochs, batch_size,
                     num_negatives=5,
                     lr_step_size=10,
                     lr_gamma=0.5,
                     device='cpu'):
    """
    R, W: scipy.sparse CSR of shape (M,N)
    Y:     scipy.sparse CSR of shape (N,L)
    """
    # 1) 参数初始化（叶子张量）
    P = torch.randn(M, K, device=device, requires_grad=True)
    with torch.no_grad(): P.mul_(0.01)
    Q = torch.randn(N, K, device=device, requires_grad=True)
    with torch.no_grad(): Q.mul_(0.01)
    X = torch.zeros(M, L, device=device, requires_grad=True)

    optim = torch.optim.Adam([P, Q, X], lr=lr)
    scheduler = StepLR(optim, step_size=lr_step_size, gamma=lr_gamma)

    # 2) 把 Y 转成 dense——若 L 太大请改用稀疏批量方式
    Y_dense = torch.from_numpy(Y.toarray()).float().to(device)

    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
        # 每轮随机打乱用户
        perm = torch.randperm(M, device=device)
        total_loss = 0.0

        for start in tqdm(range(0, M, batch_size),
                          desc=f"Epoch {epoch+1}", unit="batch", leave=False):
            optim.zero_grad()
            u_batch = perm[start : start + batch_size]
            B = u_batch.size(0)

            # 构造正样本：每个用户随机选一个他实际交互过的 item
            R_sub = R[u_batch, :].toarray()       # (B, N)
            pos_idx = []
            for row in R_sub:
                nz = np.where(row > 0)[0]
                pos_idx.append(int(np.random.choice(nz)) if nz.size>0 else int(np.random.randint(N)))
            pos_idx = torch.LongTensor(pos_idx).to(device)  # (B,)

            # 构造多负样本：每个用户采 num_negatives 个随机负样本
            neg_idx = torch.randint(0, N, (B, num_negatives), device=device)  # (B, num_neg)
            neg_idx_flat = neg_idx.view(-1)  # (B*num_neg,)

            # 取出对应的向量
            p_u    = P[u_batch]           # (B, K)
            q_pos  = Q[pos_idx]           # (B, K)
            q_neg  = Q[neg_idx_flat].view(B, num_negatives, K)  # (B, neg, K)
            x_u    = X[u_batch]           # (B, L)
            y_pos  = Y_dense[pos_idx]     # (B, L)
            y_neg  = Y_dense[neg_idx_flat].view(B, num_negatives, L)  # (B, neg, L)

            # 正/负样本分数
            pos_scores = (p_u * q_pos).sum(dim=1) + (x_u * y_pos).sum(dim=1)      # (B,)
            # (B, neg)
            neg_scores = (p_u.unsqueeze(1) * q_neg).sum(dim=2) \
                       + (x_u.unsqueeze(1) * y_neg).sum(dim=2)

            # BPR 损失：对每一个负样本对都累加
            diff = pos_scores.unsqueeze(1) - neg_scores  # (B, neg)
            bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).sum()

            # 正则项
            reg = gamma * (p_u.pow(2).sum() + 
                           q_pos.pow(2).sum() +
                           q_neg.pow(2).sum())
            l1  = lam   * x_u.abs().sum()

            loss = bpr_loss + reg + l1
            loss.backward()
            # 非负约束
            with torch.no_grad():
                X.clamp_(min=0)
            optim.step()

            total_loss += loss.item()

        # 每轮结束后衰减学习率
        scheduler.step()

        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return P.detach(), Q.detach(), X.detach()


def main(args):
    # 加载分割后的训练集
    R = sp.load_npz('data/processed/R_train.npz')
    W = sp.load_npz('data/processed/W_train.npz')
    Y = sp.load_npz('data/processed/Y.npz')

    # 子集采样
    if args.sample_users:
        R = R[:args.sample_users, :]
        W = W[:args.sample_users, :]
        print(f"Subsampled to first {args.sample_users} users.")
    if args.sample_items:
        R = R[:, :args.sample_items]
        W = W[:, :args.sample_items]
        Y = Y[:args.sample_items, :]
        print(f"Subsampled to first {args.sample_items} items.")

    M, N = R.shape
    L    = Y.shape[1]

    # 训练
    P_t, Q_t, X_t = train_pytorch_mb(
        R, W, Y, M, N, L,
        K=args.K,
        gamma=args.gamma,
        lam=args.lam,
        lr=args.eta,
        epochs=args.max_iter,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        device='cpu'
    )

    # 保存结果
    out = 'data/processed/geomf_model.npz'
    np.savez(out,
             P=P_t.cpu().numpy(),
             Q=Q_t.cpu().numpy(),
             X=X_t.cpu().numpy())
    print(f"Training complete. Model saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_users',   type=int, default=None)
    parser.add_argument('--sample_items',   type=int, default=None)
    parser.add_argument('--K',              type=int,   default=50)
    parser.add_argument('--max_iter',       type=int,   default=20)
    parser.add_argument('--gamma',          type=float, default=0.01)
    parser.add_argument('--lam',            type=float, default=0.1)
    parser.add_argument('--eta',            type=float, default=1e-3)
    parser.add_argument('--batch_size',     type=int,   default=4096)
    parser.add_argument('--num_negatives',  type=int,   default=5,
                        help="Number of negative samples per user per batch")
    parser.add_argument('--lr_step_size',   type=int,   default=10,
                        help="Epoch interval for learning rate decay")
    parser.add_argument('--lr_gamma',       type=float, default=0.5,
                        help="Multiplicative factor for LR decay")
    args = parser.parse_args()
    main(args)