# src/data/split.py

import numpy as np
import scipy.sparse as sp
import argparse
from tqdm import tqdm

def train_test_split(R, test_ratio=0.3, seed=42):
    np.random.seed(seed)
    R = R.tocsr()
    rows, cols = R.nonzero()

    # 按用户分组
    user2items = {}
    for u, i in zip(rows, cols):
        user2items.setdefault(u, []).append(i)

    train_rows, train_cols, train_data = [], [], []
    test_rows,  test_cols,  test_data  = [], [], []

    # 对每个用户抽样
    for u, items in tqdm(user2items.items(), total=len(user2items), desc="Splitting users"):
        n_test = max(1, int(len(items) * test_ratio))
        test_set = set(np.random.choice(items, size=n_test, replace=False))
        for i in items:
            if i in test_set:
                test_rows.append(u);  test_cols.append(i);  test_data.append(1)
            else:
                train_rows.append(u); train_cols.append(i); train_data.append(1)

    num_users, num_items = R.shape
    R_train = sp.csr_matrix((train_data, (train_rows, train_cols)), shape=(num_users, num_items))
    R_test  = sp.csr_matrix((test_data,  (test_rows,  test_cols)),  shape=(num_users, num_items))
    return R_train, R_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--R',          type=str, default='data/processed/R.npz')
    parser.add_argument('--W',          type=str, default='data/processed/W.npz')
    parser.add_argument('--out_dir',    type=str, default='data/processed')
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    # 加载原始矩阵
    R = sp.load_npz(args.R)
    W = sp.load_npz(args.W)

    # 切分
    R_train, R_test = train_test_split(R, test_ratio=args.test_ratio, seed=args.seed)

    # 构造 W_train：只保留训练对的权重
    train_u, train_i = R_train.nonzero()
    # 提取对应位置的权重
    w_values = [W[u, i] for u, i in zip(train_u, train_i)]
    W_train = sp.csr_matrix((w_values, (train_u, train_i)), shape=W.shape)

    # 保存
    sp.save_npz(f"{args.out_dir}/R_train.npz", R_train)
    sp.save_npz(f"{args.out_dir}/R_test.npz",  R_test)
    sp.save_npz(f"{args.out_dir}/W_train.npz", W_train)
    print("Saved R_train.npz, R_test.npz, W_train.npz in", args.out_dir)

if __name__ == "__main__":
    main()