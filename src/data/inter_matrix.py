# src/data/inter_matrix.py

import os
import json
import numpy as np
import scipy.sparse as sp
import argparse

def build_id_maps(review_path, checkin_path):
    user2idx = {}
    biz2idx  = {}
    next_u = 0
    next_i = 0

    # 扫描 review.json
    with open(review_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            u_id = r['user_id']
            b_id = r['business_id']
            if u_id not in user2idx:
                user2idx[u_id] = next_u; next_u += 1
            if b_id not in biz2idx:
                biz2idx[b_id] = next_i; next_i += 1

    # 扫描 checkin.json
    with open(checkin_path, 'r') as f:
        for line in f:
            c = json.loads(line)
            b_id = c['business_id']
            if b_id not in biz2idx:
                biz2idx[b_id] = next_i; next_i += 1

    return user2idx, biz2idx

def build_interaction_matrix(user2idx, biz2idx, review_path, checkin_path):
    # 收集 (u, i) 计数
    counts = {}
    # 来自 review.json
    with open(review_path, 'r') as f:
        for line in f:
            r = json.loads(line)
            u = user2idx[r['user_id']]
            i = biz2idx[r['business_id']]
            counts[(u, i)] = counts.get((u, i), 0) + 1

    # 来自 checkin.json
    with open(checkin_path, 'r') as f:
        for line in f:
            c = json.loads(line)
            i = biz2idx[c['business_id']]
            # Yelp checkin.json 只有 business 级别打卡次数，这里视为多个交互
            for cnt in c.get('checkin_info', {}).values():
                # 'checkin_info' 类似 {'2018-01-01': 2, ...}
                # 对每条记录累加 cnt 次
                counts[(None, i)] = counts.get((None, i), 0)  # 占位，若无 user_id 可忽略

    # 构造稀疏矩阵 R 和权重矩阵 W
    users = list(set(u for u, _ in counts.keys() if u is not None))
    items = list(set(i for _, i in counts.keys()))

    num_users = len(user2idx)
    num_items = len(biz2idx)

    rows, cols, data = [], [], []
    for (u, i), c in counts.items():
        if u is None:
            continue  # 若无对应 user，可按需处理或忽略
        rows.append(u)
        cols.append(i)
        data.append(c)

    R = sp.csr_matrix((np.ones_like(data), (rows, cols)), shape=(num_users, num_items))

    # 构造 W
    alpha = 40.0
    w_data = [1.0 + alpha * np.log1p(c) for c in data]
    W = sp.csr_matrix((w_data, (rows, cols)), shape=(num_users, num_items))

    return R, W

def save_matrix(mat, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sp.save_npz(out_path, mat)
    print(f"Saved matrix to {out_path}")

def main(args):
    # 构建 ID 映射
    user2idx, biz2idx = build_id_maps(args.review, args.checkin)
    print(f"#users: {len(user2idx)}, #items: {len(biz2idx)}")

    # 构建矩阵
    R, W = build_interaction_matrix(user2idx, biz2idx, args.review, args.checkin)

    # 保存
    save_matrix(R, args.out_dir + '/R.npz')
    save_matrix(W, args.out_dir + '/W.npz')

    # 同时可以把映射关系保存成 JSON 以便后续使用
    with open(os.path.join(args.out_dir, 'user2idx.json'), 'w') as f:
        json.dump(user2idx, f)
    with open(os.path.join(args.out_dir, 'biz2idx.json'), 'w') as f:
        json.dump(biz2idx, f)
    print("ID maps saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--review',    type=str, required=True, help="Path to review.json")
    parser.add_argument('--checkin',   type=str, required=True, help="Path to checkin.json")
    parser.add_argument('--out-dir',   type=str, default='data/processed', help="Output directory")
    args = parser.parse_args()

    main(args)
    