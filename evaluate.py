# evaluate.py

import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from scipy.sparse import csr_matrix

def load_model(path):
    data = np.load(path)
    return data['P'], data['Q'], data['X']

def recall_precision_at_k(P, Q, X, Y, R_train, R_test, K=10):
    num_users = R_test.shape[0]
    recalls, precs = [], []
    Y_csr = Y.tocsr()

    for u in tqdm(range(num_users), desc="Evaluating users"):
        _, test_items = R_test[u].nonzero()
        if len(test_items) == 0:
            continue
        train_items = set(R_train[u].nonzero()[1])
        # candidate items not in training set
        all_items = np.arange(Q.shape[0])
        cand = all_items[~np.in1d(all_items, list(train_items))]

        # latent score
        scores_latent = P[u].dot(Q[cand].T)
        # geographic score via sparse Y
        Yi = Y_csr[cand]             # sparse (len(cand), L)
        scores_geo = Yi.dot(X[u])    # dense (len(cand),)
        scores = scores_latent + scores_geo

        # compute hits
        topk = cand[np.argsort(-scores)[:K]]
        hits = len(set(topk) & set(test_items))
        recalls.append(hits / len(test_items))
        precs.append(hits / K)

    return np.mean(recalls), np.mean(precs)

def main(args):
    # 加载矩阵
    R_train = sp.load_npz('data/processed/R_train.npz')
    R_test  = sp.load_npz('data/processed/R_test.npz')
    Y       = sp.load_npz('data/processed/Y.npz')
    P, Q, X = load_model('data/processed/geomf_model.npz')

    # 子集采样
    if args.sample_users is not None:
        R_train = R_train[:args.sample_users, :]
        R_test  = R_test[:args.sample_users, :]
        X       = X[:args.sample_users]
        P       = P[:args.sample_users]
        print(f"Subsampled to first {args.sample_users} users for eval.")
    if args.sample_items is not None:
        R_train = R_train[:, :args.sample_items]
        R_test  = R_test[:, :args.sample_items]
        Q       = Q[:args.sample_items]
        Y       = Y[:args.sample_items, :]
        print(f"Subsampled to first {args.sample_items} items for eval.")

    # 评估
    for K in [5, 10, 20]:
        recall, prec = recall_precision_at_k(P, Q, X, Y, R_train, R_test, K)
        print(f"K={K}: Recall@K={recall:.4f}, Precision@K={prec:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_users', type=int, default=None,
                        help="Subsample number of users for debugging")
    parser.add_argument('--sample_items', type=int, default=None,
                        help="Subsample number of items for debugging")
    args = parser.parse_args()
    main(args)