# src/model/als.py

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

def weighted_als(R: csr_matrix, W: csr_matrix, K: int, gamma: float, num_iter: int = 10):
    """
    Weighted ALS implementation.
    R: scipy.sparse csr_matrix shape (M, N) of interactions.
    W: scipy.sparse csr_matrix shape (M, N) of weights.
    K: latent dimensionality.
    gamma: regularization parameter.
    num_iter: number of ALS iterations.
    Returns:
        P: np.ndarray shape (M, K)
        Q: np.ndarray shape (N, K)
    """
    M, N = R.shape
    # Initialize latent factors
    P = np.random.normal(scale=0.01, size=(M, K))
    Q = np.random.normal(scale=0.01, size=(N, K))
    W_csr = W.tocsr()

    for it in range(num_iter):
        # Update P for each user
        for u in tqdm(range(M), desc=f"ALS iter {it+1}/{num_iter} - Users", leave=False):
            row = R[u]
            if row.nnz == 0:
                continue
            idx = row.indices
            ru = row.data
            wu = W_csr[u, idx].toarray().ravel()
            Qi = Q[idx]  # shape (len(idx), K)
            A = (Qi.T * wu).dot(Qi) + gamma * np.eye(K)
            b = (wu * ru).dot(Qi)
            P[u] = np.linalg.solve(A, b)

        # Update Q for each item
        for i in tqdm(range(N), desc=f"ALS iter {it+1}/{num_iter} - Items", leave=False):
            col = R[:, i]
            if col.nnz == 0:
                continue
            idx = col.indices
            ri = col.data
            wi = W_csr[idx, i].toarray().ravel()
            Pi = P[idx]  # shape (len(idx), K)
            A = (Pi.T * wi).dot(Pi) + gamma * np.eye(K)
            b = (wi * ri).dot(Pi)
            Q[i] = np.linalg.solve(A, b)

    return P, Q