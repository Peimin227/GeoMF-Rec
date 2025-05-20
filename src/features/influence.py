# src/features/influence.py

import os
import numpy as np
import scipy.sparse as sp
import math
import argparse
import yaml
import json
from tqdm import tqdm

def load_config(path):
    return yaml.safe_load(open(path, 'r'))

def build_Y(biz2idx_path, centers_path, biz2grid_path, config):
    with open(biz2idx_path, 'r') as f:
        biz2idx = json.load(f)
    centers  = np.load(centers_path)    # (L,2)
    biz2grid = np.load(biz2grid_path)   # (N,)

    N = len(biz2idx)
    L = centers.shape[0]
    σ = config['grid']['sigma']
    r = config['grid']['radius']
    dlat = config['grid']['delta_lat']
    dlon = config['grid']['delta_lon']

    # Determine grid dimensions
    lat_vals = np.unique(centers[:, 0])
    lon_vals = np.unique(centers[:, 1])
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)

    # Earth radius in kilometers
    R_earth = 6371.0

    rows, cols, data = [], [], []

    for biz, idx in tqdm(biz2idx.items(), total=N, desc="Building Y rows"):
        center_cell = biz2grid[idx]
        lat_idx_center = center_cell // n_lon
        lon_idx_center = center_cell % n_lon
        lat0, lon0 = centers[center_cell]
        # Iterate only over cells within radius r
        for di in range(-r, r + 1):
            ni = lat_idx_center + di
            if ni < 0 or ni >= n_lat:
                continue
            for dj in range(-r, r + 1):
                nj = lon_idx_center + dj
                if nj < 0 or nj >= n_lon:
                    continue
                g = ni * n_lon + nj
                lat_c, lon_c = centers[g]
                # compute haversine distance (km) between grid center and biz center
                lat0_rad, lon0_rad = math.radians(lat0), math.radians(lon0)
                latc_rad, lonc_rad = math.radians(lat_c), math.radians(lon_c)
                dlat_rad = latc_rad - lat0_rad
                dlon_rad = lonc_rad - lon0_rad
                a = math.sin(dlat_rad/2)**2 + math.cos(lat0_rad) * math.cos(latc_rad) * math.sin(dlon_rad/2)**2
                dist_km = 2 * R_earth * math.asin(math.sqrt(a))
                # Gaussian kernel in km; σ is in degrees, so convert to km (approx 111 km/deg)
                val = math.exp(- (dist_km ** 2) / (2 * (σ * 111.0) ** 2))
                if val > 0:
                    rows.append(idx)
                    cols.append(g)
                    data.append(val)

    Y = sp.csr_matrix((data, (rows, cols)), shape=(N, L))
    return Y

def save_Y(Y, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sp.save_npz(out_path, Y)
    print(f"Saved Y matrix to {out_path} with shape {Y.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--biz2idx',      type=str, required=True)
    p.add_argument('--centers',      type=str, required=True)
    p.add_argument('--biz2grid',     type=str, required=True)
    p.add_argument('--config',       type=str, default='config/default.yaml')
    p.add_argument('--out',          type=str, default='data/processed/Y.npz')
    args = p.parse_args()

    config = load_config(args.config)
    Y = build_Y(
        args.biz2idx, args.centers, args.biz2grid, config
    )
    save_Y(Y, args.out)