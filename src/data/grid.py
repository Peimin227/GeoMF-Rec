# src/data/grid.py

import os
import numpy as np
import pandas as pd
import argparse
import yaml
import json

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_grid(biz_json, biz2idx_path, config):
    # 1. 读取商铺经纬度
    biz_df = pd.read_json(biz_json, lines=True)
    # 2. 读取 business_id -> index 映射
    with open(biz2idx_path, 'r') as f:
        biz2idx = json.load(f)
    # 3. 过滤仅保留映射内的商铺
    biz_df = biz_df[biz_df['business_id'].isin(biz2idx)]
    coords = biz_df[['latitude', 'longitude']].values

    # 4. 计算网格边界
    dlat = config['grid']['delta_lat']
    dlon = config['grid']['delta_lon']
    min_lat, max_lat = coords[:,0].min(), coords[:,0].max()
    min_lon, max_lon = coords[:,1].min(), coords[:,1].max()

    lat_edges = np.arange(min_lat, max_lat + dlat, dlat)
    lon_edges = np.arange(min_lon, max_lon + dlon, dlon)

    # 5. 生成所有格子中心
    centers = []
    for lat in lat_edges:
        for lon in lon_edges:
            centers.append((lat + dlat/2, lon + dlon/2))
    centers = np.array(centers)

    # 6. 每个商铺所属格子索引
    lat_idx = np.floor((coords[:,0] - min_lat) / dlat).astype(int)
    lon_idx = np.floor((coords[:,1] - min_lon) / dlon).astype(int)
    grid_w = len(lon_edges)
    biz2grid = lat_idx * grid_w + lon_idx

    return centers, biz2grid

def save_arrays(centers, biz2grid, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'grid_centers.npy'), centers)
    np.save(os.path.join(out_dir, 'biz2grid.npy'), biz2grid)
    print(f"Saved grid_centers.npy ({centers.shape}) and biz2grid.npy ({biz2grid.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--biz_json', type=str, required=True)
    parser.add_argument('--biz2idx', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--out_dir', type=str, default='data/processed')
    args = parser.parse_args()

    config = load_config(args.config)
    centers, biz2grid = build_grid(args.biz_json, args.biz2idx, config)
    save_arrays(centers, biz2grid, args.out_dir)