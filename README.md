[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

# GeoMF-Rec

Point-of-Interest recommendation via GeoMF (Geographical Matrix Factorization)  
Supports PyTorch mini-batch BPR training, multi-negative sampling, and learning rate scheduling.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** This project runs on CPU by default (no GPU required).

---

## Download Data

1. Go to the Yelp Academic Dataset page:  
   https://www.yelp.com/dataset  
2. Download the latest `yelp_academic_dataset.zip`.  
3. Unzip and copy the JSON files into `data/raw/`:
   ```bash
   unzip yelp_academic_dataset.zip -d data/raw/
4. Since,  `yelp_academic_review.json` and `yelp_academic_tips.json` have similar meaning to our project, we drop the `yelp_academic_tips.json`

## Data Preparation

Place the Yelp JSON files under `data/raw/`:
```
data/raw/
  ├── business.json
  ├── review.json
  ├── checkin.json
  ├── user.json
  └── tip.json    # optional
```

1. **Construct interaction matrices**  
   ```bash
   python src/data/inter_matrix.py \
     --review data/raw/review.json \
     --checkin data/raw/checkin.json \
     --out-dir data/processed
   ```
   Generates `R.npz` (user×item ratings) and `W.npz` (weight matrix).

2. **Generate geographic grid and influence matrix**  
   Adjust `config/default.yaml` grid parameters as needed:
   ```yaml
   grid:
     delta_lat: 0.1
     delta_lon: 0.1
     sigma:     0.2
     radius:    3
   ```
   Then run:
   ```bash
   python src/data/grid.py \
     --biz_json data/raw/business.json \
     --biz2idx data/processed/biz2idx.json \
     --config config/default.yaml \
     --out_dir data/processed

   python src/features/influence.py \
     --biz2idx  data/processed/biz2idx.json \
     --centers  data/processed/grid_centers.npy \
     --biz2grid data/processed/biz2grid.npy \
     --config   config/default.yaml \
     --out      data/processed/Y.npz
   ```
   *Optional:* For subset compression, use `src/data/compress_Y.py`.

3. **Split into train/test sets**  
   ```bash
   python src/data/split.py \
     --R data/processed/R.npz \
     --W data/processed/W.npz \
     --out_dir data/processed \
     --test_ratio 0.3 \
     --seed 123
   ```
   Produces `R_train.npz`, `R_test.npz`, and `W_train.npz`.

---

## Model Training

The project now uses a two-phase hybrid training:
1. **Strict GeoMF** (ALS + projected gradient) as per the paper.
2. **BPR fine-tuning** (mini-batch, multi-negative sampling) for ranking.

Use `train.py` with the following options:

| Argument           | Description                                           | Default |
|--------------------|-------------------------------------------------------|---------|
| `--sample_users`   | Subsample first N users for debugging                 | —       |
| `--sample_items`   | Subsample first N items for debugging                 | —       |
| `--K`              | Latent factor dimension                               | 50      |
| `--max_iter`       | Alternating ALS+PG iterations for strict GeoMF        | 20      |
| `--gamma`          | L2 regularization coefficient (γ) for P and Q         | 0.01    |
| `--lam`            | L1 regularization coefficient (λ) for X               | 0.1     |
| `--eta`            | Learning rate for X projected gradient                | 1e-3    |
| **BPR fine-tuning**|||| 
| `--bpr_epochs`     | Number of BPR fine-tuning epochs                      | 5       |
| `--bpr_lr`         | Learning rate for BPR optimizer                       | 1e-3    |
| `--bpr_neg`        | Negative samples per user in BPR                      | 5       |
| `--bpr_batch`      | Batch size for BPR DataLoader                         | 256     |
| `--bpr_workers`    | Number of DataLoader worker processes                 | 4       |

Example (5k×5k subset, hybrid training):

```bash
python train.py \
  --sample_users 5000 \
  --sample_items 5000 \
  --K 100 \
  --max_iter 20 \
  --gamma 0.01 \
  --lam 0.1 \
  --eta 1e-3 \
  --bpr_epochs 5 \
  --bpr_lr 1e-3 \
  --bpr_neg 5 \
  --bpr_batch 512 \
  --bpr_workers 4
```

The script first runs strict GeoMF (ALS + PG with `tqdm` progress bar), then performs BPR fine-tuning using a multi-worker `DataLoader` for efficient negative sampling and gradient updates.

---

## Model Evaluation

Use `evaluate.py` with options:

```bash
python evaluate.py \
  --sample_users 2000 \
  --sample_items 2000 \
  --K 5 10 20 50 100
```

Outputs Recall@K and Precision@K for specified K values.

---

## Hyperparameter Tuning & Scaling

1. **Grid Search** on a mid-scale subset (e.g., 2000×2000):
   ```
   K ∈ {50,100,200}, γ ∈ {1e-3,1e-2,1e-1}, λ ∈ {1e-2,1e-1,1}, η ∈ {1e-4,1e-3}
   ```
2. **Compress Y** for each subset via `src/data/compress_Y.py`.
3. **Increase num_negatives** for stronger ranking signals.
4. **Adjust batch_size** to balance noise and memory.
5. **Two-stage retrieval**: coarse P·Q retrieval → GeoMF-BPR reranking.

---

## Project Structure

```
GeoMF-Rec/
├── README.md
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   └── model/
├── train.py
├── evaluate.py
└── requirements.txt
```

---

## Contact

For issues or questions, please open an issue or contact the maintainer.  
zhenye2@ualberta.ca 


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
zhenye2@ualberta.ca