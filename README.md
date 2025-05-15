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

Use `train.py` with the following options:

| Argument         | Description                              | Default |
|------------------|------------------------------------------|---------|
| `--sample_users` | Subsample first N users for debugging    | —       |
| `--sample_items` | Subsample first N items for debugging    | —       |
| `--K`            | Latent factor dimension                  | 50      |
| `--max_iter`     | Number of training epochs                | 20      |
| `--gamma`        | L2 regularization coefficient (γ)        | 0.01    |
| `--lam`          | L1 regularization coefficient (λ)        | 0.1     |
| `--eta`          | Learning rate                            | 1e-3    |
| `--batch_size`   | Mini-batch size                          | 4096    |
| `--num_negatives`| Negative samples per user per batch      | 5       |
| `--lr_step_size` | Epoch interval for learning rate decay   | 10      |
| `--lr_gamma`     | Learning rate decay factor               | 0.5     |

Example (1000×1000 subset):

```bash
python train.py \
  --sample_users 1000 \
  --sample_items 1000 \
  --K 100 \
  --max_iter 40 \
  --batch_size 4096 \
  --num_negatives 5 \
  --lr_step_size 10 \
  --lr_gamma 0.5
```

Training shows per-epoch and per-batch progress bars, and logs the training loss.

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
```