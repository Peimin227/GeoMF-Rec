f
# GeoMF Recommender System

A location-aware matrix factorization model implementation using the Yelp dataset.

## Structure

- `data/`: Raw and processed data files
- `config/`: Configuration for grid size, sigma, paths, etc.
- `src/`: Source code organized by module (data processing, model, training, etc.)
- `notebooks/`: Jupyter notebooks for EDA and experiment analysis
- `scripts/`: Shell scripts for one-click data pipeline and training

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/preprocess.sh
bash scripts/train.sh