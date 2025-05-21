# GeoMF-Rec Documentation

## 1. Introduction

One of the most important concept of the GEOMF is the Points of Interest (POIs) and the spatial clustering phenomenon commonly observed in real-world location-based data.

**Points of Interest (POIs):**  
POIs are specific locations or venues—such as restaurants, shops, parks, or landmarks—that users can visit or interact with. Each POI has geographic coordinates (latitude and longitude) and various attributes (e.g., category, rating).

**Spatial Clustering Phenomenon:**  
In many urban and geographic contexts, POIs tend to cluster spatially. For example:  
- Restaurants often concentrate in dining districts.  
- Retail shops group within shopping malls or commercial streets.  
- Points such as tourist attractions form clusters in historical or scenic neighborhoods.

These clusters reflect underlying human behaviors, urban planning, and natural geography, and motivate incorporating geographic information into recommendation models. Thus, combining this principle with the concept of collaborative filtering(CF), GEOMF has been invented as a new location-based recommendation system.


<p align="center">
  <img src="formulas/prediction_formula.png" alt="Prediction Formula" width="400"/>
</p>

- **Collaborative Filtering Term ($p_u^T q_i$):**
  - Measures the alignment between user preferences and POI characteristics.
  - *Example:* If user *u* frequently visits cafes (*p_u* has high values in "cafe-related" dimensions) and POI *i* is a café (*q_i* aligns with those dimensions), this term will be high.

- **Geographical Term ($x_u^T y_i$):**
  - Quantifies the overlap between user *u*'s activity areas and POI *i*'s influence areas.
  - *Example:* If user *u* often appears in grid *l₁* (*x_u* has a high value at *l₁*) and POI *i* strongly influences *l₁* (*y_i* has a high value at *l₁*), this term will be high.

## 1.1 Critical features & Matrix

1. **R (User-POI Interaction Matrix)**  
   - **Dimensions:** M×N (M users, N POIs)  
   - **Elements:** r_u,i ∈ {0,1}, where  
     r_u,i = 1 if c_u,i > 0 (user visited POI)  
     r_u,i = 0 otherwise  
     (c_u,i is the visit count for user u and POI i)  
   - **Role:** Target matrix for training; model predicts for entries where r_u,i = 0.

2. **W (Weight Matrix)**  
   - **Dimensions:** M×N  
   - **Formula:** w_u,i = 1 + log(1 + c_u,i) if c_u,i > 0; otherwise 1  
   - **Role:** Assigns confidence weight to observed interactions.

3. **Y (POI Influence Area Matrix)**  
   - **Dimensions:** N×L (L spatial grids)  
   - **Formula:** y_i,l = (1/σ)·K(d(i,l)/σ)  
     - d(i,l) is the Euclidean distance between POI i and grid center l  
     - K(z) = (1/√(2π))·exp(−z²/2) (Gaussian kernel)  
   - **Role:** Models spatial influence of POIs.

4. **P (User Latent Factor Matrix)**  
   - **Dimensions:** M×K (K latent factors)  
   - **Update (ALS):**  
     p_u = (QᵀWᵤQ + γI)⁻¹ QᵀWᵤ (r_u − Yx_u)  
     where Wᵤ is diag(w_u,i), r_u is row u of R  
   - **Role:** Captures users’ latent preferences.

5. **Q (POI Latent Factor Matrix)**  
   - **Dimensions:** N×K  
   - **Update (ALS):**  
     q_i = (PᵀWᵢP + γI)⁻¹ PᵀWᵢ (r_i − Xy_i)  
     where Wᵢ is diag(w_u,i), r_i is column i of R  
   - **Role:** Captures POIs’ latent features.

6. **X (User Activity Area Matrix)**  
   - **Dimensions:** M×L  
   - **Update (Projected Gradient):**  
     x_u(t+1) = max(0, x_u(t) − η·grad_u)  
     grad_u = YᵀWᵤ (Yx_u − (r_u − Qp_u)) + λ·sign(x_u)  
   - **Role:** Models users’ spatial activity distribution with sparsity.

## 2. Symbol Summary Table

| Matrix | Dimensions | Definition                            | Formula                                  | Role                                           |
|--------|------------|---------------------------------------|------------------------------------------|------------------------------------------------|
| R      | M×N        | User–POI interaction matrix (0/1)     | r<sub>u,i</sub> = I(c<sub>u,i</sub> > 0)  | Target matrix for training.                    |
| W      | M×N        | Weight matrix                         | w<sub>u,i</sub> = 1 + log(1 + c<sub>u,i</sub>) | Reflects confidence in observed interactions. |
| Y      | N×L        | POI influence area matrix             | y<sub>i,l</sub> = (1/σ) K(d(i,l)/σ)       | Quantifies POIs’ spatial influence.           |
| P      | M×K        | User latent factor matrix             | Updated via ALS                          | Encodes users’ latent preferences.            |
| Q      | N×K        | POI latent factor matrix              | Updated via ALS                          | Encodes POIs’ latent attributes.              |
| X      | M×L        | User activity area matrix             | Updated via projected gradient descent   | Models users’ spatial activity patterns.      |

## 3. Overall Workflow Diagram

```mermaid
flowchart LR
  subgraph Data_Preparation
    A1[review.json & tip.json] -->|inter_matrix.py| R[R matrix]
    A1 -->|inter_matrix.py| W[W matrix]
    A2[business.json] -->|grid.py| G[Grid centers & biz2grid]
    G -->|influence.py| Y[Y matrix]
  end

  subgraph Data_Splitting
    R & W -->|split.py| R_train[Train R_train]
    R & W -->|split.py| R_test[Test R_test]
  end

  subgraph Phase_1_Strict_GeoMF
    R_train & W_train & Y -->|"train.py: GeoMFPTStrict.fit()<br/>ALS + PG with tqdm"| PQX[P, Q, X]
  end

  subgraph Phase_2_BPR_Fine_Tune
    R_train & PQX -->|"train.py: bpr_fine_tune()<br/>DataLoader + Adam"| PpQpXp[P', Q', X']
  end

  subgraph Evaluation
        PpQpXp & R_test -->|evaluate.py| Metrics["Recall@K / Precision@K"]
  end
```

---

## 4. Features Used and Their Roles

| Feature               | Source / Script         | Symbol / Shape | Description                                                       | Model Role                         |
|-----------------------|-------------------------|----------------|-------------------------------------------------------------------|------------------------------------|
| Review Interactions   | `inter_matrix.py`       | R (M×N)        | Binary user–item interactions derived from reviews & tips         | Ground truth for prediction        |
| Interaction Weights   | `inter_matrix.py`       | W (M×N)        | $1 + α·ln(1 + count_{ui})$ weighting per interaction              | Weights in weighted ALS            |
| Item Coordinates      | `business.json`         | —              | Item latitude/longitude mapped to grid centers                    | Building geographic matrix Y       |
| Geographic Grid       | `grid.py`               | —              | Partition area into L grids, output centers                       | Basis for geographic encoding      |
| Geographic Matrix     | `influence.py`          | Y (N×L)        | $Y_{i,c}=exp(-d(i,c)^2/(2σ^2))$                                   | Geo component in prediction        |
| Training Interactions | `split.py`              | R_train (M×N)  | Per-user leave-one-out split, at least one test interaction       | Input for GeoMF & BPR training     |
| Testing Interactions  | `split.py`              | R_test (M×N)   | Held-out test interactions                                        | Evaluation (Recall/Precision)      |
| User Latent Factors   | `geo_mf.py` (fit)       | P (M×K)        | Updated by ALS                                                    | CF component                       |
| Item Latent Factors   | `geo_mf.py` (fit)       | Q (N×K)        | Updated by ALS                                                    | CF component                       |
| User Geo Preferences  | `geo_mf.py` (fit)       | X (M×L)        | Updated by projected gradient + L1                                | Geo component                      |
| BPR Fine-tuning       | `train.py`              | —              | Pairwise ranking loss with multi-negative sampling and DataLoader | Improves Top-K ranking             |


