# Fire Insurance Premium Prediction
### Crédit Agricole Assurances : Agricultural Multi-Risk Insurance

> **Leaderboard RMSE: 5,603** | Master's Applied Machine Learning 2024–2025

---

## Brief Summary

A full end-to-end machine learning pipeline that predicts fire insurance premiums for agricultural firms using a two-stage frequency-severity model (XGBoost), achieving a leaderboard RMSE of **5,603** - within **0.16%** of the best submission in the course competition.

---

## Overview

This project follows the complete data science lifecycle:

1. **Exploratory Data Analysis** - understanding 383,610 records across 373 features
2. **Unsupervised Learning** - PCA + K-Means clustering to identify risk segments
3. **Supervised Modelling** - two-stage XGBoost frequency-severity model
4. **Evaluation** - 5-fold cross-validation + leaderboard scoring
5. **Deployment Design** - REST API microservice architecture proposal

The final premium is computed as:

```
CHARGE = FREQ × CM × ANNEE_ASSURANCE
```

where **FREQ** = predicted fire frequency (events/year) and **CM** = predicted average claim cost (€).

---

## Problem Statement

Crédit Agricole Assurances needs to accurately price fire insurance premiums for agricultural clients. Underpricing creates financial exposure; overpricing loses clients. The challenge is predicting premiums from 373 features covering farm location, weather, property type, and prevention measures - with **99.25% of policies having zero fire claims** (extreme class imbalance).

---

## Dataset

| Attribute | Value |
|---|---|
| Training observations | 383,610 |
| Test observations | 95,852 |
| Total features | 373 |
| Numerical features | 27 |
| Binary features | 11 |
| Categorical features | 332 |
| Target: FREQ mean | 0.0125 fires/year |
| Target: CM mean (non-zero) | €29,775 |
| Target: CHARGE mean | €186 |
| Zero-claim policies | 99.25% |

> Dataset provided by Crédit Agricole Assurances via the course competition platform. Raw data files are not included in this repository due to very large size.

---

## Tools and Technologies

- Language Python (3.12)
- Data processing (pandas, numpy)
- Machine learning (scikit-learn, XGBoost)
- Dimensionality reduction (sklearn PCA)
- Clustering (sklearn, KMeans)
- Visualisation (matplotlib, seaborn)
- Environment (Google Colab (T4 GPU))
- Github

---

## Methods

### Preprocessing
- Removed constant columns (zero variance) and duplicate rows
- Auto-classified features into numerical, binary, and categorical types
- Median imputation for numerical; `"Missing"` category for categorical
- Ordinal encoding for all 332 categorical features
- Clipped negative target values to zero

### Unsupervised Learning
- **PCA** on 27 standardised numerical features - 13 components explain 90% variance
- **K-Means clustering** (k=5, selected via elbow method) - 5 distinct risk segments identified

### Model Comparison
Three models were evaluated for the frequency stage:

- Ridge Regression - 6,799.7 ± 86.9
- Random Forest - 6,800.5 ± 86.4 
- **XGBoost** | **6,799.0 ± 87.3**

XGBoost selected for its Poisson objective, GPU acceleration, and best RMSE.

### Final Model - Two-Stage XGBoost

**Stage 1 - Frequency Model (FREQ)**
```python
XGBRegressor(
    objective     = "count:poisson",
    n_estimators  = 1200,
    max_depth     = 6,
    learning_rate = 0.05,
    subsample     = 0.8,
    colsample_bytree = 0.8
)
# Trained on ALL rows (including zero-claim policies)
```

**Stage 2 - Severity Model (CM)**
```python
XGBRegressor(
    objective     = "reg:squarederror",   # on log1p(CM)
    n_estimators  = 1500,
    max_depth     = 8,
    learning_rate = 0.03,
    subsample     = 0.8,
    colsample_bytree = 0.8
)
# Trained ONLY on policies with observed claims (FREQ > 0)
# Predictions back-transformed via expm1()
```

### Actuarial Design Principles
- **Non-negativity** enforced via `np.maximum(..., 0)` on all predictions
- **Exposure offset** - `ANNEE_ASSURANCE` multiplied at the final charge stage
- **Tail risk** - Poisson and log-normal objectives handle right-skewed distributions without manual outlier treatment
- **Claim-only severity** - CM model trained only on observed claims to avoid zero-cost bias

---

## Key Insights

### Risk Segmentation (K-Means Clusters)

| Cluster | Mean FREQ | Mean CHARGE (€) | Risk Level |
|---|---|---|---|
| 0 | 0.009 | 115 | Low |
| 1 | 0.007 | 84 | Lowest |
| **2** | **0.059** | **610** | **High** |
| 3 | 0.025 | 485 | Medium-High |
| 4 | 0.015 | 112 | Low-Medium |

→ **Cluster 2** farms are 8× more likely to have a fire and pay 7× higher premiums than Cluster 1.

### Top Predictive Features (FREQ model)
1. `NBBAT2` - Number of buildings (type 2)
2. `NBBAT10` - Number of buildings (type 10)
3. `ANNEE_ASSURANCE` - Policy duration
4. `RISK1` - Risk classification score
5. `Cluster` - K-Means risk segment

---

## Results & Conclusion

| Metric | Value |
|---|---|
| 5-Fold CV RMSE | 6,790 ± 338 |
| **Leaderboard RMSE** | **5,603** |
| Best competing score | 5,594 |
| Gap from 1st place | 0.16% |

The two-stage frequency-severity decomposition with XGBoost achieved strong generalisation, with the leaderboard score significantly better than the CV estimate - indicating the model is not overfitting. The 99.25% sparsity of fire claims was the central modelling challenge, addressed by separating frequency and severity estimation and using actuarially appropriate loss functions.

---

## How to Run This Project

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### Data Setup
Place the following files in a folder (e.g. `data/`):
```
data/
├── train_input_Z61KlZo.csv
├── train_output_DzPxaPY.csv
└── test_input_5qJzHrr.csv
```

### Run
Open `DS_Project_main.ipynb` in Google Colab or Jupyter and run all cells in order.

> **Recommended:** Use Google Colab with a T4 GPU runtime for ~15 min total training time. CPU runtime will take 2–3 hours.

### Output
The notebook generates `submission.csv` with columns:
```
FREQ | CM | CHARGE
```

---

## Future Work

- **LightGBM / CatBoost** - native categorical handling without ordinal encoding
- **Hurdle model** - explicit zero-inflation modelling with a binary classifier gate
- **Ensemble stacking** - combine FREQ×CM and direct CHARGE models
- **External data** - satellite fire risk indices, crop calendars, climate projections
- **Temporal features** - year-on-year claim history per policyholder

---

## Author & Contact

**RISHABH KUMAR**
email: rknith16@gmail.com
linkedin: www.linkedin.com/in/rishabhnith

> MSc Data Analytics and AI · EDHEC Business School · 2024 - 2025

---
