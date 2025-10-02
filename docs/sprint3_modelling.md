# Sprint 3 — Data Modelling

## Objectives
- **Classification**: predict `spend_band` (quartiles of `purchase_amount_usd`).
- **Regression**: predict exact `purchase_amount_usd`.
- **Unsupervised**: KMeans segmentation; **Generative**: Gaussian Mixture (GMM).
- **Inference**: run models on new data.

## Methods
- **Classification**: Logistic Regression (balanced), Random Forest.
- **Regression**: Linear Regression, Random Forest Regressor.
- **Clustering**: KMeans (K=2..10) with Elbow + Silhouette; PCA 2D visualization.
- **Generative**: Gaussian Mixture Model (soft clustering with log-likelihoods).
- **Preprocessing**: `StandardScaler` (numeric), `OneHotEncoder` (categorical) via `ColumnTransformer`.
- **Validation**: Train/Val/Test split + 5-fold CV (best model).
- **Metrics**:
  - Classification: Accuracy, Weighted-F1, **Macro-F1**, per-class report + confusion matrix.
  - Regression: **RMSE, MAE, R², MAPE**.
  - Clustering: Silhouette (grid), **Davies–Bouldin** for chosen K.

## Key Findings (fill after run)
- Classification: Best model = … | Test F1 (weighted) = …, Macro-F1 = …
- Regression: Best model = … | Test RMSE = …, MAE = …, R² = …, MAPE = …%
- Clustering: Best K = … | Silhouette ≈ … | DBI ≈ … | Segment profiles: …
- GMM: Components = … | Soft memberships saved in `gmm_assignments.csv`.

## Improvements Explored
- Binary variant: Top vs Other classifier → F1/ACC = …
- Small RF grid search (optional) → best params: … | ΔF1 = …
- Next ideas: feature interactions (e.g., `discount * previous_purchases`), calibration, threshold tuning.

## Inference on New Data
- Input: `data/new_customers_example.csv`
- Output: `reports/tables/new_data_predictions.csv` with `pred_spend_band`, `pred_purchase_amount_usd`.

## Artifacts
- Figures → `reports/figures/`
- Tables/JSON → `reports/tables/`
- Models → `models/`
- Script → `scripts/modeling_shopping_behaviours.py`
