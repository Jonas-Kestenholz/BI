# Sprint 3 — Data Modelling

## Objectives
- **Classification**: predict `spend_band` (quartiles of `purchase_amount_usd`).
- **Regression**: predict exact `purchase_amount_usd`.
- **Unsupervised**: KMeans segmentation; **Generative**: Gaussian Mixture (GMM).
- **Inference**: run models on new data.

## Methods
We took the cleaned data of the 'global_superstore_cleaned.pkl' and did a correlation matrix on profits to what columns have a strong corralation with profits.
It can be seen that 'Sales_Log', 'ShippingCost_Log', 'Quantity', 'Category_Technology' and 'Sub-Category_Copiers' have the highest corralations at 0.42, 0.40, 0.16, 0.16 and 0.12 respectively
Sales got a hih corralation because the more sales there is the more profit. Bigger shipping cost usuallymean bigger shipments that give bigger profits. Quantity is the more you buy the more profit they get. And Technology and Copiers are the most expensive and most profitiable.

In the other end on the negative side we have 'Discount' at -0.59. It is at minus because discount takes the profits down. So it results in more losses.
So i have decided to make the linear regression with these columns. 

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
- Regression: Best model = … | Test RMSE = …, MAE = …, MSE
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
