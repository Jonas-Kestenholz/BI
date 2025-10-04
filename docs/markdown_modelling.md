# 🛍️ Modeling Shopping Behaviours — Flow Document

##  Purpose
Automate **data cleaning, supervised learning (classification & regression), unsupervised learning (clustering), and inference** for shopping trends data.  
Outputs: trained models, evaluation metrics, and plots.

---

##  Workflow Overview

### 1. Data Loading & Cleaning
**Input**:  
- `data/shopping_trends.csv` (auto-locates if in repo tree)  

**Process**:
- Clean column names (lowercase, underscores).
- Convert numerics (`age`, `purchase_amount_usd`, `review_rating`, `previous_purchases`).
- Standardize Yes/No flags.
- Normalize purchase frequencies → derive `frequency_per_year`, `frequency_days_between`.
- Clip review ratings to 1–5.
- Drop duplicates and rows with missing purchase amounts.  

**Feature Engineering**:
- `spend_band`: quartiles of purchase amount (Low, Mid, High, Top).  
- Binary versions of Yes/No columns.  
- `loyalty_index`: subscription × previous purchases.  

**Output**:  
- Cleaned dataset → `data/processed/shopping_trends_clean_onehot.csv`.

---

### 2. Classification (Spend Band Prediction)
**Goal**: Predict `spend_band` (Low/Mid/High/Top).  

**Process**:
- Preprocess: scale numeric features, one-hot encode categorical features.  
- Train models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naïve Bayes
  - KNN
- Select best model using **validation F1 (weighted)**.  
- Evaluate with **cross-validation** and test set metrics.  
- Generate reports: classification report + confusion matrix.  
- Export a small interpretable decision tree.  

**Outputs**:
- `reports/tables/clf_validation_results.csv`  
- `reports/tables/clf_test_metrics.json`  
- `reports/tables/clf_classification_report.csv`  
- `reports/figures/clf_confusion_matrix.png`  
- Best model → `models/clf_spend_band_best.joblib`  

**Optional**:
- Binary Top vs Other classifier (`clf_top_vs_other.joblib`).  
- Permutation importance (feature ranking).  

---

### 3. Regression (Purchase Amount Prediction)
**Goal**: Predict `purchase_amount_usd`.  

**Process**:
- Preprocess: scale + one-hot encode.  
- Train models:
  - Linear Regression
  - Ridge Regression
  - Random Forest Regressor
- Pick best model using validation RMSE.  
- Evaluate with **cross-validation** and test set metrics (RMSE, MAE, R², MAPE).  
- Plot true vs predicted purchase amounts.  

**Outputs**:
- `reports/tables/reg_validation_results.csv`  
- `reports/tables/reg_test_metrics.json`  
- `reports/figures/reg_true_vs_pred.png`  
- Best model → `models/reg_purchase_amount_best.joblib`  

---

### 4. Clustering (Customer Segmentation)
**Goal**: Segment customers based on behaviours.  

**Process**:
- Use numeric (`age`, `review_rating`, `previous_purchases`) + categorical features.  
- Preprocess: scale + one-hot encode.  
- Run **KMeans** for K=2–10 → elbow & silhouette plots.  
- Select best K by silhouette score.  
- Visualize clusters with PCA (2D scatter).  
- Profile clusters: numeric means + top categories.  
- Compute Davies–Bouldin index.  

**Optional**:  
- Fit **Gaussian Mixture Model (GMM)** for softer clustering.  

**Outputs**:
- `reports/figures/cluster_elbow_inertia.png`  
- `reports/figures/cluster_silhouette.png`  
- `reports/figures/cluster_pca_scatter.png`  
- `reports/tables/cluster_sizes.csv`  
- `reports/tables/cluster_numeric_means.csv`  
- `reports/tables/cluster_top_categories.csv`  
- Preprocessor → `models/cluster_preprocessor.joblib`  
- KMeans model → `models/cluster_kmeans.joblib`  
- (Optional) GMM model → `models/gmm_segments.joblib`  

---

### 5. Inference on New Data
**Input**:  
- `data/new_customers_example.csv`  

**Process**:
- Load best classifier + regressor.  
- Predict spend band + purchase amount for each row.  

**Output**:  
- `reports/tables/new_data_predictions.csv`  

---

##  Output Directory Structure

reports/
│── figures/
│ ├── clf_confusion_matrix.png
│ ├── reg_true_vs_pred.png
│ ├── cluster_elbow_inertia.png
│ ├── cluster_silhouette.png
│ └── cluster_pca_scatter.png
│
│── tables/
│ ├── clf_validation_results.csv
│ ├── clf_test_metrics.json
│ ├── clf_classification_report.csv
│ ├── reg_validation_results.csv
│ ├── reg_test_metrics.json
│ ├── cluster_sizes.csv
│ ├── cluster_numeric_means.csv
│ ├── cluster_top_categories.csv
│ └── new_data_predictions.csv
│
models/
│── clf_spend_band_best.joblib
│── clf_top_vs_other.joblib (optional)
│── reg_purchase_amount_best.joblib
│── cluster_preprocessor.joblib
│── cluster_kmeans.joblib
│── gmm_segments.joblib (optional)
