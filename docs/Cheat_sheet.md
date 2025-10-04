# Business Intelligence Exam Cheat Sheet

This document contains explanations for **all BI terms, metrics, and outputs** from Sprints 1–3, following the BI course syllabus.  
Use this file as a quick reference during study or project work.

---

## Sprint 1 — Python Basics & Data Foundations

### Concepts
- **Anaconda & Jupyter** → Python environment & interactive notebooks.  
- **Language syntax** → Python basics (loops, conditions, functions).  
- **Data structures**:  
  - `list` = ordered collection.  
  - `dict` = key-value pairs.  
  - `numpy array` = fast numeric data.  
  - `pandas dataframe` = table-like structure.  

### Data Types
- **Numeric**: int, float.  
- **Categorical**: strings (e.g., “Male”, “Female”).  
- **Datetime**: order dates, etc.  

### Visualisation Packages
- **Matplotlib** → Flexible plotting library.  
- **Seaborn** → Statistical plots (histogram, heatmap).  
- **Plotly** → Interactive & 3D plots.  
- **Folium** → Maps & geo data.  
- **PyGWalker** → BI-style data cube exploration.

---

## Sprint 2 — Data Ingestion, Cleaning & EDA

### Common Tables Produced
- `shape.csv` → Dataset size (#rows, #cols).  
- `dtype.csv` → Data types of each column.  
- `missing_counts.csv` → Missing values per column.  
- `nunique_counts.csv` → Number of unique values per column.  
- `numeric_describe.csv` → Stats summary for numerical columns.  
- `avg_spend_by_dimension.csv` → Aggregated averages by category, gender, etc.  

### Cleaning Techniques
- **Handling Missing Values** → drop rows, fill with mean/median/mode.  
- **Wrong/Damaged Values** → check ranges, replace or drop.  
- **Binning** → grouping continuous values into ranges (e.g., age groups).  
- **Dropping Columns** → remove irrelevant info.  
- **Joining** → merging multiple dataframes.

### Visualisation
- **Scatter plot** → relationship between two variables.  
- **Bar chart** → categorical comparison.  
- **Pie chart** → proportions.  
- **Histogram** → distribution of one variable.  
- **Boxplot** → median, quartiles, and outliers.  
- **Heatmap** → matrix of correlations.

### Statistics
- **Central tendency** → Mean, Median, Mode.  
- **Dispersion** → Min, Max, Range, Std Dev, Variance, Quartiles, Outliers.  
- **Correlation** → Relationship between two variables.  
- **Covariance** → How two variables vary together.

---

## Sprint 3 — Data Modelling & Advanced BI

### Classification (Spend Band)
- **Tables**
  - `clf_validation_results.csv` → Model comparison (F1/Accuracy).  
  - `clf_crossval_summary.json` → CV averages.  
  - `clf_test_metrics.json` → Test set results.  
  - `clf_classification_report.csv` → Precision/Recall/F1 per class.  
  - `clf_binary_test_metrics.json` → Binary “Top spender” test results.  
- **Figures**
  - `clf_confusion_matrix.png` → Where predictions go wrong.  
  - `decision_tree.pdf` → Visual decision tree.  
- **Terms**
  - **Precision** = TP / (TP+FP).  
  - **Recall** = TP / (TP+FN).  
  - **F1 Score** = Balance of precision & recall.  
  - **Weighted F1** = Accounts for class size.  
  - **Macro F1** = Treats all classes equally.  

### Regression (Purchase Amount)
- **Tables**
  - `reg_validation_results.csv` → Validation errors for models.  
  - `reg_crossval_summary.json` → CV averages.  
  - `reg_test_metrics.json` → Test metrics (RMSE, MAE, R², MAPE%).  
- **Figures**
  - `reg_true_vs_pred.png` → Predicted vs actual scatter.  
- **Terms**
  - **MAE** = avg absolute error.  
  - **RMSE** = square-root of avg squared error.  
  - **R²** = how much variance explained.  
  - **MAPE** = % error, business friendly.  

### Clustering (KMeans)
- **Tables**
  - `cluster_k_grid_metrics.json` → Inertia + Silhouette for many K.  
  - `cluster_extra_metrics.json` → Best K and DBI score.  
  - `cluster_sizes.csv` → How many per cluster.  
  - `cluster_numeric_means.csv` → Avg numbers per cluster.  
  - `cluster_top_categories.csv` → Top categorical values per cluster.  
- **Figures**
  - `cluster_elbow_inertia.png` → Elbow method for K.  
  - `cluster_silhouette.png` → Silhouette per K.  
  - `cluster_pca_scatter.png` → Clusters in 2D PCA.  
- **Terms**
  - **Inertia** = compactness of clusters (lower is better).  
  - **Elbow Method** = find inflection point in inertia curve.  
  - **Silhouette Score** = [-1,1], higher = better separation.  
  - **Davies–Bouldin Index** = lower is better.  
  - **PCA** = dimensionality reduction for visualisation.  

### Generative (Gaussian Mixture Models)
- **Tables**
  - `gmm_assignments.csv` → Each customer’s cluster + log-likelihood.  
- **Terms**
  - **GMM** = probabilistic clustering method (soft clusters).  
  - **Log-likelihood** = fit score (higher = better).  

---

## Glossary of BI Terms

- **Feature Engineering** → Creating new variables (e.g., loyalty_index).  
- **One-hot encoding** → Converting categories into 0/1 columns.  
- **Label encoding** → Replacing categories with numbers.  
- **Normalization** → Rescaling numbers to 0–1 range.  
- **Standardization** → Center to mean 0, std 1.  
- **Cross-validation (CV)** → Train/test split repeated across folds.  
- **Overfitting** → Model too tailored to training data, bad generalization.  
- **Underfitting** → Model too simple, misses patterns.  
- **Supervised Learning** → Has target labels (classification/regression).  
- **Unsupervised Learning** → No labels (clustering, dimensionality reduction).  
- **Generative Models** → Probabilistic models that can simulate data.  
- **Dashboard** → Interactive BI report with multiple charts.  
- **Data Cube** → Multi-dimensional data view (e.g., sales by region by product).  

---

# Exam Strategy

- Sprint 1 = Python basics + data structures.  
- Sprint 2 = Cleaning + EDA + visualization + statistics.  
- Sprint 3 = ML models (classification, regression, clustering, GMM).  
- Know **what each metric means** and **which method applies when**.  
