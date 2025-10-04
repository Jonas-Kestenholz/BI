# Sprint 3 — Data Modelling

## Objectives
- **Classification**: predict `spend_band` (quartiles of `purchase_amount_usd`).
- **Regression**: predict exact `purchase_amount_usd`.
- **Unsupervised**: KMeans segmentation; **Generative**: Gaussian Mixture (GMM).
- **Inference**: run models on new data.

## Methods
We took the cleaned data of the 'global_superstore_cleaned.pkl' and did a correlation matrix on profits to what columns have a strong corralation with profits.
It can be seen that 'Sales_Log', 'ShippingCost_Log', 'Quantity', 'Category_Technology' and 'Sub-Category_Copiers' have the highest corralations at 0.42, 0.40, 0.16, 0.16 and 0.12 respectively
Sales got a hih corralation because the more sales there is the more profit. Bigger shipping cost usually mean bigger shipments that give bigger profits. Quantity is the more you buy the more profit they get. And Technology and Copiers are the most expensive and most profitiable.

In the other end on the negative side we have 'Discount' at -0.59. It is at minus because discount takes the profits down. So it results in more losses.
So i have decided to make the linear regression with these columns. 
A correlation heatmap is also made to have some visiual data on. 

With that we know what columns would make sense to make lineær regression with. 
After training the model by splitting the data 80 to 20 we get some output to see how it did.
Linear Regression Results:
  Test R2: 0.4362
  Test RMSE: 28.8336
  Test MAE: 22.3515
  Test MSE: 831.3756
I also made a residual plot. What i could see and what my r^2 said was that it was not very accurate. The residual plot visually confirmed the lack of accuracy.
And if i look at the coefficients i can see that `discounts` is a very powerful negative feature at -95. It makes sense but the 
coefficient is extreme. For this reason i decided to make use of the Random Forest to maybe get a better result.
{
  "Sales_Log": 4.323330751515112,
  "ShippingCost_Log": 4.457774157758003,
  "Quantity": 0.854849493903937,
  "Category_Technology": 3.9708580891735394,
  "Sub-Category_Copiers": 4.565856760672779,
  "Discount": -95.6242898756402
}

Now that i made use a Random Forrest i got some much improved results
   Random Forest Regression Results:
      Test R2: 0.6511
      Test RMSE: 22.6820
      Test MAE: 13.7603
My R2 went from .43 to .65 a .22 increase. The RMSE got lower at 22.6 compared to 28.8 and the test MAE got lower as well. 
I can see on the new plot though that the dots are kinda the same, just pressed more together. So even though the R", RMSE and MAE was much better, it is still not perfect. 







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
