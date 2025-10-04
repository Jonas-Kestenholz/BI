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
I can see on the new plot though that the dots are kinda the same, just pressed more together. So even though the R2, RMSE and MAE was much better, it is still not perfect. 

After having used Random Forest to make a predictive model i wanted to do some market segmentation. I wanna do this to take the huge superstore data set and divide it into
customer groups. To do this i use Kmeans clustering. I select the features for the clustering. I use Sales_Log, Profit_YJ, Discount and Shipping_costs, these were chosen for their 
they create. I also chose Segment_Corporate, Segment_Home Office as they seemed they would be crusial in the group dividing, as we will see later is that they didnt really do much.
I also had Market_EU in it too. 
Now before i trained the model i made use of StandardScaler to prevent features with a larg magnitude like Sales to dominate and skewing the results.
I then use the Elbow method to get a plot to see what is the most optimal amount of clusters to use. It didnt really give me the inside i wanted and i couldnt draw a 
conclusion for the optimal amount of clusters to i also made use of Silhouette Score. With the Elbow method and comparing it to the silhouette score i could conclue that
5 clusters was the best amount to use. 

Now after training the model and using 5 clusters it provided me with this data.
Sales_Log,Profit_YJ,Discount,Quantity,ShippingCost_Log,Category_Technology,Market_EU,Segment_Corporate,Segment_Home Office
5.4869,30.87106,0.09776,4.87594,3.21382,0.0,0.2458,0.30303,0.18232
5.91753,37.37136,0.11715,3.35313,3.48956,1.0,0.20918,0.2978,0.17994
5.36203,28.47662,0.11763,3.62615,3.01694,1.0,0.19299,0.30121,0.17871
3.2679,-20.99112,0.55085,2.84044,1.28109,0.05254,0.12625,0.29989,0.18213
3.50086,9.67587,0.03829,2.52566,1.38906,0.00639,0.17842,0.29956,0.18365

So what can be seen on that data is that row 4 is the big looser. Profit_YJ is at a -20.9 and Discount is at a high 0.55. This segment is made of transactions with a very high discount
that equals a high loss on average. The conclution to draw from this is to limit the discounts to not suffer losses on sales. 
On row two we can see the highest number for Profit_YJ. So this group are the premium customers. Profit_YJ is on 37.37 and a Sales_log on 5.92 that is also the highest. They drive in
the most profit for the company and would be an obvious group to taget to try to hold on to and nudge into more sales. 
Now the last row have a low Profit_YJ but they also make use of the very lowest Discount. So these are the people that make small purchase but just buys it full out. 
A strategy for this group could be to try to upscale their purchase since they almost never make use of discount could they bring in a heafty profit. 
The last two rows are the average customer. They use the average amount of discount and have good Profit_YJ. Nothing really needs to be done on those. Good profit low discount usage.

Now that the data is there and have been analized i can see that it was very good to have included Discount and Profit_YJ, but Segment_Corporate and Segment_Home Office didnt really
make a difference. They had a maximum difference on 0.06 and 0.04.

Now for my last trick i will make use of Gaussian Naive Bayes to try to predict if a order will result in profit or not. 

The result i got were pretty darn good
--- Superstore Profit/Loss: Gaussian Naive Bayes Classification ---
Gaussian Naive Bayes Summary Metrics (Classes=2):
  Test Accuracy: 0.9146
  Test F1-Score (Macro Avg): 0.8835

So the Gaussian Naive Bayes model had an accuracy on 91%. So this means that we can with high accuracy predict if an order will result in profit or not. 
Now what this could be used for is to limit the discount usage on an order if it suddenly would result in a loss. 



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
