# Sprint 2 — Data Preparation & Exploration

## Data Collection
- Source: Kaggle *Shopping Behaviours* dataset (raw CSV).
- Files: `data/shopping_trends.csv`.
- Source: https://community.tableau.com/s/question/0D54T00000C5vSDSAZ/global-superstore-data-file
- File `Global Superstore.xls`

## Cleaning & Integration (ETL/ELT)
- Standardized headers to `snake_case`.
- Coerced numeric types: `age`, `purchase_amount_usd`, `review_rating`, `previous_purchases`.
- Normalized yes/no-like fields: `discount_applied`, `promo_code_used`, `subscription_status`.
- Dropped duplicates and rows missing `purchase_amount_usd`.
- Saved cleaned data: `data/processed/shopping_trends_clean.csv`.

So for cleaning the superstore data i first looked at the head of the data to get a feel for it. I look at the col names and look at the shape.
With that i can see that some of the columns are not relevant for the analysis so i drop them from the dataframe. 
`data = data.drop(columns=['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name'])`

I check the data for missing data and could see that postal codes was missing alot of data, so i drop that one also.
I the column for date its in 12-04-13. I decide to pull them out and make columns of their own if i wanted to check for
things on days or mounths or something. Like what day has the most sales or something.

I look some more at the data like top categories and top regions. I make some boxplots for some of the numerial data
`Profit', 'Sales', 'Shipping Cost`
With that i can see that there are some clear outliers. I make use of the IQR method to cap them so they dont skew the data.
I make some boxplots again of the new data to visualise the data¨

I then make some histograms to look at the datas formation. I can see that it is kinda skewed. 
I make use of log1p on sales and shipping cost. Profit however makes use of negative numbers, so instead i use Yeo-Johnson.
This will stabelize the data and make it better suited for machine learning. 

Last but not least, i do some one hot encoding to get rid of all the strings in the data. But before that i check the columns for unique values
in each of them. I dont want to have a ridiculous amount of columns. With this i drop the columns with over 100 unique vales and even some with thousands.


## Feature Engineering (for Sprint 3)
- `spend_band` = quartiles of `purchase_amount_usd` (Low/Mid/High/Top).
- Binary flags: `*_bin` for yes/no variables.
- `loyalty_index` = `subscription_status_bin * previous_purchases`.

## Quality Assurance
- **Relevance**: kept features aligned with spend/loyalty/segments.
- **Sufficiency**: dataset overview (`reports/tables/dataset_overview.json`) + segment coverage (counts by category/season/subscription if computed).
- **Structure**: tidy tabular format, ready for BI/ML.
- **Completeness**: `missing_counts.csv` (if profiled), duplicates removed.
- **Dimensionality**: created bands, binary flags; numeric scales standardized in modelling pipelines.

## EDA (highlights)
- Distributions: age, purchase amount, rating, previous purchases.
- Categorical counts: gender, category, item, season, payment, discounts, subscription.
- Relationships: box/violin plots of purchase amount by key categories.
- Numeric correlation heatmap.

## Artifacts
- Figures → `reports/figures/`
            `figures/superstore_figures`
- Tables  → `reports/tables/`
- Clean data → `data/processed/shopping_trends_clean.csv`
