# Sprint 2 — Data Preparation & Exploration

## Data Collection
- Source: Kaggle *Shopping Behaviours* dataset (raw CSV).
- Files: `data/shopping_trends.csv`.

## Cleaning & Integration (ETL/ELT)
- Standardized headers to `snake_case`.
- Coerced numeric types: `age`, `purchase_amount_usd`, `review_rating`, `previous_purchases`.
- Normalized yes/no-like fields: `discount_applied`, `promo_code_used`, `subscription_status`.
- Dropped duplicates and rows missing `purchase_amount_usd`.
- Saved cleaned data: `data/processed/shopping_trends_clean.csv`.

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
- Tables  → `reports/tables/`
- Clean data → `data/processed/shopping_trends_clean.csv`
