# scripts/eda_shopping_behaviours.py
# -*- coding: utf-8 -*-
"""
Shopping Behaviours — Sprint 2: Data Prep + EDA
- Loads CSV
- Cleans & coerces types
- Descriptive stats & quality checks
- Visualizations: distributions, counts, box/violin, scatter, correlations
- Saves figures and tables under /reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORT_TAB_DIR = os.path.join(BASE_DIR, "reports", "tables")

os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(REPORT_TAB_DIR, exist_ok=True)

# -------------------------
# Load
# -------------------------
def load_shopping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# -------------------------
# Clean
# -------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # standardize headers to snake_case
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # expected columns (based on your dataset)
    # customer_id, age, gender, item_purchased, category, purchase_amount_usd,
    # location, size, color, season, review_rating, subscription_status,
    # shipping_type, discount_applied, promo_code_used, previous_purchases,
    # payment_method, frequency_of_purchases

    # coerce numeric
    for col in ["age", "purchase_amount_usd", "review_rating", "previous_purchases"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # strip whitespace in strings
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype(str).str.strip()

    # normalize simple yes/no fields
    normalize_yes_no = {
        "discount_applied": {"y": "Yes", "n": "No"},
        "promo_code_used": {"y": "Yes", "n": "No"},
        "subscription_status": {"active": "Yes", "inactive": "No"}
    }
    for c, mapping in normalize_yes_no.items():
        if c in out.columns:
            out[c] = out[c].str.title()
            out[c] = out[c].replace(mapping)

    # drop fully-empty rows and exact duplicates
    out = out.dropna(how="all").drop_duplicates()

    # key column we need for spend analysis
    out = out.dropna(subset=["purchase_amount_usd"])

    return out

# -------------------------
# Utilities
# -------------------------
def savefig(name: str):
    plt.tight_layout()
    path = os.path.join(REPORT_FIG_DIR, name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Figure] {path}")

def savetab(df: pd.DataFrame, name: str):
    path = os.path.join(REPORT_TAB_DIR, name)
    df.to_csv(path, index=False)
    print(f"[Table]  {path}")

# -------------------------
# Quality + Profile Tables
# -------------------------
def dataset_profile(df: pd.DataFrame):
    # basic shape
    shape = pd.DataFrame({"rows": [df.shape[0]], "cols": [df.shape[1]]})
    savetab(shape, "shape.csv")

    # dtypes
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ["column", "dtype"]
    savetab(dtypes, "dtypes.csv")

    # missing
    missing = df.isnull().sum().reset_index()
    missing.columns = ["column", "missing"]
    savetab(missing, "missing_counts.csv")

    # nunique
    nunique = df.nunique().reset_index()
    nunique.columns = ["column", "unique_values"]
    savetab(nunique, "nunique_counts.csv")

    # numeric describe
    numdesc = df.select_dtypes(include=[np.number]).describe().T.reset_index()
    numdesc = numdesc.rename(columns={"index": "column"})
    savetab(numdesc, "numeric_describe.csv")

# -------------------------
# Plot helpers
# -------------------------
def plot_numeric_dist(df: pd.DataFrame, col: str, bins: int = 25):
    plt.figure(figsize=(8,5))
    sb.histplot(df[col].dropna(), bins=bins, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    savefig(f"dist__{col}.png")

def plot_count(df: pd.DataFrame, col: str, top_n: int = None, orient: str = "v"):
    counts = df[col].value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)

    plt.figure(figsize=(10,5))
    if orient == "v":
        sb.barplot(x=counts.index, y=counts.values)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(col)
        plt.ylabel("Count")
    else:
        sb.barplot(y=counts.index, x=counts.values)
        plt.ylabel(col)
        plt.xlabel("Count")
    plt.title(f"{col} counts")
    savefig(f"count__{col}.png")

def plot_box_by_cat(df: pd.DataFrame, y: str, x: str, rotate_x: bool = False):
    plt.figure(figsize=(10,5))
    sb.boxplot(data=df, x=x, y=y)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.title(f"{y} by {x}")
    savefig(f"box__{y}__by__{x}.png")

def plot_violin_by_cat(df: pd.DataFrame, y: str, x: str, rotate_x: bool = False):
    plt.figure(figsize=(10,5))
    sb.violinplot(data=df, x=x, y=y, cut=0, inner="quartile")
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.title(f"{y} by {x} (violin)")
    savefig(f"violin__{y}__by__{x}.png")

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None):
    plt.figure(figsize=(8,5))
    sb.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
    title = f"{y} vs {x}" + (f" by {hue}" if hue else "")
    plt.title(title)
    savefig(("scatter__" + y + "_vs_" + x + (f"__by__{hue}" if hue else "") + ".png"))

def plot_corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10,7))
    sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap (numeric)")
    savefig("corr_heatmap_numeric.png")

def plot_stacked_share(df: pd.DataFrame, cat_main: str, cat_sub: str, top_n: int = 8):
    top = df[cat_main].value_counts().head(top_n).index
    d = df[df[cat_main].isin(top)]
    ct = pd.crosstab(d[cat_main], d[cat_sub], normalize="index")
    ct = ct.sort_index()
    ct.plot(kind="bar", stacked=True, figsize=(11,6))
    plt.ylabel("Share")
    plt.title(f"{cat_sub} share within {cat_main} (top {top_n})")
    plt.legend(title=cat_sub, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(f"stacked_share__{cat_main}__{cat_sub}.png")

# -------------------------
# Optional: Light FE for later Sprints
# -------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # spending bands (for classification later)
    if "purchase_amount_usd" in out.columns:
        out["spend_band"] = pd.qcut(out["purchase_amount_usd"], 4, labels=["Low","Mid","High","Top"])

    # freq ordering if present
    if "frequency_of_purchases" in out.columns:
        order = ["Daily","Weekly","Fortnightly","Bi-Weekly","Monthly","Quarterly","Annually"]
        present = [x for x in order if x in out["frequency_of_purchases"].unique().tolist()]
        out["frequency_of_purchases"] = pd.Categorical(out["frequency_of_purchases"], categories=present, ordered=True)

    # binary maps
    for c in ["discount_applied", "promo_code_used", "subscription_status"]:
        if c in out.columns:
            out[c + "_bin"] = out[c].map({"Yes":1, "No":0})

    return out

# -------------------------
# Run EDA
# -------------------------
def run():
    sb.set(style="whitegrid", context="talk")

    path = os.path.join(DATA_DIR, "shopping_trends.csv")
    df = load_shopping(path)
    df = clean(df)
    df = engineer_features(df)

    # profile tables
    dataset_profile(df)

    # distributions
    for col in ["age", "purchase_amount_usd", "review_rating", "previous_purchases"]:
        if col in df.columns:
            plot_numeric_dist(df, col)

    # categorical counts
    cat_cols = [
        "gender", "category", "item_purchased", "location", "size", "color", "season",
        "subscription_status", "shipping_type", "discount_applied", "promo_code_used",
        "payment_method", "frequency_of_purchases"
    ]
    for c in [x for x in cat_cols if x in df.columns]:
        orient = "h" if df[c].nunique() > 8 else "v"
        plot_count(df, c, orient=orient)

    # purchase amount vs key categories
    y = "purchase_amount_usd"
    for c in ["gender","category","season","subscription_status","discount_applied",
              "promo_code_used","payment_method","frequency_of_purchases","location","size","color"]:
        if c in df.columns and y in df.columns:
            rotate = df[c].nunique() > 8
            plot_box_by_cat(df, y, c, rotate_x=rotate)
            if df[c].nunique() <= 12:
                plot_violin_by_cat(df, y, c, rotate_x=rotate)

    # numeric correlations
    plot_corr_heatmap(df)

    # stacked shares: useful BI views
    if "season" in df.columns and "payment_method" in df.columns:
        plot_stacked_share(df, "season", "payment_method", top_n=8)
    if "category" in df.columns and "discount_applied" in df.columns:
        plot_stacked_share(df, "category", "discount_applied", top_n=12)

    # summary tables: avg spend by dimension + top items by revenue
    pivots = []
    for c in ["category","season","gender","subscription_status","discount_applied",
              "promo_code_used","payment_method","frequency_of_purchases"]:
        if c in df.columns:
            piv = df.groupby(c)["purchase_amount_usd"].agg(["count","mean","median","std"]).reset_index()
            piv.insert(0, "dimension", c)
            pivots.append(piv)
    if pivots:
        spend_summary = pd.concat(pivots, ignore_index=True)
        savetab(spend_summary, "avg_spend_by_dimension.csv")

    if {"item_purchased","purchase_amount_usd"}.issubset(df.columns):
        top_items = (df.groupby("item_purchased")["purchase_amount_usd"]
                       .sum()
                       .sort_values(ascending=False)
                       .reset_index()
                       .rename(columns={"purchase_amount_usd":"revenue"}))
        savetab(top_items, "top_items_by_revenue.csv")

    print("Sprint 2 EDA complete.")

if __name__ == "__main__":
    run()