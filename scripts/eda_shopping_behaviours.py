# scripts/eda_shopping_behaviours.py
# -*- coding: utf-8 -*-
"""
EDA for Shopping Behaviours dataset
- Loads & cleans data
- Summary tables
- Core visualizations
- Saves figures and tables to /reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORT_TAB_DIR = os.path.join(BASE_DIR, "reports", "tables")

os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(REPORT_TAB_DIR, exist_ok=True)

# ---------- Load ----------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# ---------- Clean ----------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize column names -> snake_case
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Expected columns (from your description)
    # customer_id, age, gender, item_purchased, category, purchase_amount_usd,
    # location, size, color, season, review_rating, subscription_status,
    # shipping_type, discount_applied, promo_code_used,
    # previous_purchases, payment_method, frequency_of_purchases

    # Coerce dtypes where sensible
    numeric_like = ["age", "purchase_amount_usd", "review_rating", "previous_purchases"]
    for c in numeric_like:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Trim strings
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip()

    # Handle obvious typos (optional; adjust as needed)
    if "discount_applied" in out.columns:
        out["discount_applied"] = out["discount_applied"].str.title().replace({"Y":"Yes","N":"No"})
    if "promo_code_used" in out.columns:
        out["promo_code_used"] = out["promo_code_used"].str.title().replace({"Y":"Yes","N":"No"})
    if "subscription_status" in out.columns:
        out["subscription_status"] = out["subscription_status"].str.title().replace({"Active":"Yes","Inactive":"No"})

    # Drop full-NA rows and duplicates
    out = out.dropna(how="all").drop_duplicates()

    # Simple missing treatment (document decisions)
    out = out.dropna(subset=["purchase_amount_usd"])  # cannot analyze spend without this

    return out

# ---------- Helpers ----------
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

# ---------- EDA Tables ----------
def basic_profile(df: pd.DataFrame):
    shape = pd.DataFrame({"rows":[df.shape[0]], "cols":[df.shape[1]]})
    desc_num = df.select_dtypes(include=[np.number]).describe().reset_index()
    miss = df.isnull().sum().reset_index().rename(columns={"index":"column", 0:"missing"})
    nunique = df.nunique().reset_index().rename(columns={"index":"column", 0:"nunique"})

    savetab(shape, "shape.csv")
    savetab(desc_num, "numeric_describe.csv")
    savetab(miss, "missing_counts.csv")
    savetab(nunique, "nunique_counts.csv")

# ---------- Categorical Orders (optional) ----------
FREQ_ORDER = [
    "Daily", "Weekly", "Fortnightly", "Bi-Weekly", "Monthly", "Quarterly", "Annually"
]
def ordered_cat(series: pd.Series, order: list):
    present = [x for x in order if x in series.unique().tolist()]
    return pd.Categorical(series, categories=present, ordered=True)

# ---------- Plots ----------
def plot_numeric_dist(df: pd.DataFrame, col: str, bins: int = 20):
    plt.figure(figsize=(8,5))
    sb.histplot(df[col].dropna(), bins=bins, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    savefig(f"dist__{col}.png")

def plot_count(df: pd.DataFrame, col: str, top_n: int = None, orient="v"):
    vc = df[col].value_counts(dropna=False)
    if top_n:
        vc = vc.head(top_n)
    plt.figure(figsize=(10,5))
    if orient == "v":
        sb.barplot(x=vc.index, y=vc.values)
        plt.xticks(rotation=45, ha="right")
    else:
        sb.barplot(y=vc.index, x=vc.values)
    plt.title(f"{col} count")
    plt.xlabel(col if orient=="v" else "Count")
    plt.ylabel("Count" if orient=="v" else col)
    savefig(f"count__{col}.png")

def plot_box_by_cat(df: pd.DataFrame, num: str, cat: str, rotate_x=False):
    plt.figure(figsize=(10,5))
    sb.boxplot(data=df, x=cat, y=num)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.title(f"{num} by {cat}")
    savefig(f"box__{num}__by__{cat}.png")

def plot_violin_by_cat(df: pd.DataFrame, num: str, cat: str, rotate_x=False):
    plt.figure(figsize=(10,5))
    sb.violinplot(data=df, x=cat, y=num, cut=0, inner="quartile")
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.title(f"{num} by {cat} (violin)")
    savefig(f"violin__{num}__by__{cat}.png")

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str = None):
    plt.figure(figsize=(8,5))
    sb.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.6)
    plt.title(f"{y} vs {x}" + (f" by {hue}" if hue else ""))
    savefig(f"scatter__{y}_vs_{x}" + (f"__by__{hue}.png" if hue else ".png"))

def plot_corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(10,7))
    sb.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap (numeric)")
    savefig("corr_heatmap_numeric.png")

def plot_stacked_share(df: pd.DataFrame, cat_main: str, cat_sub: str, top_n: int = 8):
    # Top categories for main
    top = df[cat_main].value_counts().head(top_n).index
    d = df[df[cat_main].isin(top)]
    ct = pd.crosstab(d[cat_main], d[cat_sub], normalize="index")
    ct = ct.sort_index()
    ct.plot(kind="bar", stacked=True, figsize=(11,6))
    plt.ylabel("Share")
    plt.title(f"{cat_sub} share within {cat_main} (top {top_n})")
    plt.legend(title=cat_sub, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(f"stacked_share__{cat_main}__{cat_sub}.png")

# ---------- Run EDA ----------
def run_eda():
    path = os.path.join(DATA_DIR, "shopping_trends.csv")
    df = load_data(path)
    df = clean(df)

    basic_profile(df)

    # Distributions (numeric)
    for col in ["age", "purchase_amount_usd", "review_rating", "previous_purchases"]:
        if col in df.columns:
            plot_numeric_dist(df, col)

    # Counts (categorical)
    cat_cols = [
        "gender","category","item_purchased","location","size","color","season",
        "subscription_status","shipping_type","discount_applied","promo_code_used",
        "payment_method","frequency_of_purchases"
    ]
    for c in [x for x in cat_cols if x in df.columns]:
        # long lists better as horizontal
        orient = "h" if df[c].nunique() > 8 else "v"
        plot_count(df, c, orient=orient)

    # Relationships to purchase amount
    y = "purchase_amount_usd"
    for c in ["gender","category","season","subscription_status","discount_applied",
              "promo_code_used","payment_method","frequency_of_purchases","location","size","color"]:
        if c in df.columns and y in df.columns:
            rotate = df[c].nunique() > 8
            plot_box_by_cat(df, y, c, rotate_x=rotate)
            # violin also nice for distributions
            if df[c].nunique() <= 12:
                plot_violin_by_cat(df, y, c, rotate_x=rotate)

    # Age vs spend (+ hue)
    if set(["age", y]).issubset(df.columns):
        plot_scatter(df, "age", y)
        if "gender" in df.columns:
            plot_scatter(df, "age", y, hue="gender")
        if "subscription_status" in df.columns:
            plot_scatter(df, "age", y, hue="subscription_status")

    # Correlation heatmap
    plot_corr_heatmap(df)

    # Stacked shares: payment method within season; discount within category
    if "season" in df.columns and "payment_method" in df.columns:
        plot_stacked_share(df, "season", "payment_method", top_n=8)
    if "category" in df.columns and "discount_applied" in df.columns:
        plot_stacked_share(df, "category", "discount_applied", top_n=12)

    # Export a couple of useful summary tables
    # Average spend by key dims
    pivots = []
    for c in ["category","season","gender","subscription_status","discount_applied","promo_code_used","payment_method","frequency_of_purchases"]:
        if c in df.columns:
            piv = df.groupby(c)["purchase_amount_usd"].agg(["count","mean","median","std"]).reset_index()
            piv["dimension"] = c
            pivots.append(piv)
    if pivots:
        summary = pd.concat(pivots, ignore_index=True)
        savetab(summary, "avg_spend_by_dimensions.csv")

    # Top items by revenue
    if {"item_purchased","purchase_amount_usd"}.issubset(df.columns):
        top_items = (
            df.groupby("item_purchased")["purchase_amount_usd"]
              .sum()
              .sort_values(ascending=False)
              .reset_index()
              .rename(columns={"purchase_amount_usd":"revenue"})
        )
        savetab(top_items, "top_items_by_revenue.csv")

    print("EDA complete.")

if __name__ == "__main__":
    # Set seaborn theme like your previous project
    sb.set(style="whitegrid", context="talk")
    run_eda()
