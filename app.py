import os
import glob
import json
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Figures (original)
FIG_DIR  = os.path.join(BASE_DIR, "figures")

# NEW: reports figure paths (superstore figs live here)
REPORTS_FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
SUPERSTORE_FIG_DIR = os.path.join(REPORTS_FIG_DIR, "superstore_figures")

MODEL_DIR = os.path.join(BASE_DIR, "machine_learning", "joblib")
ML_FIG_DIR = os.path.join(BASE_DIR, "machine_learning", "ml_results_diagrams")

SHOP_ONEHOT = os.path.join(DATA_DIR, "shopping_trends_clean_onehot.csv")
SHOP_RAW    = os.path.join(DATA_DIR, "shopping_trends_clean.csv")
SUPERSTORE  = os.path.join(DATA_DIR, "global_superstore_cleaned.csv")

SHOP_MODEL_FILE  = os.path.join(MODEL_DIR, "shopping_lr_model.joblib")
SHOP_FEATURES_FILE = os.path.join(MODEL_DIR, "shopping_lr_features.json")

SUPER_MODEL_FILE = os.path.join(MODEL_DIR, "superstore_lr_model.joblib")
SUPER_FEATURES_FILE = os.path.join(MODEL_DIR, "superstore_lr_features.json")

st.set_page_config(page_title="BI App — Shopping & Superstore", layout="wide")

# -------------------- Helpers --------------------
@st.cache_data(show_spinner=False)
def load_df(path):
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Could not load model ({path}): {e}")
    return None

def kpi(label, value):
    st.metric(label, value)

def _normkey(p: str) -> str:
    """Key for deduping across case and symlinks (fixes duplicate images)."""
    return os.path.normcase(os.path.realpath(p)).lower()

def discover_images(fig_dirs):
    """
    Recursively find images (png/jpg/jpeg) across one or many folders
    and de-duplicate paths across case/symlinks.
    """
    if isinstance(fig_dirs, str):
        fig_dirs = [fig_dirs]

    pats = ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.PNG", "**/*.JPG", "**/*.JPEG")
    found = []

    for d in fig_dirs:
        if not os.path.isdir(d):
            continue
        for pat in pats:
            found.extend(glob.glob(os.path.join(d, pat), recursive=True))

    # de-dupe across case and symlinks
    seen = {_normkey(p): p for p in found}
    return sorted(seen.values())

def get_training_features(model, features_json_path):
    if os.path.exists(features_json_path):
        try:
            with open(features_json_path, "r", encoding="utf-8") as f:
                feats = json.load(f)
            if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
                return feats
        except Exception as e:
            st.info(f"Could not read {os.path.relpath(features_json_path, BASE_DIR)}: {e}")
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def options_from_prefix(columns, prefix):
    return sorted({c[len(prefix):] for c in columns if c.startswith(prefix)})

def has_prefix(columns, prefix):
    return any(c.startswith(prefix) for c in columns)

# -------------------- Figure helpers --------------------
CAPTION_MAP = {
    # Descriptive / EDA
    "heatmap": "Heatmap: correlations between numeric features.",
    "pairplot": "Pair plot: relationships and distributions across variables.",
    "hist": "Histogram: value distribution; bar height = frequency.",
    "kde": "KDE: smoothed distribution; density peaks highlighted.",
    "box": "Boxplot: median, quartiles, and outliers across groups.",
    "violin": "Violin: distribution shape + median.",
    "bar": "Bar chart: categorical comparison (counts or averages).",
    "count": "Count plot: frequency by category.",
    "pie": "Pie chart: category proportions.",
    "line": "Line chart: trend over time.",
    "timeseries": "Time series: trends/seasonality over dates.",
    "map": "Map: geographic distribution.",

    # Classification
    "confusion": "Confusion matrix: actual vs predicted; diagonal = correct.",
    "class_report": "Classification report: precision, recall, F1 per class.",
    "roc": "ROC curve: TPR vs FPR; AUC summarizes performance.",
    "pr": "Precision–Recall curve: useful for imbalanced classes.",

    # Regression
    "true_vs_pred": "Predicted vs Actual: points near diagonal = good fit.",
    "residual": "Residuals: error patterns; randomness suggests good fit.",
    "error_dist": "Error distribution: spread of prediction errors.",

    # Clustering
    "cluster_pca": "Cluster scatter (PCA): colors = segments.",
    "cluster_scatter": "Cluster scatter: colored by cluster assignment.",
    "elbow": "Elbow: inertia vs K; look for the bend.",
    "silhouette": "Silhouette: cluster separation (higher is better).",
    "dbi": "Davies–Bouldin Index: lower is better.",
    "gmm": "GMM clusters: soft/elliptical segments.",
    "kmeans": "K-Means clusters: hard assignments; centroids = typical customers."
}

def guess_caption(filename: str) -> str:
    """Return a short human caption based on filename tokens."""
    name = os.path.basename(filename).lower()
    keys = [
        # clustering
        ("cluster_pca", ["pca", "cluster", "scatter"]),
        ("cluster_scatter", ["cluster", "scatter"]),
        ("elbow", ["elbow", "inertia"]),
        ("silhouette", ["silhouette"]),
        ("dbi", ["dbi", "davies"]),
        ("gmm", ["gmm", "mixture"]),
        ("kmeans", ["kmeans"]),
        # classification
        ("confusion", ["confusion"]),
        ("class_report", ["classification_report", "class_report"]),
        ("roc", ["roc"]),
        ("pr", ["precision_recall", "prcurve", "pr_curve", "pr-curve"]),
        # regression
        ("true_vs_pred", ["true_vs_pred", "y_true_vs_y_pred", "pred_vs_true"]),
        ("residual", ["residual"]),
        ("error_dist", ["error", "err_dist", "error_hist"]),
        # EDA
        ("heatmap", ["heatmap", "corr"]),
        ("pairplot", ["pairplot", "pair_plot"]),
        ("kde", ["kde"]),
        ("box", ["box"]),
        ("violin", ["violin"]),
        ("hist", ["hist", "distribution", "distplot", "dist_"]),
        ("bar", ["bar"]),
        ("count", ["countplot", "count"]),
        ("pie", ["pie"]),
        ("line", ["line", "trend"]),
        ("timeseries", ["time", "timeseries", "ts"]),
        ("map", ["map", "geo", "folium"]),
    ]
    for key, toks in keys:
        if all(tok in name for tok in toks) or any(tok in name for tok in toks):
            cap = CAPTION_MAP.get(key)
            if cap:
                return cap
    return os.path.splitext(os.path.basename(filename))[0].replace("_", " ").replace("-", " ").title()

def render_gallery(imgs, columns=3):
    cols = st.columns(columns)
    for i, img in enumerate(imgs):
        with cols[i % columns]:
            st.image(img, use_container_width=True)
            st.caption(guess_caption(img))

def group_ml_images(image_paths):
    groups = {"classification": [], "regression": [], "clustering": [], "model_selection": []}
    for p in image_paths:
        name = os.path.basename(p).lower()
        if any(k in name for k in ["cluster", "kmeans", "gmm", "mixture"]):
            groups["clustering"].append(p)
        elif any(k in name for k in ["elbow", "silhouette", "davies", "dbi"]):
            groups["model_selection"].append(p)
        elif any(k in name for k in ["confusion", "classification_report", "roc", "precision_recall", "prcurve", "pr_curve"]):
            groups["classification"].append(p)
        elif any(k in name for k in ["true_vs_pred", "residual", "error", "pred_vs_true"]):
            groups["regression"].append(p)
        else:
            groups["model_selection"].append(p)
    return groups

def group_eda_images(image_paths):
    """EDA tabs: Distributions / Relationships / Categories (dynamic; hide empties)."""
    groups = {"distributions": [], "relationships": [], "categories": []}
    for p in image_paths:
        name = os.path.basename(p).lower()
        if any(k in name for k in ["hist", "dist", "kde", "box", "violin"]):
            groups["distributions"].append(p)
        elif any(k in name for k in ["heatmap", "corr", "pairplot", "scatter"]):
            groups["relationships"].append(p)
        elif any(k in name for k in ["bar", "count", "pie", "stack", "share", "stacked"]):
            groups["categories"].append(p)
    return groups

# -------------------- Load data/models --------------------
shop_oh = load_df(SHOP_ONEHOT)
shop_raw = load_df(SHOP_RAW)
super_df = load_df(SUPERSTORE)
shop_model = load_model(SHOP_MODEL_FILE)
super_model = load_model(SUPER_MODEL_FILE)

# -------------------- Sidebar --------------------
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Preview", "Descriptive Diagrams", "ML Diagrams", "Predict (Shopping)", "Predict (Superstore)"]
)

# -------------------- Pages --------------------
if page == "Overview":
    st.title("BI App — Shopping & Superstore (Simple)")

    c1, c2, c3 = st.columns(3)
    with c1:
        if shop_raw is not None:
            kpi("Shopping rows (raw)", f"{len(shop_raw):,}")
        if shop_oh is not None:
            kpi("Shopping features (one-hot)", f"{shop_oh.shape[1]:,}")
    with c2:
        if super_df is not None:
            kpi("Superstore rows", f"{len(super_df):,}")
            if "Sales" in super_df.columns:
                kpi("Total Sales", f"{super_df['Sales'].sum():,.0f}")
            if "Profit" in super_df.columns:
                kpi("Total Profit", f"{super_df['Profit'].sum():,.0f}")
    with c3:
        any_fig_dir = any(os.path.isdir(p) for p in [FIG_DIR, REPORTS_FIG_DIR, SUPERSTORE_FIG_DIR])
        kpi("Figures folders", "Found" if any_fig_dir else "Missing")
        kpi("Model files", "Found" if os.path.exists(SHOP_MODEL_FILE) or os.path.exists(SUPER_MODEL_FILE) else "Missing")

    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        "- **Data Preview**: view head, shape, and basic describe().\n"
        "- **Descriptive Diagrams**: grouped EDA charts with captions.\n"
        "- **ML Diagrams**: classification, regression, clustering, and K selection.\n"
        "- **Predict (Shopping/Superstore)**: manual inputs → predictions."
    )

elif page == "Data Preview":
    st.title("Data Preview")
    dataset = st.radio("Select dataset", ["Shopping (raw)", "Shopping (one-hot)", "Superstore"])
    df = shop_raw if dataset == "Shopping (raw)" else shop_oh if dataset == "Shopping (one-hot)" else super_df

    if df is None:
        st.warning("Dataset not found.")
    else:
        st.write("**Shape:**", df.shape)
        st.dataframe(df.head(30), use_container_width=True)
        with st.expander("Describe (numeric columns)"):
            st.dataframe(df.describe(include="number").T, use_container_width=True)

elif page == "Descriptive Diagrams":
    st.title("Descriptive Diagrams")

    # Scan the classic figures folder + reports/figures + reports/figures/superstore_figures (recursively)
    images = discover_images([FIG_DIR, REPORTS_FIG_DIR, SUPERSTORE_FIG_DIR])

    if not images:
        st.info(
            "No figures found. Checked:\n"
            f"- `{os.path.relpath(FIG_DIR, BASE_DIR)}`\n"
            f"- `{os.path.relpath(REPORTS_FIG_DIR, BASE_DIR)}`\n"
            f"- `{os.path.relpath(SUPERSTORE_FIG_DIR, BASE_DIR)}`"
        )
    else:
        groups = group_eda_images(images)
        tabs = []
        if groups["distributions"]: tabs.append("Distributions")
        if groups["relationships"]: tabs.append("Relationships")
        if groups["categories"]: tabs.append("Categories")

        if not tabs:
            st.caption("No descriptive figures detected.")
        else:
            st_tabs = st.tabs(tabs)
            for tab_name, content in zip(tabs, st_tabs):
                with content:
                    st.subheader(tab_name)
                    if tab_name == "Distributions":
                        st.markdown("Histograms/box/violin/KDE show how values are spread; look for skew/outliers.")
                        render_gallery(groups["distributions"])
                    elif tab_name == "Relationships":
                        st.markdown("Heatmaps/pair plots/scatter show relationships and correlation strength.")
                        render_gallery(groups["relationships"])
                    elif tab_name == "Categories":
                        st.markdown("Bar/count/stacked charts compare categories (counts or averages).")
                        render_gallery(groups["categories"])

elif page == "ML Diagrams":
    st.title("Machine Learning Diagrams")

    if not os.path.exists(ML_FIG_DIR):
        st.info("ML figures folder not found.\nExpected: `machine_learning/ml_results_diagrams/`")
    else:
        # discover_images is now recursive, so nested ML result folders are supported too
        images = discover_images(ML_FIG_DIR)
        if not images:
            st.info("No images found in `machine_learning/ml_results_diagrams/`.")
        else:
            groups = group_ml_images(images)
            tabs = st.tabs(["Classification", "Regression", "Clustering (K-Means & GMM)", "Model Selection (K Choice)"])

            with tabs[0]:
                st.subheader("Classification")
                st.markdown("Guide: confusion matrix diagonal = correct; use report for Precision/Recall/F1.")
                if groups["classification"]:
                    render_gallery(groups["classification"])
                else:
                    st.caption("No classification figures detected.")

            with tabs[1]:
                st.subheader("Regression")
                st.markdown("Guide: Predicted vs Actual near diagonal is good; residuals should look random.")
                if groups["regression"]:
                    render_gallery(groups["regression"])
                else:
                    st.caption("No regression figures detected.")

            with tabs[2]:
                st.subheader("Clustering (Segments)")
                st.markdown("Guide: PCA scatter shows separation; GMM allows soft/elliptical clusters.")
                if groups["clustering"]:
                    render_gallery(groups["clustering"])
                else:
                    st.caption("No clustering figures detected.")

            with tabs[3]:
                st.subheader("Model Selection (Choosing K)")
                st.markdown("Guide: Elbow = diminishing returns in inertia; Silhouette/DBI assess separation.")
                if groups["model_selection"]:
                    render_gallery(groups["model_selection"])
                else:
                    st.caption("No model selection figures detected.")

elif page == "Predict (Shopping)":
    st.title("Predict — Shopping spend (manual input)")
    if shop_oh is None:
        st.warning("Processed shopping dataset not found (needed to infer columns).")
    elif shop_model is None:
        st.warning("Model not found. Place it at `machine_learning/joblib/shopping_lr_model.joblib`.")
    else:
        training_features = get_training_features(shop_model, SHOP_FEATURES_FILE)
        if training_features is None:
            st.warning("Provide `shopping_lr_features.json` (training feature order).")
        else:
            tf_set = set(training_features)
            wants_age = "age" in tf_set
            wants_prev = "previous_purchases" in tf_set
            wants_disc = "discount_applied_bin" in tf_set
            wants_subs = "subscription_status_bin" in tf_set

            gender_opts  = options_from_prefix(training_features, "gender_") if has_prefix(training_features, "gender_") else []
            pay_opts     = options_from_prefix(training_features, "payment_method_") if has_prefix(training_features, "payment_method_") else []
            season_opts  = options_from_prefix(training_features, "season_") if has_prefix(training_features, "season_") else []
            cat_opts     = options_from_prefix(training_features, "category_") if has_prefix(training_features, "category_") else []

            with st.form("manual_predict_shop"):
                st.subheader("Enter a simple customer profile")
                c1, c2 = st.columns(2)
                age = c1.slider("Age", 16, 90, 35) if wants_age else None
                prev = c2.slider("Previous purchases", 0, 100, 5) if wants_prev else None

                colb1, colb2 = st.columns(2)
                discount = colb1.checkbox("Discount applied?", value=False) if wants_disc else None
                subscribed = colb2.checkbox("Subscription active?", value=False) if wants_subs else None

                gender = st.selectbox("Gender", gender_opts) if gender_opts else None
                payment_method = st.selectbox("Payment method", pay_opts) if pay_opts else None
                season = st.selectbox("Season", season_opts) if season_opts else None
                category = st.selectbox("Category", cat_opts) if cat_opts else None

                submitted = st.form_submit_button("Predict")

            if submitted:
                row = {c: 0.0 for c in training_features}
                if wants_age:  row["age"] = float(age)
                if wants_prev: row["previous_purchases"] = float(prev)
                if wants_disc: row["discount_applied_bin"] = 1.0 if discount else 0.0
                if wants_subs: row["subscription_status_bin"] = 1.0 if subscribed else 0.0

                def set_one_hot(prefix, value):
                    if not value:
                        return
                    key = f"{prefix}{value}"
                    for c in (c for c in training_features if c.startswith(prefix)):
                        row[c] = 1.0 if c.lower() == key.lower() else 0.0

                set_one_hot("gender_", gender)
                set_one_hot("payment_method_", payment_method)
                set_one_hot("season_", season)
                set_one_hot("category_", category)

                Xrow = pd.DataFrame([[row[c] for c in training_features]], columns=training_features)

                try:
                    yhat = float(shop_model.predict(Xrow.values)[0])
                    st.success(f"Predicted spend: ${yhat:,.2f}")
                    with st.expander("View model input vector"):
                        st.dataframe(Xrow, use_container_width=True)
                    st.caption(f"Model expects {len(training_features)} features; provided {Xrow.shape[1]}.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.caption("Ensure the training features and order match the saved model.")

elif page == "Predict (Superstore)":
    st.title("Predict — Superstore profit (manual input)")

    if super_df is None or super_model is None:
        st.warning("Superstore dataset or model not found.")
    else:
        training_features = get_training_features(super_model, SUPER_FEATURES_FILE)
        if training_features is None:
            st.warning("Provide `superstore_lr_features.json` (training feature order).")
        else:
            # sensible defaults from data
            med_sales = float(super_df["Sales"].median()) if "Sales" in super_df.columns else 100.0
            ship_col = "Shipping Cost" if "Shipping Cost" in super_df.columns else ("Shipping_Cost" if "Shipping_Cost" in super_df.columns else None)
            med_ship = float(super_df[ship_col].median()) if ship_col else 10.0
            med_qty   = int(float(super_df["Quantity"].median())) if "Quantity" in super_df.columns else 2
            med_disc  = float(super_df["Discount"].median()) if "Discount" in super_df.columns else 0.0

            with st.form("predict_superstore"):
                st.subheader("Enter features")
                c1, c2 = st.columns(2)
                c3, c4 = st.columns(2)
                c5, c6 = st.columns(2)

                sales_raw = c1.number_input("Sales", value=med_sales, min_value=0.0, step=1.0, format="%.2f")
                ship_raw  = c2.number_input("Shipping Cost", value=med_ship, min_value=0.0, step=0.5, format="%.2f")
                qty       = c3.number_input("Quantity", value=med_qty, min_value=0, step=1)
                disc      = c4.number_input("Discount", value=med_disc, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
                cat_tech  = c5.checkbox("Category = Technology", value=False)
                sub_copy  = c6.checkbox("Sub-Category = Copiers", value=False)

                submitted = st.form_submit_button("Predict")

            if submitted:
                row = {c: 0.0 for c in training_features}
                row["Sales_Log"] = float(np.log1p(sales_raw))
                row["ShippingCost_Log"] = float(np.log1p(ship_raw))
                row["Quantity"] = float(qty)
                row["Discount"] = float(disc)
                row["Category_Technology"] = 1.0 if cat_tech else 0.0
                row["Sub-Category_Copiers"] = 1.0 if sub_copy else 0.0

                X = pd.DataFrame([[row[c] for c in training_features]], columns=training_features)

                try:
                    yhat = float(super_model.predict(X.values)[0])
                    st.success(f"Predicted profit: ${yhat:,.2f}")
                    with st.expander("Model input vector"):
                        st.dataframe(X, use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
