
import os
import glob
import json
import joblib
import pandas as pd
import streamlit as st

# -------------------- Paths (app lives inside BI/) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # points to BI/
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR  = os.path.join(BASE_DIR, "reports", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "reports", "machine_learning", "joblib")

SHOP_ONEHOT = os.path.join(DATA_DIR, "shopping_trends_clean_onehot.csv")
SHOP_RAW    = os.path.join(DATA_DIR, "shopping_trends_clean.csv")
SUPERSTORE  = os.path.join(DATA_DIR, "global_superstore_cleaned.csv")

SHOP_MODEL_FILE  = os.path.join(MODEL_DIR, "shopping_lr_model.joblib")
SHOP_FEATURES_FILE = os.path.join(MODEL_DIR, "shopping_lr_features.json")

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

def discover_images(fig_dir):
    images = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        images.extend(glob.glob(os.path.join(fig_dir, pat)))
    return sorted(images)

def get_training_features(model, features_json_path):
    # Priority 1: JSON sidecar
    if os.path.exists(features_json_path):
        try:
            with open(features_json_path, "r", encoding="utf-8") as f:
                feats = json.load(f)
            if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
                return feats
        except Exception as e:
            st.info(f"Could not read {os.path.relpath(features_json_path, BASE_DIR)}: {e}")
    # Priority 2: sklearn attribute (if trained with DataFrame)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def options_from_prefix(columns, prefix):
    return sorted({c[len(prefix):] for c in columns if c.startswith(prefix)})

def has_prefix(columns, prefix):
    return any(c.startswith(prefix) for c in columns)

# -------------------- Load data/models --------------------
shop_oh = load_df(SHOP_ONEHOT)
shop_raw = load_df(SHOP_RAW)
super_df = load_df(SUPERSTORE)
shop_model = load_model(SHOP_MODEL_FILE)

# -------------------- Sidebar --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Preview", "Descriptive Diagrams", "Predict (Shopping)"]
)

# -------------------- Pages --------------------
if page == "Overview":
    st.title("📊 BI App — Shopping & Superstore (Simple)")

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
        kpi("Figures folder", "Found" if os.path.isdir(FIG_DIR) else "Missing")
        kpi("Model file", "Found" if os.path.exists(SHOP_MODEL_FILE) else "Missing")

    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        "- **Data Preview**: view head, shape, and basic describe() for Shopping or Superstore.\n"
        "- **Descriptive Diagrams**: see saved charts from `reports/figures` (EDA/diagrams).\n"
        "- **Predict (Shopping)**: manual inputs → one-hot vector → predict spend using your saved model.\n"
    )

elif page == "Data Preview":
    st.title("Data Preview")

    dataset = st.radio("Select dataset", ["Shopping (raw)", "Shopping (one-hot)", "Superstore"])

    df = None
    if dataset == "Shopping (raw)":
        df = shop_raw
        st.caption(f"Path: {os.path.relpath(SHOP_RAW, BASE_DIR)}")
    elif dataset == "Shopping (one-hot)":
        df = shop_oh
        st.caption(f"Path: {os.path.relpath(SHOP_ONEHOT, BASE_DIR)}")
    elif dataset == "Superstore":
        df = super_df
        st.caption(f"Path: {os.path.relpath(SUPERSTORE, BASE_DIR)}")

    if df is None:
        st.warning("Dataset not found.")
    else:
        st.write("**Shape:**", df.shape)
        st.dataframe(df.head(30), use_container_width=True)
        with st.expander("Describe (numeric columns)"):
            desc = df.describe(include="number").T
            st.dataframe(desc, use_container_width=True)

elif page == "Descriptive Diagrams":
    st.title("Descriptive Diagrams")
    if not os.path.exists(FIG_DIR):
        st.info("Figures folder not found.\nExpected: `BI/reports/figures/`")
    else:
        images = discover_images(FIG_DIR)
        if not images:
            st.info("No images found in `BI/reports/figures/`.")
        else:
            cols = st.columns(3)
            for i, img in enumerate(images):
                with cols[i % 3]:
                    st.image(img, use_container_width=True)
                    st.caption(os.path.basename(img).replace("_", " ").replace(".png", "").replace(".jpg", "").replace(".jpeg", ""))

elif page == "Predict (Shopping)":
    st.title("Predict — Shopping spend (manual input)")
    if shop_oh is None:
        st.warning("Processed shopping dataset not found. Needed to infer expected columns.")
    elif shop_model is None:
        st.warning("Model not found. Place it at `BI/reports/machine_learning/joblib/shopping_lr_model.joblib`.")
    else:
        training_features = get_training_features(shop_model, SHOP_FEATURES_FILE)
        if training_features is None:
            st.warning(
                "Could not determine the exact feature list for the saved model. "
                "Provide `shopping_lr_features.json` (list of column names in training order)."
            )
            st.stop()

        tf_set = set(training_features)
        # Common fields if present
        wants_age = "age" in tf_set
        wants_prev = "previous_purchases" in tf_set
        wants_disc = "discount_applied_bin" in tf_set
        wants_subs = "subscription_status_bin" in tf_set

        gender_opts  = options_from_prefix(training_features, "gender_") if has_prefix(training_features, "gender_") else []
        pay_opts     = options_from_prefix(training_features, "payment_method_") if has_prefix(training_features, "payment_method_") else []
        season_opts  = options_from_prefix(training_features, "season_") if has_prefix(training_features, "season_") else []
        cat_opts     = options_from_prefix(training_features, "category_") if has_prefix(training_features, "category_") else []

        with st.form("manual_predict"):
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
                st.caption("Ensure the training features match the model and their order is identical.")
