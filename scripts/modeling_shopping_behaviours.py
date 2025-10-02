# scripts/modeling_shopping_behaviours.py
# -*- coding: utf-8 -*-
"""
Sprint 3 — Data Modelling for Shopping Behaviours
- Supervised: Classification (spend_band) & Regression (purchase_amount_usd)
- Unsupervised: KMeans clustering (elbow, silhouette, PCA viz)
- Generative: Gaussian Mixture Model (soft clustering + log-likelihoods)
- Inference: run best models on new data (data/new_customers_example.csv)
- Outputs: metrics (csv/json), figures (png), models (joblib)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# -------------------------------------------------
# Toggles
# -------------------------------------------------
VERBOSE = True
RUN_PERM_IMPORTANCE = False   # True → compute permutation importance (slower)
RUN_CLUSTERING = True         # False → skip clustering for faster run
RUN_TUNE_RF = False           # True → small grid search example (classification)
RUN_BINARY_VARIANT = True     # Train a binary Top vs Other classifier as improvement

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
PROC_DIR       = os.path.join(DATA_DIR, "processed")
REPORT_FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
REPORT_TAB_DIR = os.path.join(BASE_DIR, "reports", "tables")
MODEL_DIR      = os.path.join(BASE_DIR, "models")

os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(REPORT_TAB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

RAW_PATH   = os.path.join(DATA_DIR, "shopping_trends.csv")
CLEAN_PATH = os.path.join(PROC_DIR, "shopping_trends_clean.csv")
NEW_DATA_PATH = os.path.join(DATA_DIR, "new_customers_example.csv")

# Auto-locate CSV if RAW_PATH missing
if not os.path.exists(RAW_PATH):
    import glob
    candidates = glob.glob(os.path.join(BASE_DIR, "**", "shopping_trends*.csv"), recursive=True)
    if candidates:
        vprint("[Info] Could not find data/shopping_trends.csv, using:", candidates[0])
        RAW_PATH = candidates[0]
    else:
        raise FileNotFoundError(
            "Could not find shopping_trends.csv. Put it at data/shopping_trends.csv "
            "or rename the file accordingly."
        )

vprint("BASE_DIR:", BASE_DIR)
vprint("RAW_PATH:", RAW_PATH)
vprint("CLEAN_PATH:", CLEAN_PATH)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def savefig(name: str):
    plt.tight_layout()
    out = os.path.join(REPORT_FIG_DIR, name)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Figure] {out}")

def savetab(df: pd.DataFrame, name: str):
    out = os.path.join(REPORT_TAB_DIR, name)
    df.to_csv(out, index=False)
    print(f"[Table]  {out}")

def savejson(d: dict, name: str):
    out = os.path.join(REPORT_TAB_DIR, name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"[JSON]   {out}")

def savemodel(model, name: str):
    out = os.path.join(MODE_DIR := MODEL_DIR, name)
    joblib.dump(model, out)
    print(f"[Model]  {out}")

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

# -------------------------------------------------
# Load & Clean (mirrors Sprint 2 decisions)
# -------------------------------------------------
def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_headers(df)

    for c in ["age", "purchase_amount_usd", "review_rating", "previous_purchases"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # normalize Yes/No
    maps = {
        "discount_applied": {"y": "Yes", "n": "No"},
        "promo_code_used": {"y": "Yes", "n": "No"},
        "subscription_status": {"active": "Yes", "inactive": "No"},
    }
    for col, m in maps.items():
        if col in df.columns:
            df[col] = df[col].str.title().replace(m)

    df = df.dropna(subset=["purchase_amount_usd"]).drop_duplicates()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Classification target: spend quartiles
    if "purchase_amount_usd" in out.columns:
        out["spend_band"] = pd.qcut(out["purchase_amount_usd"], 4, labels=["Low", "Mid", "High", "Top"])
    # quick binaries + simple loyalty index
    for c in ["discount_applied", "promo_code_used", "subscription_status"]:
        if c in out.columns:
            out[c + "_bin"] = out[c].map({"Yes": 1, "No": 0})
    if {"subscription_status_bin", "previous_purchases"}.issubset(out.columns):
        out["loyalty_index"] = out["subscription_status_bin"] * out["previous_purchases"].fillna(0)
    return out

def load_data() -> pd.DataFrame:
    try:
        if os.path.exists(CLEAN_PATH):
            vprint(f"[Load] Reading cleaned: {CLEAN_PATH}")
            df = pd.read_csv(CLEAN_PATH)
            df = clean_headers(df)
        else:
            vprint(f"[Load] Reading raw: {RAW_PATH}")
            df = pd.read_csv(RAW_PATH)
            df = basic_clean(df)
            df.to_csv(CLEAN_PATH, index=False)
            vprint(f"[Load] Saved cleaned → {CLEAN_PATH}")

        df = engineer_features(df)
        vprint("[Load] Data shape after FE:", df.shape)
        return df
    except Exception as e:
        raise RuntimeError(f"[Load] Failed: {e}")

# -------------------------------------------------
# Column splits
# -------------------------------------------------
def split_columns(df: pd.DataFrame):
    target_reg = "purchase_amount_usd"
    target_clf = "spend_band"
    ignore = ["customer_id", target_reg, target_clf]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ignore]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in ignore]
    return target_reg, target_clf, num_cols, cat_cols

# -------------------------------------------------
# Classification (spend_band)
# -------------------------------------------------
def run_classification(df: pd.DataFrame):
    vprint("[CLF] Starting classification…")
    target_reg, target_clf, num_cols, cat_cols = split_columns(df)
    df = df.dropna(subset=[target_clf]).copy()

    X = df[num_cols + cat_cols]
    y = df[target_clf]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )

    models = {
        "logreg": LogisticRegression(max_iter=200, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

    results = []
    best_name, best_model, best_val_f1 = None, None, -np.inf

    # small validation split from train for quick selection
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_val_pred = pipe.predict(X_val)
        f1_w = f1_score(y_val, y_val_pred, average="weighted")
        acc = accuracy_score(y_val, y_val_pred)
        results.append({"model": name, "val_f1_weighted": f1_w, "val_acc": acc})
        print(f"[CLF:{name}] val_f1_weighted={f1_w:.3f}  val_acc={acc:.3f}")
        if f1_w > best_val_f1:
            best_val_f1, best_name, best_model = f1_w, name, pipe

    savetab(pd.DataFrame(results), "clf_validation_results.csv")

    # 5-fold CV on train set for the winner
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1)
    cv_acc = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    savejson({"cv_f1_weighted_mean": float(cv_f1.mean()), "cv_acc_mean": float(cv_acc.mean())},
             "clf_crossval_summary.json")

    # Final test evaluation
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_w = f1_score(y_test, y_test_pred, average="weighted")
    test_f1_m = f1_score(y_test, y_test_pred, average="macro")  # extra metric
    savejson({"test_acc": float(test_acc), "test_f1_weighted": float(test_f1_w), "test_f1_macro": float(test_f1_m)},
             "clf_test_metrics.json")
    print(f"[CLF:{best_name}] TEST f1_w={test_f1_w:.3f}  f1_m={test_f1_m:.3f}  acc={test_acc:.3f}")

    # Per-class report
    rep = classification_report(y_test, y_test_pred, output_dict=True)
    rep_df = pd.DataFrame(rep).transpose().reset_index().rename(columns={"index":"label"})
    savetab(rep_df, "clf_classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=sorted(y.unique()))
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name.upper()}")
    savefig("clf_confusion_matrix.png")

    # Optional: permutation importance
    if RUN_PERM_IMPORTANCE:
        try:
            perm = permutation_importance(best_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
            feat_names = best_model.named_steps["pre"].get_feature_names_out()
            imp = (pd.DataFrame({"feature": feat_names,
                                 "importance_mean": perm.importances_mean,
                                 "importance_std": perm.importances_std})
                   .sort_values("importance_mean", ascending=False)
                   .head(20))
            savetab(imp, "clf_feature_importance_permutation.csv")

            plt.figure(figsize=(8,6))
            plt.barh(imp["feature"][::-1], imp["importance_mean"][::-1])
            plt.title("Top Features (Permutation Importance) — Classification")
            plt.xlabel("Mean importance decrease")
            savefig("clf_feature_importance.png")
        except Exception as e:
            print(f"[CLF] Permutation importance skipped: {e}")

    savemodel(best_model, "clf_spend_band_best.joblib")

    # Optional improvement: binary target variant (Top vs Other)
    if RUN_BINARY_VARIANT:
        vprint("[CLF] Training binary Top-vs-Other variant…")
        df2 = df.copy()
        df2["high_spender"] = (df2[target_clf] == "Top").astype(int)
        Xb = df2[num_cols + cat_cols]
        yb = df2["high_spender"]
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.20, random_state=42, stratify=yb)
        preb = ColumnTransformer(
            [("num", StandardScaler(), num_cols),
             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
        )
        pipeb = Pipeline([("pre", preb), ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))])
        pipeb.fit(Xb_train, yb_train)
        yb_pred = pipeb.predict(Xb_test)
        accb = accuracy_score(yb_test, yb_pred)
        f1b  = f1_score(yb_test, yb_pred)
        savejson({"binary_top_vs_other_acc": float(accb), "binary_top_vs_other_f1": float(f1b)},
                 "clf_binary_test_metrics.json")
        savemodel(pipeb, "clf_top_vs_other.joblib")

    # Optional tuning example for RF
    if RUN_TUNE_RF:
        vprint("[CLF] Running small RF grid search…")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        pipe_rf = Pipeline([("pre", pre), ("clf", rf)])
        grid = {"clf__n_estimators": [300, 600], "clf__max_depth": [None, 12, 20]}
        gs = GridSearchCV(pipe_rf, grid, scoring="f1_weighted", cv=3, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        savejson({"best_params": gs.best_params_, "best_score": float(gs.best_score_)},
                 "clf_rf_gridsearch.json")
        savemodel(gs.best_estimator_, "clf_rf_tuned.joblib")

# -------------------------------------------------
# Regression (purchase_amount_usd)
# -------------------------------------------------
def run_regression(df: pd.DataFrame):
    vprint("[REG] Starting regression…")
    target_reg, target_clf, num_cols, cat_cols = split_columns(df)
    df = df.dropna(subset=[target_reg]).copy()

    X = df[num_cols + cat_cols]
    y = df[target_reg]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )

    models = {
        "linreg": LinearRegression(),
        "rfreg": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
    }

    results = []
    best_name, best_model, best_val_rmse = None, None, np.inf

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    for name, reg in models.items():
        pipe = Pipeline([("pre", pre), ("reg", reg)])
        pipe.fit(X_tr, y_tr)
        y_val_pred = pipe.predict(X_val)

        mae  = mean_absolute_error(y_val, y_val_pred)
        mse  = mean_squared_error(y_val, y_val_pred)   # no 'squared' arg (compat-friendly)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_val, y_val_pred)

        results.append({"model": name, "val_mae": mae, "val_rmse": rmse, "val_r2": r2})
        print(f"[REG:{name}] val_RMSE={rmse:.2f}  val_MAE={mae:.2f}  val_R2={r2:.3f}")
        if rmse < best_val_rmse:
            best_val_rmse, best_name, best_model = rmse, name, pipe

    savetab(pd.DataFrame(results), "reg_validation_results.csv")

    # 5-fold CV (RMSE via sqrt of MSE)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mses  = -cross_val_score(best_model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
    cv_rmses = np.sqrt(cv_mses)
    cv_r2    =  cross_val_score(best_model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
    savejson({"cv_rmse_mean": float(cv_rmses.mean()), "cv_r2_mean": float(cv_r2.mean())},
             "reg_crossval_summary.json")

    # Test metrics (+MAPE)
    y_test_pred = best_model.predict(X_test)
    test_mae  = mean_absolute_error(y_test, y_test_pred)
    test_mse  = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2   = r2_score(y_test, y_test_pred)
    test_mape = mape(y_test, y_test_pred)

    savejson({"test_rmse": float(test_rmse), "test_mae": float(test_mae), "test_r2": float(test_r2), "test_mape_pct": test_mape},
             "reg_test_metrics.json")
    print(f"[REG:{best_name}] TEST RMSE={test_rmse:.2f}  MAE={test_mae:.2f}  R2={test_r2:.3f}  MAPE={test_mape:.1f}%")

    # y_true vs y_pred plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    lo = min(y_test.min(), y_test_pred.min())
    hi = max(y_test.max(), y_test_pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("True Purchase Amount")
    plt.ylabel("Predicted Purchase Amount")
    plt.title(f"Regression — {best_name.upper()} (Test)")
    savefig("reg_true_vs_pred.png")

    # Optional: permutation importance
    if RUN_PERM_IMPORTANCE:
        try:
            perm = permutation_importance(best_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
            feat_names = best_model.named_steps["pre"].get_feature_names_out()
            imp = (pd.DataFrame({"feature": feat_names,
                                 "importance_mean": perm.importances_mean,
                                 "importance_std": perm.importances_std})
                   .sort_values("importance_mean", ascending=False)
                   .head(20))
            savetab(imp, "reg_feature_importance_permutation.csv")

            plt.figure(figsize=(8,6))
            plt.barh(imp["feature"][::-1], imp["importance_mean"][::-1])
            plt.title("Top Features (Permutation Importance) — Regression")
            plt.xlabel("Mean importance decrease")
            savefig("reg_feature_importance.png")
        except Exception as e:
            print(f"[REG] Permutation importance skipped: {e}")

    savemodel(best_model, "reg_purchase_amount_best.joblib")

# -------------------------------------------------
# Clustering (KMeans) + Generative (GMM)
# -------------------------------------------------
def run_clustering(df: pd.DataFrame):
    vprint("[CLU] Starting clustering…")
    # Choose interpretable features for segments
    use_num = [c for c in ["age", "review_rating", "previous_purchases"] if c in df.columns]
    use_cat = [c for c in [
        "gender","category","season","subscription_status","discount_applied","promo_code_used",
        "payment_method","frequency_of_purchases","location","size","color"
    ] if c in df.columns]

    X = df[use_num + use_cat].copy()

    pre = ColumnTransformer(
        [("num", StandardScaler(), use_num),
         ("cat", OneHotEncoder(handle_unknown="ignore"), use_cat)],
        remainder="drop",
        verbose_feature_names_out=False
    )
    Xp = pre.fit_transform(X)

    # Elbow & silhouette
    Ks, inertias, sils = list(range(2, 11)), [], []
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        lbl = km.fit_predict(Xp)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xp, lbl))

    # Elbow
    plt.figure(figsize=(7,4))
    plt.plot(Ks, inertias, "o-")
    plt.title("KMeans Elbow (Inertia)")
    plt.xlabel("K"); plt.ylabel("Inertia")
    savefig("cluster_elbow_inertia.png")

    # Silhouette
    plt.figure(figsize=(7,4))
    plt.plot(Ks, sils, "o-")
    plt.title("KMeans Silhouette Score")
    plt.xlabel("K"); plt.ylabel("Silhouette")
    savefig("cluster_silhouette.png")

    k_best = Ks[int(np.argmax(sils))]
    km = KMeans(n_clusters=k_best, n_init=30, random_state=42)
    labels = km.fit_predict(Xp)

    # PCA 2D viz
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(Xp.toarray() if hasattr(Xp, "toarray") else Xp)
    plt.figure(figsize=(7,6))
    plt.scatter(pts[:,0], pts[:,1], c=labels, cmap="viridis", s=18, alpha=0.7)
    plt.title(f"KMeans Clusters (K={k_best}) — PCA 2D")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    savefig("cluster_pca_scatter.png")

    # Profiles & metrics
    dfx = df.copy()
    dfx["cluster"] = labels
    sizes = dfx["cluster"].value_counts().reset_index()
    sizes.columns = ["cluster", "n"]
    savetab(sizes, "cluster_sizes.csv")

    if use_num:
        means = dfx.groupby("cluster")[use_num].mean(numeric_only=True).reset_index()
        savetab(means, "cluster_numeric_means.csv")

    # Top categories per cluster
    cat_profiles = []
    for c in use_cat:
        top = (dfx.groupby("cluster")[c]
               .apply(lambda s: s.value_counts().head(3).to_dict()))
        cat_profiles.append(pd.DataFrame({"cluster": top.index, f"top_{c}": top.values}))
    if cat_profiles:
        out = cat_profiles[0]
        for p in cat_profiles[1:]:
            out = out.merge(p, on="cluster", how="left")
        savetab(out, "cluster_top_categories.csv")

    dbi = davies_bouldin_score(Xp, labels)
    savejson({"silhouette_scores": [float(x) for x in sils],
              "k_best": int(k_best),
              "dbi_k_best": float(dbi)},
             "cluster_extra_metrics.json")

    savemodel(pre, "cluster_preprocessor.joblib")
    savemodel(km, "cluster_kmeans.joblib")

    # Generative: Gaussian Mixture (soft clustering)
    vprint("[CLU] Fitting GMM (generative)…")
    gmm = GaussianMixture(n_components=min(3, max(2, k_best)), random_state=42)
    gmm_labels = gmm.fit_predict(Xp)
    gmm_ll = gmm.score_samples(Xp)  # per-sample log-likelihood

    gmm_df = pd.DataFrame({"gmm_cluster": gmm_labels, "gmm_loglike": gmm_ll})
    savetab(gmm_df, "gmm_assignments.csv")
    savemodel(gmm, "gmm_segments.joblib")

# -------------------------------------------------
# Inference on new data
# -------------------------------------------------
def run_inference_on_new_data():
    if not os.path.exists(NEW_DATA_PATH):
        vprint("[Inference] No new data file found; skipping (expected at data/new_customers_example.csv).")
        return

    vprint("[Inference] Running predictions for new data…")
    df_new = pd.read_csv(NEW_DATA_PATH)
    df_new = clean_headers(df_new)

    # Load models
    clf = joblib.load(os.path.join(MODEL_DIR, "clf_spend_band_best.joblib"))
    reg = joblib.load(os.path.join(MODEL_DIR, "reg_purchase_amount_best.joblib"))

    # Predict
    try:
        pred_cls = clf.predict(df_new)
    except Exception:
        # If columns don't align well, try to drop unknowns softly with the same preprocessing
        pred_cls = clf.predict(df_new.fillna(""))
    try:
        pred_reg = reg.predict(df_new)
    except Exception:
        pred_reg = reg.predict(df_new.fillna(""))

    out = df_new.copy()
    out["pred_spend_band"] = pred_cls
    out["pred_purchase_amount_usd"] = pred_reg
    out_path = os.path.join(REPORT_TAB_DIR, "new_data_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"[Inference] Saved: {out_path}")

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    sb.set(style="whitegrid", context="talk")
    vprint("[Main] Starting Sprint 3 pipeline…")
    df = load_data()

    # quick dataset overview
    savejson({"rows": int(df.shape[0]), "cols": int(df.shape[1]),
              "has_spend_band": "spend_band" in df.columns}, "dataset_overview.json")

    vprint("[Main] Running classification…")
    run_classification(df)

    vprint("[Main] Running regression…")
    run_regression(df)

    if RUN_CLUSTERING:
        vprint("[Main] Running clustering…")
        run_clustering(df)
    else:
        vprint("[Main] Clustering skipped (RUN_CLUSTERING=False).")

    # New data inference (optional; will run only if file exists)
    run_inference_on_new_data()

    vprint("[Main] Done. Check reports/ and models/")

if __name__ == "__main__":
    main()
