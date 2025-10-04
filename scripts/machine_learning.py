import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import re
import seaborn as sns
import os
import matplotlib.pyplot as plt
import json
import joblib

# 1. PATH DEFINITION AND DIRECTORY SETUP
# Assumes the script is inside BASE_DIR/scripts/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project Root: {BASE_DIR}")


# --- DATA DIRECTORIES ---
DATA_DIR = os.path.join(BASE_DIR, "data")          # For raw data and class data
PROC_DIR = os.path.join(DATA_DIR, "processed")     # For your cleaned PKL data
os.makedirs(PROC_DIR, exist_ok=True) # Ensure processed exists if not created by a previous script

# --- REPORT DIRECTORIES ---
# The primary report folder (at the same level as 'scripts')
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Nested ML results folder
ML_REPORT_DIR = os.path.join(REPORT_DIR, "machine_learning")
os.makedirs(ML_REPORT_DIR, exist_ok=True)

# Specific directories inside machine_learning/
REPORT_FIG_DIR = os.path.join(ML_REPORT_DIR, "ml_results_diagrams")
REPORT_PRINT_DIR = os.path.join(ML_REPORT_DIR, "ml_results_print") # Renamed from 'REPORT_TAB_DIR' for print/log data
MODEL_DIR = os.path.join(ML_REPORT_DIR, "joblib")



# Create all necessary directories
os.makedirs(REPORT_FIG_DIR, exist_ok=True)
os.makedirs(REPORT_PRINT_DIR, exist_ok=True) # Use the new print directory
os.makedirs(MODEL_DIR, exist_ok=True)


# 3. FILE PATHS FOR LOADING DATA

# Your cleaned data (Global Superstore - saved in processed/)
SUPERSTORE_CLEANED_PATH = os.path.join(PROC_DIR, "global_superstore_cleaned.pkl") 

# Your class data (shopping_trends_clean.csv - assumed to be in data/processed/)
SHOPPING_DATA_PATH = os.path.join(PROC_DIR, "shopping_trends_clean_onehot.csv")

# Load your data
try:
    df_superstore = pd.read_pickle(SUPERSTORE_CLEANED_PATH)
    print("Superstore data loaded.")
except FileNotFoundError:
    print(f"ERROR: Could not find PKL file: {SUPERSTORE_CLEANED_PATH}")


# Load the new class dataset
try:
    df_shopping = pd.read_csv(SHOPPING_DATA_PATH)
    print("Shopping Trends data loaded.")
except FileNotFoundError:
    print(f"WARNING: Could not find CSV file: {SHOPPING_DATA_PATH}")
    df_new = None

# ---

# 4. UTILITY FUNCTIONS

def savelog(message: str, name: str = "ml_run_log.txt"):
    """Saves an important message (metrics, summaries) to a log file."""
    # Note: Using REPORT_PRINT_DIR for all text/log outputs
    path = os.path.join(REPORT_PRINT_DIR, name)
    
    # Open the file in append mode ('a') to add new lines
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
        
    print(f"[Log]    {message}") # Still print to console for immediate feedback

def safe_filename(name: str) -> str:
    # replace path separators and illegal chars with underscores
    s = str(name).replace("/", "_").replace("\\", "_")
    s = re.sub(r'[^A-Za-z0-9._-]+', "_", s)
    s = re.sub(r'_{2,}', "_", s).strip("_")
    return s

def savejson(d: dict, name: str):
    name = safe_filename(name)
    path = os.path.join(REPORT_PRINT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"[JSON]   {path}")

def savetab(df: pd.DataFrame, name: str):
    name = safe_filename(name)
    path = os.path.join(REPORT_PRINT_DIR, name)
    df.to_csv(path, index=False)
    print(f"[Table]  {path}")

def savefig(name: str):
    name = safe_filename(name)
    path = os.path.join(REPORT_FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] {path}")

def savemodel(model, name: str):
    name = safe_filename(name)
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, path)
    print(f"[Model]  {path}")

def correlation_with_target(df: pd.DataFrame, target: str, filename: str):
     if df is None or target not in df.columns:
         print(f"ERROR: DataFrame is None or target column '{target}' not found.")
         return
     corr_matrix = df.corr(numeric_only=True)

     if target not in corr_matrix.columns:
         print(f"ERROR: Target column '{target}' not found in correlation matrix.")
         return

     corr_series = corr_matrix[target].sort_values(ascending=False)
     corr_df = corr_series.reset_index()
     corr_df.columns = ['Feature', f'Correlation_with_{target}']

     savetab(corr_df, filename)
     print(f"Correlation with target '{target}' saved to {filename}")

     


correlation_with_target(df_superstore, 'Profit_YJ', "profit_correlation_sorted.csv")
correlation_with_target(df_shopping, 'purchase_amount_usd', "yearly_amount_spent_correlation_sorted.csv")

def correlation_with_target(df: pd.DataFrame, target: str, filename: str):
    """
    Calculates the correlation of all numeric features with the target 
    and saves the sorted result as a CSV table using savetab().
    (Original function, slightly cleaned for robustness)
    """
    if df is None or target not in df.columns:
        print(f"ERROR: DataFrame is None or target column '{target}' not found for correlation analysis.")
        return

    # Ensure only numeric columns are used for correlation
    df_numeric = df.select_dtypes(include=np.number)
    
    if target not in df_numeric.columns:
        print(f"ERROR: Target column '{target}' not found in numeric data.")
        return

    corr_matrix = df_numeric.corr()
    corr_series = corr_matrix[target].sort_values(ascending=False)
    
    # Save Correlation Table
    corr_df = corr_series.reset_index()
    corr_df.columns = ['Feature', f'Correlation_with_{target}']
    
    savetab(corr_df, filename)
    print(f"Correlation table with target '{target}' saved to {filename}")
    
    # Return the correlation series for use in other functions (like the heatmap one)
    return corr_series


def create_correlation_heatmap(df: pd.DataFrame, corr_series: pd.Series, target: str, heatmap_name: str, top_n: int = 10):
    """
    Generates and saves a heatmap showing the correlation between the target 
    and the top_n most correlated features (based on the provided corr_series).
    """
    if corr_series is None:
        print(f"ERROR: Correlation Series is None for heatmap generation of {target}.")
        return
        
    # Get the features corresponding to the top N absolute correlations
    # Exclude the target itself, take the top N features based on absolute correlation
    top_features = corr_series.drop(target, errors='ignore').abs().nlargest(top_n).index.tolist()
    top_features_with_target = top_features + [target]

    # Ensure all selected features actually exist in the DataFrame (avoiding errors from previous cleaning)
    valid_features = [f for f in top_features_with_target if f in df.columns]
    
    if len(valid_features) < 2:
        print(f"WARNING: Too few valid features ({len(valid_features)}) to create heatmap for {target}.")
        return

    correlation_for_heatmap = df[valid_features].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_for_heatmap, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Top {len(valid_features)-1} Features Correlation Heatmap with {target}')
    savefig(heatmap_name)
    print(f"Heatmap for {target} saved to {heatmap_name}")

# Step 1a: Lav korrelationstabel (bruger din oprindelige logik)
profit_corr_data = correlation_with_target(
    df=df_superstore.copy(), 
    target='Profit_YJ', 
    filename="superstore_profit_correlation_sorted.csv"
)

create_correlation_heatmap(
    df=df_superstore.copy(), 
    corr_series=profit_corr_data,
    target='Profit_YJ', 
    heatmap_name="superstore_profit_correlation_heatmap.png",
    top_n=10
)

profit_corr_data = correlation_with_target(
    df=df_shopping.copy(), 
    target='purchase_amount_usd', 
    filename="yearly_amount_spent_correlation_sorted.csv"
)

create_correlation_heatmap(
    df=df_shopping.copy(), 
    corr_series=profit_corr_data,
    target='purchase_amount_usd', 
    heatmap_name="shopping_purchase_correlation_heatmap.png",
    top_n=10
)

# --- LINEAR REGRESSION FUNCTION ---

def plot_regression_residuals(y_test: np.array, y_pred: np.array, data_name: str, model_type: str):
    """
    Generates and saves a Residual Plot to visualize the performance of the regression model.
    A good plot shows residuals randomly scattered around zero.
    """
    # Calculate residuals (Difference between actual and predicted values)
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of predicted values vs. residuals
    plt.scatter(y_pred, residuals, alpha=0.5)
    
    # Draw a horizontal line at residual = 0 (the ideal line)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='red', linestyle='--')
    
    # Add labels and title
    plt.title(f'{model_type} Residual Plot - {data_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save the figure
    safe_data_name = data_name.lower().replace(' ', '_')
    safe_model_type = model_type.lower().replace(' ', '_')
    savefig(f"{safe_data_name}_{safe_model_type}_residuals.png")
    savelog(f"Residual Plot saved: {safe_data_name}_{safe_model_type}_residuals.png")
    plt.close() # Close the plot figure to free up memory

def train_and_evaluate_lr(df: pd.DataFrame, target: str, selected_features: list, model_name: str, log_prefix: str):
    """
    Trains a Linear Regression model using specific features and saves results/model.
    """
    if df is None or target not in df.columns or not all(f in df.columns for f in selected_features):
        print(f"ERROR: Missing data or features for Linear Regression training: {log_prefix}")
        return

    print(f"\n--- {log_prefix}: Linear Regression ---")
    print(f"Features used: {', '.join(selected_features)}")

    # 1. Prepare data 
    X = df[selected_features]
    y = df[target]

    # 2. Split data (standard 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)

    # 4. Evaluate the model
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr) 
    mse_lr = mean_squared_error(y_test, y_pred_lr) 
    
    feature_coeffs = dict(zip(selected_features, linear_model.coef_))
    
    # 5. Log results
    savelog(f"Linear Regression Results:")
    savelog(f"  Test R2: {r2_lr:.4f}")
    savelog(f"  Test RMSE: {rmse_lr:.4f}")
    savelog(f"  Test MAE: {mae_lr:.4f}")
    savelog(f"  Test MSE: {mse_lr:.4f}")
    
    # Save coefficients
    savejson(feature_coeffs, f"{log_prefix.lower().replace(' ', '_')}_lr_coefficients.json")
    print(f"  Coefficients saved to {log_prefix.lower().replace(' ', '_')}_lr_coefficients.json")

    # 6. Save the trained model
    savemodel(linear_model, model_name)
    print(f"Model saved as {model_name}")

        # 7. Visualization: Residual Plot (NYT TRIN)
    plot_regression_residuals(
        y_test=y_test, 
        y_pred=y_pred_lr, 
        data_name=log_prefix, 
        model_type="Linear Regression"
    )

        # 8. Return metrics (Optional, but good practice)
    return {
        'R2': r2_lr,
        'RMSE': rmse_lr,
        'Coefficients': feature_coeffs
    }

TARGET_SUPERSTORE = 'Profit_YJ'
TARGET_SHOPPING = 'purchase_amount_usd'

SUPERSTORE_LR_FEATURES = [
        'Sales_Log', 
        'ShippingCost_Log', 
        'Quantity', 
        'Category_Technology', 
        'Sub-Category_Copiers', 
        'Discount'
]

train_and_evaluate_lr(
        df=df_superstore.copy(), 
        target=TARGET_SUPERSTORE, 
        selected_features=SUPERSTORE_LR_FEATURES,
        model_name="superstore_lr_model.joblib",
        log_prefix="Superstore Data"
)

SHOPPING_LR_FEATURES = [
    'age',
    'discount_applied_bin',
    'promo_code_used_bin',            
    'review_rating',
    'previous_purchases',
    'subscription_status_bin',
]

train_and_evaluate_lr(
        df=df_shopping.copy(), 
        target=TARGET_SHOPPING, 
        selected_features=SHOPPING_LR_FEATURES,
        model_name="shopping_lr_model.joblib",
        log_prefix="Shopping Data"
)

# --- NEW CLUSTERING UTILITY FUNCTIONS ---

def scale_features(df: pd.DataFrame, features: list):
    """
    Scales the selected features using StandardScaler and returns the scaled DataFrame 
    and the fitted scaler object.
    """
    if not all(f in df.columns for f in features):
        print("ERROR: One or more features for scaling not found in DataFrame.")
        return None, None
        
    X = df[features].copy()
    scaler = StandardScaler()
    
    # Fit and transform the data
    X_scaled = scaler.fit_transform(X)
    
    # Return as a DataFrame for easy use later
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
    
    print(f"Features successfully scaled using StandardScaler.")
    return X_scaled_df, scaler


def find_optimal_k_and_plot(X_scaled_df: pd.DataFrame, max_k: int, target_name: str):
    """
    Applies both the Elbow Method (Inertia) and Silhouette Score analysis 
    to find the optimal number of clusters (k).
    Generates and saves two separate plots for visual and objective evaluation.
    """
    inertia = []
    silhouette_scores = []
    
    # K-range for Inertia (Elbow) starts at 1
    k_range_inertia = range(1, max_k + 1)
    # K-range for Silhouette Score starts at 2
    k_range_silhouette = range(2, max_k + 1)
    
    savelog(f"Running K-Means for k=1 to {max_k} to find optimal k (Elbow & Silhouette).")

    for k in k_range_inertia:
        # random_state ensures reproducibility
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
        labels = kmeans.fit_predict(X_scaled_df)
        
        # 1. Elbow Method (Inertia)
        inertia.append(kmeans.inertia_)
        
        # 2. Silhouette Score (Kun for k >= 2)
        if k >= 2:
            score = silhouette_score(X_scaled_df, labels)
            silhouette_scores.append(score)
            savelog(f"  k={k} achieved Silhouette Score: {score:.4f}")

    # --- PLOT 1: ELBOW METHOD (INERTIA) ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_range_inertia, inertia, marker='o', linestyle='--')
    plt.title(f'Elbow Method (Inertia) for Optimal k - {target_name}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.grid(True)
    savefig(f"kmeans_elbow_plot_{target_name.lower().replace(' ', '_')}.png")

    print(f"Elbow Plot saved to kmeans_elbow_plot_{target_name.lower().replace(' ', '_')}.png")


    # --- PLOT 2: SILHOUETTE SCORE ---
    if silhouette_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(k_range_silhouette, silhouette_scores, marker='o', linestyle='--')
        plt.title(f'Silhouette Scores for Optimal k - {target_name}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Average Silhouette Score')
        plt.grid(True)
        savefig(f"kmeans_silhouette_plot_{target_name.lower().replace(' ', '_')}.png")
        
        print(f"Silhouette Plot saved to kmeans_silhouette_plot_{target_name.lower().replace(' ', '_')}.png")

        # Find the k with the highest score
        best_k_index = np.argmax(silhouette_scores)
        best_k = k_range_silhouette[best_k_index]
        savelog(f"Based on objective Silhouette Score, the best k is likely {best_k} (Score: {silhouette_scores[best_k_index]:.4f}).")
    
    # Return the calculated scores for potential inspection
    return inertia, silhouette_scores

def run_final_clustering(df: pd.DataFrame, X_scaled_df: pd.DataFrame, features: list, optimal_k: int, data_name: str):
    """
    Trains the final K-Means model, assigns labels to the DataFrame, 
    performs profiling, and visualizes the results for a given dataset.
    
    Returns the DataFrame with the new 'Cluster_KMeans' column.
    """
    print(f"\n--- {data_name}: Running Final K-Means (k={optimal_k}) ---")

    if X_scaled_df is None:
        print(f"ERROR: Scaled data is None for {data_name}. Cannot run final K-Means.")
        return df

    X_clustering = X_scaled_df.values
    
    # 1. Training and Prediction
    kmeans = KMeans(
        init='k-means++', 
        n_clusters=optimal_k, 
        n_init=20, 
        random_state=42 
    )
    y_labels = kmeans.fit_predict(X_clustering)

    # 2. Labeling and Saving the DataFrame
    df['Cluster_KMeans'] = y_labels
    print(f"Final K-Means executed. Labels assigned to {data_name} with k={optimal_k}.")

    # Use the dynamic file name for persistence
    filename = f"{data_name.lower().replace(' ', '_')}_clustered.pkl"
    df.to_pickle(os.path.join(PROC_DIR, filename))
    print(f"Clustered DataFrame saved for persistence: {filename}")
    
    # 3. Cluster Profiling: Define features dynamically
    if data_name == "Superstore Data":
        profile_features = ['Sales_Log', 'Profit_YJ', 'Discount', 'Quantity', 'ShippingCost_Log', 'Category_Technology', 'Market_EU']
    elif data_name == "Shopping Data":
        profile_features = ['purchase_amount_usd', 'review_rating', 'age', 'previous_purchases', 'subscription_status_bin']
    else:
        profile_features = features 

    valid_features = [f for f in profile_features if f in df.columns]
    if valid_features:
        # Calculate mean values for each cluster
        cluster_profile = df.groupby('Cluster_KMeans')[valid_features].mean().round(5)
        filename_tab = f"{data_name.lower().replace(' ', '_')}_kmeans_cluster_profile.csv"
        savetab(cluster_profile, filename_tab)
        print(f"Cluster profile saved to {filename_tab}.")
    else:
        print(f"WARNING: No valid profile features found for {data_name}. Skipping profiling.")

    # 4. Visualization (Plotting the first two features vs. the centroids)
    if len(features) >= 2:
        feature_0 = features[0] 
        feature_1 = features[1] 

        plt.figure(figsize=(12, 8))
        plt.scatter(
            X_clustering[:, 0], X_clustering[:, 1], c=y_labels, s=20, cmap='viridis'
        )
        centers = kmeans.cluster_centers_
        plt.scatter(
            centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label='Centroids'
        )

        plt.title(f'K-Means Clustering Visualization (k={optimal_k}) - {data_name}')
        plt.xlabel(f'Scaled {feature_0}')
        plt.ylabel(f'Scaled {feature_1}')
        plt.legend()
        plt.grid(True)
        savefig(f"{data_name.lower().replace(' ', '_')}_kmeans_final_scatter.png")
        print(f"Final K-Means scatter plot saved.")
    
    print(f"Cluster Centroids (Scaled Feature Values):\n{kmeans.cluster_centers_}")
    
    return df

SUPERSTORE_OPTIMAL_K = 5 
SHOPPING_OPTIMAL_K = 2 

RUN_CLUSTERING = True             # True: Runs K-Optimal analysis. False: Skips K-Optimal analysis.
LOAD_CLUSTERING_RESULTS = True   # True: Attempts to load final clustered data and skips all clustering steps if successful.


SUPERSTORE_CLUSTERING_FEATURES = [
    'Sales_Log', 
    'ShippingCost_Log', 
    'Quantity', 
    'Category_Technology', 
    'Sub-Category_Copiers', 
    'Discount'
]

SHOPPING_CLUSTERING_FEATURES = [
    'previous_purchases', 
    'purchase_amount_usd', 
    'promo_code_used_bin',
    'review_rating', 
    'age', 
    'subscription_status_bin',
]


X_scaled_superstore, scaler_superstore = None, None
X_scaled_shopping, scaler_shopping = None, None

df_superstore_clustered = df_superstore.copy()
df_shopping_clustered = df_shopping.copy()


# Initialize scaled variables in the global scope (Must exist outside conditional blocks)
X_scaled_superstore, scaler_superstore = None, None
X_scaled_shopping, scaler_shopping = None, None

# Create copies of DFs to be clustered (Ensures original DFs are preserved if needed)
df_superstore_clustered = df_superstore.copy()
df_shopping_clustered = df_shopping.copy()

SHOULD_CALCULATE_CLUSTERING = True # Internal flag to check if we must run the heavy calculation

# ----------------------------------------------------------------------
# 5.1 Step 1: Attempt to Load Final Clustered Data (Skip Logic)
# ----------------------------------------------------------------------

if LOAD_CLUSTERING_RESULTS:
    print("\n--- Attempting to load final clustered data (Skip Logic Active) ---")
    
    SHOULD_CALCULATE_CLUSTERING = False

    # Superstore Load
    superstore_path = os.path.join(PROC_DIR, "superstore_data_clustered.pkl")
    if os.path.exists(superstore_path):
        df_superstore_clustered = pd.read_pickle(superstore_path)
        print(f"SUCCESS: Loaded clustered data for df_superstore.")
    else:
        print("WARNING: Could not find saved Superstore clustered data. Calculation required.")
        SHOULD_CALCULATE_CLUSTERING = True
        
    # Shopping Load
    shopping_path = os.path.join(PROC_DIR, "shopping_data_clustered.pkl")
    if os.path.exists(shopping_path):
        df_shopping_clustered = pd.read_pickle(shopping_path)
        print(f"SUCCESS: Loaded clustered data for df_shopping.")
    else:
        print("WARNING: Could not find saved Shopping clustered data. Calculation required.")
        SHOULD_CALCULATE_CLUSTERING = True # Ensure calculation runs if any file is missing

    if not SHOULD_CALCULATE_CLUSTERING:
        RUN_CLUSTERING = False 
        

# ----------------------------------------------------------------------
# 5.2 Scaling and K-Optimal Analysis (Runs if calculated data is needed)
# ----------------------------------------------------------------------

# We only run this complex block if we failed to load the clustered data (SHOULD_CALCULATE_CLUSTERING = True)
# OR if the user manually wants to run the K-Optimal analysis (RUN_CLUSTERING = True)
if SHOULD_CALCULATE_CLUSTERING or RUN_CLUSTERING:
    print("\n--- K-MEANS SCALING AND K-FINDINGS ---")
    
    # Scaling for Superstore (ALWAYS RUNS IF CLUSTERING IS REQUIRED)
    X_scaled_superstore, scaler_superstore = scale_features(
        df=df_superstore.copy(), 
        features=SUPERSTORE_CLUSTERING_FEATURES
    )
    if X_scaled_superstore is not None and RUN_CLUSTERING:
        # K-Findings (The potentially slow part)
        find_optimal_k_and_plot(
            X_scaled_df=X_scaled_superstore, 
            max_k=10, 
            target_name="Superstore Customers"
        )
        
    # Scaling for Shopping
    X_scaled_shopping, scaler_shopping = scale_features(
        df=df_shopping.copy(), 
        features=SHOPPING_CLUSTERING_FEATURES
    )
    if X_scaled_shopping is not None and RUN_CLUSTERING:
        find_optimal_k_and_plot(
            X_scaled_df=X_scaled_shopping, 
            max_k=10, 
            target_name="Shopping Customers"
        )
        
    if not RUN_CLUSTERING:
        print("K-Optimal analysis skipped (RUN_CLUSTERING=False). Proceeding with final K-Means.")
        
else:
    print("Clustering steps skipped (Data successfully loaded or calculation not required).")


# ----------------------------------------------------------------------
# 5.3 Final K-Means Execution (Runs only if data was NOT loaded and scaling succeeded)
# ----------------------------------------------------------------------

if SHOULD_CALCULATE_CLUSTERING: 
    print("\n--- RUNNING FINAL K-MEANS CALCULATION ---")
    
    # Superstore Final Clustering
    if X_scaled_superstore is not None:
        df_superstore_clustered = run_final_clustering(
            df=df_superstore_clustered, 
            X_scaled_df=X_scaled_superstore, 
            features=SUPERSTORE_CLUSTERING_FEATURES, 
            optimal_k=SUPERSTORE_OPTIMAL_K,
            data_name="Superstore Data"
        )

    # Shopping Final Clustering
    if X_scaled_shopping is not None:
        df_shopping_clustered = run_final_clustering(
            df=df_shopping_clustered, 
            X_scaled_df=X_scaled_shopping, 
            features=SHOPPING_CLUSTERING_FEATURES, 
            optimal_k=SHOPPING_OPTIMAL_K,
            data_name="Shopping Data"
        )
        
# ----------------------------------------------------------------------
# 5.4 FINAL DATA ASSIGNMENT
# ----------------------------------------------------------------------
# Ensure your main DFs are updated for subsequent ML steps
df_superstore = df_superstore_clustered
df_shopping = df_shopping_clustered

print("All clustering steps completed or skipped. Data is ready for the next ML step.")


# =========================================================
# 6. CLASSIFICATION TARGET AND FEATURE DEFINITIONS
# =========================================================

# --- SUPERSTORE CLASSIFICATION TARGETS & FEATURES ---

# 1. Create the binary target column: 1 for Profit (> 0), 0 for Loss (<= 0)
# IMPORTANT: This line must run BEFORE calling the classification function!
if 'Profit_YJ' in df_superstore.columns:
    df_superstore['Profit_Binary'] = (df_superstore['Profit_YJ'] > 0).astype(int)
    
SUPERSTORE_CLASSIFICATION_TARGET = 'Profit_Binary' 
SUPERSTORE_CLASSIFICATION_FEATURES = [
    'Sales_Log', 
    'ShippingCost_Log', 
    'Quantity', 
    'Discount', 
    'Category_Technology'
]

# --- SHOPPING CLASSIFICATION TARGETS & FEATURES ---

# IMPORTANT: The target for Shopping is your newly created cluster column.
SHOPPING_CLASSIFICATION_TARGET = 'Cluster_KMeans'
SHOPPING_CLASSIFICATION_FEATURES = [
    'review_rating', 'discount_applied_bin', 'promo_code_used_bin',
    'age', 'previous_purchases', 'subscription_status_bin',
]

def train_and_evaluate_classifier(df: pd.DataFrame, target: str, features: list, model_type: str, model_name: str, log_prefix: str):
    """
    Trains and evaluates a classification model (Logistic Regression, RandomForest, or Gaussian Naive Bayes).
    Saves model, metrics, and classification report.
    """
    if df is None or target not in df.columns or not all(f in df.columns for f in features):
        print(f"ERROR: Missing data or features for Classification training: {log_prefix}")
        return

    savelog(f"\n--- {log_prefix}: {model_type} Classification ---")

    # 1. Prepare Data
    X = df[features]
    y = df[target]

    # 2. Select Model based on model_type
    
    if model_type == "Logistic Regression":
        # Logistisk Regression (Baseline 1)
        model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    elif model_type == "Random Forest Classifier":
        # Random Forest (Ensemble/Multi-class)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Gaussian Naive Bayes": # NEW MODEL TYPE
        # Gaussian Naive Bayes (Effektiv Baseline)
        model = GaussianNB()
    else:
        print(f"WARNING: Unknown model_type '{model_type}'. Skipping.")
        return

    # 3. Split Data (standard 80/20)
    # Stratify sikrer ensartet fordeling af target-klasser i train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 5. Evaluate the model
    # ... (Resten af evalueringslogikken er den samme) ...
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Extract key metrics
    metrics = {
        'Model Type': model_type,
        'Accuracy': accuracy,
        'Macro Avg Precision': report['macro avg']['precision'],
        'Macro Avg Recall': report['macro avg']['recall'],
        'Macro Avg F1-Score': report['macro avg']['f1-score'],
    }
    
    # 6. Log and Save Results
    
    # Save the detailed classification report as JSON
    savejson(report, f"{log_prefix.lower().replace(' ', '_')}_{model_name.split('_')[0]}_report.json")
    
    # Log summary metrics
    savelog(f"{model_type} Summary Metrics (Classes={len(y.unique())}):")
    savelog(f"  Test Accuracy: {accuracy:.4f}")
    savelog(f"  Test F1-Score (Macro Avg): {metrics['Macro Avg F1-Score']:.4f}")
    
    # Save the trained model
    savemodel(model, model_name)
    print(f"Model saved as {model_name}")

    return metrics, model, X_test, y_test


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, class_names: list, data_name: str, model_name: str):
    """
    Generates and saves a Confusion Matrix heatmap for a given model and test set.
    """
    savelog(f"Generating Confusion Matrix for {model_name}...")
    
    # 1. Prediction
    y_pred = model.predict(X_test)
    
    # 2. Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 3. Create Display Object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Customize the plot
    plt.title(f'Confusion Matrix - {data_name} ({model_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Ensure filename is safe (fjerner spaces og /)
    safe_data_name = data_name.lower().replace(' ', '_').replace('/', '-')
    safe_model_name = model_name.split('.')[0]
    
    # 5. Save the figure
    savefig(f"confusion_matrix_{safe_data_name}_{safe_model_name}.png")
    savelog(f"Confusion Matrix saved: confusion_matrix_{safe_data_name}_{safe_model_name}.png")


    # 6.3 Superstore Profit/Loss (Gaussian Naive Bayes)
superstore_nb_metrics, superstore_nb_model, X_test_superstore_nb, y_test_superstore_nb = train_and_evaluate_classifier(
    df=df_superstore.copy(), 
    target=SUPERSTORE_CLASSIFICATION_TARGET, 
    features=SUPERSTORE_CLASSIFICATION_FEATURES,
    model_type="Gaussian Naive Bayes", 
    model_name="superstore_profit_nb.joblib",
    log_prefix="Superstore Profit/Loss"
)

plot_confusion_matrix(
    model=superstore_nb_model, 
    X_test=X_test_superstore_nb, 
    y_test=y_test_superstore_nb, 
    class_names=['Loss (0)', 'Profit (1)'],
    data_name="Superstore Data", 
    model_name="Gaussian NB"
)

# 6.4 Shopping Cluster Prediction (Gaussian Naive Bayes)
shopping_nb_metrics, shopping_nb_model, X_test_shopping_nb, y_test_shopping_nb = train_and_evaluate_classifier(
    df=df_shopping.copy(), 
    target=SHOPPING_CLASSIFICATION_TARGET, 
    features=SHOPPING_CLASSIFICATION_FEATURES,
    model_type="Gaussian Naive Bayes", # BRUG DEN NYE TYPE
    model_name="shopping_cluster_nb.joblib",
    log_prefix="Shopping Cluster Prediction"
)

cluster_names = [f'Cluster {c}' for c in sorted(df_shopping['Cluster_KMeans'].unique())]
plot_confusion_matrix(
    model=shopping_nb_model, 
    X_test=X_test_shopping_nb, 
    y_test=y_test_shopping_nb, 
    class_names=cluster_names, # Klyngeklassenavne
    data_name="Shopping Data", 
    model_name="Gaussian NB"
)