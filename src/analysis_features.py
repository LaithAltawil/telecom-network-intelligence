import dask.dataframe as dd
import dask.array as da
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb # For feature importance

# --- CONFIGURATION ---
# Point this at your own RAN KPI export (same schema as the ETL output).
file_path = os.path.join('data', 'raw_kpis.csv')
id_cols = ['CEll ID', 'Date Id', 'Gov', 'Row Id', 'Site ID'] # Columns to ignore
# (Add your full list of 88 feature columns here)
feature_cols = [
    'NPM CSFB Prep SR', 'NPM DL PRB Utilization', 'NPM DL Traffic GB',
    'NPM DL User Throughput', 'NPM Down time Auto', 'NPM Down time Total',
    'Npm Erab Sr', 'NPM Intrafreq HO SR', 'Npm Rrc Sr', 'NPM RRC SR Retx',
    'NPM RRC Users Avg', 'NPM RRC Users Max', 'Npm S1 Sr',
    'NPM Service Drop Rate', 'NPM UL PRB Utilization', 'NPM UL Traffic GB',
    'Pm Cell Down Time Auto', 'Pm Cell Down Time Man',
    'Pm Cell Ho Exe Succ Lte Intra F', 'Pm Cell Ho Prep Att Lte Intra F',
    'Pm Erab Estab Att Added', 'Pm Erab Estab Att Init',
    'Pm Erab Estab Succ Added', 'Pm Erab Estab Succ Init',
    'Pm Erab Rel Abnormal Enb', 'Pm Erab Rel Abnormal Enb Act',
    'Pm Erab Rel Mme', 'Pm Erab Rel Normal Enb', 'Pm Ho Exe Succ',
    'Pm Ho Prep Att', 'Pm Ho Prep Succ Csfb', 'Pm Mac Harq Dl Ack Qpsk',
    'Pm Mac Harq Dl Dtx Qpsk', 'Pm Mac Harq Dl Nack Qpsk',
    'Pm Mac Harq Ul Dtx Qpsk', 'Pm Mac Harq Ul Fail Qpsk',
    'Pm Mac Harq Ul Succ Qpsk', 'Pm Pdcp Vol Dl Drb',
    'Pm Pdcp Vol Dl Drb Last TTI', 'Pm Pdcp Vol Dl Srb', 'Pm Pdcp Vol Ul Drb',
    'Pm Pdcp Vol Ul Srb', 'Pm Prb Avail Dl', 'Pm Prb Avail Ul',
    'Pm Prb Used Dl Ailg Tot', 'Pm Prb Used Dl Bcch', 'Pm Prb Used Dl Dtch',
    'Pm Prb Used Dl Pcch', 'Pm Prb Used Dl Srb First Trans',
    'Pm Prb Used Ul Ailg Tot', 'Pm Prb Used Ul Dtch', 'Pm Prb Used Ul Srb',
    'Pm Rrc Conn Estab Att', 'Pm Rrc Conn Estab Att Reatt',
    'Pm Rrc Conn Estab Succ', 'Pm Rrc Conn Lev Samp', 'Pm Rrc Conn Lev Sum',
    'Pm Rrc Conn Max', 'Pm Sched Activity Cell Dl', 'Pm Ue Ctxt Estab Att Csfb',
    'Pm Ue Ctxt Mod Att Csfb', 'Pm Ue Ctxt Rel Csfb Wcdma', 'Pm Ue Thp Time Dl',
    'Pm Ue Thp Time Ul', 'Pm Ue Thp Vol Ul', 'pmMacHarqDlAck16qam',
    'pmMacHarqDlAck256qam', 'pmMacHarqDlAck64qam', 'pmMacHarqDlDtx16qam',
    'pmMacHarqDlDtx256qam', 'pmMacHarqDlDtx64qam', 'pmMacHarqDlNack16qam',
    'pmMacHarqDlNack256qam', 'pmMacHarqDlNack64qam', 'pmMacHarqUlDtx16qam',
    'pmMacHarqUlDtx64Qam', 'pmMacHarqUlFail16qam', 'pmMacHarqUlFail64Qam',
    'pmMacHarqUlSucc16qam', 'pmMacHarqUlSucc64Qam', 'pmS1SigConnEstabAtt',
    'pmS1SigConnEstabSucc'
]

# 1. LOAD & CLEAN
df = dd.read_csv(
    file_path,
    usecols=id_cols + feature_cols,
    blocksize='64MB',
    na_values=['∞', 'inf', '-inf'],
    dtype={'NPM CSFB Prep SR': 'float64', 'NPM Service Drop Rate': 'float64'}
)

# Fill NaNs with 0 (assuming 0 traffic/activity for missing values)
df_clean = df.fillna(0)
X_dask = df_clean[feature_cols]

# Normalize (Critical for PCA and Correlation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dask.to_dask_array(lengths=True))

# Create a sampled version (e.g., 5% of data) for expensive plots
# We compute this ONCE into memory to speed up plotting
df_sample = df_clean[feature_cols].sample(frac=0.05).compute()
print(f"Sample shape for visualization: {df_sample.shape}")


def analyze_correlations(dask_df, sample_df, threshold=0.9):
    print("Computing Correlation Matrix (Full Dataset)...")
    # Dask computes correlation efficiently on large data
    corr_matrix = dask_df.corr().compute()

    # 1. HEATMAP (Grouped)
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix ({len(dask_df.columns)} Features)")
    plt.show()

    # 2. LIST HIGH CORRELATIONS (Multicollinearity)
    # We only look at the upper triangle to avoid duplicates
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column].abs() > threshold)]

    print(f"\n--- HIGHLY CORRELATED PAIRS (> {threshold}) ---")
    print(f"Detected {len(to_drop)} redundant features.")

    # Print specific pairs
    for col in upper.columns:
        high_corr = upper.index[upper[col].abs() > threshold].tolist()
        if high_corr:
            print(f"{col} is redundant with: {high_corr}")

    return corr_matrix


# Run it
corr_mat = analyze_correlations(X_dask, df_sample)


def plot_dendrogram(corr_matrix):
    # We use the Correlation Matrix distance (1 - correlation)
    # Features with correlation 1.0 have distance 0.
    # --- FIX START ---
    # 1. Replace infinite values with NaN just in case
    corr_matrix = corr_matrix.replace([np.inf, -np.inf], np.nan)

    # 2. Drop rows/columns that are completely NaN (caused by zero-variance features)
    # We drop from both index (rows) and columns to keep the matrix square (88x88 -> 85x85)
    corr_matrix = corr_matrix.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 3. Fill any remaining individual NaNs with 0 (assumes no correlation)
    # This acts as a final safety net.
    corr_matrix = corr_matrix.fillna(0)
    # --- FIX END ---
    distance_matrix = 1 - abs(corr_matrix)

    # Use Ward's linkage
    linked = linkage(distance_matrix, 'ward')

    plt.figure(figsize=(15, 8))
    dendrogram(
        linked,
        orientation='top',
        labels=corr_matrix.columns,
        distance_sort='descending',
        show_leaf_counts=True,
        leaf_rotation=90,
        leaf_font_size=8
    )
    plt.title("Feature Clustering Dendrogram")
    plt.ylabel("Distance (1 - Correlation)")
    plt.tight_layout()
    plt.show()


# Run it
plot_dendrogram(corr_mat)


def plot_pca_biplot(X_scaled_dask, feature_names):
    # Fit PCA
    pca = PCA(n_components=2, svd_solver='randomized')
    pca.power_iteration_normalizer = 'LU'  # Fix for sklearn version
    pca.fit(X_scaled_dask)

    # Get components (loadings)
    loadings = pca.components_.T  # Shape: (88, 2)

    # Transform a SAMPLE of the data (plotting 1.4M points is messy)
    # We take the first 5000 points from our sample
    X_sample_pca = pca.transform(X_scaled_dask).compute()[:5000]

    # Scale points for visualization
    xs = X_sample_pca[:, 0]
    ys = X_sample_pca[:, 1]

    # Scale arrows so they are visible over the scatter plot
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    plt.figure(figsize=(14, 14))

    # 1. Plot the Scatter points (Rows)
    plt.scatter(xs * scalex, ys * scaley, c='lightgray', alpha=0.5, s=5)

    # 2. Plot the Arrows (Features)
    # To avoid 88 messy arrows, we only plot the top 15 longest ones
    magnitude = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_indices = np.argsort(magnitude)[-15:]

    for i in top_indices:
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.9)
        plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature_names[i],
                 color='darkred', ha='center', va='center')

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA Biplot: Top Contributing Features")
    plt.grid()
    plt.show()


# Run it
plot_pca_biplot(X_scaled, feature_cols)


def feature_importance_xgb(X_sample, target_col='NPM DL User Throughput'):
    # Prepare X (Features) and y (Target)
    y = X_sample[target_col]
    X = X_sample.drop(columns=[target_col])

    # Train a quick XGBoost model on the SAMPLE (fast and sufficient for ranking)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, n_jobs=-1)
    model.fit(X, y)

    # Plot importance
    plt.figure(figsize=(10, 15))
    xgb.plot_importance(model, max_num_features=20, height=0.5)
    plt.title(f"What drives '{target_col}'?")
    plt.show()

    # Use Mutual Information as a cross-check (Captures Non-Linearity)
    mi_scores = mutual_info_regression(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    print("\nTop 5 Features by Mutual Information:")
    print(mi_series.head(5))


# Run it (Ensure 'NPM DL User Throughput' is in your columns, or swap it)
if 'NPM DL User Throughput' in df_sample.columns:
    feature_importance_xgb(df_sample, target_col='NPM DL User Throughput')


cols_to_drop = [
    "NPM RRC Users Max",
    "NPM UL PRB Utilization",
    "Pm Cell Down Time Auto",
    "Pm Cell Ho Prep Att Lte Intra F",
    "Pm Erab Estab Att Init",
    "Pm Erab Estab Succ Added",
    "Pm Erab Estab Succ Init",
    "Pm Erab Rel Normal Enb",
    "Pm Ho Prep Att",
    "Pm Mac Harq Ul Succ Qpsk",
    "Pm Pdcp Vol Dl Srb",
    "Pm Pdcp Vol Ul Srb",
    "Pm Prb Avail Ul",
    "Pm Prb Used Dl Srb First Trans",
    "Pm Rrc Conn Estab Att",
    "Pm Rrc Conn Estab Succ",
    "Pm Rrc Conn Lev Sum",
    "Pm Rrc Conn Max",
    "Pm Sched Activity Cell Dl",
    "Pm Ue Ctxt Rel Csfb Wcdma",
    "Pm Ue Thp Time Ul",
    "Pm Ue Thp Vol Ul",
    "pmMacHarqDlNack16qam",
    "pmMacHarqDlNack256qam",
    "pmMacHarqDlNack64qam",
    "pmS1SigConnEstabAtt",
    "pmS1SigConnEstabSucc",
    "Npm Rrc Sr",
    "NPM RRC SR Retx",
    "Npm S1 Sr",
    "Pm Pdcp Vol Dl Drb",
    "Pm Ue Thp Time Dl",
    "Pm Pdcp Vol Dl Drb Last TTI"
]



