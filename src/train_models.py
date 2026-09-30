import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, r2_score, mean_absolute_error

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
# Point this at your own RAN KPI export (same schema as the ETL output).
file_path = os.path.join('data', 'raw_kpis.csv')

# Common "Echo" columns to drop from everything
common_drops = [
    'CEll ID', 'Date Id', 'Gov', 'Row Id', 'Site ID',
    "NPM RRC Users Max", "NPM UL PRB Utilization", "Pm Cell Down Time Auto",
    "Pm Cell Ho Prep Att Lte Intra F", "Pm Erab Estab Att Init",
    "Pm Erab Estab Succ Added", "Pm Erab Estab Succ Init", "Pm Erab Rel Normal Enb",
    "Pm Ho Prep Att", "Pm Mac Harq Ul Succ Qpsk", "Pm Pdcp Vol Dl Srb",
    "Pm Pdcp Vol Ul Srb", "Pm Prb Avail Ul", "Pm Prb Used Dl Srb First Trans",
    "Pm Rrc Conn Estab Att", "Pm Rrc Conn Estab Succ", "Pm Rrc Conn Lev Sum",
    "Pm Rrc Conn Max", "Pm Sched Activity Cell Dl", "Pm Ue Ctxt Rel Csfb Wcdma",
    "Pm Ue Thp Time Ul", "Pm Ue Thp Vol Ul", "pmS1SigConnEstabAtt",
    "pmS1SigConnEstabSucc", "Npm Rrc Sr", "NPM RRC SR Retx", "Npm S1 Sr"
]

print("Loading data...")
df = dd.read_csv(
    file_path,
    blocksize='64MB',
    na_values=['∞', 'inf', '-inf'],
    dtype={'NPM CSFB Prep SR': 'float64', 'NPM Service Drop Rate': 'float64', 'NPM DL PRB Utilization': 'float64'}
)
df_clean = df.drop(columns=common_drops, errors='ignore').fillna(0)

# Use 100% of data for final training (or keep sampling if RAM is tight)
# df_final = df_clean.compute()
print("Sampling 100% for final training...")
df_final = df_clean.sample(frac=1).compute()

# ==========================================
# 2. FINAL VALIDATED DROP LISTS
# ==========================================

# Model 1: THROUGHPUT (Speed)
# Validated Strategy: Drop Leakage (Vol/Time) + Drop Nacks + Drop Useless Features
drops_throughput = [
    'Pm Pdcp Vol Dl Drb', 'Pm Ue Thp Time Dl', 'Pm Pdcp Vol Dl Drb Last TTI',
    'NPM DL Traffic GB', 'pmMacHarqDlNack16qam', 'pmMacHarqDlNack256qam',
    'pmMacHarqDlNack64qam', 'Pm Erab Rel Abnormal Enb', 'NPM Down time Auto',
    'Pm Ho Exe Succ', 'NPM Down time Total', 'Pm Cell Down Time Man',
    'Pm Prb Used Ul Ailg Tot', 'Pm Prb Used Dl Ailg Tot', 'Pm Ho Prep Succ Csfb'
]

# Model 2: FAULTS (Classifier)
# Validated Strategy: Drop Successes + Drop Numerator Counters + Drop Useless Features
drops_faults = [
    'pmMacHarqDlAck16qam', 'pmMacHarqDlAck64qam', 'pmMacHarqDlAck256qam',
    'Pm Mac Harq Dl Ack Qpsk', 'Pm Erab Rel Abnormal Enb',
    'Pm Erab Rel Abnormal Enb Act', 'Pm Erab Rel Mme', 'NPM Down time Auto',
    'Pm Prb Used Dl Ailg Tot', 'Pm Prb Used Ul Ailg Tot', 'Pm Ho Prep Succ Csfb',
    'NPM CSFB Prep SR'
]

# Model 3: ACCESSIBILITY (Success Rate)
# Validated Strategy: Similar to Faults (Focus on Failure Drivers)
drops_access = [
    'pmMacHarqDlAck16qam', 'pmMacHarqDlAck64qam', 'pmMacHarqDlAck256qam',
    'Pm Mac Harq Dl Ack Qpsk', 'Pm Erab Rel Abnormal Enb',
    'Pm Erab Rel Abnormal Enb Act', 'Pm Erab Rel Mme', 'NPM Down time Auto',
    'Pm Prb Used Ul Ailg Tot', 'NPM Down time Total', 'Pm Ho Exe Succ',
    'Pm Cell Down Time Man', 'Pm Ho Prep Succ Csfb', 'Pm Prb Used Dl Ailg Tot',
    'Pm Rrc Conn Lev Samp'
]


# ==========================================
# 3. TRAINING ENGINE
# ==========================================

def train_regressor(target, drops, name):
    print(f"\n--- Training Final {name} Model (Regression) ---")
    if target not in df_final.columns: return None

    X = df_final.drop(columns=[target] + drops, errors='ignore')
    y = df_final[target]

    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1)
    model.fit(X, y)

    # Simple Validation Print
    score = model.score(X, y)
    print(f"{name} Final Training R²: {score:.4f}")

    plt.figure(figsize=(15, 8))
    xgb.plot_importance(model, max_num_features=12, importance_type='weight', title=f"Top Drivers: {name}")
    plt.show()
    return model


def train_classifier(target, drops, name, threshold=0.1):
    print(f"\n--- Training Final {name} Model (Classification) ---")
    if target not in df_final.columns: return None

    # Create Binary Target
    y_binary = (df_final[target] > threshold).astype(int)
    X = df_final.drop(columns=[target] + drops, errors='ignore')

    # Calculate scale_pos_weight to handle imbalance
    scale = sum(y_binary == 0) / sum(y_binary == 1)
    print(f"Imbalance Ratio: {scale:.2f} (Using this to boost rare faults)")

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=scale, n_jobs=-1)
    model.fit(X, y_binary)

    print("\nClassification Report:")
    print(classification_report(y_binary, model.predict(X)))

    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model ,max_num_features=12, importance_type='weight', title=f"Top Drivers: {name}")
    plt.show()
    return model


# ==========================================
# 4. EXECUTION
# ==========================================

# 1. THROUGHPUT (REGRESSION)
model_throughput = train_regressor(
    target='NPM DL User Throughput',
    drops=drops_throughput,
    name="Throughput"
)

# 2. ACCESSIBILITY (REGRESSION)
model_access = train_regressor(
    target='Npm Erab Sr',
    drops=drops_access,
    name="Accessibility"
)

# 3. FAULTS (CLASSIFICATION)
# Note: Threshold 0.5 means "Drop Rate > 0.5%" is considered a Fault.
model_faults = train_classifier(
    target='NPM Service Drop Rate',
    drops=drops_faults,
    name="Fault Detection",
    threshold=0.1
)

import joblib

# Create a folder for your models
if not os.path.exists('production_models'):
    os.makedirs('production_models')

print("Saving production models to disk...")

# 1. Save Throughput Model
joblib.dump(model_throughput, 'production_models/throughput_xgb_v1.pkl')

# 2. Save Accessibility Model
joblib.dump(model_access, 'production_models/accessibility_xgb_v1.pkl')

# 3. Save Faults Model
joblib.dump(model_faults, 'production_models/fault_detector_clf_v1.pkl')

print("Models saved successfully in 'production_models/' folder.")