import math
import sqlite3
import time
import os
import joblib
import base64
from datetime import datetime
import io

# Data & UI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, cross_val_score
# ML Imports
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Network Intelligence", layout="wide", page_icon="📡")
DB_PATH = 'network_kpis.db'
TABLE_NAME = 'network_kpis'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #FF7900;}
    div.stButton > button {background-color: #FF7900; color: white;}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. OPTIMIZED DATA LOADING (Pure Pandas)
# ==========================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_filtered_data(site_id=None, date_range=None):
    """
    Optimized loader: Pushes filtering to the SQL engine.
    """
    query = f"SELECT * FROM {TABLE_NAME}"
    params = []
    conditions = []

    # 1. SQL Level Filtering (The Speed Boost)
    if site_id and site_id != 'All':
        conditions.append("site_id = ?")
        params.append(site_id)

    # Optional: Filter by date at SQL level too if needed
    # if date_range:
    #     conditions.append("date_id BETWEEN ? AND ?")
    #     params.extend([date_range[0], date_range[1]])

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # 2. Fetch Data
    with sqlite3.connect(DB_PATH) as conn:
        # Use Pandas directly - much faster than Dask for filtered SQL queries
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return df

    # 3. Fast Vectorized Feature Engineering
    # Pre-calculate useful constants
    df['date_id'] = pd.to_datetime(df['date_id'])

    # Vectorized Sum
    ack_cols = ['pm_mac_harq_dl_ack_qpsk', 'pm_mac_harq_dl_ack_16qam',
                'pm_mac_harq_dl_ack_64qam', 'pm_mac_harq_dl_ack_256qam']

    # Fill NAs only on specific columns to save time, or globally if needed
    df[ack_cols] = df[ack_cols].fillna(0)

    total_ack = df[ack_cols].sum(axis=1)

    # Avoid div by zero using numpy where (faster than mask)
    total_ack = np.where(total_ack == 0, 1, total_ack)

    df['weak_signal_ratio'] = df['pm_mac_harq_dl_ack_qpsk'] / total_ack
    df['throughput_mbps'] = df['npm_dl_user_throughput'] / 1e9

    return df.fillna(0)


# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
# --- Sidebar Controls ---
st.sidebar.title("📡 Controls")


@st.cache_data
def get_site_list():
    """Cached distinct site list"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # ONLY fetch the distinct column, not the whole table
            sites = pd.read_sql(f"SELECT DISTINCT site_id FROM {TABLE_NAME}", conn)['site_id'].tolist()
        sites.sort()
        return sites
    except Exception as e:
        st.error(f"Error fetching sites: {e}")
        return []


@st.cache_data
def get_date_bounds():
    """Cached date bounds"""
    with sqlite3.connect(DB_PATH) as conn:
        row = pd.read_sql_query(
            f"SELECT MIN(date_id) AS min_d, MAX(date_id) AS max_d FROM {TABLE_NAME}",
            conn
        ).iloc[0]
    min_d = pd.to_datetime(row["min_d"]).date()
    max_d = pd.to_datetime(row["max_d"]).date()
    return min_d, max_d


@st.cache_data
def get_random_sites(n=4):
    """Fast random sample"""
    with sqlite3.connect(DB_PATH) as conn:
        df_sites = pd.read_sql_query(
            f"SELECT DISTINCT site_id FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT {n}",
            conn
        )
    return sorted(df_sites["site_id"].dropna().astype(str).tolist())


# --- Sidebar Logic ---
if "random_sites" not in st.session_state:
    st.session_state.random_sites = get_random_sites(4)
# Initialize active tab in session state if it doesn't exist

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Network Health"

if st.sidebar.button("🎲 Reroll 4 random sites"):
    st.session_state.random_sites = get_random_sites(4)
    # No need to clear full cache, just the session state updates

random_sites = st.session_state.random_sites
selected_site = st.sidebar.selectbox(
    "Select Site ID",
    ["All (Caution)"] + random_sites
)

min_d, max_d = get_date_bounds()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

# Load Data
with st.status("Initializing Engine...", expanded=True) as status:
    st.write("Fetching SQL Records...")
    df = load_filtered_data(selected_site, date_range)
    st.write("Calculating Vectorized KPIs...")
    status.update(label="Data Ready!", state="complete", expanded=False)

with st.spinner(f"Fetching data for {selected_site}..."):
    if selected_site == 'All (Caution)':
        st.warning("⚠️ Loading ALL sites. This depends on your RAM.")
        df = load_filtered_data('All', date_range)
    else:
        df = load_filtered_data(selected_site, date_range)

# Filter by Date (in memory is fine now since the dataset is small per site)
if not df.empty and isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['date_id'].dt.date >= start_date) & (df['date_id'].dt.date <= end_date)
    df = df[mask]

if df.empty:
    st.error(f"No data found for Site {selected_site}")
    st.stop()
# ==========================================
# TAB 1: HEALTH & STATS
# ==========================================
# Create the tabs and assign a key to persist selection
tab_list = ["📊 Network Health", "🤖 Comparison Models", "🧠 Deep Learning", "🗺️ Clustering", "🚨 Anomalies", "LLM"]

# Using the 'key' parameter allows Streamlit to automatically manage the state
tabs = st.tabs(tab_list)

# Assign tabs to variables as you did before
tab1, tab2, tab3, tab4, tab5, tab6 = tabs
with tab1:
    st.header(f"Network Health: {selected_site}")
    avg_drop = df['npm_service_drop_rate'].mean()
    mpd_val = 1 / avg_drop if avg_drop > 0 else 9999 # 9999 means "Perfect/Infinity"

    # 2. Site-Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Speed", f"{df['throughput_mbps'].mean():.1f} Mbps",
              delta=f"{df['throughput_mbps'].mean() - 10:.1f} vs Target")
    c2.metric("Avg Congestion", f"{df['npm_dl_prb_utilization'].mean():.1f} %")
    c3.metric("Failure Rate", f"{avg_drop * 100:.2f} %", delta_color="inverse")
    c4.metric("Stability Score (MPD)", f"{mpd_val:.0f} mins",
              help="Minutes Per Drop: How long a user stays connected before a failure. Higher is Better.")

    st.markdown("---")

    # 3. Detailed Table (Added MPD)
    st.subheader("🔬 Cell-Level Performance")

    cell_stats = df.groupby('cell_id').agg({
        'throughput_mbps': ['mean', 'min', 'max'],
        'npm_dl_prb_utilization': ['mean', 'max'],
        'npm_service_drop_rate': ['mean']
    }).reset_index()

    # Flatten
    cell_stats.columns = ['Cell Name', 'T_Avg', 'T_Min', 'T_Max', 'L_Avg', 'L_Max', 'Drop_Rate']

    # Vectorized Calculation for Table
    cell_stats['Drop_Rate_Pct'] = cell_stats['Drop_Rate'] * 100
    # MPD for each cell (handle 0 drops -> set to 9999)
    cell_stats['MPD'] = cell_stats['Drop_Rate'].apply(lambda x: 1/x if x > 0.0001 else 9999)

    # Rename
    final_view = cell_stats[[
        'Cell Name', 'T_Avg', 'T_Min', 'T_Max',
        'L_Avg', 'L_Max', 'Drop_Rate_Pct', 'MPD'
    ]].rename(columns={
        'T_Avg': 'Avg Speed (Mbps)',
        'T_Min': 'Worst Speed',
        'T_Max': 'Best Speed',
        'L_Avg': 'Avg Congestion (%)',
        'L_Max': 'Peak Congestion (%)',
        'Drop_Rate_Pct': 'Failure Rate (%)',
        'MPD': 'Mins Per Drop'
    })

    # Style (MPD: Green = High/Long Time, Red = Low/Short Time)
    styler = final_view.style.background_gradient(subset=['Avg Speed (Mbps)', 'Worst Speed'], cmap='RdYlGn') \
                             .background_gradient(subset=['Avg Congestion (%)', 'Peak Congestion (%)', 'Failure Rate (%)'], cmap='RdYlGn_r') \
                             .background_gradient(subset=['Mins Per Drop'], cmap='RdYlGn') \
                             .format("{:.1f}")

    st.dataframe(styler, use_container_width=True)

    # 3. Drill-Down Charts
    st.subheader("📉 Time-Series Drill Down")
    selected_cell = st.selectbox("Select Cell to Inspect", df['cell_id'].unique())

    cell_df = df[df['cell_id'] == selected_cell]

    fig_ts = px.line(cell_df, x='date_id', y=['throughput_mbps', 'npm_dl_prb_utilization'],
                     title=f"Traffic vs Speed: {selected_cell}",
                     labels={'value': 'Value', 'variable': 'Metric'})
    st.plotly_chart(fig_ts, use_container_width=True)

    # 4. Correlation Heatmap (Moved to bottom)
    with st.expander("Show Correlation Heatmap"):
        corr_cols = ['throughput_mbps', 'npm_dl_prb_utilization', 'npm_dl_traffic_gb',
                     'npm_service_drop_rate', 'npm_rrc_users_avg', 'weak_signal_ratio']
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ==========================================
# TAB 2: ML COMPARISON
# ==========================================
with tab2:
    st.header("🤖 Model Comparison (Regression)")

    if st.button("🚀 Train & Compare Models"):
        features = ['npm_dl_prb_utilization', 'npm_dl_traffic_gb', 'weak_signal_ratio', 'npm_rrc_users_avg']
        target = 'throughput_mbps'

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = {
            "Linear Reg": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
        }

        results = []
        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):
            start = time.time()

            # 1. Train
            model.fit(X_train_s, y_train)

            # 2. Predict & Basic Metrics
            pred = model.predict(X_test_s)
            r2 = r2_score(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))

            # 3. Perform Cross-Validation (New Requirement)
            # Splits data 5 times to ensure results aren't just luck
            cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='r2')
            avg_cv = cv_scores.mean()

            results.append({
                "Model": name,
                "R2 Score": r2,
                "CV Score (Avg)": avg_cv,  # Added CV
                "RMSE": rmse,
                "Time (s)": time.time() - start
            })
            progress.progress((i + 1) / 3)

        res_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
        st.subheader("🏆 Model Leaderboard")
        st.dataframe(res_df.style.highlight_max(axis=0, subset=['R2 Score', 'CV Score (Avg)'], color='lightgreen'))

        # Get Best Model
        best_name = res_df.iloc[0]['Model']
        best_model = models[best_name]

        # 4. Explain Model Outputs (New Requirement: Feature Importance)
        st.subheader(f"🧠 Explaining the Best Model: {best_name}")
        c1, c2 = st.columns(2)

        with c1:
            st.write("### Predicted vs Actual")
            fig_pred = px.scatter(x=y_test, y=best_model.predict(X_test_s),
                                  labels={'x': 'Actual Speed', 'y': 'Predicted Speed'},
                                  title=f"Prediction Accuracy")
            fig_pred.add_shape(type="line", line=dict(dash='dash', color='grey'),
                               x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
            st.plotly_chart(fig_pred, use_container_width=True)

        with c2:
            st.write("### 🔍 Feature Importance")
            # Only Tree-based models (RF/XGB) have feature_importances_
            if hasattr(best_model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=True)

                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                 title="What drives network speed?",
                                 color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Linear Regression uses coefficients (shown below).")
                # Fallback for Linear Reg
                if hasattr(best_model, 'coef_'):
                    st.write(dict(zip(features, best_model.coef_)))

# ==========================================
# TAB 3: DEEP LEARNING
# ==========================================
with tab3:
    st.header("🧠 Advanced Multivariate LSTM")
    st.markdown("This model uses **Congestion, Traffic, and User Count** to predict future Throughput.")

    # --- 1. Hyperparameter Settings (New!) ---
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Training Epochs", min_value=10, max_value=500, value=50)
    lr = c2.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1], value=0.01)
    hidden_size = c3.selectbox("Hidden Neurons", [32, 64, 128], index=1)
    seq_len = c4.slider("Sequence Length (Hours)", 12, 168, 24)

    if st.button("⚡ Train Multivariate Model"):
        # --- 2. Multivariate Data Prep ---
        # We use 3 features now instead of just 1
        features = ['throughput_mbps', 'npm_dl_prb_utilization', 'npm_rrc_users_avg']

        # Resample to hourly to ensure consistent time-steps (Crucial for LSTM)
        # We assume data is roughly hourly; if not, this aggregates it nicely.
        ts_df = df.groupby('date_id')[features].mean().sort_index().fillna(0)

        if len(ts_df) < seq_len + 10:
            st.error(f"Not enough data points ({len(ts_df)}). Need at least {seq_len + 10}.")
            st.stop()

        # Scaling (Normalizing)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(ts_df)

        # Create Sequences
        X_all, y_all = [], []
        # Target is always index 0 (throughput_mbps)
        for i in range(len(scaled_data) - seq_len):
            X_all.append(scaled_data[i:i + seq_len])
            y_all.append(scaled_data[i + seq_len, 0])  # Predict only Throughput

        X_t = torch.tensor(np.array(X_all), dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(np.array(y_all), dtype=torch.float32).view(-1, 1).to(DEVICE)

        # Split Train/Test for Validation
        split_idx = int(len(X_t) * 0.8)
        X_train, X_val = X_t[:split_idx], X_t[split_idx:]
        y_train, y_val = y_t[:split_idx], y_t[split_idx:]


        # --- 3. Robust LSTM Architecture ---
        class MultiLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers=2, output_size=1):
                super(MultiLSTM, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # Auto-initialize hidden states
                h0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)
                c0 = torch.zeros(2, x.size(0), hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                return self.fc(out[:, -1, :])


        model = MultiLSTM(input_size=len(features), hidden_size=hidden_size, num_layers=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # --- 4. Training Loop with Visuals ---
        status_text = st.empty()
        progress_bar = st.progress(0)
        loss_history = []
        val_history = []

        model.train()
        for ep in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            # Validation check
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_loss = criterion(val_out, y_val)
            model.train()

            loss_history.append(loss.item())
            val_history.append(val_loss.item())

            if ep % 5 == 0:
                progress_bar.progress((ep + 1) / epochs)
                status_text.text(f"Epoch {ep + 1} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")

        st.success("Training Complete!")

        # --- 5. Advanced Visualizations ---

        # A. Loss Curve (Did it learn?)
        st.subheader("1. Training Diagnostics")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=loss_history, name="Train Loss", line=dict(color='blue')))
        fig_loss.add_trace(go.Scatter(y=val_history, name="Validation Loss", line=dict(color='orange')))
        fig_loss.update_layout(title="Learning Curve (Lower is Better)", xaxis_title="Epochs", yaxis_title="MSE Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

        # B. Forecast vs Actual
        st.subheader("2. Forecast Performance")
        model.eval()
        with torch.no_grad():
            train_preds = model(X_t).cpu().numpy()

        # Inverse Transform (Tricky part: we scaled 3 dims, need to inverse all 3)
        # We create a dummy matrix to allow inverse_transform to work
        dummy_mat = np.zeros((len(train_preds), len(features)))
        dummy_mat[:, 0] = train_preds.flatten()  # Fill 0th col with predictions
        preds_inv = scaler.inverse_transform(dummy_mat)[:, 0]

        dummy_mat_y = np.zeros((len(y_t), len(features)))
        dummy_mat_y[:, 0] = y_t.cpu().numpy().flatten()
        actuals_inv = scaler.inverse_transform(dummy_mat_y)[:, 0]

        # Dates for plotting
        dates = ts_df.index[seq_len:]

        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=dates, y=actuals_inv, name="Actual Data", line=dict(color='gray', width=1)))
        fig_res.add_trace(go.Scatter(x=dates, y=preds_inv, name="AI Prediction", line=dict(color='red', width=2)))
        st.plotly_chart(fig_res, use_container_width=True)

        # --- 6. Executive Interpretation (For Non-Tech Stakeholders) ---
        st.markdown("---")
        st.subheader("🔍 Executive Interpretation")

        # 1. ENSURE RESIDUALS IS DEFINED CORRECTLY HERE
        # We need to recalculate it just to be safe, or ensure it flows from the previous block
        residuals = actuals_inv - preds_inv

        with st.expander("📖 How to read these results (Click to Open)", expanded=True):
            st.markdown("""
                    **1. Is the AI learning? (The Learning Curve)**
                    * **Goal:** You want both lines (Blue and Orange) to go **down** over time.
                    * **Good:** If both lines flatten out at a low number, the AI has learned the patterns.
                    * **Bad:** If the Blue line goes down but the **Orange line goes up**, the AI is "memorizing" the data (Overfitting) rather than understanding it.

                    **2. How accurate is it? (Forecast vs Actual)**
                    * **Goal:** The **Red Line** (AI) should closely follow the **Grey Line** (Actual).
                    * **Insight:** If the Red line reacts to spikes (sharp ups and downs) at the same time as the Grey line, the AI successfully predicts network congestion events.

                    **3. Is the AI biased? (Error Distribution)**
                    * **Goal:** We want a "Bell Curve" centered at **0**.
                    * **Insight:** * **Centered at 0:** The AI is fair. Sometimes it guesses high, sometimes low, but on average it's correct.
                        * **Leaning Left/Right:** The AI has a "bias." It consistently thinks the speed is faster (or slower) than it really is.
                    """)

            # Dynamic Conclusion
            avg_rmse = np.sqrt(np.mean(residuals ** 2))
            avg_actual = np.mean(actuals_inv)

            # Avoid division by zero if actuals are 0
            if avg_actual == 0:
                error_percentage = 100
            else:
                error_percentage = (avg_rmse / avg_actual) * 100

            if error_percentage < 15:
                st.success(
                    f"✅ **Verdict:** The model is performing **Excellently**. The average error is only {error_percentage:.1f}% of the typical speed.")
            elif error_percentage < 30:
                st.warning(
                    f"⚠️ **Verdict:** The model is **Acceptable**, but could be improved. The error margin is around {error_percentage:.1f}%.")
            else:
                st.error(
                    f"❌ **Verdict:** The model is **Unreliable** (Error margin: {error_percentage:.1f}%). It may need more data or different settings.")
        # C. Residual Analysis (Is the model biased?)
        st.subheader("3. Error Analysis")
        residuals = actuals_inv - preds_inv

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            fig_hist = px.histogram(x=residuals, nbins=30, title="Error Distribution (Should be centered at 0)")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_res2:
            st.metric("Mean Absolute Error (MAE)", f"{np.mean(np.abs(residuals)):.2f} Mbps")
            st.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(np.mean(residuals ** 2)):.2f} Mbps")

# ==========================================
# TAB 4: CLUSTERING
# ==========================================
with tab4:
    st.header("🗺️ Intelligent Site Segmentation")
    st.markdown(
        "We use **K-Means Clustering** to group cells with similar behavior, then apply **Logic Rules** to tag them with business categories.")

    if st.button("📍 Segment Cells"):
        # 1. Prepare Data
        # Group by cell to get site-level averages
        clus_df = df.groupby('cell_id')[
            ['npm_dl_prb_utilization', 'throughput_mbps', 'npm_service_drop_rate']].mean().reset_index()

        # 2. Normalize Data
        # We use MinMaxScaler to bring Load (0-100) and Speed (0-300) into the same 0-1 range
        scaler = MinMaxScaler()
        clus_df[['load_scaled', 'speed_scaled']] = scaler.fit_transform(
            clus_df[['npm_dl_prb_utilization', 'throughput_mbps']])

        if len(clus_df) < 3:
            st.error("⚠️ Not enough data. Need at least 3 distinct cells to perform clustering.")
        else:
            # 3. Fit K-Means
            # We use 4 clusters to try and catch: Healthy, Congested, Interference, Idle
            kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(clus_df[['load_scaled', 'speed_scaled']])
            clus_df['cluster'] = cluster_labels

            # --- METRIC: Silhouette Score (New Requirement!) ---
            # Measures how well-separated the clusters are (-1 to 1)
            sil_score = silhouette_score(clus_df[['load_scaled', 'speed_scaled']], cluster_labels)

            # 4. Business Logic Labeling (Hybrid Approach)
            # We use the clusters to find groups, but apply rules to NAME them accurately

            # Detect Load Scale (Some systems use 0-1, others 0-100)
            is_percent = clus_df['npm_dl_prb_utilization'].max() > 1.0
            thresh_load = 75.0 if is_percent else 0.75
            thresh_speed_low = 10.0  # Mbps


            def get_profile(row):
                # RULE 1: Congestion (High Load, Low Speed)
                if row['npm_dl_prb_utilization'] > thresh_load:
                    return 'Congested 🔴'

                # RULE 2: Quality Risk (Low Load but Low Speed) - Likely Interference/Coverage
                elif row['throughput_mbps'] < thresh_speed_low:
                    # Check if it's just idle (no traffic) or actually bad
                    if row['npm_dl_prb_utilization'] > (5.0 if is_percent else 0.05):
                        return 'Interference/Coverage 🟠'
                    else:
                        return 'Idle/Sleep ⚪'

                # RULE 3: Healthy
                else:
                    return 'Healthy 🟢'


            clus_df['Profile'] = clus_df.apply(get_profile, axis=1)

            # 5. Advanced Visualization
            st.subheader("📊 Network Risk Map")

            color_map = {
                'Congested 🔴': '#FF4B4B',  # Red
                'Interference/Coverage 🟠': '#FFA500',  # Orange
                'Healthy 🟢': '#00CC96',  # Green
                'Idle ⚪': '#E0E0E0'  # Grey
            }

            fig_c = px.scatter(clus_df, x='npm_dl_prb_utilization', y='throughput_mbps',
                               color='Profile', color_discrete_map=color_map,
                               hover_data=['cell_id', 'npm_service_drop_rate'],
                               title=f"Cluster Analysis (Silhouette Score: {sil_score:.2f})",
                               labels={'npm_dl_prb_utilization': 'Congestion (Load %)',
                                       'throughput_mbps': 'User Speed (Mbps)'})

            # Add Threshold Lines (The "Danger Zones")
            fig_c.add_vline(x=thresh_load, line_width=1, line_dash="dash", line_color="red",
                            annotation_text="High Load Limit")
            fig_c.add_hline(y=thresh_speed_low, line_width=1, line_dash="dash", line_color="orange",
                            annotation_text="Min Speed Limit")

            st.plotly_chart(fig_c, use_container_width=True)

            # 6. Cluster Summaries (New!)
            st.subheader("📋 Cluster Profiles")
            summary = clus_df.groupby('Profile').agg({
                'cell_id': 'count',
                'throughput_mbps': 'mean',
                'npm_dl_prb_utilization': 'mean',
                'npm_service_drop_rate': 'mean'
            }).rename(columns={
                'cell_id': 'Count',
                'throughput_mbps': 'Avg Speed',
                'npm_dl_prb_utilization': 'Avg Load',
                'npm_service_drop_rate': 'Avg Drops'
            })
            st.dataframe(summary.style.highlight_max(axis=0, color='lightgrey').format("{:.2f}"))

            # 7. Executive Interpretation
            st.markdown("---")
            with st.expander("🧐 How to interpret these results (Click to Open)", expanded=True):
                st.markdown(f"""
                ### 1. The Categories
                * **Congested 🔴 (Load > {thresh_load}%):** These cells are "full." Too many users are sharing the bandwidth.
                    * **Action:** Needs Capacity Expansion (New spectrum, new carriers, or physical tilt).
                * **Interference/Coverage 🟠 (Low Speed, Low Load):** These cells are empty but *still slow*. This is usually technical.
                    * **Action:** Check for Interference, Overshooting, or Transmission links.
                * **Healthy 🟢:** Fast speeds and reasonable load. No action needed.

                ### 2. The Metric (Silhouette Score: {sil_score:.2f})
                * **What it means:** Measures how distinct the groups are.
                * **Range:** -1 (Wrong) to +1 (Perfect). 
                * **Your Score:** A score above **0.4** is generally considered "Good" for real-world data.
                """)

            # 8. Actionable Downloads
            c1, c2 = st.columns(2)
            with c1:
                bad_cells = clus_df[clus_df['Profile'].str.contains('Congested')]
                if not bad_cells.empty:
                    st.download_button("📥 Download Congested List",
                                       bad_cells.to_csv(index=False),
                                       "congested_cells.csv", "text/csv")
                else:
                    st.info("No Congested Cells found.")

            with c2:
                risk_cells = clus_df[clus_df['Profile'].str.contains('Interference')]
                if not risk_cells.empty:
                    st.download_button("📥 Download Quality Risk List",
                                       risk_cells.to_csv(index=False),
                                       "risk_cells.csv", "text/csv")
                else:
                    st.info("No Quality Risk Cells found.")
with tab5:
    st.header("🚨 Intelligent Anomaly Detection")
    st.markdown("""
    **System Status:** Running **Isolation Forest** Algorithm.
    This AI looks for data points that are "rare" and "different" from the standard network pattern.
    """)

    col_cfg1, col_cfg2 = st.columns([1, 2])
    with col_cfg1:
        contamination = st.slider("Sensitivity (How strict?)", 0.01, 0.2, 0.05,
                                  help="0.01 = Only flag the craziest 1%. 0.2 = Flag top 20% deviations.")

    # Run Analysis
    if st.button("🔍 Scan Network for Anomalies"):
        # 1. Feature Selection & Model Fitting
        features_iso = ['throughput_mbps', 'npm_service_drop_rate', 'npm_dl_prb_utilization']

        # Train Model
        iso = IsolationForest(contamination=contamination, random_state=42)
        df_iso = df.copy()  # Work on a copy to be safe

        # Fit & Predict
        iso.fit(df_iso[features_iso])
        df_iso['anomaly'] = iso.predict(df_iso[features_iso])
        df_iso['anomaly_score'] = iso.decision_function(df_iso[features_iso])  # Lower = More Anomalous

        # Filter Outliers
        outliers = df_iso[df_iso['anomaly'] == -1].copy()

        # 2. Results Display
        if outliers.empty:
            st.success("✅ System Clean. No significant anomalies detected.")
        else:
            st.error(f"⚠️ **Alert:** Detected {len(outliers)} Anomalies ({len(outliers) / len(df) * 100:.1f}% of data)")

            # --- VISUAL 1: The Timeline (NEW & CRITICAL) ---
            st.subheader("1. 📅 When did the anomalies occur?")
            fig_time = go.Figure()

            # Normal Data (Grey Line)
            fig_time.add_trace(go.Scatter(
                x=df_iso['date_id'], y=df_iso['throughput_mbps'],
                mode='lines', name='Normal Traffic',
                line=dict(color='lightgrey', width=1)
            ))

            # Anomalies (Red Markers)
            fig_time.add_trace(go.Scatter(
                x=outliers['date_id'], y=outliers['throughput_mbps'],
                mode='markers', name='Anomaly Detected',
                marker=dict(color='red', size=8, symbol='x')
            ))

            fig_time.update_layout(title="Anomaly Timeline (Throughput Spikes/Drops)", xaxis_title="Date",
                                   yaxis_title="Speed (Mbps)")
            st.plotly_chart(fig_time, use_container_width=True)

            # --- VISUAL 2: The Scatter Map (Existing) ---
            st.subheader("2. 📉 Risk Map (Congestion vs Speed)")
            fig_anom = px.scatter(df_iso, x='npm_dl_prb_utilization', y='throughput_mbps',
                                  color=df_iso['anomaly'].map({1: 'Normal', -1: 'Critical Anomaly'}),
                                  color_discrete_map={'Normal': '#636EFA', 'Critical Anomaly': '#EF553B'},
                                  hover_data=['cell_id', 'date_id', 'anomaly_score'],
                                  title="Contextual Risk Map")
            st.plotly_chart(fig_anom, use_container_width=True)

            # --- 3. LOGIC & EXPLANATION ---
            st.subheader("3. 🧐 Root Cause Analysis")

            # Calculate Baseline (Median of Normal Data)
            normal_data = df_iso[df_iso['anomaly'] == 1]
            medians = normal_data[features_iso].median()


            def categorize_issue(row):
                reasons = []
                severity = "Warning"

                # Check specifics
                if row['npm_dl_prb_utilization'] > medians['npm_dl_prb_utilization'] * 1.5:
                    reasons.append("Extreme Congestion")
                    severity = "Critical"
                if row['throughput_mbps'] < medians['throughput_mbps'] * 0.3:
                    reasons.append("Speed Collapse")
                    severity = "Critical"
                if row['npm_service_drop_rate'] > medians['npm_service_drop_rate'] * 3:
                    reasons.append("High Drop Rate")

                # Fallback
                if not reasons:
                    reasons.append("Unusual Pattern Combination")

                return pd.Series([" & ".join(reasons), severity])


            outliers[['Reason', 'Severity']] = outliers.apply(categorize_issue, axis=1)

            # Display Table
            st.dataframe(
                outliers[['date_id', 'cell_id', 'Severity', 'Reason', 'throughput_mbps', 'npm_dl_prb_utilization']]
                .sort_values('throughput_mbps', ascending=True)
                .style.applymap(lambda v: 'color: red; font-weight: bold;' if v == 'Critical' else 'color: orange;',
                                subset=['Severity']),
                use_container_width=True
            )

            # --- 4. EXECUTIVE INTERPRETATION (NEW) ---
            st.markdown("---")
            st.subheader("📋 Executive Action Plan")

            with st.expander("🚀 Click to see Business Recommendations", expanded=True):
                # Dynamic Stats for the summary
                avg_anom_speed = outliers['throughput_mbps'].mean()
                avg_norm_speed = medians['throughput_mbps']
                speed_drop_pct = (1 - (avg_anom_speed / avg_norm_speed)) * 100

                st.markdown(f"""
                ### 🔍 What does this mean?
                The AI identified **{len(outliers)} instances** where the network behaved abnormally. 
                On average, during these anomalies, **speed dropped by {speed_drop_pct:.1f}%**.

                ### 🛠️ Recommended Actions
                1.  **Check the Timeline:** Look at the first chart. 
                    * *Are the red X's clustered together?* -> **System Outage** (Power/Transmission failure).
                    * *Are they random?* -> **Transient Congestion** (User spikes).
                2.  **Investigate 'Speed Collapse':** * If load is LOW but Speed is LOW (Bottom-Left of Scatter), check for **Interference** or **Hardware Faults**.
                3.  **Investigate 'Extreme Congestion':**
                    * If load is HIGH (Right side of Scatter), the cell needs **Capacity Upgrade** or **Traffic Balancing**.
                """)
with tab6:
    st.header("🤖 AI Data Analyst (Full Context)")
    st.info(f"Analyzing currently filtered data: {len(df)} rows.")

    user_query = st.text_input("Ask a question about this site's data:",
                               placeholder="e.g., Identify any anomalies in the throughput.")

    # Optional: Select specific columns to save tokens if your DF is wide
    # If your df has 50 columns, this prevents the prompt from exploding
    selected_cols_for_ai = st.multiselect("Select columns for AI context (optional)",
                                          options=df.columns.tolist(),
                                          default=df.columns[:5].tolist())  # Default first 5 cols

    model_name = st.selectbox("Select Model", ["llama3", "mistral", "gemma"], index=0)

    if st.button("🚀 Analyze Full Dataset"):
        if not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Sending full dataset to Ollama..."):
                try:
                    # 1. PREPARE CONTEXT (Full DF)
                    # Convert filtered DF to CSV string
                    # using selected columns only to be efficient
                    cols_to_use = selected_cols_for_ai if selected_cols_for_ai else df.columns

                    csv_context = df[cols_to_use].to_csv(index=False)

                    # 2. CONSTRUCT PROMPT
                    prompt = f"""
                    You are a Network Data Analyst. I am providing you with a dataset in CSV format.
                    Analyze this data to answer the user's question who is a less technical individual.

                    DATASET (CSV format):
                    {csv_context}

                    USER QUESTION:
                    {user_query}

                    INSTRUCTIONS:
                    - Reference specific dates and values from the data.
                    - Be concise.
                    - If the data doesn't answer the question, say so.
                    - convert throughput to mbps by dividing by 1e9
                    """

                    # 3. CALL OLLAMA
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        # Increase context window if possible (Ollama default is 2048 or 4096)
                        "options": {
                            "num_ctx": 32768
                        }
                    }

                    response = requests.post("http://localhost:11434/api/generate", json=payload)

                    if response.status_code == 200:
                        ai_response = response.json()['response']
                        st.markdown("### 🧠 AI Response")
                        st.write(ai_response)
                    else:
                        st.error(f"Ollama Error: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"Connection failed. Ensure Ollama is running. Error: {e}")

# ==========================================
# REPORT GENERATION LOGIC
# ==========================================

def generate_html_report(dataframe, site_name):
    """Generates a professional HTML report with embedded charts."""

    # 1. Calculate Aggregates
    avg_speed = dataframe['throughput_mbps'].mean()
    avg_load = dataframe['npm_dl_prb_utilization'].mean()  # Keep as ratio (0-1 or 0-100 based on data)

    # Handle load scaling (if it's 0.8 instead of 80%)
    if dataframe['npm_dl_prb_utilization'].max() <= 1.0:
        avg_load_display = avg_load * 100
        load_threshold = 0.8
    else:
        avg_load_display = avg_load
        load_threshold = 80

    total_traffic = dataframe['npm_dl_traffic_gb'].sum()
    avg_drop_rate = dataframe['npm_service_drop_rate'].mean() * 100

    # Identify "Worst" Cells
    congested_df = dataframe[dataframe['npm_dl_prb_utilization'] > load_threshold]
    congested_count = len(congested_df)

    # 2. Create Embedded Charts (Plotly to HTML)
    # Chart A: Traffic Trend
    fig_trend = px.line(dataframe.groupby('date_id')['throughput_mbps'].mean().reset_index(),
                        x='date_id', y='throughput_mbps', title="Average Speed Trend (Mbps)")
    fig_trend.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    chart_trend_html = fig_trend.to_html(full_html=False, include_plotlyjs='cdn')

    # Chart B: Load Distribution
    fig_hist = px.histogram(dataframe, x='npm_dl_prb_utilization', nbins=20, title="Congestion Distribution")
    fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    chart_hist_html = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')

    # 3. Automated Recommendations
    recommendation = "✅ Network is performing within optimal parameters."
    rec_color = "#28a745"  # Green
    if congested_count > 0:
        recommendation = f"⚠️ <b>Action Required:</b> Detected {congested_count} intervals of high congestion. Consider capacity upgrades for Site {site_name}."
        rec_color = "#ffc107"  # Orange
    if avg_drop_rate > 1.0:
        recommendation = f"🚨 <b>CRITICAL:</b> High drop rate ({avg_drop_rate:.2f}%). Immediate physical inspection of antenna/cabling required."
        rec_color = "#dc3545"  # Red

    # 4. Detailed Table Styling
    cell_summary = dataframe.groupby('cell_id').agg({
        'throughput_mbps': 'mean',
        'npm_dl_prb_utilization': 'mean',
        'npm_service_drop_rate': 'mean',
        'npm_dl_traffic_gb': 'sum'
    }).reset_index()

    # Normalize cols for display
    cell_summary['npm_service_drop_rate'] = (cell_summary['npm_service_drop_rate'] * 100).round(2)
    cell_summary['npm_dl_prb_utilization'] = cell_summary['npm_dl_prb_utilization'].round(2)
    cell_summary['throughput_mbps'] = cell_summary['throughput_mbps'].round(2)
    cell_summary['npm_dl_traffic_gb'] = cell_summary['npm_dl_traffic_gb'].round(1)

    # Highlight High Load Cells in Table
    def style_row(row):
        val = row['npm_dl_prb_utilization']
        # Check against threshold (handle 0-1 vs 0-100)
        is_high = val > 0.8 if val <= 1.0 else val > 80
        return 'background-color: #ffe6e6;' if is_high else ''

    # We manually build the table rows to apply styles easily
    table_rows = ""
    for _, row in cell_summary.iterrows():
        bg_style = "background-color: #ffebee;" if (
                    row['npm_dl_prb_utilization'] > (0.8 if row['npm_dl_prb_utilization'] < 2 else 80)) else ""
        table_rows += f"""
        <tr style="{bg_style}">
            <td>{row['cell_id']}</td>
            <td>{row['throughput_mbps']}</td>
            <td>{row['npm_dl_prb_utilization']}</td>
            <td>{row['npm_service_drop_rate']}%</td>
            <td>{row['npm_dl_traffic_gb']}</td>
        </tr>
        """

    # 5. CSS Styling (Modern & Clean)
    css = """
    <style>
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; background-color: #f9f9f9; margin: 0; }
        .container { width: 85%; margin: 40px auto; background: white; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; }
        .header { border-bottom: 3px solid #ff7900; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { margin: 0; color: #202945; font-size: 28px; }
        .meta { color: #666; font-size: 14px; }

        .grid-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }
        .stat-val { font-size: 24px; font-weight: bold; color: #ff7900; margin-top: 5px; }
        .stat-label { color: #6c757d; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }

        .rec-box { padding: 15px; border-radius: 5px; color: white; margin-bottom: 30px; font-weight: 500; }

        .charts-container { display: flex; gap: 20px; margin-bottom: 40px; }
        .chart-box { flex: 1; background: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }

        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }
        th { background-color: #202945; color: white; padding: 12px; text-align: left; }
        td { border-bottom: 1px solid #eee; padding: 10px; }

        .footer { margin-top: 50px; text-align: center; color: #999; font-size: 12px; border-top: 1px solid #eee; padding-top: 20px; }
    </style>
    """

    # 6. Assemble HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Network Report - {site_name}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>📡 Network Intelligence Report</h1>
                    <div class="meta">Site ID: <b>{site_name}</b></div>
                </div>
                <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
            </div>

            <div class="grid-stats">
                <div class="stat-card">
                    <div class="stat-label">Avg Throughput</div>
                    <div class="stat-val">{avg_speed:.1f} Mbps</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Congestion</div>
                    <div class="stat-val">{avg_load_display:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Traffic</div>
                    <div class="stat-val">{total_traffic:.0f} GB</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Drop Rate</div>
                    <div class="stat-val" style="color: {'red' if avg_drop_rate > 1 else 'green'}">{avg_drop_rate:.2f}%</div>
                </div>
            </div>

            <div class="rec-box" style="background-color: {rec_color};">
                {recommendation}
            </div>

            <h3>📊 Performance Visuals</h3>
            <div class="charts-container">
                <div class="chart-box">{chart_trend_html}</div>
                <div class="chart-box">{chart_hist_html}</div>
            </div>

            <h3>🔬 Cell-Level Breakdown</h3>
            <p style="color: #666; font-size: 13px;">Rows highlighted in red indicate high congestion intervals.</p>
            <table>
                <thead>
                    <tr>
                        <th>Cell ID</th>
                        <th>Avg Speed (Mbps)</th>
                        <th>Avg Load (Prb %)</th>
                        <th>Drop Rate</th>
                        <th>Total Volume (GB)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>

            <div class="footer">
                Generated by Network Intelligence Platform | Confidential Internal Document
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


# --- Sidebar Download Button (Update this part too) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Export Report")

if not df.empty:
    # We call the NEW function here
    report_html = generate_html_report(df, selected_site)

    st.sidebar.download_button(
        label="📥 Download Smart Report (HTML)",
        data=report_html,
        file_name=f"Network_SmartReport_{selected_site}_{datetime.now().date()}.html",
        mime="text/html"
    )
else:
    st.sidebar.warning("Please select a site to generate a report.")
