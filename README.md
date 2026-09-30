# Telecom Network Intelligence Platform 📡

An end-to-end data science platform for mobile (RAN) network performance analytics. It takes raw cell-level KPI exports through a full pipeline — ingestion, cleaning, exploratory analysis, machine learning — and serves the results in an interactive Streamlit dashboard with forecasting, segmentation, anomaly detection, and automated reporting.

## ✨ Highlights

- **Full pipeline, not just a notebook** — ETL (Dask → SQLite), domain-aware cleaning, gap auditing, EDA, model training, and a production-style dashboard.
- **Multiple ML paradigms in one place** — gradient boosting (XGBoost), multivariate LSTM forecasting (PyTorch), K-Means segmentation, Isolation Forest anomaly detection.
- **Built for decision-makers** — every model output ships with plain-language interpretation, business rule labeling, and downloadable HTML reports, not just metrics.

## 🖥️ Dashboard tabs

| Tab | What it does |
|---|---|
| 📊 Network Health | Site/cell KPIs (throughput, congestion, drop rate, Minutes-Per-Drop stability score), time-series drill-down, correlation heatmap |
| 🤖 Comparison Models | Train & compare Linear Regression / Random Forest / XGBoost with train/test split, 5-fold CV, leaderboard, and feature importance |
| 🧠 Deep Learning | Configurable multivariate LSTM (epochs, LR, hidden size, sequence length) with learning curves, forecast-vs-actual, residual analysis, and an auto-generated verdict |
| 🗺️ Clustering | K-Means cell segmentation with silhouette scoring, threshold overlays, and business-profile labeling (Congested / Interference / Healthy / Idle) + CSV exports |
| 🚨 Anomalies | Isolation Forest scan with sensitivity control, anomaly timeline, risk map, root-cause categorization, and an executive action plan |
| 🤖 AI Analyst | Optional natural-language Q&A over the filtered data via a local Ollama model |

## 📂 Project structure

```
telecom-network-intelligence/
├── app_dashboard.py        # Streamlit dashboard (the main app)
├── requirements.txt
├── src/
│   ├── etl_pipeline.py       # CSV → cleaned SQLite warehouse (snake_case, indexes)
│   ├── clean_logic_errors.py # Domain fixes: impossible handovers, throughput glitches
│   ├── audit_data_gaps.py    # Time-continuity audit per cell (outage log → CSV)
│   ├── analysis_features.py  # EDA: correlations, dendrogram, PCA biplot, importances
│   └── train_models.py       # Trains & saves production XGBoost models
└── data/                   # (not committed) your raw KPI export goes here
```

## 🛠️ Tech stack

Python • Pandas • Dask • SQLite/SQLAlchemy • Scikit-learn • XGBoost • PyTorch • Plotly • Seaborn • Streamlit • Ollama (optional)

## 🚀 Getting started

### 1. Prerequisites

- Python 3.10+
- (Optional, for the AI Analyst tab) [Ollama](https://ollama.com/) running locally with a chat model pulled, e.g. `ollama pull llama3`

### 2. Install

```bash
git clone https://github.com/LaithAltawil/telecom-network-intelligence.git
cd telecom-network-intelligence
pip install -r requirements.txt
```

> PyTorch install varies by platform/CUDA setup — see the [official guide](https://pytorch.org/get-started/locally/) if the default wheel doesn't suit you.

### 3. Bring your own data

The repo ships **without data**. Place your RAN KPI export at `data/raw_kpis.csv`, then build the warehouse:

```bash
python src/etl_pipeline.py
```

This creates `network_kpis.db` (table `network_kpis`). Expected schema — keys plus standard performance counters:

| Column | Meaning |
|---|---|
| `date_id` | Timestamp of the measurement interval |
| `cell_id`, `site_id` | Cell and site identifiers |
| `npm_dl_user_throughput` | Downlink user throughput (bps) |
| `npm_dl_prb_utilization` | DL PRB utilization (congestion) |
| `npm_dl_traffic_gb` | DL traffic volume |
| `npm_service_drop_rate` | Service drop/failure rate |
| `npm_rrc_users_avg` | Average RRC-connected users |
| `npm_erab_sr`, `npm_intrafreq_ho_sr` | Accessibility / handover success rates |
| `pm_mac_harq_dl_ack_*` | HARQ ACK counters per modulation (QPSK/16/64/256QAM) |
| `pm_ho_exe_succ`, `pm_ho_prep_att` | Handover success / attempt counters |

Column names are normalized to `snake_case` by the ETL; the dashboard derives `throughput_mbps` and a weak-signal ratio from the HARQ counters automatically.

### 4. (Optional) Run the supporting scripts

```bash
python src/clean_logic_errors.py   # fix impossible counters & glitches
python src/audit_data_gaps.py      # audit time continuity → network_data_gaps.csv
python src/analysis_features.py    # correlation / PCA / feature-importance EDA
python src/train_models.py         # train production models → production_models/
```

### 5. Launch the dashboard

```bash
streamlit run app_dashboard.py
```

Then open http://localhost:8501. Use the sidebar to pick a site and date range; the **📄 Export Report** panel downloads a self-contained HTML smart report for the current selection.

## 🤖 Production models (`src/train_models.py`)

| Model | Type | Target |
|---|---|---|
| Throughput | XGBoost regressor | DL user throughput |
| Accessibility | XGBoost regressor | E-RAB success rate |
| Fault detection | XGBoost classifier | Drop-rate threshold breach (with `scale_pos_weight` for imbalance) |

Feature lists are leakage-aware (e.g. volume/time-derived columns excluded from the throughput model). Models are versioned artifacts under `production_models/` (git-ignored — train them yourself).

## ⚙️ Configuration

Key constants at the top of `app_dashboard.py` / `src/*.py`:

- `DB_PATH` (`network_kpis.db`), `TABLE_NAME` (`network_kpis`)
- LSTM hyperparameters are tunable live in the dashboard (epochs, learning rate, hidden size, sequence length)
- Anomaly sensitivity via the `contamination` slider

## 📝 Notes & limitations

- The dashboard trains comparison/LSTM models interactively for exploration; a production deployment should pre-train, version, and serve models instead.
- The AI Analyst tab sends filtered data to a **local** LLM — keep it local when working with sensitive network data.
- SQLite suits single-node exploration; scale-out would move to a columnar/warehouse store.

## 📄 License

MIT — see [LICENSE](LICENSE).
