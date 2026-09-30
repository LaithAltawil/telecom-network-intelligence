import dask.dataframe as dd
import pandas as pd
import sqlalchemy
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = 'data/raw_kpis.csv'  # <-- point this at your own RAN KPI export
DB_NAME = 'network_kpis.db'
TABLE_NAME = 'network_kpis'

# EXACT LIST OF COLUMNS NEEDED (Model Features + Keys)
# This matches your XGBoost training script exactly
required_columns = [
    # Keys
    'Date Id', 'CEll ID', 'Site ID', 'Gov',

    # Target Variables
    'NPM DL User Throughput',

    # Model Features (The 39 predictors)
    'NPM CSFB Prep SR', 'NPM DL PRB Utilization', 'Npm Erab Sr', 'NPM Intrafreq HO SR',
    'NPM RRC Users Avg', 'NPM Service Drop Rate', 'NPM UL Traffic GB', 'Pm Cell Ho Exe Succ Lte Intra F',
    'Pm Erab Estab Att Added', 'Pm Erab Rel Abnormal Enb Act', 'Pm Erab Rel Mme',
    'Pm Mac Harq Dl Ack Qpsk', 'Pm Mac Harq Dl Dtx Qpsk', 'Pm Mac Harq Dl Nack Qpsk',
    'Pm Mac Harq Ul Dtx Qpsk', 'Pm Mac Harq Ul Fail Qpsk', 'Pm Pdcp Vol Ul Drb',
    'Pm Prb Avail Dl', 'Pm Prb Used Dl Bcch', 'Pm Prb Used Dl Dtch', 'Pm Prb Used Dl Pcch',
    'Pm Prb Used Ul Dtch', 'Pm Prb Used Ul Srb', 'Pm Rrc Conn Estab Att Reatt',
    'Pm Rrc Conn Lev Samp', 'Pm Ue Ctxt Estab Att Csfb', 'Pm Ue Ctxt Mod Att Csfb',
    'pmMacHarqDlAck16qam', 'pmMacHarqDlAck256qam', 'pmMacHarqDlAck64qam',
    'pmMacHarqDlDtx16qam', 'pmMacHarqDlDtx256qam', 'pmMacHarqDlDtx64qam',
    'pmMacHarqUlDtx16qam', 'pmMacHarqUlDtx64Qam', 'pmMacHarqUlFail16qam',
    'pmMacHarqUlFail64Qam', 'pmMacHarqUlSucc16qam', 'pmMacHarqUlSucc64Qam',

    # Additional useful metrics for Dashboard (optional)
    'NPM DL Traffic GB', 'Pm Ho Exe Succ', 'Pm Ho Prep Att'
]

# Rename mapping (CSV Name -> DB Name)
# We convert to snake_case for SQL best practices
rename_map = {
    'Date Id': 'date_id',
    'CEll ID': 'cell_id',
    'Site ID': 'site_id',
    'Gov': 'governorate',
    'NPM DL User Throughput': 'npm_dl_user_throughput',
    'NPM CSFB Prep SR': 'npm_csfb_prep_sr',
    'NPM DL PRB Utilization': 'npm_dl_prb_utilization',
    'Npm Erab Sr': 'npm_erab_sr',
    'NPM Intrafreq HO SR': 'npm_intrafreq_ho_sr',
    'NPM RRC Users Avg': 'npm_rrc_users_avg',
    'NPM Service Drop Rate': 'npm_service_drop_rate',
    'NPM UL Traffic GB': 'npm_ul_traffic_gb',
    'Pm Cell Ho Exe Succ Lte Intra F': 'pm_cell_ho_exe_succ_lte_intra_f',
    'Pm Erab Estab Att Added': 'pm_erab_estab_att_added',
    'Pm Erab Rel Abnormal Enb Act': 'pm_erab_rel_abnormal_enb_act',
    'Pm Erab Rel Mme': 'pm_erab_rel_mme',
    'Pm Mac Harq Dl Ack Qpsk': 'pm_mac_harq_dl_ack_qpsk',
    'Pm Mac Harq Dl Dtx Qpsk': 'pm_mac_harq_dl_dtx_qpsk',
    'Pm Mac Harq Dl Nack Qpsk': 'pm_mac_harq_dl_nack_qpsk',
    'Pm Mac Harq Ul Dtx Qpsk': 'pm_mac_harq_ul_dtx_qpsk',
    'Pm Mac Harq Ul Fail Qpsk': 'pm_mac_harq_ul_fail_qpsk',
    'Pm Pdcp Vol Ul Drb': 'pm_pdcp_vol_ul_drb',
    'Pm Prb Avail Dl': 'pm_prb_avail_dl',
    'Pm Prb Used Dl Bcch': 'pm_prb_used_dl_bcch',
    'Pm Prb Used Dl Dtch': 'pm_prb_used_dl_dtch',
    'Pm Prb Used Dl Pcch': 'pm_prb_used_dl_pcch',
    'Pm Prb Used Ul Dtch': 'pm_prb_used_ul_dtch',
    'Pm Prb Used Ul Srb': 'pm_prb_used_ul_srb',
    'Pm Rrc Conn Estab Att Reatt': 'pm_rrc_conn_estab_att_reatt',
    'Pm Rrc Conn Lev Samp': 'pm_rrc_conn_lev_samp',
    'Pm Ue Ctxt Estab Att Csfb': 'pm_ue_ctxt_estab_att_csfb',
    'Pm Ue Ctxt Mod Att Csfb': 'pm_ue_ctxt_mod_att_csfb',
    'pmMacHarqDlAck16qam': 'pm_mac_harq_dl_ack_16qam',
    'pmMacHarqDlAck256qam': 'pm_mac_harq_dl_ack_256qam',
    'pmMacHarqDlAck64qam': 'pm_mac_harq_dl_ack_64qam',
    'pmMacHarqDlDtx16qam': 'pm_mac_harq_dl_dtx_16qam',
    'pmMacHarqDlDtx256qam': 'pm_mac_harq_dl_dtx_256qam',
    'pmMacHarqDlDtx64qam': 'pm_mac_harq_dl_dtx_64qam',
    'pmMacHarqUlDtx16qam': 'pm_mac_harq_ul_dtx_16qam',
    'pmMacHarqUlDtx64Qam': 'pm_mac_harq_ul_dtx_64qam',
    'pmMacHarqUlFail16qam': 'pm_mac_harq_ul_fail_16qam',
    'pmMacHarqUlFail64Qam': 'pm_mac_harq_ul_fail_64qam',
    'pmMacHarqUlSucc16qam': 'pm_mac_harq_ul_succ_16qam',
    'pmMacHarqUlSucc64Qam': 'pm_mac_harq_ul_succ_64qam',
    'NPM DL Traffic GB': 'npm_dl_traffic_gb',
    'Pm Ho Exe Succ': 'pm_ho_exe_succ',
    'Pm Ho Prep Att': 'pm_ho_prep_att'
}


# In the rebuild_database() function, after loading with Dask:

def rebuild_database():
    print(f"🚀 Starting Database Rebuild from {CSV_PATH}...")

    ddf = dd.read_csv(
        CSV_PATH,
        usecols=lambda x: x in required_columns,
        blocksize="64MB",
        dtype={'NPM CSFB Prep SR': 'float64', 'NPM Service Drop Rate': 'float64'},
        na_values=['∞', 'inf', '-inf']  # <--- CRITICAL FIX
    )

    # 2. **RENAME TO SNAKE_CASE**
    ddf = ddf.rename(columns=rename_map)

    # 3. Clean Data
    print("🧹 Cleaning Data...")
    ddf = ddf.fillna(0)

    # 4. Write to SQLite
    print(f"💾 Writing to {DB_NAME}...")
    ddf.to_sql(
        TABLE_NAME,
        uri=f'sqlite:///{DB_NAME}',
        if_exists='replace',
        index=False,
        chunksize=1000,
        parallel=False
    )

    # 5. Add Indexes
    print("⚡ Creating Indexes...")
    engine = sqlalchemy.create_engine(f'sqlite:///{DB_NAME}')
    with engine.connect() as con:
        con.execute(sqlalchemy.text(f"CREATE INDEX idx_date ON {TABLE_NAME} (date_id)"))
        con.execute(sqlalchemy.text(f"CREATE INDEX idx_site ON {TABLE_NAME} (site_id)"))
        con.execute(sqlalchemy.text(f"CREATE INDEX idx_cell ON {TABLE_NAME} (cell_id)"))
        con.commit()

    print("✅ Database Ready with snake_case columns!")


if __name__ == "__main__":
    rebuild_database()
