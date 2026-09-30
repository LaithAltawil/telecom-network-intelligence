import sqlite3
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = 'network_kpis.db'
TABLE_NAME = 'network_kpis'
PHYSICAL_MAX_THROUGHPUT = 1000 * 1e9  # 1000 Mbps = 1 Gbps (assuming Mbps scale)


def safe_clean_database():
    print("[1/4] Loading Data for Safe Cleaning...")
    conn = sqlite3.connect(DB_PATH)

    # We load everything to fix logic across rows
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

    # Sort
    df['date_timestamp'] = pd.to_datetime(df['date_id'])
    df = df.sort_values(by=['site_id', 'cell_id', 'date_timestamp'])

    initial_count = len(df)
    print(f"      > Processing {initial_count:,} time-series records...")

    # ==========================================
    # FIX 1: IMPOSSIBLE HANDOVERS
    # ==========================================
    # Use correct columns for Intra-Freq Handover (common scenario)
    # Success: pm_cell_ho_exe_succ_lte_intra_f
    # Attempts: pm_cell_ho_prep_att_lte_intra_f (Check if this exists, or use generic pm_ho_prep_att)

    # Assuming generic mapping from your provided list:
    succ_col = 'pm_ho_exe_succ'
    att_col = 'pm_ho_prep_att'

    bad_ho_mask = df[succ_col] > df[att_col]
    print(f"[2/4] Correcting {bad_ho_mask.sum()} rows with Impossible Handovers...")

    # Cap Successes to Attempts
    df.loc[bad_ho_mask, succ_col] = df.loc[bad_ho_mask, att_col]

    # Recalculate SR (Save to DB column npm_intrafreq_ho_sr if that's the intention)
    # Note: npm_intrafreq_ho_sr is the KPI, usually % (0-100 or 0-1).
    # Let's ensure it's consistent.
    df['npm_intrafreq_ho_sr'] = df[succ_col] / df[att_col]
    df['npm_intrafreq_ho_sr'] = df['npm_intrafreq_ho_sr'].fillna(1.0)

    # ==========================================
    # FIX 2: THROUGHPUT GLITCHES
    # ==========================================
    bad_thp_mask = df['npm_dl_user_throughput'] > PHYSICAL_MAX_THROUGHPUT
    print(f"[3/4] Correcting {bad_thp_mask.sum()} rows with Glitched Throughput...")

    df.loc[bad_thp_mask, 'npm_dl_user_throughput'] = np.nan
    df['npm_dl_user_throughput'] = df.groupby('cell_id')['npm_dl_user_throughput'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

    # ==========================================
    # FIX 3: LOGIC CONSISTENCY
    # ==========================================
    # Traffic exists but PRB Load is near zero -> Impossible
    bad_logic_mask = (df['npm_dl_traffic_gb'] > 10) & (df['npm_dl_prb_utilization'] < 1.0)
    print(f"[4/4] Correcting {bad_logic_mask.sum()} rows with Logic Errors...")

    df.loc[bad_logic_mask, 'npm_dl_prb_utilization'] = np.nan
    df['npm_dl_prb_utilization'] = df.groupby('cell_id')['npm_dl_prb_utilization'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

    # ==========================================
    # SAVE BACK
    # ==========================================
    print(f"\n[DONE] Safe Cleaning Complete.")

    # Drop the temporary date_timestamp column we created for sorting
    df = df.drop(columns=['date_timestamp'])

    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()


if __name__ == "__main__":
    safe_clean_database()
