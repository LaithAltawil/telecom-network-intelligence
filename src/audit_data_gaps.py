import sqlite3
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = 'network_kpis.db'
TABLE_NAME = 'network_kpis'


def check_gaps():
    print("1. Loading Timestamp Data...")

    with sqlite3.connect(DB_PATH) as conn:
        # Use date_id as it maps to timestamp
        query = f"SELECT site_id, cell_id, date_id FROM {TABLE_NAME}"
        df = pd.read_sql(query, conn)

    # Convert date_id to datetime
    df['date_timestamp'] = pd.to_datetime(df['date_id'])

    print(f"   > Loaded {len(df):,} records.")
    print("2. Auditing Time Continuity for All Cells...")

    gap_report = []
    cells = df['cell_id'].unique()

    for cell in cells:
        cell_data = df[df['cell_id'] == cell].sort_values('date_timestamp')
        if cell_data.empty: continue

        start_time = cell_data['date_timestamp'].min()
        end_time = cell_data['date_timestamp'].max()

        # Assuming hourly data ('h')
        expected_range = pd.date_range(start=start_time, end=end_time, freq='h')
        actual_timestamps = set(cell_data['date_timestamp'])
        missing_hours = [t for t in expected_range if t not in actual_timestamps]

        if missing_hours:
            site_id = cell_data['site_id'].iloc[0]
            missing_series = pd.Series(missing_hours).sort_values()
            diffs = missing_series.diff() > pd.Timedelta(hours=1)
            group_ids = diffs.cumsum()

            for _, group in missing_series.groupby(group_ids):
                gap_start = group.min()
                gap_end = group.max()
                duration_hrs = (gap_end - gap_start).total_seconds() / 3600 + 1

                gap_report.append({
                    'Site ID': site_id,
                    'Cell ID': cell,
                    'Gap Start': gap_start,
                    'Gap End': gap_end,
                    'Missing Hours Count': int(duration_hrs)
                })

    if not gap_report:
        print("\n[SUCCESS] No missing timestamps found! Data is 100% continuous.")
    else:
        report_df = pd.DataFrame(gap_report)
        print(f"\n[WARNING] Found {len(report_df)} discontinuity events.")

        print("\n--- TOP 5 SITES WITH DATA LOSS ---")
        summary = report_df.groupby('Site ID')['Missing Hours Count'].sum().sort_values(ascending=False).head(5)
        print(summary)

        report_df.to_csv('network_data_gaps.csv', index=False)
        print("\n   > Full outage log saved to 'network_data_gaps.csv'")


if __name__ == "__main__":
    check_gaps()
