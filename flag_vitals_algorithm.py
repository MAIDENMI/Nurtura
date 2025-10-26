import pandas as pd
import snowflake.connector
import os

SNOWFLAKE_ACCOUNT = "tgusboh-sub09046"
SNOWFLAKE_USER = "aidenmiah"
SNOWFLAKE_PASSWORD = "Oscar890890890"
SNOWFLAKE_DATABASE = "HEALTH_MONITORING"
SNOWFLAKE_SCHEMA = "PUBLIC"
SNOWFLAKE_WAREHOUSE = "COMPUTE_WH"
SNOWFLAKE_TABLE = "VITAL_SIGNS_DATA"

HR_COL = "HEART_RATE"
BR_COL = "BREATHING_RATE"
TIME_COL = "TIMESTAMP"

def query_vital_signs_data():
    conn = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        warehouse=SNOWFLAKE_WAREHOUSE
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {SNOWFLAKE_TABLE}")
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    df = df[~((df[HR_COL] == 0) | (df[BR_COL] == 0) | 
              df[HR_COL].isnull() | df[BR_COL].isnull())]
    cursor.close()
    conn.close()
    return df.tail(10)


def flag_vitals() -> bool:
    df = query_vital_signs_data()
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)
    MAX_HR = 130
    MIN_HR = 50
    MAX_BR = 50
    MIN_BR = 8
    
    out_of_range_count = 0
    abnormal_stats = []
    for i in range(len(df)):
        hr = df.iloc[i][HR_COL]
        br = df.iloc[i][BR_COL]
        
        hr_out_of_range = hr < MIN_HR or hr > MAX_HR
        br_out_of_range = br < MIN_BR or br > MAX_BR
        is_out_of_range = hr_out_of_range or br_out_of_range
        
        if is_out_of_range:
            out_of_range_count += 1
            abnormal_stats.append({
                "timestamp": df.iloc[i][TIME_COL],
                "hr": hr,
                "br": br
            })
        else:
            out_of_range_count = 0
        
        if out_of_range_count > 2:
            return True
    return False
    




if __name__ == "__main__":
    flagged_stats = flag_vitals()
    print(flagged_stats)
    