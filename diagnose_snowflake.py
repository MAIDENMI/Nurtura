import os
import config

try:
    import snowflake.connector
    
    print("Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        account=config.SNOWFLAKE_ACCOUNT,
        user=config.SNOWFLAKE_USER,
        password=config.SNOWFLAKE_PASSWORD,
        warehouse=config.SNOWFLAKE_WAREHOUSE
    )
    
    cursor = conn.cursor()
    print("✓ Connected!")
    
    print(f"\nCreating database {config.SNOWFLAKE_DATABASE}...")
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.SNOWFLAKE_DATABASE}")
    print("✓ Database ready")
    
    print(f"\nUsing database {config.SNOWFLAKE_DATABASE}...")
    cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
    
    print(f"Using schema {config.SNOWFLAKE_SCHEMA}...")
    cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
    
    print(f"\nCreating table {config.SNOWFLAKE_TABLE}...")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {config.SNOWFLAKE_TABLE} (
            TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            HEART_RATE FLOAT,
            BREATHING_RATE FLOAT
        )
    """)
    conn.commit()
    print("✓ Table ready")
    
    print("\nInserting test data...")
    cursor.execute(f"""
        INSERT INTO {config.SNOWFLAKE_TABLE} (HEART_RATE, BREATHING_RATE) 
        VALUES (120.5, 35.2)
    """)
    conn.commit()
    print("✓ Test data inserted")
    
    print(f"\nQuerying {config.SNOWFLAKE_TABLE}...")
    cursor.execute(f"SELECT * FROM {config.SNOWFLAKE_TABLE}")
    rows = cursor.fetchall()
    
    print(f"\n✓ SUCCESS! Found {len(rows)} records:")
    print("\nTIMESTAMP                | HEART_RATE | BREATHING_RATE")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]} | {row[1]:10.1f} | {row[2]:14.1f}")
    
    cursor.close()
    conn.close()
    print("\n✓ All systems working! Your scripts will now save data to Snowflake.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

