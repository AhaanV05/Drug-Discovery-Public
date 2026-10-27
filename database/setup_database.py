#!/usr/bin/env python3
"""
Quick script to run the Supabase schema setup via psycopg2

Usage:
    python setup_database.py
"""

import sys
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Read the SQL schema file
with open('setup_supabase_schema.sql', 'r', encoding='utf-8') as f:
    schema_sql = f.read()

# Connect to Supabase
conn_params = {
    "host": os.getenv("SUPABASE_HOST"),
    "port": int(os.getenv("SUPABASE_PORT", 5432)),
    "database": os.getenv("SUPABASE_DATABASE", "postgres"),
    "user": os.getenv("SUPABASE_USER"),
    "password": os.getenv("SUPABASE_PASSWORD"),
    "sslmode": "require",
}

try:
    print("Connecting to Supabase...")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    print("Executing schema setup...")
    # Execute the entire SQL file
    cur.execute(schema_sql)
    conn.commit()
    
    print("✅ Schema setup complete!")
    
    # Verify tables exist
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'dd' ORDER BY table_name")
    tables = cur.fetchall()
    
    print(f"\n📊 Created {len(tables)} tables in dd schema:")
    for table in tables:
        print(f"  - dd.{table[0]}")
    
    cur.close()
    conn.close()
    print("\n✅ Database is ready for data ingestion!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
