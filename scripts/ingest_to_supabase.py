#!/usr/bin/env python3
"""
Ingest ChEMBL curated dataset into PostgreSQL database (Supabase or Local).

This script:
1. Reads the CSV/Parquet files (molecules_diverse_enriched.csv with 24 features)
2. Connects to PostgreSQL (Supabase remote or local)
3. Inserts data into the dd schema tables:
   - dd.compounds (molecules with base ChEMBL properties)
   - dd.compound_features (computed features, filters, scaffolds in JSONB)
   - dd.activities (bioactivity data - optional)
   - dd.assays (assay information - optional)
   - dd.targets (target information - optional)
   - dd.dataset_versions (metadata about this dataset)
   - dd.dataset_compounds (links compounds to dataset version)

Feature Sets Inserted:
  - computed_v1: qed, sa_score, num_rings, num_aromatic_rings, num_rotatable_bonds, 
                 formal_charge, num_heteroatoms, fraction_csp3
  - filters_scaffolds_v1: lipinski_violations, passes_lipinski, pains_alert, 
                          brenk_alert, passes_filters, bemis_murcko_scaffold

Install dependencies:
  pip install psycopg2-binary pandas pyarrow tqdm python-dotenv

Usage:
  # Configure .env file first (copy from .env.example)
  
  # Ingest using .env defaults
  python ingest_to_supabase.py
  
  # Override .env with CLI arguments
  python ingest_to_supabase.py --target local --db_name drugdiscovery
  
  # Specify dataset name
  python ingest_to_supabase.py --dataset_name week1_200k_v1
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =========================
# Database connection
# =========================

def get_db_connection(
    target: str = None,
    password: str = None,
    host: str = None,
    port: int = None,
    database: str = None,
    user: str = None
):
    """Connect to PostgreSQL (Supabase or Local) using .env or CLI arguments"""
    
    # Determine target from args or env
    target = target or os.getenv("DB_TARGET", "supabase")
    
    if target == "supabase":
        # Supabase connection parameters (from .env or args)
        conn_params = {
            "host": host or os.getenv("SUPABASE_HOST"),
            "port": port or int(os.getenv("SUPABASE_PORT", 5432)),
            "database": database or os.getenv("SUPABASE_DATABASE", "postgres"),
            "user": user or os.getenv("SUPABASE_USER"),
            "password": password or os.getenv("SUPABASE_PASSWORD"),
            "sslmode": "require",
        }
        db_label = "Supabase PostgreSQL"
    
    elif target == "local":
        # Local PostgreSQL connection parameters (from .env or args)
        conn_params = {
            "host": host or os.getenv("LOCAL_HOST", "localhost"),
            "port": port or int(os.getenv("LOCAL_PORT", 5432)),
            "database": database or os.getenv("LOCAL_DATABASE", "postgres"),
            "user": user or os.getenv("LOCAL_USER", "postgres"),
            "password": password or os.getenv("LOCAL_PASSWORD", "postgres"),
            "sslmode": "prefer",
        }
        db_label = f"Local PostgreSQL ({conn_params['host']}:{conn_params['port']}/{conn_params['database']})"
    
    else:
        print(f"[ERROR] Invalid target: {target}. Must be 'supabase' or 'local'")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = False
        print(f"[OK] Connected to {db_label}")
        return conn
    except Exception as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        print(f"[DEBUG] Connection params: {', '.join([f'{k}={v}' for k, v in conn_params.items() if k != 'password'])}")
        sys.exit(1)


# =========================
# Data loading functions
# =========================

def load_dataframe(data_dir: str, name: str, fmt: str = "parquet") -> pd.DataFrame:
    """Load dataframe from parquet or csv"""
    path = os.path.join(data_dir, f"{name}.{fmt}")
    
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}, trying alternate format...")
        alt_fmt = "csv" if fmt == "parquet" else "parquet"
        path = os.path.join(data_dir, f"{name}.{alt_fmt}")
        
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            return pd.DataFrame()
        fmt = alt_fmt
    
    print(f"[INFO] Loading {path}...")
    
    if fmt == "parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def read_filter_config(data_dir: str) -> Dict[str, Any]:
    """Read the filter configuration used to generate the dataset"""
    config_path = os.path.join(data_dir, "filter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


# =========================
# Insertion functions
# =========================

def insert_dataset_version(
    conn,
    name: str,
    description: str,
    filters_json: Dict[str, Any]
) -> int:
    """Insert or get dataset version and return its ID"""
    cur = conn.cursor()
    
    try:
        # Check if exists
        cur.execute(
            "SELECT dataset_version_id FROM dd.dataset_versions WHERE name = %s",
            (name,)
        )
        row = cur.fetchone()
        
        if row:
            print(f"[INFO] Dataset version '{name}' already exists with ID {row[0]}")
            return row[0]
        
        # Insert new version
        cur.execute(
            """
            INSERT INTO dd.dataset_versions (name, description, filters_json)
            VALUES (%s, %s, %s)
            RETURNING dataset_version_id
            """,
            (name, description, json.dumps(filters_json))
        )
        
        dataset_id = cur.fetchone()[0]
        conn.commit()
        print(f"[OK] Created dataset version '{name}' with ID {dataset_id}")
        return dataset_id
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to insert dataset version: {e}")
        raise


def insert_compounds(conn, df: pd.DataFrame, batch_size: int = 1000):
    """Insert compounds into dd.compounds"""
    if df.empty:
        print("[WARN] No compounds to insert")
        return
    
    cur = conn.cursor()
    
    # Prepare data
    records = []
    for _, row in df.iterrows():
        records.append((
            row.get("molecule_chembl_id"),
            row.get("canonical_smiles"),
            float(row.get("mw")) if pd.notna(row.get("mw")) else None,
            float(row.get("alogp")) if pd.notna(row.get("alogp")) else None,
            int(row.get("hbd")) if pd.notna(row.get("hbd")) else None,
            int(row.get("hba")) if pd.notna(row.get("hba")) else None,
            float(row.get("tpsa")) if pd.notna(row.get("tpsa")) else None,
            int(row.get("max_phase")) if pd.notna(row.get("max_phase")) else None,
            int(row.get("first_approval")) if pd.notna(row.get("first_approval")) else None,
            row.get("pref_name") if pd.notna(row.get("pref_name")) else None,
        ))
    
    # Insert with ON CONFLICT (upsert)
    sql = """
        INSERT INTO dd.compounds (
            molecule_chembl_id, canonical_smiles, mw, alogp, hbd, hba, tpsa,
            max_phase, first_approval, pref_name
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (molecule_chembl_id) DO UPDATE SET
            canonical_smiles = EXCLUDED.canonical_smiles,
            mw = EXCLUDED.mw,
            alogp = EXCLUDED.alogp,
            hbd = EXCLUDED.hbd,
            hba = EXCLUDED.hba,
            tpsa = EXCLUDED.tpsa,
            max_phase = EXCLUDED.max_phase,
            first_approval = EXCLUDED.first_approval,
            pref_name = EXCLUDED.pref_name
    """
    
    try:
        print(f"[INFO] Inserting {len(records):,} compounds...")
        for i in tqdm(range(0, len(records), batch_size), desc="Insert compounds"):
            batch = records[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=batch_size)
        
        conn.commit()
        print(f"[OK] Inserted/updated {len(records):,} compounds")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to insert compounds: {e}")
        raise


def insert_activities(conn, df: pd.DataFrame, batch_size: int = 1000):
    """Insert activities into dd.activities"""
    if df.empty:
        print("[WARN] No activities to insert")
        return
    
    cur = conn.cursor()
    
    # Prepare data
    records = []
    for _, row in df.iterrows():
        records.append((
            int(row.get("activity_id")) if pd.notna(row.get("activity_id")) else None,
            row.get("molecule_chembl_id"),
            row.get("assay_chembl_id") if pd.notna(row.get("assay_chembl_id")) else None,
            row.get("target_chembl_id") if pd.notna(row.get("target_chembl_id")) else None,
            row.get("standard_type") if pd.notna(row.get("standard_type")) else None,
            row.get("standard_relation") if pd.notna(row.get("standard_relation")) else None,
            float(row.get("standard_value")) if pd.notna(row.get("standard_value")) else None,
            row.get("standard_units") if pd.notna(row.get("standard_units")) else None,
            float(row.get("pchembl_value")) if pd.notna(row.get("pchembl_value")) else None,
            row.get("data_validity_comment") if pd.notna(row.get("data_validity_comment")) else None,
            row.get("bao_endpoint") if pd.notna(row.get("bao_endpoint")) else None,
        ))
    
    sql = """
        INSERT INTO dd.activities (
            activity_id, molecule_chembl_id, assay_chembl_id, target_chembl_id,
            standard_type, standard_relation, standard_value, standard_units,
            pchembl_value, data_validity_comment, bao_endpoint
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (activity_id) DO UPDATE SET
            molecule_chembl_id = EXCLUDED.molecule_chembl_id,
            assay_chembl_id = EXCLUDED.assay_chembl_id,
            target_chembl_id = EXCLUDED.target_chembl_id,
            standard_type = EXCLUDED.standard_type,
            standard_relation = EXCLUDED.standard_relation,
            standard_value = EXCLUDED.standard_value,
            standard_units = EXCLUDED.standard_units,
            pchembl_value = EXCLUDED.pchembl_value,
            data_validity_comment = EXCLUDED.data_validity_comment,
            bao_endpoint = EXCLUDED.bao_endpoint
    """
    
    try:
        print(f"[INFO] Inserting {len(records):,} activities...")
        for i in tqdm(range(0, len(records), batch_size), desc="Insert activities"):
            batch = records[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=batch_size)
        
        conn.commit()
        print(f"[OK] Inserted/updated {len(records):,} activities")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to insert activities: {e}")
        raise


def insert_assays(conn, df: pd.DataFrame, batch_size: int = 1000):
    """Insert assays into dd.assays"""
    if df.empty:
        print("[WARN] No assays to insert")
        return
    
    cur = conn.cursor()
    
    records = []
    for _, row in df.iterrows():
        records.append((
            row.get("assay_chembl_id"),
            row.get("assay_type") if pd.notna(row.get("assay_type")) else None,
            int(row.get("confidence_score")) if pd.notna(row.get("confidence_score")) else None,
            row.get("bao_format") if pd.notna(row.get("bao_format")) else None,
            row.get("bao_endpoint") if pd.notna(row.get("bao_endpoint")) else None,
            row.get("target_chembl_id") if pd.notna(row.get("target_chembl_id")) else None,
            row.get("description") if pd.notna(row.get("description")) else None,
        ))
    
    sql = """
        INSERT INTO dd.assays (
            assay_chembl_id, assay_type, confidence_score, bao_format,
            bao_endpoint, target_chembl_id, description
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (assay_chembl_id) DO UPDATE SET
            assay_type = EXCLUDED.assay_type,
            confidence_score = EXCLUDED.confidence_score,
            bao_format = EXCLUDED.bao_format,
            bao_endpoint = EXCLUDED.bao_endpoint,
            target_chembl_id = EXCLUDED.target_chembl_id,
            description = EXCLUDED.description
    """
    
    try:
        print(f"[INFO] Inserting {len(records):,} assays...")
        for i in tqdm(range(0, len(records), batch_size), desc="Insert assays"):
            batch = records[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=batch_size)
        
        conn.commit()
        print(f"[OK] Inserted/updated {len(records):,} assays")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to insert assays: {e}")
        raise


def insert_targets(conn, df: pd.DataFrame, batch_size: int = 1000):
    """Insert targets into dd.targets"""
    if df.empty:
        print("[WARN] No targets to insert")
        return
    
    cur = conn.cursor()
    
    records = []
    for _, row in df.iterrows():
        records.append((
            row.get("target_chembl_id"),
            row.get("target_type") if pd.notna(row.get("target_type")) else None,
            row.get("pref_name") if pd.notna(row.get("pref_name")) else None,
            row.get("organism") if pd.notna(row.get("organism")) else None,
            int(row.get("taxonomy_id")) if pd.notna(row.get("taxonomy_id")) else None,
        ))
    
    sql = """
        INSERT INTO dd.targets (
            target_chembl_id, target_type, pref_name, organism, taxonomy_id
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (target_chembl_id) DO UPDATE SET
            target_type = EXCLUDED.target_type,
            pref_name = EXCLUDED.pref_name,
            organism = EXCLUDED.organism,
            taxonomy_id = EXCLUDED.taxonomy_id
    """
    
    try:
        print(f"[INFO] Inserting {len(records):,} targets...")
        for i in tqdm(range(0, len(records), batch_size), desc="Insert targets"):
            batch = records[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=batch_size)
        
        conn.commit()
        print(f"[OK] Inserted/updated {len(records):,} targets")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to insert targets: {e}")
        raise


def insert_compound_features(conn, df: pd.DataFrame, feature_set: str, feature_columns: List[str], batch_size: int = 1000):
    """Insert compound features into dd.compound_features with JSONB storage"""
    if df.empty:
        print(f"[WARN] No compounds to insert features for feature_set='{feature_set}'")
        return
    
    cur = conn.cursor()
    
    # Prepare data
    records = []
    for _, row in df.iterrows():
        feature_data = {}
        for col in feature_columns:
            if col in row and pd.notna(row[col]):
                val = row[col]
                # Convert numpy types to native Python types
                if isinstance(val, (pd.Int64Dtype, pd.Float64Dtype)):
                    val = val.item() if hasattr(val, 'item') else val
                if isinstance(val, bool):
                    val = bool(val)
                elif isinstance(val, (int, float)):
                    val = float(val) if isinstance(val, float) else int(val)
                feature_data[col] = val
        
        records.append((
            row.get("molecule_chembl_id"),
            feature_set,
            json.dumps(feature_data)
        ))
    
    sql = """
        INSERT INTO dd.compound_features (molecule_chembl_id, feature_set, feature_data)
        VALUES (%s, %s, %s::jsonb)
        ON CONFLICT (molecule_chembl_id, feature_set) DO UPDATE SET
            feature_data = EXCLUDED.feature_data
    """
    
    try:
        print(f"[INFO] Inserting {len(records):,} compound features for feature_set='{feature_set}'...")
        for i in tqdm(range(0, len(records), batch_size), desc=f"Insert {feature_set}"):
            batch = records[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=batch_size)
        
        conn.commit()
        print(f"[OK] Inserted {len(records):,} features for feature_set='{feature_set}'")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to insert compound features: {e}")
        raise


def link_compounds_to_dataset(
    conn,
    dataset_version_id: int,
    molecule_ids: List[str],
    batch_size: int = 1000
):
    """Link compounds to a dataset version"""
    if not molecule_ids:
        print("[WARN] No compounds to link")
        return
    
    cur = conn.cursor()
    
    records = [(dataset_version_id, mol_id) for mol_id in molecule_ids]
    
    sql = """
        INSERT INTO dd.dataset_compounds (dataset_version_id, molecule_chembl_id)
        VALUES (%s, %s)
        ON CONFLICT (dataset_version_id, molecule_chembl_id) DO NOTHING
    """
    
    try:
        print(f"[INFO] Linking {len(records):,} compounds to dataset...")
        for i in tqdm(range(0, len(records), batch_size), desc="Link compounds"):
            batch = records[i:i + batch_size]
            execute_batch(cur, sql, batch, page_size=batch_size)
        
        conn.commit()
        print(f"[OK] Linked {len(records):,} compounds to dataset version {dataset_version_id}")
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to link compounds: {e}")
        raise


# =========================
# Main ingestion pipeline
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest ChEMBL data to PostgreSQL (Supabase or Local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest to Supabase
  python ingest_to_supabase.py --target supabase --data_dir chembl_curated_out
  
  # Ingest to local PostgreSQL
  python ingest_to_supabase.py --target local --data_dir chembl_curated_out --db_name drugdiscovery --password mypass
        """
    )
    
    # Target selection
    parser.add_argument(
        "--target",
        type=str,
        choices=["supabase", "local"],
        help="Target database: 'supabase' (remote) or 'local' (localhost). Default: from .env (DB_TARGET)"
    )
    
    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the ChEMBL parquet/csv files. Default: from .env (DATA_DIR)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name for this dataset version. Default: from .env (DATASET_NAME)"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description of the dataset. Default: from .env (DATASET_DESCRIPTION)"
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        help="File format to read. Default: from .env (DATA_FORMAT)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for insertions. Default: from .env (BATCH_SIZE)"
    )
    
    # Database connection parameters (for both supabase and local)
    parser.add_argument(
        "--password",
        type=str,
        help="Database password. Default: from .env (SUPABASE_PASSWORD or LOCAL_PASSWORD)"
    )
    
    # Local PostgreSQL specific parameters
    parser.add_argument(
        "--host",
        type=str,
        help="Database host. Default: from .env"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Database port. Default: from .env"
    )
    parser.add_argument(
        "--db_name",
        type=str,
        help="Database name. Default: from .env"
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Database user. Default: from .env"
    )
    
    args = parser.parse_args()
    
    # Load defaults from .env if not provided via CLI
    args.target = args.target or os.getenv("DB_TARGET", "supabase")
    args.data_dir = args.data_dir or os.getenv("DATA_DIR", "chembl_curated_out")
    args.dataset_name = args.dataset_name or os.getenv("DATASET_NAME", "week1_200k_v1")
    args.description = args.description or os.getenv("DATASET_DESCRIPTION", "Week 1: ChEMBL 200k diverse molecules with 24 features (QED, SA, filters, scaffolds)")
    args.format = args.format or os.getenv("DATA_FORMAT", "csv")
    args.batch_size = args.batch_size or int(os.getenv("BATCH_SIZE", 1000))
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"ChEMBL to PostgreSQL Ingestion Pipeline")
    print(f"{'='*60}")
    print(f"Target: {args.target.upper()}")
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Format: {args.format}")
    if args.target == "local":
        print(f"Database: {args.host or os.getenv('LOCAL_HOST', 'localhost')}:{args.port or os.getenv('LOCAL_PORT', '5432')}/{args.db_name or os.getenv('LOCAL_DATABASE', 'postgres')}")
        print(f"User: {args.user or os.getenv('LOCAL_USER', 'postgres')}")
    print(f"{'='*60}\n")
    
    # Connect to database
    conn = get_db_connection(
        target=args.target,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.db_name,
        user=args.user
    )
    
    try:
        # Read filter config (metadata)
        filters = read_filter_config(args.data_dir)
        
        # Create dataset version
        dataset_id = insert_dataset_version(
            conn,
            args.dataset_name,
            args.description,
            filters
        )
        
        # Load molecules dataframe
        print("\n[1/4] Loading molecules...")
        mol_df = load_dataframe(args.data_dir, "molecules_diverse_enriched", args.format)
        
        if mol_df.empty:
            print("[ERROR] No molecules found in dataset")
            sys.exit(1)
        
        # Check for bioactivity data (optional)
        print("\n[2/4] Checking for bioactivity data (optional)...")
        act_df = load_dataframe(args.data_dir, "activities_filtered", args.format)
        assay_df = load_dataframe(args.data_dir, "assays_linked", args.format)
        target_df = load_dataframe(args.data_dir, "targets_linked", args.format)
        
        has_bioactivity = not (act_df.empty and assay_df.empty and target_df.empty)
        if not has_bioactivity:
            print("[INFO] No bioactivity data found (activities/assays/targets) - skipping")
        
        # Insert data
        print("\n" + "="*60)
        print("Starting data insertion...")
        print("="*60 + "\n")
        
        print("[Step 1/7] Inserting compounds (base properties)...")
        insert_compounds(conn, mol_df, args.batch_size)
        
        # Define feature sets for compound_features table
        computed_features = [
            'qed', 'sa_score', 'num_rings', 'num_aromatic_rings',
            'num_rotatable_bonds', 'formal_charge', 'num_heteroatoms', 'fraction_csp3'
        ]
        filter_scaffold_features = [
            'lipinski_violations', 'passes_lipinski', 'pains_alert',
            'brenk_alert', 'passes_filters', 'bemis_murcko_scaffold'
        ]
        
        print("\n[Step 2/7] Inserting computed features (QED, SA, descriptors)...")
        insert_compound_features(conn, mol_df, 'computed_v1', computed_features, args.batch_size)
        
        print("\n[Step 3/7] Inserting filter/scaffold features (Lipinski, PAINS, Brenk, scaffolds)...")
        insert_compound_features(conn, mol_df, 'filters_scaffolds_v1', filter_scaffold_features, args.batch_size)
        
        step = 4
        if has_bioactivity:
            print(f"\n[Step {step}/7] Inserting targets...")
            insert_targets(conn, target_df, args.batch_size)
            step += 1
            
            print(f"\n[Step {step}/7] Inserting assays...")
            insert_assays(conn, assay_df, args.batch_size)
            step += 1
            
            print(f"\n[Step {step}/7] Inserting activities...")
            insert_activities(conn, act_df, args.batch_size)
            step += 1
        else:
            print("\n[Step 4-6/7] Skipping bioactivity data (not found)...")
            step = 7
        
        print(f"\n[Step {step}/7] Linking compounds to dataset...")
        molecule_ids = mol_df["molecule_chembl_id"].tolist()
        link_compounds_to_dataset(conn, dataset_id, molecule_ids, args.batch_size)
        
        print("\n" + "="*60)
        print("✅ INGESTION COMPLETE!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  Compounds:        {len(mol_df):,}")
        print(f"  Computed features: {len(mol_df):,} (feature_set='computed_v1')")
        print(f"  Filter/scaffolds:  {len(mol_df):,} (feature_set='filters_scaffolds_v1')")
        if has_bioactivity:
            print(f"  Activities:       {len(act_df):,}")
            print(f"  Assays:           {len(assay_df):,}")
            print(f"  Targets:          {len(target_df):,}")
        print(f"  Dataset ID:       {dataset_id}")
        print(f"\nYou can now query your data:")
        print(f"  SELECT * FROM dd.compounds LIMIT 10;")
        print(f"  SELECT * FROM dd.compound_features WHERE feature_set='computed_v1' LIMIT 10;")
        print(f"  SELECT * FROM dd.dataset_versions;")
        print()
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        conn.close()
        print("[INFO] Database connection closed")


if __name__ == "__main__":
    main()
