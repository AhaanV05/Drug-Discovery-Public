#!/usr/bin/env python3
"""
Extract bioactivity data from ChEMBL SQLite database for Week 4 RL training
Focuses on EGFR target with high-quality pChEMBL values
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("BIOACTIVITY DATA EXTRACTION FROM ChEMBL 34")
print("="*100)

# Paths
base_dir = Path(__file__).parent.parent  # Go up to project root
chembl_db = base_dir / 'chembl_34' / 'chembl_34_sqlite' / 'chembl_34.db'
molecules_file = base_dir / 'chembl_curated_out' / 'molecules_filtered_for_rl.csv'
output_dir = base_dir / 'chembl_curated_out'

# Verify database exists
if not chembl_db.exists():
    raise FileNotFoundError(f"ChEMBL database not found: {chembl_db}")

print(f"\n[1/7] Loading our 200k molecules...")
df_molecules = pd.read_csv(molecules_file)
molecule_ids = df_molecules['molecule_chembl_id'].tolist()
print(f"  ✓ Loaded {len(molecule_ids):,} molecule IDs")

# Connect to ChEMBL database
print(f"\n[2/7] Connecting to ChEMBL database...")
conn = sqlite3.connect(str(chembl_db))
print(f"  ✓ Connected to {chembl_db}")

# Query 1: Extract activities for our molecules
print(f"\n[3/7] Extracting bioactivity data...")
print(f"  Targeting: ALL human single protein targets (expanded coverage)")
print(f"  Activity types: IC50, Ki, EC50, Kd")
print(f"  Quality filters:")
print(f"    • pChEMBL value present (standardized -log10 binding affinity)")
print(f"    • Assay confidence score >= 7 (medium-high confidence)")
print(f"    • Human organism (target organism)")
print(f"    • Single protein target (not complex)")
print(f"    • All protein classes (kinases, GPCRs, ion channels, enzymes, etc.)")

# Create temporary table with our molecule IDs for efficient querying
print(f"\n  Creating temporary table with our {len(molecule_ids):,} molecules...")
placeholders = ','.join(['?' for _ in molecule_ids])
conn.execute("DROP TABLE IF EXISTS temp_molecules")
conn.execute("CREATE TEMP TABLE temp_molecules (molecule_chembl_id TEXT PRIMARY KEY)")
conn.executemany("INSERT INTO temp_molecules VALUES (?)", [(mid,) for mid in molecule_ids])
print(f"  ✓ Temporary table created")

# Main query: Join activities with our molecules - ALL human single protein targets
query = """
SELECT DISTINCT
    act.molregno,
    mol.chembl_id AS molecule_chembl_id,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.activity_comment,
    ass.assay_id,
    ass.chembl_id AS assay_chembl_id,
    ass.description AS assay_description,
    ass.assay_type,
    ass.confidence_score,
    td.tid,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.target_type,
    td.organism AS target_organism,
    cs.accession AS uniprot_accession
FROM activities act
INNER JOIN temp_molecules tm ON tm.molecule_chembl_id = (
    SELECT chembl_id FROM molecule_dictionary WHERE molregno = act.molregno
)
INNER JOIN molecule_dictionary mol ON mol.molregno = act.molregno
INNER JOIN assays ass ON ass.assay_id = act.assay_id
INNER JOIN target_dictionary td ON td.tid = ass.tid
LEFT JOIN target_components tc ON tc.tid = td.tid
LEFT JOIN component_sequences cs ON cs.component_id = tc.component_id
WHERE 
    act.pchembl_value IS NOT NULL
    AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
    AND ass.confidence_score >= 7
    AND td.organism = 'Homo sapiens'
    AND td.target_type = 'SINGLE PROTEIN'
ORDER BY act.pchembl_value DESC
"""

print(f"\n  Executing bioactivity query...")
df_activities = pd.read_sql_query(query, conn)
print(f"  ✓ Retrieved {len(df_activities):,} activity records")

if len(df_activities) == 0:
    print(f"\n  ⚠️ WARNING: No EGFR activities found for our molecules!")
    print(f"  Expanding search to ALL high-quality activities...")
    
    # Fallback: Get all high-quality activities for our molecules
    query_all = """
    SELECT DISTINCT
        act.molregno,
        mol.chembl_id AS molecule_chembl_id,
        act.standard_type,
        act.standard_relation,
        act.standard_value,
        act.standard_units,
        act.pchembl_value,
        act.activity_comment,
        ass.assay_id,
        ass.chembl_id AS assay_chembl_id,
        ass.description AS assay_description,
        ass.assay_type,
        ass.confidence_score,
        td.tid,
        td.chembl_id AS target_chembl_id,
        td.pref_name AS target_name,
        td.target_type,
        td.organism AS target_organism,
        cs.accession AS uniprot_accession
    FROM activities act
    INNER JOIN temp_molecules tm ON tm.molecule_chembl_id = (
        SELECT chembl_id FROM molecule_dictionary WHERE molregno = act.molregno
    )
    INNER JOIN molecule_dictionary mol ON mol.molregno = act.molregno
    INNER JOIN assays ass ON ass.assay_id = act.assay_id
    INNER JOIN target_dictionary td ON td.tid = ass.tid
    LEFT JOIN target_components tc ON tc.tid = td.tid
    LEFT JOIN component_sequences cs ON cs.component_id = tc.component_id
    WHERE 
        act.pchembl_value IS NOT NULL
        AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
        AND ass.confidence_score >= 7
        AND td.organism = 'Homo sapiens'
        AND td.target_type = 'SINGLE PROTEIN'
    ORDER BY act.pchembl_value DESC
    LIMIT 100000
    """
    
    df_activities = pd.read_sql_query(query_all, conn)
    print(f"  ✓ Retrieved {len(df_activities):,} activity records (all targets)")

# Query 2: Get detailed assay information
print(f"\n[4/7] Extracting assay metadata...")
assay_ids = df_activities['assay_id'].unique().tolist()
assay_placeholders = ','.join(['?' for _ in assay_ids])

query_assays = f"""
SELECT DISTINCT
    ass.assay_id,
    ass.chembl_id AS assay_chembl_id,
    ass.description,
    ass.assay_type,
    ass.assay_organism,
    ass.assay_tissue,
    ass.assay_cell_type,
    ass.assay_subcellular_fraction,
    ass.confidence_score,
    ass.src_id,
    src.src_description AS data_source,
    doc.chembl_id AS document_chembl_id,
    doc.journal,
    doc.year,
    doc.doi
FROM assays ass
LEFT JOIN source src ON src.src_id = ass.src_id
LEFT JOIN docs doc ON doc.doc_id = ass.doc_id
WHERE ass.assay_id IN ({assay_placeholders})
"""

df_assays = pd.read_sql_query(query_assays, conn, params=assay_ids)
print(f"  ✓ Retrieved {len(df_assays):,} unique assays")

# Query 3: Get target information
print(f"\n[5/7] Extracting target metadata...")
target_ids = df_activities['tid'].unique().tolist()
target_placeholders = ','.join(['?' for _ in target_ids])

query_targets = f"""
SELECT DISTINCT
    td.tid,
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name,
    td.target_type,
    td.organism AS target_organism,
    td.tax_id,
    cs.component_id,
    cs.accession AS uniprot_accession,
    cs.component_type,
    cs.description AS protein_description,
    cs.sequence AS protein_sequence,
    pc.protein_class_desc
FROM target_dictionary td
LEFT JOIN target_components tc ON tc.tid = td.tid
LEFT JOIN component_sequences cs ON cs.component_id = tc.component_id
LEFT JOIN component_class cc ON cc.component_id = cs.component_id
LEFT JOIN protein_classification pc ON pc.protein_class_id = cc.protein_class_id
WHERE td.tid IN ({target_placeholders})
"""

df_targets = pd.read_sql_query(query_targets, conn, params=target_ids)
print(f"  ✓ Retrieved {len(df_targets):,} target records")

conn.close()
print(f"  ✓ Database connection closed")

# Data processing and filtering
print(f"\n[6/7] Processing and filtering data...")

# Filter activities: Keep only high-quality records
print(f"  Initial activities: {len(df_activities):,}")
df_activities_filtered = df_activities[
    (df_activities['pchembl_value'].notnull()) &
    (df_activities['pchembl_value'] >= 4) &  # Minimum pChEMBL of 4 (IC50 = 100 μM)
    (df_activities['pchembl_value'] <= 12) &  # Maximum pChEMBL of 12 (IC50 = 1 pM)
    (df_activities['standard_relation'].isin(['=', '<', '<='])) &  # Exclude '>' (weak binders)
    (df_activities['confidence_score'] >= 7)  # Medium-high confidence
].copy()

print(f"  After filtering: {len(df_activities_filtered):,} activities")
print(f"  Quality filters applied:")
print(f"    • pChEMBL range: 4-12")
print(f"    • Relations: =, <, <= (excluding weak '>' binders)")
print(f"    • Confidence: >= 7")

# Calculate statistics per molecule
print(f"\n  Calculating per-molecule statistics...")
molecule_stats = df_activities_filtered.groupby('molecule_chembl_id').agg({
    'pchembl_value': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'target_chembl_id': 'nunique',
    'assay_chembl_id': 'nunique'
}).reset_index()

molecule_stats.columns = [
    'molecule_chembl_id', 
    'activity_count', 
    'pchembl_mean', 
    'pchembl_median', 
    'pchembl_std',
    'pchembl_min',
    'pchembl_max',
    'target_count',
    'assay_count'
]

print(f"  ✓ Statistics calculated for {len(molecule_stats):,} molecules")
print(f"\n  pChEMBL value distribution:")
print(f"    Mean: {molecule_stats['pchembl_mean'].mean():.2f}")
print(f"    Median: {molecule_stats['pchembl_median'].median():.2f}")
print(f"    Min: {molecule_stats['pchembl_min'].min():.2f}")
print(f"    Max: {molecule_stats['pchembl_max'].max():.2f}")

# Merge statistics back to activities
df_activities_final = df_activities_filtered.merge(
    molecule_stats[['molecule_chembl_id', 'pchembl_mean', 'pchembl_median', 'activity_count']], 
    on='molecule_chembl_id', 
    how='left'
)

# Save outputs
print(f"\n[7/7] Saving output files...")

# Save activities
output_activities = output_dir / 'activities_filtered.parquet'
df_activities_final.to_parquet(output_activities, index=False, engine='pyarrow')
print(f"  ✓ Saved: {output_activities}")
print(f"    Rows: {len(df_activities_final):,}")
print(f"    Columns: {len(df_activities_final.columns)}")

# Save assays
output_assays = output_dir / 'assays_linked.parquet'
df_assays.to_parquet(output_assays, index=False, engine='pyarrow')
print(f"  ✓ Saved: {output_assays}")
print(f"    Rows: {len(df_assays):,}")
print(f"    Columns: {len(df_assays.columns)}")

# Save targets
output_targets = output_dir / 'targets_linked.parquet'
df_targets.to_parquet(output_targets, index=False, engine='pyarrow')
print(f"  ✓ Saved: {output_targets}")
print(f"    Rows: {len(df_targets):,}")
print(f"    Columns: {len(df_targets.columns)}")

# Save molecule-level statistics
output_stats = output_dir / 'bioactivity_statistics.json'
stats_summary = {
    'total_activities': int(len(df_activities_final)),
    'unique_molecules': int(len(molecule_stats)),
    'unique_targets': int(df_activities_final['target_chembl_id'].nunique()),
    'unique_assays': int(df_activities_final['assay_chembl_id'].nunique()),
    'pchembl_stats': {
        'mean': float(molecule_stats['pchembl_mean'].mean()),
        'median': float(molecule_stats['pchembl_median'].median()),
        'min': float(molecule_stats['pchembl_min'].min()),
        'max': float(molecule_stats['pchembl_max'].max()),
        'std': float(molecule_stats['pchembl_mean'].std())
    },
    'activity_types': df_activities_final['standard_type'].value_counts().to_dict(),
    'target_types': df_activities_final['target_type'].value_counts().to_dict(),
    'assay_types': df_activities_final['assay_type'].value_counts().to_dict(),
    'top_10_targets': df_activities_final['target_name'].value_counts().head(10).to_dict()
}

import json
with open(output_stats, 'w') as f:
    json.dump(stats_summary, f, indent=2)
print(f"  ✓ Saved: {output_stats}")

print("\n" + "="*100)
print("BIOACTIVITY EXTRACTION COMPLETE ✅")
print("="*100)

print(f"\n📊 SUMMARY:")
print(f"  • Activities extracted: {len(df_activities_final):,}")
print(f"  • Molecules with bioactivity: {len(molecule_stats):,} / {len(molecule_ids):,} ({len(molecule_stats)/len(molecule_ids)*100:.1f}%)")
print(f"  • Unique targets: {df_activities_final['target_chembl_id'].nunique():,}")
print(f"  • Unique assays: {df_activities_final['assay_chembl_id'].nunique():,}")
print(f"  • pChEMBL range: {molecule_stats['pchembl_min'].min():.2f} - {molecule_stats['pchembl_max'].max():.2f}")
print(f"  • Mean pChEMBL: {molecule_stats['pchembl_mean'].mean():.2f} (IC50 ~ {10**(-molecule_stats['pchembl_mean'].mean()) * 1e9:.1f} nM)")

print(f"\n📁 OUTPUT FILES:")
print(f"  1. {output_activities} - Activity data with pChEMBL values")
print(f"  2. {output_assays} - Assay metadata")
print(f"  3. {output_targets} - Target information")
print(f"  4. {output_stats} - Summary statistics (JSON)")

print(f"\n🎯 NEXT STEPS:")
print(f"  ✅ Bioactivity data ready for Week 4 RL training")
print(f"  ✅ Use 'pchembl_mean' column for binding affinity reward (30% weight)")
print(f"  ✅ Filter molecules with pchembl_mean >= 6.0 for good binders (IC50 < 1 μM)")
print(f"  ✅ Update data_requirements_check.md status to COMPLETE")

print("="*100)
