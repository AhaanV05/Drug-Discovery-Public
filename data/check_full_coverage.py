#!/usr/bin/env python3
"""
Check bioactivity coverage if we expand to ALL human targets (not just kinases)
"""

import sqlite3
import pandas as pd
from pathlib import Path

print("="*80)
print("CHECKING FULL BIOACTIVITY COVERAGE (ALL TARGETS)")
print("="*80)

# Paths
base_dir = Path(__file__).parent.parent
chembl_db = base_dir / 'chembl_34' / 'chembl_34_sqlite' / 'chembl_34.db'
molecules_file = base_dir / 'chembl_curated_out' / 'molecules_filtered_for_rl.csv'

# Load molecules
print(f"\n[1/3] Loading our 200k molecules...")
df_molecules = pd.read_csv(molecules_file)
molecule_ids = df_molecules['molecule_chembl_id'].tolist()
print(f"  ✓ {len(molecule_ids):,} molecules")

# Connect to database
print(f"\n[2/3] Querying ChEMBL for ALL targets...")
conn = sqlite3.connect(str(chembl_db))

# Create temp table
conn.execute("DROP TABLE IF EXISTS temp_molecules")
conn.execute("CREATE TEMP TABLE temp_molecules (molecule_chembl_id TEXT PRIMARY KEY)")
conn.executemany("INSERT INTO temp_molecules VALUES (?)", [(mid,) for mid in molecule_ids])

# Query: ALL human single protein targets with good quality data
query = """
SELECT COUNT(DISTINCT mol.chembl_id) as molecule_count,
       COUNT(DISTINCT td.tid) as target_count,
       COUNT(*) as activity_count
FROM activities act
INNER JOIN temp_molecules tm ON tm.molecule_chembl_id = (
    SELECT chembl_id FROM molecule_dictionary WHERE molregno = act.molregno
)
INNER JOIN molecule_dictionary mol ON mol.molregno = act.molregno
INNER JOIN assays ass ON ass.assay_id = act.assay_id
INNER JOIN target_dictionary td ON td.tid = ass.tid
WHERE 
    act.pchembl_value IS NOT NULL
    AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
    AND ass.confidence_score >= 7
    AND td.organism = 'Homo sapiens'
    AND td.target_type = 'SINGLE PROTEIN'
    AND act.pchembl_value >= 4.0
    AND act.pchembl_value <= 12.0
    AND act.standard_relation IN ('=', '<', '<=')
"""

result = pd.read_sql_query(query, conn)
print(f"\n[3/3] COVERAGE WITH ALL HUMAN TARGETS:")
print(f"  Molecules with bioactivity: {result['molecule_count'].iloc[0]:,} / {len(molecule_ids):,}")
print(f"  Coverage: {result['molecule_count'].iloc[0] / len(molecule_ids) * 100:.1f}%")
print(f"  Unique targets: {result['target_count'].iloc[0]:,}")
print(f"  Total activities: {result['activity_count'].iloc[0]:,}")

# Get top target classes
query_classes = """
SELECT pc.protein_class_desc, COUNT(DISTINCT mol.chembl_id) as molecule_count
FROM activities act
INNER JOIN temp_molecules tm ON tm.molecule_chembl_id = (
    SELECT chembl_id FROM molecule_dictionary WHERE molregno = act.molregno
)
INNER JOIN molecule_dictionary mol ON mol.molregno = act.molregno
INNER JOIN assays ass ON ass.assay_id = act.assay_id
INNER JOIN target_dictionary td ON td.tid = ass.tid
LEFT JOIN target_components tc ON tc.tid = td.tid
LEFT JOIN component_class cc ON cc.component_id = tc.component_id
LEFT JOIN protein_classification pc ON pc.protein_class_id = cc.protein_class_id
WHERE 
    act.pchembl_value IS NOT NULL
    AND act.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
    AND ass.confidence_score >= 7
    AND td.organism = 'Homo sapiens'
    AND td.target_type = 'SINGLE PROTEIN'
    AND act.pchembl_value >= 4.0
    AND act.pchembl_value <= 12.0
    AND act.standard_relation IN ('=', '<', '<=')
    AND pc.protein_class_desc IS NOT NULL
GROUP BY pc.protein_class_desc
ORDER BY molecule_count DESC
LIMIT 20
"""

df_classes = pd.read_sql_query(query_classes, conn)
print(f"\n  Top 20 protein classes by molecule coverage:")
for idx, row in df_classes.iterrows():
    print(f"    {row['protein_class_desc']}: {row['molecule_count']:,} molecules")

conn.close()

print("\n" + "="*80)
print("RECOMMENDATION:")
if result['molecule_count'].iloc[0] / len(molecule_ids) >= 0.20:
    print("  ✅ Coverage >= 20% - SUFFICIENT for RL training")
    print("  → Re-extract with ALL targets (not just kinases)")
elif result['molecule_count'].iloc[0] / len(molecule_ids) >= 0.10:
    print("  ⚠️ Coverage 10-20% - MARGINAL, but usable with caveats")
    print("  → Consider: (1) Re-extract all targets, (2) Reduce binding weight to 10%")
else:
    print("  ❌ Coverage < 10% - INSUFFICIENT for RL training")
    print("  → MUST use alternative strategy (see recommendations below)")

print("="*80)
