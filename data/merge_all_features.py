#!/usr/bin/env python3
"""
Merge molecules_diverse_enriched.csv with ADMET results to create master dataset.

This script:
1. Loads molecules_diverse_enriched.csv (24 features, high-precision QED/SA)
2. Loads results.csv (112 ADMET features)
3. Merges on molecule_chembl_id
4. Excludes duplicate columns (QED, Synth, SMILES) from ADMET results
5. Creates molecules_all_features.csv with all unique features
"""

import pandas as pd
import sys
from pathlib import Path

print("="*70)
print("MERGING ALL FEATURES INTO MASTER DATASET")
print("="*70)

# File paths
base_file = "chembl_curated_out/molecules_diverse_enriched.csv"
admet_file = "results.csv"
output_file = "chembl_curated_out/molecules_all_features.csv"

# Check files exist
if not Path(base_file).exists():
    print(f"[ERROR] Base file not found: {base_file}")
    sys.exit(1)

if not Path(admet_file).exists():
    print(f"[ERROR] ADMET file not found: {admet_file}")
    sys.exit(1)

print(f"\n[1/5] Loading base dataset: {base_file}")
base_df = pd.read_csv(base_file)
print(f"  ✓ Loaded {len(base_df):,} molecules with {len(base_df.columns)} features")
print(f"  Columns: {list(base_df.columns)}")

print(f"\n[2/5] Loading ADMET results: {admet_file}")
admet_df = pd.read_csv(admet_file)
print(f"  ✓ Loaded {len(admet_df):,} molecules with {len(admet_df.columns)} features")

# Columns to exclude from ADMET (duplicates or less precise)
exclude_cols = ['SMILES', 'QED', 'Synth']  # We have better versions in base_df
admet_cols_to_keep = [col for col in admet_df.columns if col not in exclude_cols and col != 'molecule_chembl_id']

print(f"\n[3/5] Excluding duplicate columns from ADMET results:")
print(f"  Excluded: {exclude_cols}")
print(f"  Keeping {len(admet_cols_to_keep)} ADMET-specific columns")

print(f"\n[4/5] Merging datasets on molecule_chembl_id...")
# Select only the columns we want from ADMET
admet_subset = admet_df[['molecule_chembl_id'] + admet_cols_to_keep]

# Merge
merged_df = base_df.merge(admet_subset, on='molecule_chembl_id', how='inner')

print(f"  ✓ Merged successfully!")
print(f"  Result: {len(merged_df):,} molecules with {len(merged_df.columns)} features")

# Verify no molecules were lost
if len(merged_df) != len(base_df):
    lost = len(base_df) - len(merged_df)
    print(f"  ⚠️ WARNING: Lost {lost} molecules during merge!")
    print(f"  Base: {len(base_df):,}, ADMET: {len(admet_df):,}, Merged: {len(merged_df):,}")
else:
    print(f"  ✓ All molecules preserved (no data loss)")

print(f"\n[5/5] Saving master dataset: {output_file}")
merged_df.to_csv(output_file, index=False)

# Calculate file size
file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
print(f"  ✓ Saved {len(merged_df):,} molecules × {len(merged_df.columns)} features")
print(f"  File size: {file_size_mb:.1f} MB")

print("\n" + "="*70)
print("MASTER DATASET SUMMARY")
print("="*70)

print(f"\nTotal molecules: {len(merged_df):,}")
print(f"Total features: {len(merged_df.columns)}")

print(f"\nFeature breakdown:")
print(f"  • Base features (molecules_diverse_enriched.csv): {len(base_df.columns)}")
print(f"  • ADMET features added (results.csv): {len(admet_cols_to_keep)}")
print(f"  • Excluded duplicates: {len(exclude_cols)}")

print(f"\nKey features retained (high precision):")
print(f"  • QED: From base dataset (16 decimal places)")
print(f"  • SA Score: From base dataset (15 decimal places)")
print(f"  • SMILES: From base dataset (canonical_smiles)")

print(f"\nADMET features added:")
print(f"  • Total ADMET properties: {len(admet_cols_to_keep)}")
print(f"  • Includes: hERG, Ames, CYP450s, BBB, HIA, Caco2, PPB, DILI, etc.")

print(f"\nColumn list (first 20):")
for i, col in enumerate(merged_df.columns[:20], 1):
    print(f"  {i:2d}. {col}")
if len(merged_df.columns) > 20:
    print(f"  ... and {len(merged_df.columns) - 20} more columns")

print(f"\n✅ SUCCESS! Master dataset created: {output_file}")
print("="*70)
