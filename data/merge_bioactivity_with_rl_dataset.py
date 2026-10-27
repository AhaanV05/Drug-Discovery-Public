#!/usr/bin/env python3
"""
Merge bioactivity data with molecules_filtered_for_rl.csv
Adds pChEMBL values for Week 4 RL training
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("MERGING BIOACTIVITY DATA WITH RL DATASET")
print("="*100)

# Paths
base_dir = Path(__file__).parent.parent
molecules_file = base_dir / 'chembl_curated_out' / 'molecules_filtered_for_rl.csv'
activities_file = base_dir / 'chembl_curated_out' / 'activities_filtered.parquet'
output_file = base_dir / 'chembl_curated_out' / 'molecules_with_bioactivity.csv'

print(f"\n[1/4] Loading datasets...")
df_molecules = pd.read_csv(molecules_file)
print(f"  ✓ Loaded {len(df_molecules):,} molecules with {len(df_molecules.columns)} features")

df_activities = pd.read_parquet(activities_file)
print(f"  ✓ Loaded {len(df_activities):,} activity records")

# Calculate per-molecule bioactivity statistics
print(f"\n[2/4] Calculating per-molecule bioactivity statistics...")
molecule_bioactivity = df_activities.groupby('molecule_chembl_id').agg({
    'pchembl_value': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'target_chembl_id': 'nunique',
    'assay_chembl_id': 'nunique'
}).reset_index()

molecule_bioactivity.columns = [
    'molecule_chembl_id',
    'pchembl_mean',
    'pchembl_median', 
    'pchembl_std',
    'pchembl_min',
    'pchembl_max',
    'bioactivity_count',
    'target_count',
    'assay_count'
]

print(f"  ✓ Calculated statistics for {len(molecule_bioactivity):,} molecules")
print(f"\n  pChEMBL Statistics:")
print(f"    Mean: {molecule_bioactivity['pchembl_mean'].mean():.2f} ± {molecule_bioactivity['pchembl_mean'].std():.2f}")
print(f"    Median: {molecule_bioactivity['pchembl_median'].median():.2f}")
print(f"    Range: {molecule_bioactivity['pchembl_min'].min():.2f} - {molecule_bioactivity['pchembl_max'].max():.2f}")
print(f"\n  Bioactivity Distribution:")
print(f"    Molecules with >= 1 activity: {len(molecule_bioactivity):,}")
print(f"    Molecules with >= 3 activities: {(molecule_bioactivity['bioactivity_count'] >= 3).sum():,}")
print(f"    Molecules with >= 5 activities: {(molecule_bioactivity['bioactivity_count'] >= 5).sum():,}")
print(f"    Molecules with >= 10 activities: {(molecule_bioactivity['bioactivity_count'] >= 10).sum():,}")

# Merge with RL dataset
print(f"\n[3/4] Merging bioactivity data with RL dataset...")
df_merged = df_molecules.merge(
    molecule_bioactivity,
    on='molecule_chembl_id',
    how='left'
)

# Add flags for molecules with bioactivity
df_merged['has_bioactivity'] = df_merged['pchembl_mean'].notnull()
df_merged['is_active'] = (df_merged['pchembl_mean'] >= 6.0) & df_merged['pchembl_mean'].notnull()  # IC50 < 1 μM
df_merged['is_potent'] = (df_merged['pchembl_mean'] >= 7.0) & df_merged['pchembl_mean'].notnull()  # IC50 < 100 nM
df_merged['is_highly_potent'] = (df_merged['pchembl_mean'] >= 8.0) & df_merged['pchembl_mean'].notnull()  # IC50 < 10 nM

print(f"  ✓ Merged dataset: {len(df_merged):,} molecules × {len(df_merged.columns)} columns")
print(f"\n  Coverage Statistics:")
print(f"    Molecules with bioactivity: {df_merged['has_bioactivity'].sum():,} / {len(df_merged):,} ({df_merged['has_bioactivity'].sum()/len(df_merged)*100:.1f}%)")
print(f"    Active molecules (pChEMBL >= 6.0): {df_merged['is_active'].sum():,} ({df_merged['is_active'].sum()/len(df_merged)*100:.1f}%)")
print(f"    Potent molecules (pChEMBL >= 7.0): {df_merged['is_potent'].sum():,} ({df_merged['is_potent'].sum()/len(df_merged)*100:.1f}%)")
print(f"    Highly potent (pChEMBL >= 8.0): {df_merged['is_highly_potent'].sum():,} ({df_merged['is_highly_potent'].sum()/len(df_merged)*100:.1f}%)")

# Fill missing pChEMBL values with neutral value (5.0 = IC50 = 10 μM)
df_merged['pchembl_mean_filled'] = df_merged['pchembl_mean'].fillna(5.0)
df_merged['pchembl_median_filled'] = df_merged['pchembl_median'].fillna(5.0)

print(f"\n  Missing Value Handling:")
print(f"    Missing pChEMBL values filled with 5.0 (IC50 = 10 μM, neutral binding)")

# Save merged dataset
print(f"\n[4/4] Saving merged dataset...")
df_merged.to_csv(output_file, index=False)
print(f"  ✓ Saved: {output_file}")
print(f"    Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
print(f"    Dimensions: {len(df_merged):,} × {len(df_merged.columns)} columns")

print("\n" + "="*100)
print("BIOACTIVITY MERGE COMPLETE ✅")
print("="*100)

print(f"\n📊 FINAL DATASET SUMMARY:")
print(f"  Total molecules: {len(df_merged):,}")
print(f"  Total features: {len(df_merged.columns)}")
print(f"    • 84 base features (QED, SA, ADMET)")
print(f"    • 9 bioactivity features (pchembl_mean/median/std/min/max/count, target/assay_count, bioactivity_count)")
print(f"    • 4 activity flags (has_bioactivity, is_active, is_potent, is_highly_potent)")
print(f"    • 2 filled features (pchembl_mean_filled, pchembl_median_filled)")
print(f"  Bioactivity coverage: {df_merged['has_bioactivity'].sum():,} / {len(df_merged):,} ({df_merged['has_bioactivity'].sum()/len(df_merged)*100:.1f}%)")

print(f"\n🎯 RL TRAINING READY:")
print(f"  ✅ R_Binding(m) = sigmoid((pchembl_mean_filled - 6.0) / 2.0)")
print(f"  ✅ For molecules with bioactivity: Use actual pChEMBL mean")
print(f"  ✅ For molecules without bioactivity: Use neutral value (5.0)")
print(f"  ✅ 30% of reward function now complete")

print(f"\n📁 OUTPUT FILE:")
print(f"  {output_file}")

print("="*100)
