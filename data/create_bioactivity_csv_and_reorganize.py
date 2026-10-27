#!/usr/bin/env python3
"""
Create separate bioactivity-only CSV and reorganize files
"""

import pandas as pd
import shutil
from pathlib import Path

print("="*80)
print("CREATING BIOACTIVITY CSV & REORGANIZING FILES")
print("="*80)

base_dir = Path('.')
chembl_dir = base_dir / 'chembl_curated_out'

# 1. Load activities and create bioactivity-only CSV
print(f"\n[1/4] Creating bioactivity-only CSV...")
df_activities = pd.read_parquet(chembl_dir / 'activities_filtered.parquet')

# Calculate per-molecule statistics
bioactivity_stats = df_activities.groupby('molecule_chembl_id').agg({
    'pchembl_value': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'target_chembl_id': 'nunique',
    'assay_chembl_id': 'nunique',
    'target_name': lambda x: '; '.join(sorted(set(x)))[:500]  # Top targets (truncated)
}).reset_index()

bioactivity_stats.columns = [
    'molecule_chembl_id',
    'activity_count',
    'pchembl_mean',
    'pchembl_median',
    'pchembl_std',
    'pchembl_min',
    'pchembl_max',
    'target_count',
    'assay_count',
    'target_names'
]

# Add activity flags
bioactivity_stats['has_bioactivity'] = True
bioactivity_stats['is_active'] = bioactivity_stats['pchembl_mean'] >= 6.0
bioactivity_stats['is_potent'] = bioactivity_stats['pchembl_mean'] >= 7.0
bioactivity_stats['is_highly_potent'] = bioactivity_stats['pchembl_mean'] >= 8.0

# Save bioactivity-only CSV
bioactivity_file = chembl_dir / 'bioactivity_data_only.csv'
bioactivity_stats.to_csv(bioactivity_file, index=False)
print(f"  ✓ Saved: {bioactivity_file}")
print(f"    Rows: {len(bioactivity_stats):,}")
print(f"    Coverage: {len(bioactivity_stats):,} / 200,000 = {len(bioactivity_stats)/200000*100:.1f}%")

# 2. Rename existing files for clarity
print(f"\n[2/4] Renaming files for clarity...")

rename_map = {
    'molecules_diverse.csv': 'molecules_200k_week1_diverse.csv',
    'molecules_diverse_enriched.csv': 'molecules_200k_week1_with_24_features.csv',
    'molecules_all_features.csv': 'molecules_200k_all_134_features.csv',
    'molecules_filtered_for_rl.csv': 'molecules_200k_filtered_84_features.csv',
    'molecules_with_bioactivity.csv': 'OLD_molecules_with_bioactivity_kinases_only.csv'  # Old kinase-only version
}

for old_name, new_name in rename_map.items():
    old_path = chembl_dir / old_name
    new_path = chembl_dir / new_name
    
    if old_path.exists():
        if new_path.exists():
            new_path.unlink()  # Remove if exists
        shutil.move(str(old_path), str(new_path))
        print(f"  ✓ Renamed: {old_name} → {new_name}")
    else:
        print(f"  ⊘ Skipped: {old_name} (not found)")

# 3. Merge bioactivity with filtered RL dataset
print(f"\n[3/4] Merging bioactivity with filtered RL dataset...")
df_filtered = pd.read_csv(chembl_dir / 'molecules_200k_filtered_84_features.csv')
print(f"  ✓ Loaded: molecules_200k_filtered_84_features.csv ({len(df_filtered):,} × {len(df_filtered.columns)} columns)")

# Merge with bioactivity (left join - keep all 200k molecules)
df_merged = df_filtered.merge(
    bioactivity_stats,
    on='molecule_chembl_id',
    how='left'
)

# Fill missing values for molecules without bioactivity
df_merged['has_bioactivity'] = df_merged['has_bioactivity'].fillna(False)
df_merged['is_active'] = df_merged['is_active'].fillna(False)
df_merged['is_potent'] = df_merged['is_potent'].fillna(False)
df_merged['is_highly_potent'] = df_merged['is_highly_potent'].fillna(False)
df_merged['activity_count'] = df_merged['activity_count'].fillna(0).astype(int)
df_merged['target_count'] = df_merged['target_count'].fillna(0).astype(int)
df_merged['assay_count'] = df_merged['assay_count'].fillna(0).astype(int)

# IMPORTANT: Do NOT fill pchembl values yet - leave as NaN for ML model training
# df_merged['pchembl_mean'] stays NaN for molecules without bioactivity

print(f"  ✓ Merged dataset: {len(df_merged):,} × {len(df_merged.columns)} columns")
print(f"    With bioactivity: {df_merged['has_bioactivity'].sum():,} ({df_merged['has_bioactivity'].sum()/len(df_merged)*100:.1f}%)")
print(f"    Without bioactivity: {(~df_merged['has_bioactivity']).sum():,} ({(~df_merged['has_bioactivity']).sum()/len(df_merged)*100:.1f}%)")

# Save merged dataset (WITHOUT imputation - keep NaN for ML training)
merged_file = chembl_dir / 'molecules_200k_with_bioactivity_21pct_coverage.csv'
df_merged.to_csv(merged_file, index=False)
print(f"  ✓ Saved: {merged_file}")
print(f"    Size: {merged_file.stat().st_size / 1024**2:.1f} MB")

# 4. Summary
print(f"\n[4/4] Summary of files in chembl_curated_out/:")
print(f"\n  PRIMARY DATASETS (use these):")
print(f"    1. molecules_200k_filtered_84_features.csv")
print(f"       → 200k molecules × 84 features (QED, SA, ADMET)")
print(f"       → Use for: Week 2-3 training (Diffusion, VAE)")
print(f"")
print(f"    2. bioactivity_data_only.csv ⭐ NEW")
print(f"       → 42,113 molecules × 14 bioactivity features")
print(f"       → Use for: ML model training (predict missing pChEMBL)")
print(f"")
print(f"    3. molecules_200k_with_bioactivity_21pct_coverage.csv ⭐ NEW")
print(f"       → 200k molecules × 97 features (84 base + 13 bioactivity)")
print(f"       → pChEMBL is NaN for 158k molecules (to be predicted by ML)")
print(f"       → Use for: Week 4 RL training (after ML prediction)")
print(f"")
print(f"  INTERMEDIATE FILES:")
print(f"    • molecules_200k_week1_diverse.csv (200k × 2: SMILES + ChEMBL ID)")
print(f"    • molecules_200k_week1_with_24_features.csv (200k × 24: Week 1 output)")
print(f"    • molecules_200k_all_134_features.csv (200k × 134: All ADMET features)")
print(f"")
print(f"  BIOACTIVITY METADATA:")
print(f"    • activities_filtered.parquet (88,447 activities)")
print(f"    • assays_linked.parquet (37,009 assays)")
print(f"    • targets_linked.parquet (1,982 targets)")
print(f"    • bioactivity_statistics.json (summary stats)")
print(f"")
print(f"  OLD FILES (archived):")
print(f"    • OLD_molecules_with_bioactivity_kinases_only.csv (kinase-only, 4.7% coverage)")

print("\n" + "="*80)
print("REORGANIZATION COMPLETE ✅")
print("="*80)
print(f"\n📊 NEXT STEPS:")
print(f"  1. ✅ Bioactivity extracted (21.1% coverage)")
print(f"  2. ✅ Files reorganized and renamed")
print(f"  3. ⏳ Implement masked reward function")
print(f"  4. ⏳ Train ML model to predict missing pChEMBL (158k molecules)")
print("="*80)
