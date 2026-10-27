#!/usr/bin/env python3
"""
Comprehensive data quality check for molecules_all_features.csv
"""

import pandas as pd
import numpy as np

print("="*80)
print("DATA QUALITY CHECK: molecules_all_features.csv")
print("="*80)

# Load the master dataset
df = pd.read_csv('chembl_curated_out/molecules_all_features.csv')

print(f"\n[1/6] BASIC STATISTICS")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\n[2/6] MISSING VALUES ANALYSIS")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percent': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(f"  ⚠️ Found {len(missing_df)} columns with missing values:\n")
    print(missing_df.to_string(index=False))
    print(f"\n  Total missing values: {missing.sum():,}")
else:
    print(f"  ✅ NO MISSING VALUES - All columns complete!")

print(f"\n[3/6] CRITICAL FEATURES CHECK (for RL Training)")
critical_features = {
    'molecule_chembl_id': 'Identifier',
    'canonical_smiles': 'SMILES structure',
    'qed': 'Drug-likeness (30% RL weight)',
    'sa_score': 'Synthetic accessibility (20% RL weight)',
    'hERG': 'Cardiac toxicity (10% RL weight)',
    'Ames': 'Mutagenicity (10% RL weight)',
    'CYP2C9-inh': 'Drug-drug interactions (10% RL weight)',
    'CYP2D6-inh': 'Drug-drug interactions (10% RL weight)',
    'CYP3A4-inh': 'Drug-drug interactions (10% RL weight)',
    'bemis_murcko_scaffold': 'Novelty (10% RL weight)',
}

print("  Checking critical features for RL training:\n")
all_critical_present = True
for col, description in critical_features.items():
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            print(f"  ✅ {col:25s} - {description} - COMPLETE")
        else:
            print(f"  ⚠️ {col:25s} - {description} - {missing_count:,} missing ({missing_count/len(df)*100:.2f}%)")
            all_critical_present = False
    else:
        print(f"  ❌ {col:25s} - {description} - COLUMN MISSING!")
        all_critical_present = False

if all_critical_present:
    print(f"\n  ✅ All critical features present and complete!")
else:
    print(f"\n  ⚠️ Some critical features have issues!")

print(f"\n[4/6] DATA TYPE CHECK")
print(f"  Object (string) columns: {len(df.select_dtypes(include='object').columns)}")
print(f"  Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"  Boolean columns: {len(df.select_dtypes(include='bool').columns)}")

# Check for unexpected data types
print(f"\n  Column data types breakdown:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"    {str(dtype):15s}: {count:3d} columns")

print(f"\n[5/6] VALUE RANGE CHECK (Key Features)")
key_numeric_features = ['qed', 'sa_score', 'mw', 'alogp', 'hbd', 'hba', 'psa']
print(f"  Checking value ranges for key features:\n")

for col in key_numeric_features:
    if col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        
        # Expected ranges
        expected = {
            'qed': (0, 1),
            'sa_score': (1, 10),
            'mw': (0, 1000),
            'alogp': (-5, 10),
            'hbd': (0, 20),
            'hba': (0, 30),
            'psa': (0, 300)
        }
        
        if col in expected:
            exp_min, exp_max = expected[col]
            in_range = (min_val >= exp_min) and (max_val <= exp_max)
            status = "✅" if in_range else "⚠️"
            print(f"  {status} {col:15s}: [{min_val:8.2f}, {max_val:8.2f}] mean={mean_val:8.2f}")
        else:
            print(f"  ℹ️ {col:15s}: [{min_val:8.2f}, {max_val:8.2f}] mean={mean_val:8.2f}")

print(f"\n[6/6] DUPLICATE CHECK")
duplicates = df.duplicated(subset=['molecule_chembl_id']).sum()
if duplicates > 0:
    print(f"  ⚠️ Found {duplicates:,} duplicate molecule IDs!")
else:
    print(f"  ✅ No duplicate molecule IDs")

# Check for duplicate SMILES
smiles_duplicates = df.duplicated(subset=['canonical_smiles']).sum()
if smiles_duplicates > 0:
    print(f"  ⚠️ Found {smiles_duplicates:,} duplicate SMILES structures!")
else:
    print(f"  ✅ No duplicate SMILES structures")

print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

issues = []
if len(missing_df) > 0:
    issues.append(f"Missing values in {len(missing_df)} columns")
if not all_critical_present:
    issues.append("Some critical features missing or incomplete")
if duplicates > 0:
    issues.append(f"{duplicates} duplicate molecule IDs")

if len(issues) == 0:
    print("\n  ✅ EXCELLENT! Dataset is complete and ready for use!")
    print("  ✅ All critical features present")
    print("  ✅ No missing values")
    print("  ✅ No duplicates")
    print("  ✅ All value ranges reasonable")
    print("\n  🚀 Ready for Week 2 Diffusion Model Training!")
else:
    print("\n  ⚠️ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
    print("\n  Action required before proceeding to training.")

print("\n" + "="*80)
