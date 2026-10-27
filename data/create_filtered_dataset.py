#!/usr/bin/env python3
"""
Create filtered dataset with essential features for RL training
Keeps: 24 base + 30 CRITICAL + 14 IMPORTANT + 16 USEFUL = 84 features
"""

import pandas as pd

print("="*100)
print("CREATING FILTERED DATASET FOR RL TRAINING")
print("="*100)

# Load full dataset
print("\n[1/4] Loading molecules_all_features.csv...")
df = pd.read_csv('chembl_curated_out/molecules_all_features.csv')
print(f"  Loaded: {len(df):,} molecules × {len(df.columns)} features")

# Define feature groups
base_features = [
    'molecule_chembl_id', 'pref_name', 'canonical_smiles',
    'mw', 'alogp', 'hbd', 'hba', 'psa',
    'max_phase', 'first_approval',
    'qed', 'sa_score',
    'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds',
    'formal_charge', 'num_heteroatoms', 'fraction_csp3',
    'lipinski_violations', 'passes_lipinski',
    'pains_alert', 'brenk_alert', 'passes_filters',
    'bemis_murcko_scaffold'
]

critical_features = [
    # Safety (6)
    'hERG', 'hERG-10um', 'Ames', 'DILI', 'Carcinogenicity', 'Genotoxicity',
    # Metabolism - CYP450s (13)
    'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C9-inh', 'CYP2C9-sub',
    'CYP2C19-inh', 'CYP2C19-sub', 'CYP2D6-inh', 'CYP2D6-sub',
    'CYP3A4-inh', 'CYP3A4-sub', 'CYP2B6-inh', 'CYP2B6-sub', 'CYP2C8-inh',
    # ADME Properties (11) - LOWERCASE column names from ADMET results
    'caco2', 'hia', 'f20', 'f30', 'f50', 'BBB', 'PPB', 'logVDss', 'Fu', 'cl-plasma', 't0.5'
]

important_features = [
    # Organ Toxicity (7)
    'Respiratory', 'H-HT', 'Nephrotoxicity-DI', 'Hematotoxicity',
    'Ototoxicity', 'Neurotoxicity-DI', 'SkinSen',
    # Transporters (7)
    'pgp_inh', 'pgp_sub', 'OATP1B1', 'OATP1B3', 'BCRP', 'BSEP', 'MRP1'
]

useful_features = [
    # Physicochemical (7)
    'logS', 'logD', 'logP', 'mp', 'bp', 'pka_acidic', 'pka_basic',
    # Permeability (2)
    'MDCK', 'PAMPA',
    # Environmental/Additional (7)
    'BCF', 'IGC50', 'LC50DM', 'LC50FM', 'LM-human', 'ROA', 'FDAMDD'
]

print("\n[2/4] Feature selection breakdown:")
print(f"  • Base features (Week 1):        {len(base_features):2d}")
print(f"  • CRITICAL ADMET (must have):    {len(critical_features):2d}")
print(f"  • IMPORTANT ADMET (should have): {len(important_features):2d}")
print(f"  • USEFUL ADMET (nice to have):   {len(useful_features):2d}")
print(f"  {'─'*50}")
print(f"  Total selected features:         {len(base_features) + len(critical_features) + len(important_features) + len(useful_features)}")

# Combine all selected features
selected_features = base_features + critical_features + important_features + useful_features

# Verify all features exist
print("\n[3/4] Verifying feature availability...")
missing_features = [f for f in selected_features if f not in df.columns]
if missing_features:
    print(f"  ⚠️ WARNING: {len(missing_features)} features not found:")
    for f in missing_features:
        print(f"    • {f}")
    # Remove missing features
    selected_features = [f for f in selected_features if f in df.columns]
    print(f"  Proceeding with {len(selected_features)} available features")
else:
    print(f"  ✅ All {len(selected_features)} features found!")

# Create filtered dataset
print("\n[4/4] Creating filtered dataset...")
df_filtered = df[selected_features].copy()

# Save to new file
output_file = 'chembl_curated_out/molecules_filtered_for_rl.csv'
df_filtered.to_csv(output_file, index=False)

file_size_mb = pd.io.common.file_path_to_url(output_file)
import os
file_size = os.path.getsize(output_file) / 1024**2

print(f"  ✅ Saved: {output_file}")
print(f"  Rows: {len(df_filtered):,}")
print(f"  Columns: {len(df_filtered.columns)}")
print(f"  File size: {file_size:.1f} MB")

print("\n" + "="*100)
print("FEATURE BREAKDOWN IN FILTERED DATASET")
print("="*100)

print("\n🔵 BASE FEATURES (24):")
for i, f in enumerate(base_features, 1):
    if f in df_filtered.columns:
        print(f"  {i:2d}. {f}")

print("\n🔴 CRITICAL ADMET - Safety (6):")
safety_features = ['hERG', 'hERG-10um', 'Ames', 'DILI', 'Carcinogenicity', 'Genotoxicity']
for i, f in enumerate(safety_features, 1):
    if f in df_filtered.columns:
        print(f"  {i}. {f}")

print("\n🔴 CRITICAL ADMET - Metabolism/CYP450s (13):")
cyp_features = [f for f in critical_features if 'CYP' in f]
for i, f in enumerate(cyp_features, 1):
    if f in df_filtered.columns:
        print(f"  {i:2d}. {f}")

print("\n🔴 CRITICAL ADMET - ADME Properties (11):")
adme_features = ['caco2', 'hia', 'f20', 'f30', 'f50', 'BBB', 'PPB', 'logVDss', 'Fu', 'cl-plasma', 't0.5']
for i, f in enumerate(adme_features, 1):
    if f in df_filtered.columns:
        print(f"  {i:2d}. {f}")

print("\n🟠 IMPORTANT ADMET - Organ Toxicity (7):")
organ_tox = ['Respiratory', 'H-HT', 'Nephrotoxicity-DI', 'Hematotoxicity', 'Ototoxicity', 'Neurotoxicity-DI', 'SkinSen']
for i, f in enumerate(organ_tox, 1):
    if f in df_filtered.columns:
        print(f"  {i}. {f}")

print("\n🟠 IMPORTANT ADMET - Transporters (7):")
transporters = ['pgp_inh', 'pgp_sub', 'OATP1B1', 'OATP1B3', 'BCRP', 'BSEP', 'MRP1']
for i, f in enumerate(transporters, 1):
    if f in df_filtered.columns:
        print(f"  {i}. {f}")

print("\n🟡 USEFUL ADMET - Physicochemical (7):")
physico = ['logS', 'logD', 'logP', 'mp', 'bp', 'pka_acidic', 'pka_basic']
for i, f in enumerate(physico, 1):
    if f in df_filtered.columns:
        print(f"  {i}. {f}")

print("\n🟡 USEFUL ADMET - Permeability (2):")
perm = ['MDCK', 'PAMPA']
for i, f in enumerate(perm, 1):
    if f in df_filtered.columns:
        print(f"  {i}. {f}")

print("\n🟡 USEFUL ADMET - Environmental/Additional (7):")
env = ['BCF', 'IGC50', 'LC50DM', 'LC50FM', 'LM-human', 'ROA', 'FDAMDD']
for i, f in enumerate(env, 1):
    if f in df_filtered.columns:
        print(f"  {i}. {f}")

print("\n" + "="*100)
print("✅ SUCCESS!")
print("="*100)
print(f"\nFiltered dataset ready: {output_file}")
print(f"  • {len(df_filtered):,} molecules")
print(f"  • {len(df_filtered.columns)} essential features")
print(f"  • {file_size:.1f} MB (reduced from 374.9 MB)")
print(f"\n🚀 Ready for Week 2-4 training pipeline!")
print("="*100)
