#!/usr/bin/env python3
"""
Extract molecules from checkpoint and save as final output
"""

import pickle
from pathlib import Path
import pandas as pd

# Load the diverse molecules from checkpoint
cache_dir = Path("chembl_curated_out/.cache")
with open(cache_dir / "step2_diverse.pkl", "rb") as f:
    mol_df = pickle.load(f)

print(f"Loaded {len(mol_df):,} diverse molecules from checkpoint")

# Save as parquet
output_dir = Path("chembl_curated_out")
mol_df.to_parquet(output_dir / "molecules_diverse.parquet", index=False)
print(f"✓ Saved: molecules_diverse.parquet ({len(mol_df):,} molecules)")

# Save SMILES file for training
mol_df[["canonical_smiles", "molecule_chembl_id"]].to_csv(
    output_dir / "molecules_diverse.smi", sep="\t", index=False, header=False
)
print(f"✓ Saved: molecules_diverse.smi")

# Save CSV for easy viewing
mol_df.to_csv(output_dir / "molecules_diverse.csv", index=False)
print(f"✓ Saved: molecules_diverse.csv")

print(f"\n{'='*60}")
print(f"✅ EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"Total molecules: {len(mol_df):,}")
print(f"Unique ChEMBL IDs: {mol_df['molecule_chembl_id'].nunique():,}")
print(f"\nFiles ready for training:")
print(f"  - molecules_diverse.smi (for generative models)")
print(f"  - molecules_diverse.parquet (full data)")
print(f"  - molecules_diverse.csv (human-readable)")
print()
