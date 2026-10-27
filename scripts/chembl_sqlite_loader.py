#!/usr/bin/env python3
"""
ChEMBL SQLite Database Loader

Queries the local ChEMBL SQLite database to extract diverse, drug-like molecules.
Much faster than API calls - queries 2.4M molecules in minutes instead of days.

Usage:
    python chembl_sqlite_loader.py --db_path chembl_34/chembl_34_sqlite/chembl_34.db --n_molecules 200000
"""

import sqlite3
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


@dataclass
class FilterConfig:
    """Drug-like filtering configuration (same as API loader)"""
    mw_min: float = 150.0
    mw_max: float = 500.0
    logp_max: float = 5.0
    hbd_max: int = 5
    hba_max: int = 10
    psa_max: float = 140.0
    heavy_atoms_max: int = 50
    diversity_min_tanimoto: float = 0.75
    fingerprint_radius: int = 2
    fingerprint_nbits: int = 2048


def mol_from_smiles(smiles: str):
    """Parse SMILES, return None if invalid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Sanitize
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def passes_structure_filters(mol: Chem.Mol, cfg: FilterConfig) -> bool:
    """Check if molecule passes RDKit structure validation"""
    if mol is None:
        return False
    
    # Must have atoms
    if mol.GetNumAtoms() == 0:
        return False
    
    # Single connected component (no salts)
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    
    # Heavy atom limit
    if mol.GetNumHeavyAtoms() > cfg.heavy_atoms_max:
        return False

    return True


def morgan_fp(mol: Chem.Mol, radius: int, nbits: int):
    """Generate Morgan fingerprint"""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def greedy_diverse_subset(
    items: List[Tuple[str, str]],  # (chembl_id, smiles)
    cfg: FilterConfig,
    target_n: int,
) -> List[Tuple[str, str]]:
    """
    Greedy diversity selection using Morgan fingerprints.
    Keeps molecules with Tanimoto similarity < threshold.
    """
    kept: List[Tuple[str, str]] = []
    kept_fps = []

    print(f"\n[DIVERSITY] Selecting {target_n:,} diverse molecules from {len(items):,} candidates...")
    
    for chembl_id, smiles in tqdm(items, desc="Diversity selection"):
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        fp = morgan_fp(mol, cfg.fingerprint_radius, cfg.fingerprint_nbits)

        if not kept_fps:
            # First molecule always kept
            kept.append((chembl_id, smiles))
            kept_fps.append(fp)
        else:
            # Calculate max similarity to kept set
            sims = DataStructs.BulkTanimotoSimilarity(fp, kept_fps)
            if (max(sims) if sims else 0.0) < cfg.diversity_min_tanimoto:
                kept.append((chembl_id, smiles))
                kept_fps.append(fp)

        if len(kept) >= target_n:
            break

    print(f"✓ Selected {len(kept):,} diverse molecules")
    return kept


def query_chembl_database(db_path: str, cfg: FilterConfig) -> pd.DataFrame:
    """
    Query ChEMBL SQLite database for drug-like molecules.
    
    ChEMBL Schema (key tables):
    - compound_structures: canonical_smiles
    - compound_properties: molecular properties (mw, alogp, hbd, hba, psa, etc.)
    - molecule_dictionary: chembl_id, pref_name, max_phase, first_approval
    """
    print(f"\n{'='*60}")
    print(f"ChEMBL SQLite Database Loader")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Target molecules: {cfg.target_n:,}")
    print(f"{'='*60}\n")
    
    # Connect to database
    print("[1/4] Connecting to ChEMBL database...")
    conn = sqlite3.connect(db_path)
    
    # Query for drug-like molecules with all required data
    print("[2/4] Querying drug-like molecules (MW 150-500, LogP ≤5, etc.)...")
    print("       This may take 2-5 minutes for 2.4M molecules...")
    
    query = """
    SELECT 
        md.chembl_id,
        md.pref_name,
        cs.canonical_smiles,
        cp.full_mwt as mw,
        cp.alogp,
        cp.hbd,
        cp.hba,
        cp.psa,
        md.max_phase,
        md.first_approval
    FROM molecule_dictionary md
    INNER JOIN compound_structures cs ON md.molregno = cs.molregno
    INNER JOIN compound_properties cp ON md.molregno = cp.molregno
    WHERE 
        cs.canonical_smiles IS NOT NULL
        AND cp.full_mwt BETWEEN ? AND ?
        AND cp.alogp <= ?
        AND cp.hbd <= ?
        AND cp.hba <= ?
        AND cp.psa <= ?
        AND md.chembl_id IS NOT NULL
    ORDER BY RANDOM()
    LIMIT ?
    """
    
    # Fetch 3-4x target to ensure enough after diversity filtering
    fetch_limit = cfg.target_n * 4
    
    params = (
        cfg.mw_min,
        cfg.mw_max,
        cfg.logp_max,
        cfg.hbd_max,
        cfg.hba_max,
        cfg.psa_max,
        fetch_limit
    )
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    print(f"✓ Fetched {len(df):,} drug-like molecules from database")
    
    # Apply RDKit structure validation
    print("\n[3/4] Validating molecular structures with RDKit...")
    valid_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Structure validation"):
        smiles = row['canonical_smiles']
        mol = mol_from_smiles(smiles)
        
        if passes_structure_filters(mol, cfg):
            valid_rows.append(row)
    
    df = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"✓ {len(df):,} molecules passed structure validation")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Load drug-like molecules from ChEMBL SQLite database")
    parser.add_argument("--db_path", type=str, default="chembl_34/chembl_34_sqlite/chembl_34.db",
                        help="Path to ChEMBL SQLite database file")
    parser.add_argument("--n_molecules", type=int, default=200000,
                        help="Target number of diverse molecules")
    parser.add_argument("--outdir", type=str, default="chembl_curated_out",
                        help="Output directory")
    parser.add_argument("--no_diversity", action="store_true",
                        help="Skip diversity selection (faster)")
    
    args = parser.parse_args()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    cfg = FilterConfig()
    cfg.target_n = args.n_molecules
    
    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"[ERROR] Database not found: {args.db_path}")
        print("\nExpected location after extraction:")
        print("  chembl_34/chembl_34_sqlite/chembl_34.db")
        print("\nMake sure you've extracted the tar.gz file first!")
        return
    
    # Query database
    mol_df = query_chembl_database(args.db_path, cfg)
    
    if mol_df.empty:
        print("[ERROR] No molecules found! Check database schema.")
        return
    
    # Diversity selection
    if not args.no_diversity:
        print(f"\n[4/4] Applying diversity selection (target: {args.n_molecules:,})...")
        items = list(zip(mol_df["chembl_id"].tolist(), mol_df["canonical_smiles"].tolist()))
        diverse = greedy_diverse_subset(items, cfg, args.n_molecules)
        
        keep_ids = set([x[0] for x in diverse])
        mol_df = mol_df[mol_df["chembl_id"].isin(keep_ids)].copy()
        print(f"✓ Selected {len(mol_df):,} diverse molecules")
    else:
        print(f"\n[4/4] Skipping diversity selection, taking top {args.n_molecules:,}...")
        mol_df = mol_df.head(args.n_molecules).copy()
    
    # Rename chembl_id to molecule_chembl_id for consistency
    mol_df = mol_df.rename(columns={"chembl_id": "molecule_chembl_id"})
    
    # Save outputs
    print(f"\n[5/5] Saving outputs to {outdir}/...")
    
    # Parquet (efficient binary format)
    parquet_path = outdir / "molecules_diverse.parquet"
    mol_df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved: {parquet_path} ({len(mol_df):,} molecules)")
    
    # CSV (human-readable)
    csv_path = outdir / "molecules_diverse.csv"
    mol_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # SMILES file (for generative models)
    smi_path = outdir / "molecules_diverse.smi"
    with open(smi_path, 'w') as f:
        for smiles in mol_df['canonical_smiles']:
            f.write(f"{smiles}\n")
    print(f"✓ Saved: {smi_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total molecules: {len(mol_df):,}")
    print(f"Unique ChEMBL IDs: {mol_df['molecule_chembl_id'].nunique():,}")
    print(f"\nFiles ready for training:")
    print(f"  - {smi_path.name} (for generative models)")
    print(f"  - {parquet_path.name} (full data)")
    print(f"  - {csv_path.name} (human-readable)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
