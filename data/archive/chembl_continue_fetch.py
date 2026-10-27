#!/usr/bin/env python3
"""
ChEMBL Continue Fetch Script

Continues fetching molecules from a specified offset to avoid duplicates.
Use this to fetch additional molecules after the initial run completes.

Usage:
    # Continue from offset 193,100 (after 10 rounds) to get 60k more molecules
    python chembl_continue_fetch.py --start_offset 193100 --additional_rounds 3
    
    # The output will be saved as chembl_curated_out/molecules_continue.parquet
    # You can then merge with the original cached data
"""

import os
import time
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors


def fetch_from_offset(start_offset: int, num_rounds: int, page_size: int = 1000, outdir: str = "chembl_curated_out"):
    """
    Fetch molecules starting from a specific offset.
    
    Args:
        start_offset: ChEMBL API offset to start from (e.g., 193100)
        num_rounds: Number of fetch rounds to execute
        page_size: Molecules per API call
        outdir: Output directory
    """
    molecule = new_client.molecule
    
    query = molecule.filter(
        max_phase__isnull=False
    ).only(
        "molecule_chembl_id",
        "pref_name",
        "molecule_structures",
        "molecule_properties",
        "max_phase",
        "first_approval",
    )
    
    rows: List[Dict[str, Any]] = []
    offset = start_offset
    
    print(f"\n{'='*60}")
    print(f"ChEMBL Continue Fetch")
    print(f"{'='*60}")
    print(f"Starting offset: {start_offset:,}")
    print(f"Fetch rounds: {num_rounds}")
    print(f"Expected molecules: ~{num_rounds * 19310:,}")
    print(f"{'='*60}\n")
    
    pbar = tqdm(desc="Fetching molecules", unit="rows", initial=0)
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n[DEBUG] Fetch round {round_num}/{num_rounds}, calling API: offset={offset}, limit={page_size}", flush=True)
        
        # Retry logic
        max_retries = 6
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[DEBUG] API attempt {attempt}/{max_retries}...", flush=True)
                batch = query[offset : offset + page_size]
                print(f"[DEBUG] API call successful", flush=True)
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                print(f"[DEBUG] API error: {e}, retrying in {wait_time}s...", flush=True)
                time.sleep(wait_time)
        
        # Convert batch to list with progress tracking
        print(f"[DEBUG] API call returned, converting to list...", flush=True)
        print(f"[DEBUG] Starting batch iteration...", flush=True)
        
        batch_list = []
        for idx, item in enumerate(batch, 1):
            batch_list.append(item)
            if idx % 1000 == 0:
                print(f"[DEBUG] Converted {idx} items from batch...", flush=True)
        
        batch_size = len(batch_list)
        print(f"[DEBUG] Received {batch_size} molecules", flush=True)
        
        # Check if we got a partial batch (API exhausted)
        if batch_size < page_size:
            print(f"\n[INFO] Received partial batch ({batch_size} < {page_size}), ChEMBL database exhausted")
            if batch_size == 0:
                print(f"[INFO] No more molecules available, stopping fetch")
                break
        
        time.sleep(0.05)
        
        # Extract and filter molecules
        print(f"[DEBUG] Extending rows with {batch_size} molecules...", flush=True)
        for mol in batch_list:
            try:
                if not mol.get("molecule_structures"):
                    continue
                    
                smiles = mol["molecule_structures"].get("canonical_smiles")
                if not smiles:
                    continue
                
                props = mol.get("molecule_properties", {})
                if not props:
                    continue
                
                # Basic validation
                rdkit_mol = Chem.MolFromSmiles(smiles)
                if rdkit_mol is None:
                    continue
                
                # Client-side drug-like filtering
                mw = props.get("full_mwt")
                alogp = props.get("alogp")
                hbd = props.get("hbd")
                hba = props.get("hba")
                psa = props.get("psa")
                
                if mw is None or alogp is None or hbd is None or hba is None or psa is None:
                    continue
                
                # Drug-like filters
                if not (150 <= mw <= 500):
                    continue
                if alogp > 5:
                    continue
                if hbd > 5:
                    continue
                if hba > 10:
                    continue
                if psa > 140:
                    continue
                
                # Additional RDKit checks
                num_heavy = rdkit_mol.GetNumHeavyAtoms()
                if num_heavy > 50:
                    continue
                
                # Single connected component
                frags = Chem.GetMolFrags(rdkit_mol, asMols=False)
                if len(frags) > 1:
                    continue
                
                rows.append({
                    "molecule_chembl_id": mol.get("molecule_chembl_id"),
                    "pref_name": mol.get("pref_name"),
                    "canonical_smiles": smiles,
                    "molecular_weight": mw,
                    "alogp": alogp,
                    "hbd": hbd,
                    "hba": hba,
                    "psa": psa,
                    "max_phase": mol.get("max_phase"),
                    "first_approval": mol.get("first_approval"),
                })
                
            except Exception as e:
                continue
        
        pbar.update(batch_size)
        print(f"[DEBUG] Total molecules so far: {len(rows)}\n", flush=True)
        
        # Update offset for next round
        offset += batch_size
        
        # If we got a partial batch, stop
        if batch_size < page_size:
            break
    
    pbar.close()
    
    # Save results
    if not rows:
        print("\n[ERROR] No molecules fetched!")
        return None
    
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)
    
    # Save to file
    os.makedirs(outdir, exist_ok=True)
    output_file = os.path.join(outdir, "molecules_continue.parquet")
    df.to_parquet(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Fetched {len(df):,} additional molecules")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Final offset reached: {offset:,}")
    print(f"{'='*60}\n")
    
    return df


def merge_with_existing(outdir: str = "chembl_curated_out"):
    """
    Merge the continue fetch results with existing cached data.
    """
    cache_dir = Path(outdir) / ".cache"
    continue_file = Path(outdir) / "molecules_continue.parquet"
    
    if not continue_file.exists():
        print(f"[ERROR] Continue file not found: {continue_file}")
        return
    
    # Load continue data
    continue_df = pd.read_parquet(continue_file)
    print(f"[INFO] Loaded {len(continue_df):,} molecules from continue fetch")
    
    # Check if cached step1 exists
    step1_cache = cache_dir / "step1_molecules.pkl"
    if step1_cache.exists():
        with open(step1_cache, 'rb') as f:
            original_df = pickle.load(f)
        print(f"[INFO] Loaded {len(original_df):,} molecules from original cache")
        
        # Merge
        merged_df = pd.concat([original_df, continue_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)
        
        print(f"[INFO] Merged total: {len(merged_df):,} unique molecules")
        
        # Save merged back to cache
        with open(step1_cache, 'wb') as f:
            pickle.dump(merged_df, f)
        
        print(f"[SUCCESS] Updated cache with merged data: {step1_cache}")
        print(f"[INFO] You can now run the main script with --resume to continue from Step 2 (diversity selection)")
    else:
        print(f"[WARNING] Original cache not found at {step1_cache}")
        print(f"[INFO] Continue data is available at {continue_file}")
        print(f"[INFO] Run the main script first to generate the original cache, then merge manually")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue ChEMBL molecule fetch from offset")
    parser.add_argument("--start_offset", type=int, required=True, 
                        help="Offset to start from (e.g., 193100 after 10 rounds)")
    parser.add_argument("--additional_rounds", type=int, default=3,
                        help="Number of additional fetch rounds (default: 3 for ~60k molecules)")
    parser.add_argument("--page_size", type=int, default=1000,
                        help="Molecules per API call (default: 1000)")
    parser.add_argument("--outdir", type=str, default="chembl_curated_out",
                        help="Output directory (default: chembl_curated_out)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge results with existing cache after fetch")
    
    args = parser.parse_args()
    
    # Fetch additional molecules
    df = fetch_from_offset(
        start_offset=args.start_offset,
        num_rounds=args.additional_rounds,
        page_size=args.page_size,
        outdir=args.outdir
    )
    
    # Optionally merge with existing cache
    if args.merge and df is not None:
        print("\n[INFO] Merging with existing cache...")
        merge_with_existing(args.outdir)
