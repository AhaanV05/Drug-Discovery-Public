#!/usr/bin/env python3
"""
Auto-batching molecule fetcher: Fetch 3 rounds → Filter → Extract → Combine → Repeat
This prevents RAM issues by processing in small chunks
"""

import os
import sys
import subprocess
import pickle
import pandas as pd
from pathlib import Path
import time

def run_command(cmd):
    """Run shell command and return success"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def get_molecule_count(csv_path):
    """Get current molecule count from master CSV"""
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    return len(df)

def main():
    target_molecules = 50000
    batch_size = 3  # Fetch 3 rounds at a time
    start_round = 10  # Start from round 10 (you already have 0-9)
    output_dir = Path("chembl_curated_out")
    master_csv = output_dir / "molecules_diverse.csv"
    
    # Load existing molecules
    current_count = get_molecule_count(master_csv)
    print(f"\n{'='*60}")
    print(f"AUTO-BATCHING MOLECULE FETCHER")
    print(f"{'='*60}")
    print(f"Current molecules: {current_count:,}")
    print(f"Target molecules: {target_molecules:,}")
    print(f"Starting from round: {start_round}")
    print(f"Batch size: {batch_size} rounds")
    print(f"{'='*60}\n")
    
    current_round = start_round
    iteration = 1
    
    while current_count < target_molecules:
        print(f"\n{'#'*60}")
        print(f"ITERATION {iteration} - Fetching 3 rounds starting from {current_round}")
        print(f"Progress: {current_count:,}/{target_molecules:,} molecules")
        print(f"{'#'*60}\n")
        
        # Step 1: Fetch 3 rounds (the main script handles this with --continue-from-round)
        print(f"\n[FETCH] Starting round {current_round} (will fetch {current_round}, {current_round+1}, {current_round+2})")
        cmd = f"python data/chembl_data_loader.py --n_molecules 50000 --continue-from-round {current_round}"
        
        if not run_command(cmd):
            print(f"[WARN] Fetch failed, stopping")
            break
        
        # Step 2: Filter the fetched data
        print(f"\n[FILTER] Processing 3 fetched rounds")
        if not run_command("python data/chembl_data_loader.py --n_molecules 50000 --filter-only"):
            print("[ERROR] Filtering failed!")
            break
        
        # Step 3: Load filtered batch and extract molecules
        print(f"\n[EXTRACT] Loading filtered batch")
        cache_dir = output_dir / ".cache"
        fetch_progress = cache_dir / "fetch_progress.pkl"
        
        if not fetch_progress.exists():
            print("[ERROR] No fetch progress found!")
            break
        
        with open(fetch_progress, "rb") as f:
            progress = pickle.load(f)
            last_round = progress['round_num']
        
        # Find the filtered batch file for this round
        filtered_batch = output_dir / f"filtered_batch_{last_round}.parquet"
        if not filtered_batch.exists():
            print(f"[ERROR] Filtered batch not found: {filtered_batch}")
            break
        
        new_molecules = pd.read_parquet(filtered_batch)
        print(f"[EXTRACT] Found {len(new_molecules):,} new molecules from round {last_round}")
        
        # Step 4: Combine with master CSV
        if master_csv.exists():
            existing = pd.read_csv(master_csv)
            print(f"[COMBINE] Existing: {len(existing):,}, New: {len(new_molecules):,}")
            
            combined = pd.concat([existing, new_molecules], ignore_index=True)
            combined = combined.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)
            
            print(f"[COMBINE] After deduplication: {len(combined):,} molecules")
            combined.to_csv(master_csv, index=False)
        else:
            new_molecules.to_csv(master_csv, index=False)
            combined = new_molecules
        
        # Update count
        current_count = len(combined)
        
        # Step 5: Clean up cache to free RAM
        print(f"\n[CLEANUP] Removing cache files to free RAM")
        if fetch_progress.exists():
            fetch_progress.unlink()
        
        # Also save progress checkpoints
        combined.to_parquet(output_dir / "molecules_progress.parquet", index=False)
        
        # Update for next iteration (move forward 3 rounds)
        current_round = last_round + 1
        iteration += 1
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration-1} complete!")
        print(f"Total molecules: {current_count:,}/{target_molecules:,}")
        print(f"{'='*60}\n")
        
        # Small delay before next batch
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"✅ TARGET REACHED!")
    print(f"{'='*60}")
    print(f"Final count: {current_count:,} molecules")
    print(f"Saved to: {master_csv}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
