"""
Test version of auto_fetch_molecules.py - runs only ONE iteration (3 rounds)
to verify the pipeline works before committing to the full overnight run.
"""
import subprocess
import pickle
from pathlib import Path
import pandas as pd

def run_command(cmd):
    """Run a shell command and return success status"""
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def main():
    output_dir = Path("chembl_curated_out")
    master_csv = output_dir / "molecules_diverse.csv"
    
    # Get current state
    if master_csv.exists():
        existing = pd.read_csv(master_csv)
        current_count = len(existing)
        print(f"\n[START] Current molecules: {current_count:,}")
        
        # Figure out next round to fetch
        cache_dir = output_dir / ".cache"
        fetch_progress = cache_dir / "fetch_progress.pkl"
        if fetch_progress.exists():
            with open(fetch_progress, "rb") as f:
                progress = pickle.load(f)
                current_round = progress['round_num'] + 1
        else:
            # Look at existing filtered batches
            batches = list(output_dir.glob("filtered_batch_*.parquet"))
            if batches:
                rounds = [int(b.stem.split("_")[-1]) for b in batches]
                current_round = max(rounds) + 1
            else:
                current_round = 10  # Start from round 10 (we have 0-9)
    else:
        current_count = 0
        current_round = 0
        print(f"\n[START] No existing data, starting fresh")
    
    print(f"[TEST] Will fetch rounds {current_round}-{current_round+2} (ONE iteration only)")
    print(f"[TEST] This will take ~45-60 minutes")
    print(f"[TEST] Press Ctrl+C to cancel\n")
    
    # Step 1: Fetch 3 rounds
    print(f"[FETCH] Starting fetch from round {current_round}")
    if not run_command(f"python data/chembl_data_loader.py --n_molecules 50000 --continue-from-round {current_round}"):
        print("[ERROR] Fetch failed!")
        return
    
    # Step 2: Filter the fetched data
    print(f"\n[FILTER] Processing 3 fetched rounds")
    if not run_command("python data/chembl_data_loader.py --n_molecules 50000 --filter-only"):
        print("[ERROR] Filtering failed!")
        return
    
    # Step 3: Load filtered batch and extract molecules
    print(f"\n[EXTRACT] Loading filtered batch")
    cache_dir = output_dir / ".cache"
    fetch_progress = cache_dir / "fetch_progress.pkl"
    
    if not fetch_progress.exists():
        print("[ERROR] No fetch progress found!")
        return
    
    with open(fetch_progress, "rb") as f:
        progress = pickle.load(f)
        last_round = progress['round_num']
    
    # Find the filtered batch file for this round
    filtered_batch = output_dir / f"filtered_batch_{last_round}.parquet"
    if not filtered_batch.exists():
        print(f"[ERROR] Filtered batch not found: {filtered_batch}")
        return
    
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
    
    final_count = len(combined)
    
    # Step 5: Clean up cache to free RAM
    print(f"\n[CLEANUP] Removing cache files to free RAM")
    if fetch_progress.exists():
        fetch_progress.unlink()
    
    # Save progress checkpoint
    combined.to_parquet(output_dir / "molecules_progress.parquet", index=False)
    
    print(f"\n[SUCCESS] Test iteration complete!")
    print(f"[RESULT] Before: {current_count:,} → After: {final_count:,} (+{final_count - current_count:,})")
    print(f"\n[NEXT] If this worked, run: python data/auto_fetch_molecules.py")

if __name__ == "__main__":
    main()
