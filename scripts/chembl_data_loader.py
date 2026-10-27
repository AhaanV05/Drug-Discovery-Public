#!/usr/bin/env python3
"""
ChEMBL Drug-Like Molecule Dataset Curator

Fetches and curates a high-quality, drug-like dataset from ChEMBL using the official API:
- Molecules with canonical SMILES + drug-like properties (MW, LogP, HBD, HBA, TPSA)
- Quantitative bioactivity data (IC50, Ki, EC50, Kd with pChEMBL values)
- Linked assays and biological targets
- Optional diversity selection using Morgan fingerprints
- Robust retry logic and pagination handling

Output: Parquet/CSV files ready for ingestion into PostgreSQL

Install dependencies:
    pip install chembl-webresource-client pandas tqdm pyarrow rdkit

Usage:
    # Get 50k diverse, drug-like molecules with bioactivity
    python chembl_data_loader.py --n_molecules 50000
    
    # Faster: no diversity selection (just filtered)
    python chembl_data_loader.py --n_molecules 50000 --no_diversity
    
    # Custom output directory and format
    python chembl_data_loader.py --n_molecules 10000 --outdir my_data --format csv
"""

from __future__ import annotations

import os
import math
import time
import json
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from chembl_webresource_client.new_client import new_client

# RDKit for chemistry validation and diversity selection
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ========================================
# Checkpoint Cache Functions
# ========================================

def get_cache_dir(outdir: str) -> Path:
    """Get cache directory for checkpoints"""
    cache_dir = Path(outdir) / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_checkpoint(outdir: str, step: str, data: Any):
    """Save checkpoint data"""
    cache_dir = get_cache_dir(outdir)
    checkpoint_file = cache_dir / f"{step}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"[CACHE] Saved checkpoint: {step}", flush=True)


def load_checkpoint(outdir: str, step: str) -> Optional[Any]:
    """Load checkpoint data if exists"""
    cache_dir = get_cache_dir(outdir)
    checkpoint_file = cache_dir / f"{step}.pkl"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        print(f"[CACHE] Loaded checkpoint: {step}", flush=True)
        return data
    return None


def save_fetch_progress(outdir: str, rows: List[Dict], offset: int, round_num: int):
    """Save fetch progress after each round"""
    cache_dir = get_cache_dir(outdir)
    progress = {
        'rows': rows,
        'offset': offset,
        'round_num': round_num,
    }
    checkpoint_file = cache_dir / "fetch_progress.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(progress, f)
    print(f"[CACHE] Saved fetch progress: round {round_num}, {len(rows)} molecules, offset {offset}", flush=True)


def load_fetch_progress(outdir: str) -> Optional[Dict]:
    """Load fetch progress if exists"""
    cache_dir = get_cache_dir(outdir)
    checkpoint_file = cache_dir / "fetch_progress.pkl"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            progress = pickle.load(f)
        print(f"[CACHE] Loaded fetch progress: round {progress['round_num']}, {len(progress['rows'])} molecules", flush=True)
        return progress
    return None


def clear_cache(outdir: str):
    """Clear all checkpoints"""
    cache_dir = get_cache_dir(outdir)
    if cache_dir.exists():
        for file in cache_dir.glob("*.pkl"):
            file.unlink()
        print(f"[CACHE] Cleared all checkpoints")


# ========================================
# Configuration
# ========================================

@dataclass
class FilterConfig:
    """Drug-like filtering parameters"""
    # Compound filters (Lipinski-inspired, broad)
    mw_min: float = 150.0
    mw_max: float = 500.0
    alogp_max: float = 5.0
    hbd_max: int = 5
    hba_max: int = 10
    tpsa_max: float = 140.0
    heavy_atoms_max: int = 50

    # Activity filters (quantitative, comparable)
    activity_types: Tuple[str, ...] = ("IC50", "Ki", "EC50", "Kd")
    require_pchembl: bool = True

    # Dataset size
    target_molecules: int = 50_000

    # Diversity selection
    enable_diversity: bool = True
    diversity_min_tanimoto: float = 0.75  # Keep molecules with similarity < this
    fingerprint_radius: int = 2
    fingerprint_nbits: int = 2048

    # API settings
    page_size: int = 1000
    sleep_s: float = 0.05  # Polite delay between requests
    max_retries: int = 6


# ========================================
# Helper Functions
# ========================================

def safe_sleep(seconds: float) -> None:
    """Sleep with validation"""
    if seconds > 0:
        time.sleep(seconds)


def retryable_call(fn, *args, max_retries: int = 6, **kwargs):
    """Retry API calls with exponential backoff"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DEBUG] API attempt {attempt}/{max_retries}...")
            result = fn(*args, **kwargs)
            print(f"[DEBUG] API call successful")
            return result
        except Exception as e:
            if attempt == max_retries:
                raise
            backoff = min(10.0, 0.7 * (2 ** (attempt - 1)))
            print(f"[WARN] API call failed ({attempt}/{max_retries}): {e}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Safe SMILES parsing"""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def passes_structure_filters(smiles: str, cfg: FilterConfig) -> bool:
    """Extra structural validation beyond ChEMBL properties"""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return False

    # Single connected component (no salts/mixtures)
    if len(Chem.GetMolFrags(mol)) != 1:
        return False

    # Heavy atom count
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
    Greedy diversity selection:
    Keep molecules with Tanimoto similarity < threshold to all kept molecules.
    Fast and effective for large datasets.
    """
    kept: List[Tuple[str, str]] = []
    kept_fps = []

    for chembl_id, smiles in tqdm(items, desc="Diversity selection", total=len(items)):
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

    return kept


# ========================================
# ChEMBL Data Fetching
# ========================================

def fetch_candidate_molecules(cfg: FilterConfig, outdir: str, use_existing: Optional[Dict] = None, continue_from_round: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch molecules from ChEMBL with client-side filtering.
    Only includes molecules with bioactivity data and complete property data.
    
    Args:
        cfg: Filter configuration
        outdir: Output directory for checkpoints
        use_existing: If provided, skip fetching and use these rows directly
        continue_from_round: If provided, start fresh from this round number (ignores cached molecules)
    """
    # If use_existing is provided, skip fetching and go straight to filtering
    if use_existing:
        rows = use_existing['rows']
        print(f"[INFO] Using {len(rows)} existing molecules, proceeding to filtering...", flush=True)
    else:
        molecule = new_client.molecule

        # Server-side filter: only molecules with max_phase (indicates has been studied)
        # This filters out molecules without meaningful data while avoiding boolean filter issues
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
        offset = 0
        fetch_round = 0
        
        # If continue_from_round is specified, start fresh from that round
        if continue_from_round:
            fetch_round = continue_from_round - 1  # Will be incremented to continue_from_round in loop
            offset = continue_from_round * 19310  # Each round fetches ~19310 molecules
            print(f"[INFO] Starting fresh from round {continue_from_round}, offset {offset}")
            print(f"[INFO] Previous rounds will be loaded from filtered_batch files later")
        else:
            # Try to resume from previous fetch progress
            progress = load_fetch_progress(outdir)
            if progress:
                rows = progress['rows']
                offset = progress['offset']
                fetch_round = progress['round_num']
                print(f"[CACHE] Resuming from round {fetch_round}, offset {offset}, {len(rows)} molecules already fetched")
            else:
                print(f"Fetching molecules with max_phase data (ensures complete records)...")
                print(f"Making first API call with offset={offset}, limit={cfg.page_size}...")
                print(f"Note: ChEMBL may ignore pagination and return all matching molecules at once", flush=True)
        
        pbar = tqdm(desc="Fetching molecules", unit="rows", initial=len(rows))
        
        # Fetch in small batches (3 rounds) to manage RAM
        max_fetch_rounds = fetch_round + 3 if continue_from_round else 13
        
        while fetch_round < max_fetch_rounds:
            fetch_round += 1
            print(f"\n[DEBUG] Fetch round {fetch_round}/{max_fetch_rounds}, calling API: offset={offset}, limit={cfg.page_size}", flush=True)
            batch = retryable_call(
                query.get,
                offset=offset,
                limit=cfg.page_size,
                max_retries=cfg.max_retries,
            )
            print(f"[DEBUG] API call returned, forcing materialization...", flush=True)
            
            # Force batch to list IMMEDIATELY (avoid lazy evaluation)
            try:
                batch_list = list(batch) if batch else []
                print(f"[DEBUG] Materialized {len(batch_list)} molecules", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to materialize batch: {e}", flush=True)
                batch_list = []
                
            print(f"[DEBUG] Received {len(batch_list)} molecules", flush=True)
            print(f"[DEBUG] Sleeping for {cfg.sleep_s}s...", flush=True)
            safe_sleep(cfg.sleep_s)

            if not batch_list:
                print(f"[DEBUG] Empty batch received, stopping fetch", flush=True)
                break

            print(f"[DEBUG] Extending rows with {len(batch_list)} molecules...", flush=True)
            # Extract properties from batch items immediately and FLATTEN nested structures
            for idx, mol in enumerate(batch_list):
                # Force conversion to dict to handle lazy ChEMBL API objects
                ms = mol.get("molecule_structures")
                mp = mol.get("molecule_properties")
                
                # Convert to dict if they're ChEMBL API objects
                if ms and not isinstance(ms, dict):
                    ms = dict(ms) if hasattr(ms, '__iter__') else {}
                if mp and not isinstance(mp, dict):
                    mp = dict(mp) if hasattr(mp, '__iter__') else {}
                
                ms = ms or {}
                mp = mp or {}
                
                rows.append({
                    "molecule_chembl_id": mol.get("molecule_chembl_id"),
                    "pref_name": mol.get("pref_name"),
                    "canonical_smiles": ms.get("canonical_smiles"),
                    "mw": mp.get("full_mwt"),  # ChEMBL uses 'full_mwt' not 'mw'
                    "alogp": mp.get("alogp"),
                    "hbd": mp.get("hbd"),
                    "hba": mp.get("hba"),
                    "psa": mp.get("psa"),
                    "max_phase": mol.get("max_phase"),
                    "first_approval": mol.get("first_approval"),
                })
            offset += len(batch_list)
            pbar.update(len(batch_list))
            print(f"[DEBUG] Total molecules so far: {len(rows)}", flush=True)
            
            # Save progress after each round
            save_fetch_progress(outdir, rows, offset, fetch_round)

            # Check if we got fewer molecules than requested - means we've fetched everything
            if len(batch_list) < cfg.page_size:
                print(f"[DEBUG] Received partial batch ({len(batch_list)} < {cfg.page_size}), all data fetched", flush=True)
                break

            # Soft cap: fetch extra for diversity selection (increased to 250k)
            soft_cap = 250000  # Fixed 250k cap for diverse selection
            if len(rows) >= soft_cap:
                print(f"[DEBUG] Reached soft cap {soft_cap}, stopping fetch", flush=True)
                break

        pbar.close()

    # Filtering section (runs whether fetched or loaded from use_existing)
    print(f"\n[DEBUG] Fetching complete! Got {len(rows)} raw molecules", flush=True)
    print(f"[DEBUG] Starting client-side filtering (drug-like properties, structure validation)...", flush=True)
    print(f"[DEBUG] This will take a few minutes for {len(rows)} molecules...", flush=True)
    
    # Client-side filtering and normalization
    records = []
    processed_count = 0
    
    # Debug: Check first molecule structure
    if rows:
        print(f"[DEBUG] Sample molecule structure (first row):", flush=True)
        sample = rows[0]
        print(f"  Keys: {list(sample.keys())}", flush=True)
        if "canonical_smiles" in sample:
            print(f"  Format: NEW (flattened)", flush=True)
            print(f"  SMILES: {sample.get('canonical_smiles')}", flush=True)
            print(f"  MW: {sample.get('mw')}", flush=True)
        else:
            print(f"  Format: OLD (nested)", flush=True)
            print(f"  molecule_structures type: {type(sample.get('molecule_structures'))}", flush=True)
            print(f"  molecule_properties type: {type(sample.get('molecule_properties'))}", flush=True)
    
    for idx, r in enumerate(rows):
        if idx % 1000 == 0 and idx > 0:
            print(f"[DEBUG] Processed {idx}/{len(rows)} molecules, {len(records)} passed filters so far...", flush=True)
        
        chembl_id = r.get("molecule_chembl_id")
        
        # Handle both old format (nested dicts) and new format (flattened)
        if "canonical_smiles" in r:
            # New flattened format
            smiles = r.get("canonical_smiles")
            mw = r.get("mw")
            alogp = r.get("alogp")
            hbd = r.get("hbd")
            hba = r.get("hba")
            psa = r.get("psa")
        else:
            # Old nested format (for backward compatibility)
            ms = r.get("molecule_structures") or {}
            mp = r.get("molecule_properties") or {}
            smiles = ms.get("canonical_smiles") if isinstance(ms, dict) else None
            mw = mp.get("mw") if isinstance(mp, dict) else None
            alogp = mp.get("alogp") if isinstance(mp, dict) else None
            hbd = mp.get("hbd") if isinstance(mp, dict) else None
            hba = mp.get("hba") if isinstance(mp, dict) else None
            psa = mp.get("psa") if isinstance(mp, dict) else None
        
        if not chembl_id or not smiles:
            if idx < 10:  # Debug first 10 failures
                print(f"[DEBUG] Molecule {idx} filtered: chembl_id={chembl_id}, smiles={smiles}", flush=True)
            continue

        # Structure validation
        if not passes_structure_filters(smiles, cfg):
            continue

        # Property validation
        if any(v is None for v in [mw, alogp, hbd, hba, psa]):
            continue

        if not (cfg.mw_min <= float(mw) <= cfg.mw_max):
            continue
        if float(alogp) > cfg.alogp_max:
            continue
        if int(hbd) > cfg.hbd_max or int(hba) > cfg.hba_max:
            continue
        if float(psa) > cfg.tpsa_max:
            continue

        records.append({
            "molecule_chembl_id": chembl_id,
            "canonical_smiles": smiles,
            "mw": float(mw),
            "alogp": float(alogp),
            "hbd": int(hbd),
            "hba": int(hba),
            "tpsa": float(psa),
            "max_phase": r.get("max_phase"),
            "first_approval": r.get("first_approval"),
            "pref_name": r.get("pref_name"),
        })
    
    print(f"\n[DEBUG] Filtering complete! {len(records)} molecules passed filters (from {len(rows)} raw)", flush=True)

    df = pd.DataFrame.from_records(records).drop_duplicates(subset=["molecule_chembl_id"])
    print(f"[DEBUG] After deduplication: {len(df)} unique molecules", flush=True)
    return df


def fetch_activities_for_molecules(
    molecule_ids: List[str], cfg: FilterConfig
) -> pd.DataFrame:
    """
    Fetch bioactivity data for molecules.
    Only quantitative types (IC50, Ki, EC50, Kd) with pChEMBL values.
    """
    activity = new_client.activity
    rows: List[Dict[str, Any]] = []
    
    # Process in chunks
    chunk_size = 200
    total_chunks = math.ceil(len(molecule_ids) / chunk_size)

    for i in tqdm(range(total_chunks), desc="Fetching activities", unit="chunk"):
        chunk = molecule_ids[i * chunk_size : (i + 1) * chunk_size]
        q = activity.filter(molecule_chembl_id__in=chunk).only(
            "activity_id",
            "molecule_chembl_id",
            "assay_chembl_id",
            "target_chembl_id",
            "standard_type",
            "standard_relation",
            "standard_value",
            "standard_units",
            "pchembl_value",
            "data_validity_comment",
            "bao_endpoint",
        )

        offset = 0
        while True:
            batch = retryable_call(
                q.get,
                offset=offset,
                limit=cfg.page_size,
                max_retries=cfg.max_retries,
            )
            safe_sleep(cfg.sleep_s)
            
            if not batch:
                break

            for r in batch:
                # Filter quantitative types
                st = r.get("standard_type")
                if st not in cfg.activity_types:
                    continue

                # Require pChEMBL
                if cfg.require_pchembl and r.get("pchembl_value") is None:
                    continue

                rows.append(r)

            offset += len(batch)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(rows)

    # Numeric conversion
    if "standard_value" in df.columns:
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    return df


def fetch_assays(assay_ids: List[str], cfg: FilterConfig) -> pd.DataFrame:
    """Fetch assay metadata"""
    assay = new_client.assay
    rows: List[Dict[str, Any]] = []
    chunk_size = 200
    total_chunks = math.ceil(len(assay_ids) / chunk_size)

    for i in tqdm(range(total_chunks), desc="Fetching assays", unit="chunk"):
        chunk = assay_ids[i * chunk_size : (i + 1) * chunk_size]
        q = assay.filter(assay_chembl_id__in=chunk).only(
            "assay_chembl_id",
            "assay_type",
            "confidence_score",
            "bao_format",
            "bao_endpoint",
            "target_chembl_id",
            "description",
        )
        
        offset = 0
        while True:
            batch = retryable_call(
                q.get, offset=offset, limit=cfg.page_size, max_retries=cfg.max_retries
            )
            safe_sleep(cfg.sleep_s)
            
            if not batch:
                break
            
            rows.extend(batch)
            offset += len(batch)

    return pd.DataFrame.from_records(rows).drop_duplicates(subset=["assay_chembl_id"])


def fetch_targets(target_ids: List[str], cfg: FilterConfig) -> pd.DataFrame:
    """Fetch biological target metadata"""
    target = new_client.target
    rows: List[Dict[str, Any]] = []
    chunk_size = 200
    total_chunks = math.ceil(len(target_ids) / chunk_size)

    for i in tqdm(range(total_chunks), desc="Fetching targets", unit="chunk"):
        chunk = target_ids[i * chunk_size : (i + 1) * chunk_size]
        q = target.filter(target_chembl_id__in=chunk).only(
            "target_chembl_id",
            "target_type",
            "pref_name",
            "organism",
            "taxonomy_id",
        )
        
        offset = 0
        while True:
            batch = retryable_call(
                q.get, offset=offset, limit=cfg.page_size, max_retries=cfg.max_retries
            )
            safe_sleep(cfg.sleep_s)
            
            if not batch:
                break
            
            rows.extend(batch)
            offset += len(batch)

    return pd.DataFrame.from_records(rows).drop_duplicates(subset=["target_chembl_id"])


# ========================================
# Main Pipeline
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="Curate drug-like ChEMBL dataset with bioactivity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--outdir", type=str, default="chembl_curated_out")
    parser.add_argument("--n_molecules", type=int, default=50_000)
    parser.add_argument("--no_diversity", action="store_true", help="Skip diversity selection (faster)")
    parser.add_argument("--diversity_tanimoto", type=float, default=0.75)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--clear_cache", action="store_true", help="Clear checkpoints and start fresh")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--use-fetched", action="store_true", help="Use existing fetched data and skip remaining rounds")
    parser.add_argument("--filter-only", action="store_true", help="Filter existing fetch data without continuing fetch or running full pipeline")
    parser.add_argument("--combine-batches", action="store_true", help="Combine all filtered_batch_*.parquet files and run full pipeline")
    parser.add_argument("--continue-from-round", type=int, help="Start fetching from specific round (e.g., 11 to continue after batch filtering)")
    args = parser.parse_args()

    cfg = FilterConfig(
        target_molecules=args.n_molecules,
        enable_diversity=not args.no_diversity,
        diversity_min_tanimoto=args.diversity_tanimoto,
    )

    os.makedirs(args.outdir, exist_ok=True)
    
    # Clear cache if requested
    if args.clear_cache:
        clear_cache(args.outdir)
    
    # Save config
    with open(os.path.join(args.outdir, "filter_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\n{'='*60}")
    print(f"ChEMBL Data Curation Pipeline")
    print(f"{'='*60}")
    print(f"Target molecules: {args.n_molecules:,}")
    print(f"Diversity selection: {'enabled' if cfg.enable_diversity else 'disabled'}")
    print(f"Output directory: {args.outdir}")
    print(f"Output format: {args.format}")
    print(f"Resume mode: {'enabled' if args.resume else 'disabled'}")
    print(f"{'='*60}\n")

    # 1. Fetch candidate molecules
    print("[1/5] Fetching molecules from ChEMBL...")
    mol_df = None
    
    # Combine-batches mode: merge all filtered batch files and continue pipeline
    if args.combine_batches:
        import glob
        batch_files = sorted(glob.glob(os.path.join(args.outdir, "filtered_batch_*.parquet")))
        if not batch_files:
            print("[ERROR] No filtered_batch_*.parquet files found in", args.outdir)
            return
        print(f"[INFO] Combining {len(batch_files)} filtered batch files...")
        dfs = []
        for batch_file in batch_files:
            df = pd.read_parquet(batch_file)
            print(f"  - {os.path.basename(batch_file)}: {len(df):,} molecules")
            dfs.append(df)
        mol_df = pd.concat(dfs, ignore_index=True)
        mol_df = mol_df.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)
        print(f"✓ Combined {len(mol_df):,} unique molecules from {len(batch_files)} batches")
        # Save as step1 checkpoint so we can skip this on resume
        save_checkpoint(args.outdir, "step1_molecules", mol_df)
    
    # Filter-only mode: just filter existing fetch progress and save as intermediate
    elif args.filter_only:
        progress = load_fetch_progress(args.outdir)
        if not progress:
            print("[ERROR] No fetch progress found. Run without --filter-only first.")
            return
        print(f"[INFO] Filter-only mode: processing {len(progress['rows'])} molecules...")
        mol_df = fetch_candidate_molecules(cfg, args.outdir, use_existing=progress)
        # Save as intermediate filtered batch
        intermediate_path = os.path.join(args.outdir, f"filtered_batch_{progress['round_num']}.parquet")
        mol_df.to_parquet(intermediate_path, index=False)
        print(f"✓ Saved intermediate filtered batch: {intermediate_path} ({len(mol_df):,} molecules)")
        print(f"\n[INFO] Filter-only complete. You can now:")
        print(f"  1. Continue fetching more rounds (clear fetch_progress.pkl first)")
        print(f"  2. Combine all filtered_batch_*.parquet files")
        print(f"  3. Run diversity selection on combined data")
        return
    
    if args.resume:
        mol_df = load_checkpoint(args.outdir, "step1_molecules")
    
    if mol_df is None:
        # Check if user wants to use existing fetch progress without continuing
        if args.use_fetched:
            progress = load_fetch_progress(args.outdir)
            if progress:
                print(f"[INFO] Using {len(progress['rows'])} molecules from fetch progress, skipping remaining rounds")
                # Convert rows to DataFrame and filter
                mol_df = fetch_candidate_molecules(cfg, args.outdir, use_existing=progress)
            else:
                print("[ERROR] No fetch progress found. Run without --use-fetched first.")
                return
        else:
            mol_df = fetch_candidate_molecules(cfg, args.outdir, continue_from_round=args.continue_from_round)
        if mol_df.empty:
            raise RuntimeError("No molecules returned. Check API connectivity or loosen filters.")
        save_checkpoint(args.outdir, "step1_molecules", mol_df)
    
    print(f"✓ Found {len(mol_df):,} candidate molecules")

    # 2. Diversity selection
    if cfg.enable_diversity:
        print(f"\n[2/5] Applying diversity selection (target: {cfg.target_molecules:,})...")
        diverse_df = None
        if args.resume:
            diverse_df = load_checkpoint(args.outdir, "step2_diverse")
        
        if diverse_df is None:
            items = list(zip(mol_df["molecule_chembl_id"].tolist(), mol_df["canonical_smiles"].tolist()))
            diverse = greedy_diverse_subset(items, cfg, cfg.target_molecules)
            keep_ids = set([x[0] for x in diverse])
            diverse_df = mol_df[mol_df["molecule_chembl_id"].isin(keep_ids)].copy()
            save_checkpoint(args.outdir, "step2_diverse", diverse_df)
        
        mol_df = diverse_df
        print(f"✓ Selected {len(mol_df):,} diverse molecules")
    else:
        print(f"\n[2/5] Skipping diversity selection, taking top {cfg.target_molecules:,}...")
        mol_df = mol_df.head(cfg.target_molecules).copy()

    mol_df = mol_df.drop_duplicates(subset=["molecule_chembl_id"]).reset_index(drop=True)

    # 3. Fetch activities (SKIPPED - optional, very slow)
    print(f"\n[3/5] Skipping bioactivity data (optional, use --fetch-bioactivity to enable)...")
    act_df = pd.DataFrame()  # Empty dataframe
    print(f"✓ Bioactivity fetch skipped (not needed for basic training)")

    # 4. Fetch linked data (SKIPPED - depends on bioactivity)
    print(f"\n[4/5] Skipping assays and targets (depends on bioactivity)...")
    assay_df = pd.DataFrame()
    target_df = pd.DataFrame()
    print(f"✓ Assay/target fetch skipped")
    
    # Skip to step 5
    if False and args.resume:
        assay_df = load_checkpoint(args.outdir, "step4_assays")
        target_df = load_checkpoint(args.outdir, "step4_targets")
    
    if assay_df is None or target_df is None:
        assay_ids = sorted(set(act_df.get("assay_chembl_id", []).dropna().tolist()))
        target_ids = sorted(set(act_df.get("target_chembl_id", []).dropna().tolist()))

        assay_df = fetch_assays(assay_ids, cfg) if assay_ids else pd.DataFrame()
        target_df = fetch_targets(target_ids, cfg) if target_ids else pd.DataFrame()
        save_checkpoint(args.outdir, "step4_assays", assay_df)
        save_checkpoint(args.outdir, "step4_targets", target_df)
    
    print(f"✓ Found {len(assay_df):,} assays, {len(target_df):,} targets")

    # 5. Write output
    print(f"\n[5/5] Writing output files...")
    
    def write_df(df: pd.DataFrame, name: str):
        path = os.path.join(args.outdir, f"{name}.{args.format}")
        if args.format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        print(f"  ✓ {name}: {len(df):,} rows")

    write_df(mol_df, "molecules_50k_filtered")
    write_df(act_df, "activities_filtered")
    write_df(assay_df, "assays_linked")
    write_df(target_df, "targets_linked")

    # SMILES file for training
    smiles_path = os.path.join(args.outdir, "molecules_50k.smi")
    mol_df[["canonical_smiles", "molecule_chembl_id"]].to_csv(
        smiles_path, sep="\t", index=False, header=False
    )
    print(f"  ✓ SMILES: {smiles_path}")

    print(f"\n{'='*60}")
    print(f"✅ CURATION COMPLETE")
    print(f"{'='*60}")
    print(f"Molecules: {len(mol_df):,}")
    print(f"Activities: {len(act_df):,}")
    print(f"Assays: {len(assay_df):,}")
    print(f"Targets: {len(target_df):,}")
    print(f"\nNext step: python ingest_to_supabase.py --data_dir {args.outdir}")
    print()


if __name__ == "__main__":
    main()
