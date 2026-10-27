#!/usr/bin/env python3
"""
Compute additional molecular features for RL reward functions.

Adds to existing ChEMBL data:
- QED (Quantitative Estimate of Drug-likeness)
- SA Score (Synthetic Accessibility)
- Additional RDKit descriptors

Run AFTER chembl_data_loader.py to enrich the dataset.

Install:
    pip install rdkit

Usage:
    python compute_features.py --data_dir chembl_curated_out
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer


def compute_qed(smiles: str) -> float:
    """Calculate QED (drug-likeness)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return QED.qed(mol)
    except:
        pass
    return None


def compute_sa(smiles: str) -> float:
    """Calculate Synthetic Accessibility (1-10, lower is better)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return sascorer.calculateScore(mol)
    except:
        pass
    return None


def compute_additional_descriptors(smiles: str) -> dict:
    """Compute additional RDKit descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return {
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'formal_charge': Chem.GetFormalCharge(mol),
                'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
                'fraction_csp3': Descriptors.FractionCSP3(mol),
            }
    except Exception as e:
        print(f"Error computing descriptors for {smiles}: {e}")
    return {}


def enrich_dataset(data_dir: str, format: str = 'parquet'):
    """Add QED, SA, and extra descriptors to molecules"""
    
    # Load molecules
    mol_file = os.path.join(data_dir, f'molecules_diverse.{format}')
    if not os.path.exists(mol_file):
        print(f"[ERROR] File not found: {mol_file}")
        return
    
    print(f"Loading molecules from {mol_file}...")
    if format == 'parquet':
        df = pd.read_parquet(mol_file)
    else:
        df = pd.read_csv(mol_file)
    
    print(f"Computing features for {len(df):,} molecules...")
    
    # Compute features
    qed_scores = []
    sa_scores = []
    extra_features = []
    
    for smiles in tqdm(df['canonical_smiles'], desc="Computing features"):
        qed_scores.append(compute_qed(smiles))
        sa_scores.append(compute_sa(smiles))
        extra_features.append(compute_additional_descriptors(smiles))
    
    # Add to dataframe
    df['qed'] = qed_scores
    df['sa_score'] = sa_scores
    
    # Add extra descriptors
    for key in ['num_rings', 'num_aromatic_rings', 'num_rotatable_bonds', 
                'formal_charge', 'num_heteroatoms', 'fraction_csp3']:
        df[key] = [feat.get(key) for feat in extra_features]
    
    # Save enriched dataset
    output_file = os.path.join(data_dir, f'molecules_diverse_enriched.{format}')
    if format == 'parquet':
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    
    print(f"\n✅ Enriched dataset saved to: {output_file}")
    
    # Print statistics
    print(f"\n📊 Feature Statistics:")
    print(f"  QED (drug-likeness):")
    print(f"    Mean: {df['qed'].mean():.3f}")
    print(f"    Range: {df['qed'].min():.3f} - {df['qed'].max():.3f}")
    print(f"  SA Score (synthetic accessibility, lower is better):")
    print(f"    Mean: {df['sa_score'].mean():.3f}")
    print(f"    Range: {df['sa_score'].min():.3f} - {df['sa_score'].max():.3f}")
    
    # High drug-likeness candidates
    high_qed = df[df['qed'] > 0.7]
    print(f"\n🎯 High drug-likeness (QED > 0.7): {len(high_qed):,} molecules ({len(high_qed)/len(df)*100:.1f}%)")
    
    # Easy to synthesize
    easy_synth = df[df['sa_score'] < 3.0]
    print(f"⚗️  Easy to synthesize (SA < 3.0): {len(easy_synth):,} molecules ({len(easy_synth)/len(df)*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='chembl_curated_out')
    parser.add_argument('--format', choices=['parquet', 'csv'], default='parquet')
    args = parser.parse_args()
    
    enrich_dataset(args.data_dir, args.format)
