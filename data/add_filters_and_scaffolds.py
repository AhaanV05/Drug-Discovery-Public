"""
Add Lipinski/PAINS/Brenk filter flags and Bemis-Murcko scaffolds to molecules dataset.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog
from tqdm import tqdm

def compute_lipinski_violations(mol):
    """Compute number of Lipinski Rule of 5 violations."""
    violations = 0
    
    # MW <= 500
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    
    # LogP <= 5
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    
    # HBD <= 5
    if Lipinski.NumHDonors(mol) > 5:
        violations += 1
    
    # HBA <= 10
    if Lipinski.NumHAcceptors(mol) > 10:
        violations += 1
    
    return violations

def check_pains(mol, params):
    """Check for PAINS (Pan-Assay Interference Structures)."""
    if mol is None:
        return True
    return params.HasMatch(mol)

def check_brenk(mol, params):
    """Check for Brenk unwanted/toxic substructures."""
    if mol is None:
        return True
    return params.HasMatch(mol)

def get_bemis_murcko_scaffold(mol):
    """Get Bemis-Murcko scaffold (core molecular framework)."""
    try:
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def main():
    # Load existing dataset
    print("Loading molecules_diverse_enriched.csv...")
    df = pd.read_csv('chembl_curated_out/molecules_diverse_enriched.csv')
    print(f"Loaded {len(df)} molecules")
    
    # Initialize FilterCatalog for PAINS and Brenk
    print("\nInitializing filter catalogs...")
    params_pains = FilterCatalog.FilterCatalogParams()
    params_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog_pains = FilterCatalog.FilterCatalog(params_pains)
    
    params_brenk = FilterCatalog.FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog_brenk = FilterCatalog.FilterCatalog(params_brenk)
    
    # Initialize new columns
    lipinski_violations = []
    passes_lipinski = []
    pains_alert = []
    brenk_alert = []
    passes_filters = []
    scaffolds = []
    
    print("\nComputing filters and scaffolds...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        smiles = row['canonical_smiles']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # Invalid molecule
            lipinski_violations.append(None)
            passes_lipinski.append(False)
            pains_alert.append(True)
            brenk_alert.append(True)
            passes_filters.append(False)
            scaffolds.append(None)
            continue
        
        # Lipinski violations
        violations = compute_lipinski_violations(mol)
        lipinski_violations.append(violations)
        passes_lipinski.append(violations == 0)
        
        # PAINS check
        has_pains = check_pains(mol, catalog_pains)
        pains_alert.append(has_pains)
        
        # Brenk check
        has_brenk = check_brenk(mol, catalog_brenk)
        brenk_alert.append(has_brenk)
        
        # Passes all filters
        passes_all = (violations == 0) and (not has_pains) and (not has_brenk)
        passes_filters.append(passes_all)
        
        # Bemis-Murcko scaffold
        scaffold = get_bemis_murcko_scaffold(mol)
        scaffolds.append(scaffold)
    
    # Add new columns to dataframe
    print("\nAdding new columns to dataframe...")
    df['lipinski_violations'] = lipinski_violations
    df['passes_lipinski'] = passes_lipinski
    df['pains_alert'] = pains_alert
    df['brenk_alert'] = brenk_alert
    df['passes_filters'] = passes_filters
    df['bemis_murcko_scaffold'] = scaffolds
    
    # Save updated dataset
    output_path = 'chembl_curated_out/molecules_diverse_enriched.csv'
    print(f"\nSaving updated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("\n" + "="*60)
    print("FILTER STATISTICS")
    print("="*60)
    
    print(f"\nLipinski's Rule of 5:")
    print(f"  0 violations: {(df['lipinski_violations'] == 0).sum():,} ({(df['lipinski_violations'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  1 violation:  {(df['lipinski_violations'] == 1).sum():,} ({(df['lipinski_violations'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"  2+ violations: {(df['lipinski_violations'] >= 2).sum():,} ({(df['lipinski_violations'] >= 2).sum()/len(df)*100:.1f}%)")
    
    print(f"\nPAINS alerts: {df['pains_alert'].sum():,} ({df['pains_alert'].sum()/len(df)*100:.1f}%)")
    print(f"Brenk alerts: {df['brenk_alert'].sum():,} ({df['brenk_alert'].sum()/len(df)*100:.1f}%)")
    
    print(f"\nPass ALL filters: {df['passes_filters'].sum():,} ({df['passes_filters'].sum()/len(df)*100:.1f}%)")
    
    # Scaffold diversity
    unique_scaffolds = df['bemis_murcko_scaffold'].nunique()
    print(f"\nScaffold diversity:")
    print(f"  Unique scaffolds: {unique_scaffolds:,}")
    print(f"  Diversity: {unique_scaffolds/len(df)*100:.1f}%")
    
    print(f"\nDataset now has {len(df.columns)} columns")
    print(f"Saved to: {output_path}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
