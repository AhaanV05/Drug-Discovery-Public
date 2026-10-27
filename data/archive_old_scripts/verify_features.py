import pandas as pd

df = pd.read_parquet('chembl_curated_out/molecules_diverse_enriched.parquet')
df.to_csv('chembl_curated_out/molecules_diverse_enriched.csv', index=False)

print(f"✅ CSV updated: {len(df):,} molecules, {len(df.columns)} columns")
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nFirst row all descriptors:")
print(df.iloc[0])
print(f"\nDescriptor statistics:")
print(df[['num_rings', 'num_aromatic_rings', 'num_rotatable_bonds', 'formal_charge', 'num_heteroatoms', 'fraction_csp3']].describe())
