import pandas as pd
import numpy as np

# Load both files
original = pd.read_csv('chembl_curated_out/molecules_diverse_enriched.csv')
admet = pd.read_csv('results.csv')

# Merge on molecule_chembl_id
merged = original.merge(admet[['molecule_chembl_id', 'QED', 'Synth']], on='molecule_chembl_id', how='inner')

print(f'Merged {len(merged):,} molecules')
print()

# Compare QED
print('=== QED COMPARISON ===')
print(f'Our QED mean: {merged["qed"].mean():.6f}')
print(f'ADMET QED mean: {merged["QED"].mean():.6f}')
print(f'Difference: {abs(merged["qed"].mean() - merged["QED"].mean()):.6f}')
print()

# Check correlation
qed_corr = merged[['qed', 'QED']].corr().iloc[0, 1]
print(f'QED Correlation: {qed_corr:.6f}')
print()

# Sample comparison
print('First 10 molecules QED comparison:')
print(merged[['molecule_chembl_id', 'qed', 'QED']].head(10))
print()

# Compare SA Score
print('=== SA SCORE COMPARISON ===')
print(f'Our SA mean: {merged["sa_score"].mean():.6f}')
print(f'ADMET Synth mean: {merged["Synth"].mean():.6f}')
print(f'Difference: {abs(merged["sa_score"].mean() - merged["Synth"].mean()):.6f}')
print()

# Check correlation
sa_corr = merged[['sa_score', 'Synth']].corr().iloc[0, 1]
print(f'SA Score Correlation: {sa_corr:.6f}')
print()

# Sample comparison
print('First 10 molecules SA comparison:')
print(merged[['molecule_chembl_id', 'sa_score', 'Synth']].head(10))
print()

# Check if they're identical
qed_identical = np.allclose(merged['qed'], merged['QED'], rtol=1e-5)
sa_identical = np.allclose(merged['sa_score'], merged['Synth'], rtol=1e-5)

print('=== VERDICT ===')
print(f'QED values identical (within 0.001%): {qed_identical}')
print(f'SA values identical (within 0.001%): {sa_identical}')
