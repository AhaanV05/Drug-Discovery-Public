import pandas as pd

df = pd.read_csv('chembl_curated_out/molecules_filtered_for_rl.csv')

optional_features = [
    'DILI', 'Carcinogenicity', 'Genotoxicity',
    'BBB', 'caco2', 'hia', 'PPB',
    'Respiratory', 'Nephrotoxicity-DI', 'Hematotoxicity'
]

print("="*70)
print("CHECKING 'OPTIONAL' FEATURES IN molecules_filtered_for_rl.csv")
print("="*70)

print("\nFeature availability:")
present = 0
for feat in optional_features:
    if feat in df.columns:
        print(f"  ✅ {feat:25s} - PRESENT")
        present += 1
    else:
        print(f"  ❌ {feat:25s} - MISSING")

print("\n" + "="*70)
print(f"RESULT: {present}/{len(optional_features)} features are ALREADY INCLUDED!")
print("="*70)

if present == len(optional_features):
    print("\n🎉 ALL 'OPTIONAL' FEATURES ARE ACTUALLY INCLUDED!")
    print("   These are NOT optional - they're part of the 84 features.")
    print("\n   UPDATE DOCUMENTATION:")
    print("   • Remove these from 'OPTIONAL' section")
    print("   • They're already in CRITICAL/IMPORTANT categories")
