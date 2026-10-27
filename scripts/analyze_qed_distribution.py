"""
Analyze QED Distribution in Training Data

This script checks how many molecules in the training dataset
have QED values at different thresholds (0.5, 0.7, 0.85, etc.)
to understand if the model has enough high-QED examples to learn from.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path — resolves to <repo>/notebooks/data by default.
# Override with the DATA_PATH environment variable if your data lives elsewhere.
DATA_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "data"
DATA_PATH = os.environ.get("DATA_PATH", str(DATA_DIR / "molecules_200k_selfies_clean.csv"))

print("="*70)
print("QED DISTRIBUTION ANALYSIS")
print("="*70)

# Load data
print(f"\nLoading data from:\n{DATA_PATH}\n")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} molecules")
except FileNotFoundError:
    print(f"❌ File not found: {DATA_PATH}")
    print("\nPlease update DATA_PATH in the script to point to your actual data file.")
    exit(1)

# Check if 'qed' column exists
if 'qed' not in df.columns:
    print(f"\n❌ 'qed' column not found in dataset!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Get QED statistics
qed = df['qed'].dropna()
print(f"\nTotal molecules with QED: {len(qed):,}")
print(f"Missing QED values: {df['qed'].isna().sum():,}")

print("\n" + "="*70)
print("QED STATISTICS")
print("="*70)
print(f"Mean:   {qed.mean():.4f}")
print(f"Median: {qed.median():.4f}")
print(f"Std:    {qed.std():.4f}")
print(f"Min:    {qed.min():.4f}")
print(f"Max:    {qed.max():.4f}")

# Analyze distribution at key thresholds
print("\n" + "="*70)
print("QED DISTRIBUTION AT KEY THRESHOLDS")
print("="*70)

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
print(f"{'Threshold':<12} {'Count':>10} {'Percentage':>12} {'Notes':<30}")
print("-"*70)

for threshold in thresholds:
    count = (qed >= threshold).sum()
    percentage = (count / len(qed)) * 100
    
    # Add notes for critical thresholds
    note = ""
    if threshold == 0.5:
        note = "← Your target (mode collapse)"
    elif threshold == 0.85:
        note = "← Your target (control fails)"
    elif percentage < 1:
        note = "⚠️ VERY FEW EXAMPLES"
    elif percentage < 5:
        note = "⚠️ LOW REPRESENTATION"
    
    print(f"QED >= {threshold:<5} {count:>10,} {percentage:>11.2f}%  {note}")

# Quartile analysis
print("\n" + "="*70)
print("QUARTILE BREAKDOWN")
print("="*70)
quartiles = qed.quantile([0.25, 0.50, 0.75])
print(f"Q1 (25th percentile): {quartiles[0.25]:.4f}")
print(f"Q2 (50th percentile): {quartiles[0.50]:.4f}")
print(f"Q3 (75th percentile): {quartiles[0.75]:.4f}")

# Specific range counts for your targets
print("\n" + "="*70)
print("SPECIFIC RANGES (Your CFG Targets)")
print("="*70)
ranges = [
    (0.0, 0.2, "Very Low"),
    (0.2, 0.3, "Low (your target 0.2)"),
    (0.3, 0.5, "Low-Medium"),
    (0.5, 0.6, "Medium (your target 0.5)"),
    (0.6, 0.7, "Medium-High"),
    (0.7, 0.85, "High"),
    (0.85, 1.0, "Very High (your target 0.85)"),
]

for low, high, label in ranges:
    count = ((qed >= low) & (qed < high)).sum()
    percentage = (count / len(qed)) * 100
    print(f"{low:.2f} ≤ QED < {high:.2f}  |  {count:>7,} molecules ({percentage:>5.2f}%)  |  {label}")

# Visualization (optional - commented out by default)
# Uncomment to generate histogram
"""
print("\n" + "="*70)
print("Generating histogram...")
plt.figure(figsize=(12, 6))
plt.hist(qed, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Target: 0.5 (mode collapse)')
plt.axvline(0.85, color='red', linestyle='--', linewidth=2, label='Target: 0.85 (control fails)')
plt.xlabel('QED Score')
plt.ylabel('Count')
plt.title('QED Distribution in Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('qed_distribution.png', dpi=150)
print("✅ Saved histogram: qed_distribution.png")
plt.show()
"""

# Final verdict
print("\n" + "="*70)
print("VERDICT FOR MODEL TRAINING")
print("="*70)

high_qed_count = (qed >= 0.85).sum()
high_qed_pct = (high_qed_count / len(qed)) * 100
mid_qed_count = ((qed >= 0.45) & (qed < 0.55)).sum()
mid_qed_pct = (mid_qed_count / len(qed)) * 100

print(f"\n📊 For QED = 0.85 target:")
print(f"   Training examples with QED ≥ 0.85: {high_qed_count:,} ({high_qed_pct:.2f}%)")
if high_qed_pct < 1:
    print(f"   ⚠️  CRITICAL: Less than 1% of training data has QED ≥ 0.85!")
    print(f"   Model cannot learn to generate high-QED molecules without examples.")
elif high_qed_pct < 5:
    print(f"   ⚠️  WARNING: Only {high_qed_pct:.2f}% of training data has high QED.")
    print(f"   Model will struggle to generate QED=0.85 molecules.")
else:
    print(f"   ✅ Sufficient training examples for high-QED generation.")

print(f"\n📊 For QED = 0.5 target:")
print(f"   Training examples around QED=0.5 (0.45-0.55): {mid_qed_count:,} ({mid_qed_pct:.2f}%)")
if mid_qed_pct < 5:
    print(f"   ⚠️  WARNING: Low representation around QED=0.5")
else:
    print(f"   ✅ Sufficient training examples around QED=0.5")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if high_qed_pct < 1:
    print("""
⚠️  DATA IMBALANCE DETECTED!

Your model cannot generate high-QED (0.85) molecules because:
1. Less than 1% of training data has QED ≥ 0.85
2. Model learns distribution of training data
3. If training data is low-QED, model generates low-QED

SOLUTIONS:
1. **Retrain with property loss** (already implemented) - helps but won't fully solve
2. **Oversample high-QED molecules** during training
3. **Add data augmentation** for high-QED molecules
4. **Use a different dataset** with more high-QED molecules
5. **Accept limitation** - model can only reach QED ~0.4-0.5 max

Property loss will help, but fundamental issue is data distribution.
""")
elif high_qed_pct < 5:
    print("""
⚠️  LOW REPRESENTATION AT HIGH QED

Property loss + CFG should help, but:
- Model has limited high-QED examples to learn from
- May still struggle to reach QED=0.85
- Expect best result: QED ~0.6-0.7 (not 0.85)

SOLUTIONS:
1. ✅ Retrain with property loss (already implemented)
2. Oversample high-QED molecules during training
3. Lower expectations: target QED=0.7 instead of 0.85
""")
else:
    print("""
✅ Data distribution looks OK.

Property loss + increased COND_DROPOUT + CFG rescale should enable
control across the full QED range. Proceed with retraining!
""")

print("="*70)
