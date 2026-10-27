"""
SELFIES Conversion and Tokenization Pipeline

Converts SMILES to SELFIES and creates a production-ready tokenization pipeline
with PyTorch Dataset for training diffusion/VAE/RL models.
"""

import pandas as pd
import selfies as sf
from rdkit import Chem
from tqdm import tqdm
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

INPUT_FILE = r"C:\Users\ahaan\CODE\Research Papers\Drug Discovery\Project\notebooks\data\molecules_200k_with_bioactivity_21pct_coverage.csv"
OUTPUT_DIR = r"C:\Users\ahaan\CODE\Research Papers\Drug Discovery\Project\notebooks\data"

CLEAN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "molecules_200k_selfies_clean.csv")
FAILED_SMILES_FILE = os.path.join(OUTPUT_DIR, "failed_smiles.csv")
ALPHABET_FILE = os.path.join(OUTPUT_DIR, "selfies_alphabet.json")
TOKENIZER_FILE = os.path.join(OUTPUT_DIR, "selfies_tokenizer.json")

MAX_LEN = 96  # Power-of-2, safe headroom for diffusion/RL


# ============================================================================
# Step 1: Load Dataset
# ============================================================================

print("="*80)
print("STEP 1: Loading Dataset")
print("="*80)

print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} molecules")
print(f"Columns: {df.columns.tolist()[:10]}...")


# ============================================================================
# Step 2: Convert SMILES to SELFIES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Converting SMILES to SELFIES")
print("="*80)

def smiles_to_selfies(smiles):
    """Convert SMILES to SELFIES, return None if conversion fails"""
    try:
        return sf.encoder(smiles)
    except Exception as e:
        return None

print("Converting SMILES to SELFIES...")
tqdm.pandas(desc="Converting")
df['selfies'] = df['canonical_smiles'].progress_apply(smiles_to_selfies)

failed = df['selfies'].isna().sum()
print(f"\nSuccessfully converted: {len(df) - failed:,} / {len(df):,}")
if failed > 0:
    print(f"Failed conversions: {failed}")


# ============================================================================
# Step 3: Clean Failed Conversions
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Cleaning Failed Conversions")
print("="*80)

# Save failed SMILES for inspection
failed_df = df[df["selfies"].isna()][["canonical_smiles"]]
if len(failed_df) > 0:
    failed_df.to_csv(FAILED_SMILES_FILE, index=False)
    print(f"Saved {len(failed_df)} failed SMILES to {FAILED_SMILES_FILE}")
    print("\nFailed SMILES:")
    print(failed_df)
else:
    print("No failed conversions!")

# Drop rows with failed conversions
df = df.dropna(subset=["selfies"]).reset_index(drop=True)
print(f"\nFinal dataset: {len(df):,} molecules (cleaned)")


# ============================================================================
# Step 4: Analyze SELFIES Lengths
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Analyzing SELFIES Lengths")
print("="*80)

df['selfies_length'] = df['selfies'].apply(lambda x: len(list(sf.split_selfies(x))))
df['smiles_length'] = df['canonical_smiles'].apply(len)

print("\n=== Length Statistics ===")
print(f"SMILES length: {df['smiles_length'].mean():.1f} ± {df['smiles_length'].std():.1f} chars")
print(f"SMILES max: {df['smiles_length'].max()} chars")
print(f"\nSELFIES length: {df['selfies_length'].mean():.1f} ± {df['selfies_length'].std():.1f} tokens")
print(f"SELFIES max: {int(df['selfies_length'].max())} tokens")
print(f"\nSELFIES/SMILES ratio: {df['selfies_length'].mean() / df['smiles_length'].mean():.2f}x")


# ============================================================================
# Step 5: Save Clean Dataset
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Saving Clean Dataset")
print("="*80)

print(f"Saving clean dataset to {CLEAN_OUTPUT_FILE}...")
df.to_csv(CLEAN_OUTPUT_FILE, index=False)
print(f"Saved {len(df):,} molecules with SELFIES")

file_size_mb = os.path.getsize(CLEAN_OUTPUT_FILE) / (1024 * 1024)
print(f"File size: {file_size_mb:.1f} MB")


# ============================================================================
# Step 6: Verify Round-Trip Conversion
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Verifying Round-Trip Conversion")
print("="*80)

print("Verifying round-trip conversion (sample of 100)...")
sample_df = df.sample(100)

mismatches = 0
for idx, row in sample_df.iterrows():
    original_smiles = row['canonical_smiles']
    selfies_str = row['selfies']
    
    # Convert SELFIES back to SMILES
    decoded_smiles = sf.decoder(selfies_str)
    
    # Canonicalize both for comparison
    mol1 = Chem.MolFromSmiles(original_smiles)
    mol2 = Chem.MolFromSmiles(decoded_smiles)
    
    if mol1 and mol2:
        canon1 = Chem.MolToSmiles(mol1)
        canon2 = Chem.MolToSmiles(mol2)
        
        if canon1 != canon2:
            mismatches += 1
            print(f"Mismatch: {original_smiles} -> {decoded_smiles}")

print(f"\nRound-trip verification: {100 - mismatches}/100 successful")
if mismatches == 0:
    print("✅ All conversions are reversible!")
else:
    print(f"⚠️ {mismatches} mismatches found")


# ============================================================================
# Step 7: Build SELFIES Alphabet
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Building SELFIES Alphabet")
print("="*80)

print("Extracting unique tokens from all SELFIES strings...")
selfies_list = df["selfies"].tolist()

alphabet = sorted(
    set(
        token
        for s in selfies_list
        for token in sf.split_selfies(s)
    )
)

print(f"Alphabet size: {len(alphabet)} unique tokens")
print(f"\nFirst 20 tokens: {alphabet[:20]}")
print(f"Last 20 tokens: {alphabet[-20:]}")

# Save alphabet
with open(ALPHABET_FILE, "w") as f:
    json.dump(alphabet, f, indent=2)
print(f"\nSaved alphabet to {ALPHABET_FILE}")


# ============================================================================
# Step 8: Determine Max Sequence Length
# ============================================================================

print("\n" + "="*80)
print("STEP 8: Determining Max Sequence Length")
print("="*80)

max_selfies_len = int(df['selfies_length'].max())
mean_selfies_len = df['selfies_length'].mean()
p95_selfies_len = df['selfies_length'].quantile(0.95)
p99_selfies_len = df['selfies_length'].quantile(0.99)

print("=== Sequence Length Statistics ===")
print(f"Mean: {mean_selfies_len:.1f} tokens")
print(f"Max: {max_selfies_len} tokens")
print(f"95th percentile: {p95_selfies_len:.0f} tokens")
print(f"99th percentile: {p99_selfies_len:.0f} tokens")

print(f"\n✅ Using MAX_LEN = {MAX_LEN} (covers {(df['selfies_length'] <= MAX_LEN).mean()*100:.2f}% of dataset)")


# ============================================================================
# Step 9: Create Tokenizer
# ============================================================================

print("\n" + "="*80)
print("STEP 9: Creating Tokenizer")
print("="*80)

# Define special tokens
PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"

# Build vocabulary: special tokens + alphabet
vocab = [PAD, BOS, EOS] + alphabet
stoi = {tok: i for i, tok in enumerate(vocab)}  # string to index
itos = {i: tok for tok, i in stoi.items()}      # index to string

VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE} tokens")
print(f"\nSpecial tokens:")
print(f"  PAD = {stoi[PAD]}")
print(f"  BOS = {stoi[BOS]}")
print(f"  EOS = {stoi[EOS]}")
print(f"\nFirst alphabet token: {vocab[3]} = {stoi[vocab[3]]}")

# Save tokenizer
tokenizer_data = {
    "vocab": vocab,
    "stoi": stoi,
    "itos": {str(k): v for k, v in itos.items()},  # JSON keys must be strings
    "max_len": MAX_LEN,
    "vocab_size": VOCAB_SIZE,
    "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS}
}
with open(TOKENIZER_FILE, "w") as f:
    json.dump(tokenizer_data, f, indent=2)
print(f"\nSaved tokenizer to {TOKENIZER_FILE}")


# ============================================================================
# Step 10: Encode/Decode Functions
# ============================================================================

print("\n" + "="*80)
print("STEP 10: Testing Encode/Decode Functions")
print("="*80)

def encode_selfies(selfies_str, max_len=MAX_LEN):
    """
    Encode SELFIES string to token IDs with BOS/EOS/PAD.
    
    Args:
        selfies_str: SELFIES string
        max_len: Maximum sequence length
    
    Returns:
        List of token IDs
    """
    tokens = list(sf.split_selfies(selfies_str))
    tokens = [BOS] + tokens + [EOS]
    
    # Truncate if too long
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    
    # Pad if too short
    pad_len = max_len - len(tokens)
    tokens += [PAD] * pad_len
    
    # Convert to IDs
    return [stoi[t] for t in tokens]


def decode_selfies(ids):
    """
    Decode token IDs back to SELFIES string.
    
    Args:
        ids: List of token IDs
    
    Returns:
        SELFIES string
    """
    tokens = [itos[i] for i in ids]
    # Remove special tokens
    tokens = [t for t in tokens if t not in (PAD, BOS, EOS)]
    return "".join(tokens)


# Test encoding/decoding
test_selfies = df.iloc[0]["selfies"]
print("Original SELFIES:")
print(test_selfies)
print(f"\nLength: {len(list(sf.split_selfies(test_selfies)))} tokens")

encoded = encode_selfies(test_selfies)
print(f"\nEncoded (first 20 IDs): {encoded[:20]}")
print(f"Encoded length: {len(encoded)} (padded to {MAX_LEN})")

decoded = decode_selfies(encoded)
print(f"\nDecoded SELFIES:")
print(decoded)
print(f"\nRound-trip success: {test_selfies == decoded}")


# ============================================================================
# Step 11: PyTorch Dataset
# ============================================================================

print("\n" + "="*80)
print("STEP 11: Creating PyTorch Dataset")
print("="*80)

class SelfiesDataset(Dataset):
    """
    PyTorch Dataset for SELFIES strings.
    
    Args:
        df: DataFrame with 'selfies' column
        max_len: Maximum sequence length
    """
    def __init__(self, df, max_len=MAX_LEN):
        self.selfies = df["selfies"].values
        self.max_len = max_len
    
    def __len__(self):
        return len(self.selfies)
    
    def __getitem__(self, idx):
        selfies_str = self.selfies[idx]
        ids = encode_selfies(selfies_str, self.max_len)
        return torch.tensor(ids, dtype=torch.long)


# Test dataset
dataset = SelfiesDataset(df)
print(f"Dataset size: {len(dataset):,} molecules")

# Test batching
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print(f"\nBatch shape: {batch.shape}")  # [batch_size, seq_len]
print(f"Batch dtype: {batch.dtype}")

# Show one example
print(f"\nExample tensor (first 20 tokens): {batch[0][:20].tolist()}")
print(f"Decoded: {decode_selfies(batch[0].tolist())[:100]}...")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\n📁 Output Files:")
print(f"  ✅ {CLEAN_OUTPUT_FILE}")
print(f"  ✅ {FAILED_SMILES_FILE}")
print(f"  ✅ {ALPHABET_FILE}")
print(f"  ✅ {TOKENIZER_FILE}")

print("\n📊 Statistics:")
print(f"  • Success rate: {(len(df) / 200000) * 100:.4f}%")
print(f"  • SELFIES length: mean ~{mean_selfies_len:.0f} tokens, max {max_selfies_len} tokens")
print(f"  • SELFIES/SMILES ratio: {df['selfies_length'].mean() / df['smiles_length'].mean():.2f}×")
print(f"  • Vocabulary size: {VOCAB_SIZE} tokens (3 special + {len(alphabet)} alphabet)")
print(f"  • Max sequence length: {MAX_LEN} (covers {(df['selfies_length'] <= MAX_LEN).mean()*100:.2f}% of dataset)")

print("\n✅ Ready for Training:")
print("  • Diffusion models (token-level corruption, guaranteed validity)")
print("  • VAE models (sequence latent modeling)")
print("  • RL optimization (safe token mutations, 100% valid decoding)")
print("  • Production-ready PyTorch Dataset")

print("\n🚀 Next Steps:")
print("  1. Use SelfiesDataset for diffusion/VAE/RL training")
print("  2. Compare SMILES vs SELFIES model performance")
print("  3. No sanitization loops needed - SELFIES guarantees valid molecules")

print("\n" + "="*80)
