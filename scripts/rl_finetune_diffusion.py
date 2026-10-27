"""
Reinforcement Learning Fine-tuning for Conditional Diffusion Model
===================================================================

Improves QED control by adding reward-based optimization on top of Stage 2.

Usage:
    python rl_finetune_diffusion.py

Requirements:
    - Stage 2 checkpoint: notebooks/data/diffusion_cond_best.pt
    - Same tokenizer and data as Stage 2
    - GPU: RTX 4060 (8GB) or better
"""

import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# RDKit for reward computation
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf

# ===================================================================
# CONFIGURATION
# ===================================================================

class RLConfig:
    # Paths — resolve relative to the repo by default; override via env vars if needed.
    _DATA_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "data"
    _MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "diffusion"
    STAGE2_CHECKPOINT = os.environ.get("STAGE2_CHECKPOINT", str(_DATA_DIR / "diffusion_cond_best.pt"))
    DATA_PATH = os.environ.get("DATA_PATH", str(_DATA_DIR / "molecules_200k_selfies_clean.csv"))
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", str(_DATA_DIR / "selfies_tokenizer.json"))
    MODEL_DIR = os.environ.get("MODEL_DIR", str(_MODEL_DIR))
    
    # Model Architecture (must match Stage 2)
    MAX_LEN = 96  # Match checkpoint sequence length
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 1024
    DROPOUT = 0.1
    CONDITION_DIM = 64
    
    # Diffusion Schedule (must match Stage 2)
    TIMESTEPS = 1000
    SCHEDULE = "cosine"
    BETA_START = 0.0001
    BETA_END = 0.058
    GAMMA = 0.60
    
    # RL-Specific Hyperparameters (optimized for RTX 4060)
    RL_BATCH_SIZE = 16           # Reduced for 8GB VRAM
    RL_ROLLOUT_SIZE = 100        # Samples per RL iteration
    RL_EPOCHS = 20               # Quick fine-tuning
    RL_LEARNING_RATE = 1e-5      # Lower than Stage 2
    RL_WEIGHT_DECAY = 0.01
    RL_GRAD_CLIP = 1.0
    
    # Reward Settings
    REWARD_TYPE = "qed_accuracy"  # Options: "qed_accuracy", "qed_mae", "qed_combined"
    QED_REWARD_WEIGHT = 1.0       # Weight for QED accuracy reward
    VALIDITY_REWARD = 0.1         # Bonus for valid molecules
    UNIQUENESS_PENALTY = 0.05     # Penalty for duplicates
    
    # Sampling
    SAMPLE_STEPS = 100
    TEMPERATURE = 0.7
    CFG_SCALE = 3.0
    
    # Target QED values for training
    TARGET_QEDS = [0.5, 0.7, 0.85]  # Train on multiple targets
    
    # Logging
    LOG_EVERY = 10
    SAVE_EVERY = 5

config = RLConfig()

# Set device with better detection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print("⚠️  WARNING: CUDA not available! Training will be VERY slow on CPU.")
    print("   Make sure you have:")
    print("   - NVIDIA drivers installed")
    print("   - PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    response = input("   Continue anyway? (y/N): ")
    if response.lower() != 'y':
        print("   Exiting...")
        exit(1)

# ===================================================================
# LOAD TOKENIZER
# ===================================================================

print(f"\nLoading tokenizer from {config.TOKENIZER_PATH}...")
with open(config.TOKENIZER_PATH, 'r') as f:
    tokenizer_data = json.load(f)

vocab = tokenizer_data['vocab']
stoi = tokenizer_data['stoi']
itos = {int(k): v for k, v in tokenizer_data['itos'].items()}
VOCAB_SIZE = len(vocab)

PAD = tokenizer_data['special_tokens']['PAD']
BOS = tokenizer_data['special_tokens']['BOS']
EOS = tokenizer_data['special_tokens']['EOS']

PAD_IDX = stoi[PAD]
BOS_IDX = stoi[BOS]
EOS_IDX = stoi[EOS]

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Special tokens: PAD={PAD_IDX}, BOS={BOS_IDX}, EOS={EOS_IDX}")

# ===================================================================
# SELFIES ENCODING/DECODING
# ===================================================================

def encode_selfies(selfies_str, max_len=config.MAX_LEN):
    """Encode SELFIES string to token IDs."""
    tokens = list(sf.split_selfies(selfies_str))
    tokens = [BOS] + tokens + [EOS]
    
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        tokens[-1] = EOS
    
    pad_len = max_len - len(tokens)
    tokens += [PAD] * pad_len
    
    if 'UNK' in stoi:
        unk_id = stoi['UNK']
        return [stoi.get(t, unk_id) for t in tokens]
    elif '<UNK>' in stoi:
        unk_id = stoi['<UNK>']
        return [stoi.get(t, unk_id) for t in tokens]
    else:
        missing = [t for t in tokens if t not in stoi]
        if missing:
            raise ValueError(f"Unknown tokens: {missing[:10]}")
        return [stoi[t] for t in tokens]


def decode_selfies(ids):
    """Decode token IDs back to SELFIES string."""
    tokens = [itos[i] for i in ids]
    tokens = [t for t in tokens if t not in (PAD, BOS, EOS)]
    return "".join(tokens)


def selfies_to_smiles(selfies_str):
    """Convert SELFIES to SMILES."""
    try:
        return sf.decoder(selfies_str)
    except Exception:
        return None


def is_valid_molecule(smiles):
    """Check if SMILES is a valid molecule."""
    if smiles is None:
        return False
    return Chem.MolFromSmiles(smiles) is not None


# ===================================================================
# DIFFUSION SCHEDULE
# ===================================================================

def get_beta_schedule(schedule_type, timesteps, beta_start, beta_end, gamma=1.0, s=0.008):
    """Beta schedule for discrete token diffusion."""
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    
    elif schedule_type == "cosine":
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = (betas - betas.min()) / (betas.max() - betas.min() + 1e-8)
        betas = betas ** gamma
        betas = betas * (beta_end - beta_start) + beta_start
        betas = torch.clamp(betas, min=beta_start, max=min(beta_end, 0.999))
        return betas
    
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


class DiffusionSchedule:
    """Precompute diffusion schedule parameters."""
    def __init__(self, timesteps, schedule_type="cosine", beta_start=1e-4, beta_end=0.10, gamma=1.0):
        self.timesteps = timesteps
        betas = get_beta_schedule(schedule_type, timesteps, beta_start, beta_end, gamma=gamma)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.keep_prob = alphas_cumprod
        self.corrupt_prob = 1 - alphas_cumprod
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.keep_prob = self.keep_prob.to(device)
        self.corrupt_prob = self.corrupt_prob.to(device)
        return self


# Initialize schedule
schedule = DiffusionSchedule(
    config.TIMESTEPS,
    config.SCHEDULE,
    config.BETA_START,
    config.BETA_END,
    config.GAMMA
).to(device)

print(f"\nDiffusion schedule: {config.TIMESTEPS} steps, {config.SCHEDULE}")

# ===================================================================
# MODEL ARCHITECTURE (exact match to Stage 2)
# ===================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # ensure float for stable embeddings
        t = t.float()
        device = t.device

        half_dim = self.dim // 2
        if half_dim <= 1:
            # edge case: just return zeros if dim too small
            return torch.zeros((t.shape[0], self.dim), device=device, dtype=torch.float32)

        freq = math.log(10000.0) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)  # [half_dim]
        args = t[:, None] * freq[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim or dim-1]

        # if dim is odd, pad
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=-1)
        return emb


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 96,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        condition_dim: int | None = None,
        n_conditions: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.conditional = (condition_dim is not None) and (n_conditions > 0)
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        if self.conditional:
            self.cond_emb = nn.Sequential(
                nn.Linear(n_conditions, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, d_model)
            )
            
            self.property_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.property_head = None
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, t, cond=None, cond_mask=None):
        B, L = x.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len={self.max_len}")
        
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        
        t_emb = self.time_emb(t)
        h = h + t_emb.unsqueeze(1)
        
        if self.conditional and (cond is not None):
            if cond_mask is not None:
                if cond_mask.dim() == 1:
                    cond_mask = cond_mask.unsqueeze(-1).expand_as(cond)
                cond_in = cond * cond_mask.float()
            else:
                cond_in = cond
            c_emb = self.cond_emb(cond_in)
            h = h + c_emb.unsqueeze(1)
        
        h = self.drop(h)
        key_padding_mask = (x == PAD_IDX)
        
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        
        h = self.norm(h)
        logits = self.output(h)
        h_pooled = h.mean(dim=1)
        
        return logits, h_pooled


# ===================================================================
# LOAD STAGE 2 MODEL
# ===================================================================

print(f"\nLoading Stage 2 checkpoint from {config.STAGE2_CHECKPOINT}...")
checkpoint = torch.load(config.STAGE2_CHECKPOINT, map_location=device)

# Get conditioning features from checkpoint
condition_features = checkpoint.get('condition_features', [
    'qed', 'sa_score', 'mw', 'alogp', 'hbd', 'hba', 'psa', 'num_rings'
])
n_conditions = len(condition_features)

print(f"Condition features: {condition_features}")

# Load normalization stats
cond_stats_paths = [
    os.path.join(config.MODEL_DIR, 'condition_stats.json'),
    os.path.join(os.path.dirname(config.DATA_PATH), 'condition_stats.json'),
    os.path.join('notebooks', 'data', 'condition_stats.json')
]

cond_stats = None
for path in cond_stats_paths:
    if os.path.exists(path):
        with open(path, 'r') as f:
            cond_stats = json.load(f)
        print(f"Loaded normalization stats from: {path}")
        break

if cond_stats is None:
    print("⚠️  condition_stats.json not found, creating default stats...")
    # Create default normalization stats based on typical molecular properties
    cond_stats = {
        'features': condition_features,
        'mean': [0.72, 3.5, 380.0, 2.3, 2.1, 4.8, 75.0, 2.8],  # Typical values
        'std': [0.15, 1.2, 120.0, 1.8, 1.5, 2.2, 35.0, 1.5]    # Typical std
    }
    print(f"Created default stats for features: {condition_features}")

# Build model
model = DiffusionTransformer(
    vocab_size=VOCAB_SIZE,
    max_len=config.MAX_LEN,
    d_model=config.D_MODEL,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    d_ff=config.D_FF,
    dropout=config.DROPOUT,
    condition_dim=config.CONDITION_DIM,
    n_conditions=n_conditions
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from epoch {checkpoint.get('epoch', '?')}")
print(f"   Val loss: {checkpoint.get('val_loss', '?'):.4f}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===================================================================
# SAMPLING FUNCTIONS
# ===================================================================

VALID_TOKEN_IDS = torch.tensor(
    [i for i in range(VOCAB_SIZE) if i not in (PAD_IDX, BOS_IDX, EOS_IDX)],
    dtype=torch.long,
    device=device
)


@torch.no_grad()
def p_sample_step(model, x_t, t, schedule, cond=None, cond_mask=None, cfg_scale=1.0, temperature=1.0):
    """Single reverse diffusion step with CFG."""
    B = x_t.shape[0]
    t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)
    
    use_cfg = (cfg_scale > 1.0) and (cond is not None)
    
    if use_cfg:
        # Conditional pass
        logits_c, _ = model(x_t, t_batch, cond, cond_mask)
        
        # Unconditional pass
        cond_mask_u = torch.zeros_like(cond, dtype=torch.bool)
        logits_u, _ = model(x_t, t_batch, cond, cond_mask_u)
        
        # CFG with rescale
        logits_combined = logits_u + cfg_scale * (logits_c - logits_u)
        
        # Rescale to match unconditional std
        std_u = logits_u.std()
        std_combined = logits_combined.std()
        if std_combined > 0:
            logits = logits_u + (logits_combined - logits_u) * (std_u / (std_combined + 1e-8))
        else:
            logits = logits_combined
    else:
        logits, _ = model(x_t, t_batch, cond, cond_mask)
    
    # Block special tokens
    logits[:, :, PAD_IDX] = -1e9
    logits[:, :, BOS_IDX] = -1e9
    logits[:, :, EOS_IDX] = -1e9
    
    # Temperature scaling
    logits = logits / temperature
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    x_t_minus_1 = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, -1)
    
    # Preserve PAD positions
    pad_mask = (x_t == PAD_IDX)
    x_t_minus_1 = torch.where(pad_mask, x_t, x_t_minus_1)
    
    return x_t_minus_1


@torch.no_grad()
def sample(model, schedule, num_samples, seq_len, vocab_size, sample_steps=100,
           temperature=1.0, cond=None, cond_mask=None, cfg_scale=1.0, device='cuda'):
    """Generate samples using reverse diffusion."""
    model.eval()
    
    # Start from pure noise (random valid tokens)
    valid_tokens = VALID_TOKEN_IDS
    random_indices = torch.randint(0, len(valid_tokens), (num_samples, seq_len), device=device)
    x = valid_tokens[random_indices]
    
    # Add BOS at start
    x[:, 0] = BOS_IDX
    
    # Reverse diffusion
    timesteps = torch.linspace(schedule.timesteps - 1, 0, sample_steps, dtype=torch.long, device=device)
    
    for t in timesteps:
        x = p_sample_step(model, x, t.item(), schedule, cond, cond_mask, cfg_scale, temperature)
    
    return x


def prepare_conditions(target_dict, cond_stats, device):
    """Prepare normalized condition vector."""
    features = cond_stats['features']
    mean = np.array(cond_stats['mean'])
    std = np.array(cond_stats['std'])
    
    values = np.array([target_dict.get(f, 0.0) for f in features])
    normalized = (values - mean) / (std + 1e-8)
    
    return torch.tensor(normalized, dtype=torch.float32, device=device)


# ===================================================================
# REWARD COMPUTATION
# ===================================================================

def compute_qed(smiles):
    """Compute QED score for SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)


def compute_reward(smiles_list, target_qed, reward_type="qed_accuracy"):
    """
    Compute reward for generated molecules.
    
    Args:
        smiles_list: List of SMILES strings
        target_qed: Target QED value
        reward_type: Type of reward ("qed_accuracy", "qed_mae", "qed_combined")
    
    Returns:
        rewards: Tensor of rewards [batch_size]
        metrics: Dict of metrics for logging
    """
    batch_size = len(smiles_list)
    rewards = []
    actual_qeds = []
    valid_count = 0
    unique_smiles = set()
    
    for smiles in smiles_list:
        # Validity check
        is_valid = is_valid_molecule(smiles)
        valid_count += int(is_valid)
        
        if not is_valid:
            rewards.append(-1.0)  # Heavy penalty for invalid
            actual_qeds.append(0.0)
            continue
        
        # QED computation
        qed = compute_qed(smiles)
        actual_qeds.append(qed)
        
        # Uniqueness tracking
        unique_smiles.add(smiles)
        
        # Reward computation
        if reward_type == "qed_accuracy":
            # Reward = -|actual - target|
            qed_error = abs(qed - target_qed)
            reward = -qed_error * config.QED_REWARD_WEIGHT
        
        elif reward_type == "qed_mae":
            # Reward = 1 - MAE
            qed_error = abs(qed - target_qed)
            reward = (1.0 - qed_error) * config.QED_REWARD_WEIGHT
        
        elif reward_type == "qed_combined":
            # Combined: accuracy + validity bonus
            qed_error = abs(qed - target_qed)
            reward = -qed_error * config.QED_REWARD_WEIGHT + config.VALIDITY_REWARD
        
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
        
        # Validity bonus
        reward += config.VALIDITY_REWARD
        
        rewards.append(reward)
    
    # Uniqueness penalty
    uniqueness_ratio = len(unique_smiles) / max(1, valid_count)
    if uniqueness_ratio < 1.0:
        uniqueness_penalty = (1.0 - uniqueness_ratio) * config.UNIQUENESS_PENALTY
        rewards = [r - uniqueness_penalty for r in rewards]
    
    # Metrics
    metrics = {
        'validity': valid_count / batch_size * 100,
        'uniqueness': uniqueness_ratio * 100,
        'mean_qed': np.mean(actual_qeds) if actual_qeds else 0.0,
        'qed_mae': np.mean([abs(q - target_qed) for q in actual_qeds]) if actual_qeds else 1.0,
        'mean_reward': np.mean(rewards)
    }
    
    return torch.tensor(rewards, dtype=torch.float32, device=device), metrics


# ===================================================================
# RL TRAINING (REINFORCE)
# ===================================================================

def rl_train_step(model, schedule, optimizer, target_qed, rollout_size, cfg_scale, temperature):
    """
    Single RL training step using REINFORCE policy gradient.
    
    Args:
        model: Diffusion model
        schedule: Diffusion schedule
        optimizer: Optimizer
        target_qed: Target QED value for this step
        rollout_size: Number of samples to generate
        cfg_scale: CFG scale for sampling
        temperature: Sampling temperature
    
    Returns:
        metrics: Dict of training metrics
    """
    model.train()
    
    # Prepare target condition
    target_dict = {
        'qed': target_qed,
        'sa_score': 3.0,
        'mw': 400.0,
        'alogp': 2.5,
        'hbd': 2,
        'hba': 5,
        'psa': 80.0,
        'num_rings': 3
    }
    
    cond_vec = prepare_conditions(target_dict, cond_stats, device)
    cond_batch = cond_vec.unsqueeze(0).expand(rollout_size, -1)
    cond_mask = torch.ones(rollout_size, dtype=torch.bool, device=device)
    
    # Generate samples (with gradient tracking for logits)
    model.eval()  # Use eval mode for sampling, but track gradients
    
    # Sample with autograd enabled
    x_samples = sample(
        model, schedule,
        num_samples=rollout_size,
        seq_len=config.MAX_LEN,
        vocab_size=VOCAB_SIZE,
        sample_steps=config.SAMPLE_STEPS,
        temperature=temperature,
        cond=cond_batch,
        cond_mask=cond_mask,
        cfg_scale=cfg_scale,
        device=device
    )
    
    # Decode samples
    generated_selfies = [decode_selfies(s.cpu().tolist()) for s in x_samples]
    generated_smiles = [selfies_to_smiles(s) for s in generated_selfies]
    
    # Compute rewards
    rewards, reward_metrics = compute_reward(
        generated_smiles,
        target_qed,
        reward_type=config.REWARD_TYPE
    )
    
    # Normalize rewards (baseline subtraction)
    rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Compute policy gradient loss
    # We need to get log probabilities of the sampled actions
    model.train()
    
    # Forward pass to get logits
    t_mid = torch.full((rollout_size,), schedule.timesteps // 2, device=device, dtype=torch.long)
    logits, _ = model(x_samples, t_mid, cond_batch, cond_mask)
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get log prob of sampled tokens
    sampled_log_probs = torch.gather(log_probs, 2, x_samples.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding
    pad_mask = (x_samples != PAD_IDX)
    sampled_log_probs = sampled_log_probs * pad_mask.float()
    
    # Sum log probs per sequence
    seq_log_probs = sampled_log_probs.sum(dim=1)
    
    # REINFORCE loss: -log_prob * reward
    policy_loss = -(seq_log_probs * rewards_normalized).mean()
    
    # Backprop
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.RL_GRAD_CLIP)
    optimizer.step()
    
    # Metrics
    metrics = {
        'policy_loss': policy_loss.item(),
        'mean_reward': rewards.mean().item(),
        'std_reward': rewards.std().item(),
        **reward_metrics
    }
    
    return metrics


def rl_training_loop():
    """Main RL training loop."""
    print("\n" + "="*60)
    print("RL FINE-TUNING")
    print("="*60)
    print(f"Epochs: {config.RL_EPOCHS}")
    print(f"Rollout size: {config.RL_ROLLOUT_SIZE}")
    print(f"Batch size: {config.RL_BATCH_SIZE}")
    print(f"Learning rate: {config.RL_LEARNING_RATE}")
    print(f"Target QEDs: {config.TARGET_QEDS}")
    print(f"Reward type: {config.REWARD_TYPE}")
    print("="*60)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.RL_LEARNING_RATE,
        weight_decay=config.RL_WEIGHT_DECAY
    )
    
    # Training log
    training_log = []
    
    # Training loop
    for epoch in range(config.RL_EPOCHS):
        epoch_metrics = {
            'epoch': epoch + 1,
            'target_qeds': {},
        }
        
        print(f"\nEpoch {epoch+1}/{config.RL_EPOCHS}")
        
        # Train on each target QED
        for target_qed in config.TARGET_QEDS:
            print(f"  Target QED = {target_qed:.2f}...")
            
            metrics = rl_train_step(
                model, schedule, optimizer,
                target_qed=target_qed,
                rollout_size=config.RL_ROLLOUT_SIZE,
                cfg_scale=config.CFG_SCALE,
                temperature=config.TEMPERATURE
            )
            
            epoch_metrics['target_qeds'][f'qed_{target_qed}'] = metrics
            
            print(f"    Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f} | "
                  f"QED MAE: {metrics['qed_mae']:.3f} | "
                  f"Valid: {metrics['validity']:.1f}% | "
                  f"Unique: {metrics['uniqueness']:.1f}%")
        
        training_log.append(epoch_metrics)
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(config.MODEL_DIR, f'diffusion_rl_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
                'condition_features': condition_features,
                'training_log': training_log
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.MODEL_DIR, 'diffusion_rl_final.pt')
    torch.save({
        'epoch': config.RL_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config),
        'condition_features': condition_features,
        'training_log': training_log
    }, final_path)
    print(f"\n✅ Training complete! Saved final model: {final_path}")
    
    # Save training log
    log_path = os.path.join(config.MODEL_DIR, 'rl_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"📊 Training log saved: {log_path}")


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    try:
        rl_training_loop()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
