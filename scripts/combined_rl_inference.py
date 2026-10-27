"""
Combined RL Fine-tuning + Parallel GPU/CPU Inference for Conditional Diffusion
==============================================================================

This unified script handles both:
1. RL fine-tuning for precise QED control 
2. High-speed parallel inference using GPU+CPU

Usage:
    # RL training only
    python combined_rl_inference.py --mode train
    
    # Parallel inference only  
    python combined_rl_inference.py --mode inference
    
    # Full pipeline: train then inference
    python combined_rl_inference.py --mode both

Requirements:
    - Stage 2 checkpoint: notebooks/data/diffusion_cond_best.pt
    - NVIDIA GPU with CUDA support
    - Multi-core CPU for parallel processing
"""

import os
import sys
import json
import math
import time
import argparse
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# RDKit for molecular properties
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf

# ===================================================================
# CONFIGURATION
# ===================================================================

@dataclass 
class Config:
    # Paths
    STAGE2_CHECKPOINT: str = r"notebooks\data\diffusion_cond_best.pt"
    DATA_PATH: str = r"notebooks\data\molecules_200k_selfies_clean.csv"
    TOKENIZER_PATH: str = r"notebooks\data\selfies_tokenizer.json"
    CONDITION_STATS_PATH: str = r"notebooks\data\condition_stats.json"
    MODEL_DIR: str = r"models\diffusion"
    
    # Model Architecture
    MAX_LEN: int = 96
    D_MODEL: int = 256
    N_HEADS: int = 8
    N_LAYERS: int = 6
    D_FF: int = 1024
    DROPOUT: float = 0.1
    CONDITION_DIM: int = 64
    
    # Diffusion Schedule
    TIMESTEPS: int = 1000
    SCHEDULE: str = "cosine"
    BETA_START: float = 0.0001
    BETA_END: float = 0.058
    GAMMA: float = 0.60
    
    # RL Training
    RL_EPOCHS: int = 20
    RL_ROLLOUT_SIZE: int = 100
    RL_LEARNING_RATE: float = 1e-5
    RL_WEIGHT_DECAY: float = 0.01
    RL_GRAD_CLIP: float = 1.0
    REWARD_TYPE: str = "qed_accuracy"
    QED_REWARD_WEIGHT: float = 1.0
    VALIDITY_REWARD: float = 0.1
    UNIQUENESS_PENALTY: float = 0.05
    TARGET_QEDS: List[float] = None
    
    # Parallel Inference
    GPU_BATCH_SIZE: int = 32
    CPU_WORKERS: int = 4
    NUM_SAMPLES: int = 1000
    SAMPLE_STEPS: int = 50
    TEMPERATURE: float = 0.8
    CFG_SCALE: float = 3.0
    TARGET_CONDITIONS: List[Dict] = None
    
    # Logging
    SAVE_EVERY: int = 5
    
    def __post_init__(self):
        if self.TARGET_QEDS is None:
            self.TARGET_QEDS = [0.5, 0.7, 0.85]
        if self.TARGET_CONDITIONS is None:
            self.TARGET_CONDITIONS = [
                {"qed": 0.5, "mw": 350, "alogp": 2.5},
                {"qed": 0.7, "mw": 300, "alogp": 2.0}, 
                {"qed": 0.85, "mw": 400, "alogp": 3.0}
            ]

config = Config()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===================================================================
# LOAD RESOURCES
# ===================================================================

def load_tokenizer():
    """Load SELFIES tokenizer."""
    print(f"Loading tokenizer from {config.TOKENIZER_PATH}...")
    with open(config.TOKENIZER_PATH, 'r') as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data['vocab']
    stoi = tokenizer_data['stoi'] 
    itos = {int(k): v for k, v in tokenizer_data['itos'].items()}
    
    special_tokens = {
        'PAD': tokenizer_data['special_tokens']['PAD'],
        'BOS': tokenizer_data['special_tokens']['BOS'],
        'EOS': tokenizer_data['special_tokens']['EOS']
    }
    
    special_ids = {k: stoi[v] for k, v in special_tokens.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: {special_ids}")
    
    return vocab, stoi, itos, special_tokens, special_ids

def load_condition_stats():
    """Load molecular property normalization stats."""
    paths = [
        os.path.join(config.MODEL_DIR, 'condition_stats.json'),
        config.CONDITION_STATS_PATH,
        os.path.join('notebooks', 'data', 'condition_stats.json')
    ]
    
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                cond_stats = json.load(f)
            print(f"Loaded condition stats from: {path}")
            return cond_stats
    
    print("⚠️  Creating default condition stats...")
    return {
        'features': ['qed', 'sa_score', 'mw', 'alogp', 'hbd', 'hba', 'psa', 'num_rings'],
        'mean': [0.72, 3.5, 380.0, 2.3, 2.1, 4.8, 75.0, 2.8],
        'std': [0.15, 1.2, 120.0, 1.8, 1.5, 2.2, 35.0, 1.5]
    }

# Load global resources
vocab, stoi, itos, special_tokens, special_ids = load_tokenizer()
cond_stats = load_condition_stats()
VOCAB_SIZE = len(vocab)

# ===================================================================
# MOLECULAR UTILITIES
# ===================================================================

def encode_selfies(selfies_str, max_len=config.MAX_LEN):
    """Encode SELFIES string to token IDs."""
    tokens = list(sf.split_selfies(selfies_str))
    tokens = [special_tokens['BOS']] + tokens + [special_tokens['EOS']]
    
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        tokens[-1] = special_tokens['EOS']
    
    pad_len = max_len - len(tokens)
    tokens += [special_tokens['PAD']] * pad_len
    
    return [stoi.get(t, stoi.get('UNK', 0)) for t in tokens]

def decode_selfies(ids):
    """Decode token IDs back to SELFIES string."""
    tokens = [itos[i] for i in ids if i in itos]
    tokens = [t for t in tokens if t not in special_tokens.values()]
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

def compute_molecular_properties(smiles):
    """Compute molecular properties."""
    if not is_valid_molecule(smiles):
        return {'qed': 0.0, 'mw': 0.0, 'alogp': 0.0, 'hbd': 0, 'hba': 0, 'psa': 0.0, 'num_rings': 0, 'sa_score': 5.0}
    
    mol = Chem.MolFromSmiles(smiles)
    return {
        'qed': QED.qed(mol),
        'mw': Descriptors.MolWt(mol),
        'alogp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'psa': Descriptors.TPSA(mol),
        'num_rings': Descriptors.RingCount(mol),
        'sa_score': 3.0  # Simplified
    }

def prepare_conditions(target_dict, device):
    """Prepare normalized condition vector."""
    features = cond_stats['features']
    mean = np.array(cond_stats['mean'])
    std = np.array(cond_stats['std'])
    
    values = np.array([target_dict.get(f, 0.0) for f in features])
    normalized = (values - mean) / (std + 1e-8)
    
    return torch.tensor(normalized, dtype=torch.float32, device=device)

# ===================================================================
# DIFFUSION SCHEDULE
# ===================================================================

def get_beta_schedule(schedule_type, timesteps, beta_start, beta_end, gamma=1.0):
    """Beta schedule for discrete token diffusion.""" 
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule_type == "cosine":
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
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
    def __init__(self, timesteps, schedule_type="cosine", beta_start=1e-4, beta_end=0.058, gamma=0.6):
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

# ===================================================================
# MODEL ARCHITECTURE
# ===================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        device = t.device
        half_dim = self.dim // 2
        if half_dim <= 1:
            return torch.zeros((t.shape[0], self.dim), device=device, dtype=torch.float32)
        freq = math.log(10000.0) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
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
    def __init__(self, vocab_size: int, max_len: int = 96, d_model: int = 256, 
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 1024,
                 dropout: float = 0.1, condition_dim: int = 64, n_conditions: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.conditional = (n_conditions > 0)
        
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
        key_padding_mask = (x == special_ids['PAD'])
        
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_padding_mask)
        
        h = self.norm(h)
        logits = self.output(h)
        h_pooled = h.mean(dim=1)
        
        return logits, h_pooled

# ===================================================================
# MODEL LOADING
# ===================================================================

def load_model(checkpoint_path: str) -> DiffusionTransformer:
    """Load diffusion model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    condition_features = checkpoint.get('condition_features', cond_stats['features'])
    n_conditions = len(condition_features)
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', '?')
    print(f"✅ Model loaded from epoch {epoch}, val loss: {val_loss}")
    return model, condition_features

# ===================================================================
# SAMPLING
# ===================================================================

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
    for special_id in special_ids.values():
        logits[:, :, special_id] = -1e9
    
    # Temperature scaling
    logits = logits / temperature
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    x_t_minus_1 = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, -1)
    
    # Preserve PAD positions
    pad_mask = (x_t == special_ids['PAD'])
    x_t_minus_1 = torch.where(pad_mask, x_t, x_t_minus_1)
    
    return x_t_minus_1

@torch.no_grad()
def sample(model, schedule, num_samples, seq_len, vocab_size, sample_steps=100,
           temperature=1.0, cond=None, cond_mask=None, cfg_scale=1.0, device='cuda'):
    """Generate samples using reverse diffusion."""
    model.eval()
    
    # Start from noise (valid tokens only)
    valid_token_ids = torch.tensor([i for i in range(vocab_size) 
                                   if i not in special_ids.values()], device=device)
    random_indices = torch.randint(0, len(valid_token_ids), (num_samples, seq_len), device=device)
    x = valid_token_ids[random_indices]
    
    # Add BOS at start
    x[:, 0] = special_ids['BOS']
    
    # Reverse diffusion
    timesteps = torch.linspace(schedule.timesteps - 1, 0, sample_steps, dtype=torch.long, device=device)
    
    for t in timesteps:
        x = p_sample_step(model, x, t.item(), schedule, cond, cond_mask, cfg_scale, temperature)
    
    return x

# ===================================================================
# RL TRAINING
# ===================================================================

def compute_reward(smiles_list, target_qed):
    """Compute reward for generated molecules."""
    batch_size = len(smiles_list)
    rewards = []
    actual_qeds = []
    valid_count = 0
    unique_smiles = set()
    
    for smiles in smiles_list:
        is_valid = is_valid_molecule(smiles)
        valid_count += int(is_valid)
        
        if not is_valid:
            rewards.append(-1.0)
            actual_qeds.append(0.0)
            continue
        
        qed = compute_molecular_properties(smiles)['qed']
        actual_qeds.append(qed)
        unique_smiles.add(smiles)
        
        # QED accuracy reward
        qed_error = abs(qed - target_qed)
        reward = -qed_error * config.QED_REWARD_WEIGHT + config.VALIDITY_REWARD
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

def rl_train_step(model, schedule, optimizer, target_qed):
    """Single RL training step using REINFORCE."""
    model.train()
    
    # Prepare target condition
    target_dict = {
        'qed': target_qed, 'sa_score': 3.0, 'mw': 400.0, 'alogp': 2.5,
        'hbd': 2, 'hba': 5, 'psa': 80.0, 'num_rings': 3
    }
    
    cond_vec = prepare_conditions(target_dict, device)
    cond_batch = cond_vec.unsqueeze(0).expand(config.RL_ROLLOUT_SIZE, -1)
    cond_mask = torch.ones(config.RL_ROLLOUT_SIZE, dtype=torch.bool, device=device)
    
    # Generate samples
    model.eval()
    x_samples = sample(
        model, schedule,
        num_samples=config.RL_ROLLOUT_SIZE,
        seq_len=config.MAX_LEN,
        vocab_size=VOCAB_SIZE,
        sample_steps=100,
        temperature=config.TEMPERATURE,
        cond=cond_batch,
        cond_mask=cond_mask,
        cfg_scale=config.CFG_SCALE,
        device=device
    )
    
    # Decode samples  
    generated_selfies = [decode_selfies(s.cpu().tolist()) for s in x_samples]
    generated_smiles = [selfies_to_smiles(s) for s in generated_selfies]
    
    # Compute rewards
    rewards, reward_metrics = compute_reward(generated_smiles, target_qed)
    rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Policy gradient loss
    model.train()
    t_mid = torch.full((config.RL_ROLLOUT_SIZE,), schedule.timesteps // 2, device=device, dtype=torch.long)
    logits, _ = model(x_samples, t_mid, cond_batch, cond_mask)
    
    log_probs = F.log_softmax(logits, dim=-1)
    sampled_log_probs = torch.gather(log_probs, 2, x_samples.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding
    pad_mask = (x_samples != special_ids['PAD'])
    sampled_log_probs = sampled_log_probs * pad_mask.float()
    seq_log_probs = sampled_log_probs.sum(dim=1)
    
    # REINFORCE loss
    policy_loss = -(seq_log_probs * rewards_normalized).mean()
    
    # Backprop
    optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.RL_GRAD_CLIP)
    optimizer.step()
    
    metrics = {
        'policy_loss': policy_loss.item(),
        'mean_reward': rewards.mean().item(),
        'std_reward': rewards.std().item(),
        **reward_metrics
    }
    
    return metrics

def rl_training_loop(model, condition_features):
    """Main RL training loop."""
    print("\n" + "="*60)
    print("RL FINE-TUNING") 
    print("="*60)
    print(f"Epochs: {config.RL_EPOCHS}")
    print(f"Target QEDs: {config.TARGET_QEDS}")
    print("="*60)
    
    # Initialize diffusion schedule
    schedule = DiffusionSchedule(
        config.TIMESTEPS, config.SCHEDULE,
        config.BETA_START, config.BETA_END, config.GAMMA
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.RL_LEARNING_RATE,
        weight_decay=config.RL_WEIGHT_DECAY
    )
    
    # Training loop
    training_log = []
    
    for epoch in range(config.RL_EPOCHS):
        epoch_metrics = {'epoch': epoch + 1, 'target_qeds': {}}
        print(f"\nEpoch {epoch+1}/{config.RL_EPOCHS}")
        
        # Train on each target QED
        for target_qed in config.TARGET_QEDS:
            print(f"  Target QED = {target_qed:.2f}...")
            
            metrics = rl_train_step(model, schedule, optimizer, target_qed)
            epoch_metrics['target_qeds'][f'qed_{target_qed}'] = metrics
            
            print(f"    Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f} | "
                  f"QED MAE: {metrics['qed_mae']:.3f} | "
                  f"Valid: {metrics['validity']:.1f}% | "
                  f"Unique: {metrics['uniqueness']:.1f}%")
        
        training_log.append(epoch_metrics)
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
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
    print(f"\n✅ RL training complete! Saved final model: {final_path}")
    
    # Save training log
    log_path = os.path.join(config.MODEL_DIR, 'rl_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"📊 Training log saved: {log_path}")

# ===================================================================
# PARALLEL INFERENCE
# ===================================================================

class CPUPropertyPredictor:
    """CPU-optimized parallel property prediction."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        print(f"✅ CPU property predictor initialized with {num_workers} workers")
    
    def predict_properties_parallel(self, molecules_batch: List[str]) -> List[Dict]:
        """Parallel property prediction on CPU."""
        if len(molecules_batch) <= 4:
            return [compute_molecular_properties(mol) for mol in molecules_batch]
        
        # Large batch, use parallel processing
        chunk_size = max(1, len(molecules_batch) // self.num_workers)
        chunks = [molecules_batch[i:i + chunk_size] 
                 for i in range(0, len(molecules_batch), chunk_size)]
        
        # Submit parallel tasks
        futures = [self.executor.submit(self._compute_chunk_properties, chunk) 
                  for chunk in chunks]
        
        # Collect results
        all_results = []
        for future in as_completed(futures):
            all_results.extend(future.result())
        
        return all_results
    
    def _compute_chunk_properties(self, molecules: List[str]) -> List[Dict]:
        """Compute properties for a chunk of molecules."""
        return [compute_molecular_properties(mol) for mol in molecules]

class ParallelInferenceEngine:
    """Coordinated GPU+CPU inference system."""
    
    def __init__(self, model, condition_features):
        self.gpu_model = model
        self.condition_features = condition_features
        self.cpu_predictor = CPUPropertyPredictor(config.CPU_WORKERS)
        
        # Initialize diffusion schedule
        self.schedule = DiffusionSchedule(
            config.TIMESTEPS, config.SCHEDULE,
            config.BETA_START, config.BETA_END, config.GAMMA
        ).to(device)
        
        # Thread coordination
        self.gpu_queue = Queue(maxsize=100)
        self.cpu_queue = Queue(maxsize=100)
        self.results_queue = Queue()
    
    @torch.no_grad()
    def _gpu_diffusion_worker(self):
        """GPU worker for diffusion sampling."""
        while True:
            try:
                request = self.gpu_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break
                
                batch_size, target_conditions, request_id = request
                
                # Generate samples
                samples = self._sample_batch(batch_size, target_conditions)
                
                # Decode to SELFIES/SMILES
                molecules = []
                for sample in samples:
                    selfies_str = decode_selfies(sample.cpu().tolist())
                    smiles = selfies_to_smiles(selfies_str) if selfies_str else "INVALID"
                    molecules.append(smiles)
                
                # Send to CPU queue for property analysis
                self.cpu_queue.put((molecules, target_conditions, request_id))
                
            except Empty:
                continue
            except Exception as e:
                print(f"GPU worker error: {e}")
                continue
    
    def _sample_batch(self, batch_size: int, target_conditions: Dict) -> torch.Tensor:
        """Sample a batch of molecules using diffusion.""" 
        # Prepare conditions
        cond_vec = prepare_conditions(target_conditions, device)
        cond_batch = cond_vec.unsqueeze(0).expand(batch_size, -1)
        cond_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Generate samples
        samples = sample(
            self.gpu_model, self.schedule,
            num_samples=batch_size,
            seq_len=config.MAX_LEN,
            vocab_size=VOCAB_SIZE,
            sample_steps=config.SAMPLE_STEPS,
            temperature=config.TEMPERATURE,
            cond=cond_batch,
            cond_mask=cond_mask,
            cfg_scale=config.CFG_SCALE,
            device=device
        )
        
        return samples
    
    def _cpu_property_worker(self):
        """CPU worker for property prediction."""
        while True:
            try:
                request = self.cpu_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break
                
                molecules, target_conditions, request_id = request
                
                # Batch property prediction on CPU
                properties = self.cpu_predictor.predict_properties_parallel(molecules)
                
                # Compile results
                results = []
                for mol, props in zip(molecules, properties):
                    result = {
                        'smiles': mol,
                        'valid': mol != "INVALID" and is_valid_molecule(mol),
                        'properties': props,
                        'target': target_conditions,
                        'qed_error': abs(props['qed'] - target_conditions.get('qed', 0.5)),
                        'request_id': request_id
                    }
                    results.append(result)
                
                self.results_queue.put(results)
                
            except Empty:
                continue
            except Exception as e:
                print(f"CPU worker error: {e}")
                continue
    
    def generate_parallel(self, num_samples: int, target_conditions_list: List[Dict]) -> List[Dict]:
        """Generate molecules using parallel GPU+CPU inference."""
        print(f"\n🚀 Starting parallel generation: {num_samples} samples")
        print(f"   GPU batch size: {config.GPU_BATCH_SIZE}")
        print(f"   CPU workers: {config.CPU_WORKERS}")
        
        # Start workers
        gpu_thread = threading.Thread(target=self._gpu_diffusion_worker)
        cpu_thread = threading.Thread(target=self._cpu_property_worker)
        
        gpu_thread.daemon = True
        cpu_thread.daemon = True
        
        gpu_thread.start()
        cpu_thread.start()
        
        # Submit generation requests
        all_results = []
        request_id = 0
        
        samples_per_condition = num_samples // len(target_conditions_list)
        
        with tqdm(total=num_samples, desc="Generating molecules") as pbar:
            for target_conditions in target_conditions_list:
                remaining = samples_per_condition
                
                while remaining > 0:
                    batch_size = min(config.GPU_BATCH_SIZE, remaining)
                    self.gpu_queue.put((batch_size, target_conditions, request_id))
                    request_id += 1
                    remaining -= batch_size
            
            # Collect results
            completed_samples = 0
            while completed_samples < num_samples:
                try:
                    results = self.results_queue.get(timeout=30)
                    all_results.extend(results)
                    completed_samples += len(results)
                    pbar.update(len(results))
                except Empty:
                    print("⚠️  Timeout waiting for results")
                    break
        
        # Shutdown workers
        self.gpu_queue.put(None)
        self.cpu_queue.put(None)
        
        return all_results

def run_parallel_inference(model, condition_features):
    """Run parallel inference pipeline."""
    print("="*70)
    print("PARALLEL GPU + CPU MOLECULAR GENERATION")
    print("="*70)
    
    # Initialize parallel inference engine
    engine = ParallelInferenceEngine(model, condition_features)
    
    # Generate molecules
    start_time = time.time()
    results = engine.generate_parallel(config.NUM_SAMPLES, config.TARGET_CONDITIONS)
    end_time = time.time()
    
    # Analyze results
    valid_count = sum(1 for r in results if r['valid'])
    unique_smiles = set(r['smiles'] for r in results if r['valid'])
    
    qed_errors = [r['qed_error'] for r in results if r['valid']]
    mean_qed_error = np.mean(qed_errors) if qed_errors else 1.0
    
    print(f"\n📊 GENERATION RESULTS")
    print(f"   Total samples: {len(results)}")
    print(f"   Valid molecules: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    print(f"   Unique molecules: {len(unique_smiles)} ({len(unique_smiles)/valid_count*100:.1f}%)")
    print(f"   Mean QED error: {mean_qed_error:.3f}")
    print(f"   Generation time: {end_time - start_time:.1f}s")
    print(f"   Speed: {len(results)/(end_time - start_time):.1f} molecules/sec")
    
    # Save results
    output_file = "parallel_generation_results.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")
    
    # Print sample molecules
    print(f"\n🧪 SAMPLE GENERATED MOLECULES:")
    valid_results = [r for r in results if r['valid']][:10]
    for i, result in enumerate(valid_results):
        print(f"   {i+1:2d}. {result['smiles']}")
        print(f"       QED: {result['properties']['qed']:.3f} (target: {result['target']['qed']:.3f})")
        print(f"       MW: {result['properties']['mw']:.1f}")

# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Combined RL Training + Parallel Inference")
    parser.add_argument('--mode', choices=['train', 'inference', 'both'], default='both',
                       help='Mode: train (RL only), inference (generation only), both (train then inference)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to existing model (for inference mode)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("COMBINED RL FINE-TUNING + PARALLEL INFERENCE")
    print("="*70)
    
    # Ensure model directory exists
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    model = None
    condition_features = None
    
    if args.mode in ['train', 'both']:
        print("🚀 Starting RL fine-tuning...")
        
        # Load Stage 2 model for RL training
        if not os.path.exists(config.STAGE2_CHECKPOINT):
            print(f"❌ Stage 2 checkpoint not found: {config.STAGE2_CHECKPOINT}")
            print("   Please complete Stage 2 conditional training first")
            return
        
        model, condition_features = load_model(config.STAGE2_CHECKPOINT)
        
        # Run RL training
        try:
            rl_training_loop(model, condition_features)
        except KeyboardInterrupt:
            print("\n⚠️  RL training interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during RL training: {e}")
            import traceback
            traceback.print_exc()
            return
    
    if args.mode in ['inference', 'both']:
        print("🚀 Starting parallel inference...")
        
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        elif args.mode == 'both':
            # Use the RL final model that was just saved
            model_path = os.path.join(config.MODEL_DIR, 'diffusion_rl_final.pt')
        else:
            # Look for RL final model
            rl_final = os.path.join(config.MODEL_DIR, 'diffusion_rl_final.pt')
            stage2_fallback = config.STAGE2_CHECKPOINT
            
            if os.path.exists(rl_final):
                model_path = rl_final
            elif os.path.exists(stage2_fallback):
                model_path = stage2_fallback
                print("⚠️  Using Stage 2 model (no RL model found)")
            else:
                print("❌ No model found for inference")
                return
        
        # Load model fresh for inference
        model, condition_features = load_model(model_path)
        
        # Run parallel inference
        try:
            run_parallel_inference(model, condition_features)
        except Exception as e:
            print(f"\n❌ Error during parallel inference: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()