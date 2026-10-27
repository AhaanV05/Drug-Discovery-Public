"""
Parallel GPU + CPU Inference for Maximum Speed Molecule Generation
==================================================================

This script demonstrates parallel inference using both GPU and CPU for maximum
molecular generation throughput.

Architecture:
- GPU: Handles complex diffusion sampling with attention mechanisms
- CPU: Handles property prediction and lightweight post-processing in parallel
- Async coordination: Maximizes utilization of both processors

NOTE: NPU support requires Intel NPU drivers and runtime. Currently optimized
for GPU + CPU parallel processing.

Requirements:
- NVIDIA GPU with CUDA support
- Multi-core CPU for parallel property calculation
- Completed RL model weights

Usage:
    python parallel_gpu_cpu_inference.py --model_path models/diffusion/diffusion_rl_final.pt
"""

import os
import sys
import json
import math
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# For NPU support (conditional imports)
try:
    import openvino as ov
    NPU_AVAILABLE = True
    print("✅ OpenVINO available for NPU inference")
except ImportError:
    NPU_AVAILABLE = False
    print("⚠️  OpenVINO not available - NPU inference disabled")

# RDKit for molecule validation and property computation
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import selfies as sf

# ===================================================================
# CONFIGURATION
# ===================================================================

@dataclass
class InferenceConfig:
    # Model paths
    MODEL_PATH: str = r"models\diffusion\diffusion_rl_final.pt"
    TOKENIZER_PATH: str = r"notebooks\data\selfies_tokenizer.json"
    CONDITION_STATS_PATH: str = r"notebooks\data\condition_stats.json"
    
    # Hardware allocation
    GPU_BATCH_SIZE: int = 32        # Larger batches for GPU
    NPU_BATCH_SIZE: int = 8         # Smaller batches for NPU
    MAX_PARALLEL_REQUESTS: int = 4   # Number of parallel inference threads
    
    # Generation parameters
    NUM_SAMPLES: int = 1000         # Total molecules to generate
    SAMPLE_STEPS: int = 50          # Reduced for faster sampling
    TEMPERATURE: float = 0.8
    CFG_SCALE: float = 3.0
    
    # Target conditions for generation
    TARGET_CONDITIONS: List[Dict] = None
    
    def __post_init__(self):
        if self.TARGET_CONDITIONS is None:
            self.TARGET_CONDITIONS = [
                {"qed": 0.8, "mw": 350, "alogp": 2.5},  # Drug-like
                {"qed": 0.9, "mw": 300, "alogp": 2.0},  # Highly drug-like
                {"qed": 0.7, "mw": 400, "alogp": 3.0},  # Lipophilic
            ]

config = InferenceConfig()

# Device setup
GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU device: {GPU_DEVICE}")

# ===================================================================
# LOAD SHARED RESOURCES
# ===================================================================

# Load tokenizer
print(f"Loading tokenizer from {config.TOKENIZER_PATH}...")
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

# Load condition stats
with open(config.CONDITION_STATS_PATH, 'r') as f:
    cond_stats = json.load(f)

print(f"Loaded tokenizer (vocab: {VOCAB_SIZE}) and condition stats")

# ===================================================================
# MODEL ARCHITECTURE (same as RL script)
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
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
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
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, t, cond=None, cond_mask=None):
        B, L = x.shape
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
# NPU MODEL WRAPPER
# ===================================================================

class CPUPropertyPredictor:
    """CPU-optimized parallel property prediction for molecules."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        print(f"✅ CPU property predictor initialized with {num_workers} workers")
    
    def predict_properties_parallel(self, molecules_batch: List[str]) -> List[Dict]:
        """Parallel property prediction on CPU."""
        if len(molecules_batch) <= 4:
            # Small batch, compute directly
            return [self._compute_single_properties(mol) for mol in molecules_batch]
        
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
    
    def _default_props(self):
        return {'qed': 0.0, 'mw': 0.0, 'alogp': 0.0, 'hbd': 0, 'hba': 0, 'psa': 0.0, 'num_rings': 0, 'sa_score': 5.0}

# ===================================================================
# PARALLEL INFERENCE SYSTEM
# ===================================================================

class ParallelInferenceEngine:
    """Coordinated GPU+NPU inference system."""
    
    def __init__(self, model_path: str):
        # Load GPU model
        print(f"Loading GPU model from {model_path}...")
        self.gpu_model = self._load_gpu_model(model_path)
        
        # Initialize NPU predictor
        self.npu_predictor = NPUPropertyPredictor()
        
        # Diffusion schedule (simplified)
        self.schedule = self._init_schedule()
        
        # Thread coordination
        self.gpu_queue = Queue(maxsize=100)
        self.npu_queue = Queue(maxsize=100)
        self.results_queue = Queue()
        
    def _load_gpu_model(self, model_path: str):
        """Load the RL-trained diffusion model."""
        checkpoint = torch.load(model_path, map_location=GPU_DEVICE)
        
        condition_features = checkpoint.get('condition_features', cond_stats['features'])
        n_conditions = len(condition_features)
        
        model = DiffusionTransformer(
            vocab_size=VOCAB_SIZE,
            max_len=96,
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            dropout=0.1,
            condition_dim=64,
            n_conditions=n_conditions
        ).to(GPU_DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ GPU model loaded (epoch {checkpoint.get('epoch', '?')})")
        return model
    
    def _init_schedule(self):
        """Initialize diffusion schedule."""
        # Simplified schedule for demo
        timesteps = 1000
        betas = torch.linspace(0.0001, 0.058, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod.to(GPU_DEVICE)
    
    def _prepare_conditions(self, target_dict: Dict) -> torch.Tensor:
        """Prepare normalized condition vector."""
        features = cond_stats['features']
        mean = np.array(cond_stats['mean'])
        std = np.array(cond_stats['std'])
        
        values = np.array([target_dict.get(f, 0.0) for f in features])
        normalized = (values - mean) / (std + 1e-8)
        
        return torch.tensor(normalized, dtype=torch.float32, device=GPU_DEVICE)
    
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
                    selfies_str = self._decode_sample(sample)
                    smiles = sf.decoder(selfies_str) if selfies_str else "INVALID"
                    molecules.append(smiles)
                
                # Send to NPU queue for property analysis
                self.npu_queue.put((molecules, target_conditions, request_id))
                
            except Empty:
                continue
            except Exception as e:
                print(f"GPU worker error: {e}")
                continue
    
    def _sample_batch(self, batch_size: int, target_conditions: Dict) -> torch.Tensor:
        """Sample a batch of molecules using diffusion."""
        # Prepare conditions
        cond_vec = self._prepare_conditions(target_conditions)
        cond_batch = cond_vec.unsqueeze(0).expand(batch_size, -1)
        cond_mask = torch.ones(batch_size, dtype=torch.bool, device=GPU_DEVICE)
        
        # Start from noise
        seq_len = 96
        valid_tokens = torch.arange(3, VOCAB_SIZE, device=GPU_DEVICE)  # Skip PAD, BOS, EOS
        x = valid_tokens[torch.randint(0, len(valid_tokens), (batch_size, seq_len))]
        x[:, 0] = BOS_IDX  # Start with BOS
        
        # Simplified reverse diffusion (faster sampling)
        timesteps = torch.linspace(999, 0, config.SAMPLE_STEPS, dtype=torch.long, device=GPU_DEVICE)
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t.item(), device=GPU_DEVICE, dtype=torch.long)
            
            # CFG sampling
            logits_c, _ = self.gpu_model(x, t_batch, cond_batch, cond_mask)
            cond_mask_u = torch.zeros_like(cond_batch, dtype=torch.bool)
            logits_u, _ = self.gpu_model(x, t_batch, cond_batch, cond_mask_u)
            
            # Apply CFG
            logits = logits_u + config.CFG_SCALE * (logits_c - logits_u)
            
            # Temperature and sampling
            logits = logits / config.TEMPERATURE
            logits[:, :, PAD_IDX] = -1e9
            logits[:, :, BOS_IDX] = -1e9
            logits[:, :, EOS_IDX] = -1e9
            
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(batch_size, -1)
            
            # Preserve padding
            pad_mask = (x == PAD_IDX)
            x = torch.where(pad_mask, x, x)
        
        return x
    
    def _decode_sample(self, sample: torch.Tensor) -> str:
        """Decode token IDs to SELFIES."""
        tokens = [itos[i.item()] for i in sample]
        tokens = [t for t in tokens if t not in (PAD, BOS, EOS)]
        return "".join(tokens)
    
    def _npu_property_worker(self):
        """NPU worker for property prediction."""
        while True:
            try:
                request = self.npu_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break
                
                molecules, target_conditions, request_id = request
                
                # Batch property prediction on NPU
                properties = self.npu_predictor.predict_properties(molecules)
                
                # Compile results
                results = []
                for mol, props in zip(molecules, properties):
                    result = {
                        'smiles': mol,
                        'valid': mol != "INVALID" and Chem.MolFromSmiles(mol) is not None,
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
                print(f"NPU worker error: {e}")
                continue
    
    def generate_parallel(self, num_samples: int, target_conditions_list: List[Dict]) -> List[Dict]:
        """Generate molecules using parallel GPU+NPU inference."""
        print(f"\n🚀 Starting parallel generation: {num_samples} samples")
        print(f"   GPU batch size: {config.GPU_BATCH_SIZE}")
        print(f"   NPU batch size: {config.NPU_BATCH_SIZE}")
        print(f"   Parallel threads: {config.MAX_PARALLEL_REQUESTS}")
        
        # Start workers
        gpu_thread = threading.Thread(target=self._gpu_diffusion_worker)
        npu_thread = threading.Thread(target=self._npu_property_worker)
        
        gpu_thread.daemon = True
        npu_thread.daemon = True
        
        gpu_thread.start()
        npu_thread.start()
        
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
        self.npu_queue.put(None)
        
        return all_results

# ===================================================================
# MAIN EXECUTION
# ===================================================================

def main():
    print("="*70)
    print("PARALLEL GPU + NPU MOLECULAR GENERATION")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ Model not found: {config.MODEL_PATH}")
        print("   Please complete RL training first or adjust MODEL_PATH")
        return
    
    # Initialize parallel inference engine
    engine = ParallelInferenceEngine(config.MODEL_PATH)
    
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


if __name__ == "__main__":
    main()