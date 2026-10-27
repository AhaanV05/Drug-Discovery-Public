# RL-Enhanced Molecular Diffusion for Drug Discovery

A research framework for AI-driven drug candidate generation. A discrete diffusion model (D3PM) trained on 200k ChEMBL molecules is steered toward drug-like, synthesizable, target-active compounds via reinforcement learning (PPO) and classifier-free guidance. The project includes a complete data pipeline, two-stage training curriculum, ablation experiments, and a diagnostic study of why native conditioning underperforms and what RL fixes.

## What this project does

Drug discovery is bottlenecked by the enormous size of chemical space (~10⁶⁰ molecules) and the cost of experimental screening. This project trains a generative model to **propose novel drug-like molecules on demand** by combining three ideas:

1. **Discrete diffusion (D3PM) on SELFIES strings** — learns the distribution of 200k real drug-like molecules from ChEMBL, represented as SELFIES sequences that are guaranteed to be chemically valid by construction.
2. **Two-stage conditioning** — Stage 1 trains an unconditional prior on 200k molecules; Stage 2 fine-tunes with pChEMBL bioactivity values as a conditioning signal, so the model learns to generate molecules likely to bind specific protein targets.
3. **Reinforcement learning fine-tuning (PPO)** — a multi-objective reward (QED drug-likeness + synthetic accessibility + predicted binding + novelty) is used to steer the pretrained model toward the Pareto frontier of all objectives simultaneously. This is the empirically motivated fix for weak native conditioning: the diagnostic study in `notebooks/results/steering_investigation/` shows that on-demand QED control via conditioning alone achieves a slope of ~0.34 (versus ideal 1.0), and RL is the mechanism that closes that gap.

The ablation study in `notebooks/results/stage1_ablation/` quantifies the contribution of each component: best-of-N baseline, native steering (QED-only vs multi-property), and the full RL-enhanced pipeline.

## Repository layout

```
.
├── configs/
│   ├── model_config.yaml       # Diffusion + VAE architecture/training hyperparameters
│   ├── rl_config.yaml          # PPO hyperparameters and reward function weights
│   └── docking_config.yaml     # Docking validation config (Week 5+)
│
├── data/                       # Data collection and feature engineering
│   ├── extract_molecules.py
│   ├── compute_features.py     # QED, SA, ADMET predictions
│   ├── extract_bioactivity.py
│   ├── merge_*.py
│   └── create_*.py
│
├── database/
│   ├── setup_database.py       # Automated schema creation
│   └── setup_supabase_schema.sql
│
├── scripts/
│   ├── chembl_sqlite_loader.py       # Fetch 200k diverse molecules from ChEMBL SQLite
│   ├── selfies_conversion_pipeline.py
│   ├── strong_conditional_diffusion.py
│   ├── rl_finetune_diffusion.py      # PPO fine-tuning loop
│   ├── combined_rl_inference.py
│   └── parallel_gpu_npu_inference.py
│
├── notebooks/
│   ├── d3pm_ultimate_conditional_diffusion_ORGANIZED.ipynb  # Main training notebook
│   ├── dit_conditional_generation.ipynb                     # DiT variant
│   ├── steering_investigation.ipynb                         # Conditioning analysis
│   ├── results/
│   │   ├── stage1_ablation/     # Full ablation: figures, tables, CSVs, configs
│   │   └── steering_investigation/  # Diagnostic findings and figures
│   └── Old_Notebooks/           # Earlier iterations kept for reference
│
├── requirements.txt
├── .env.example                 # Template — copy to .env and fill in your keys
└── .gitignore
```

## Quick start

### 1. Install dependencies

```bash
python -m venv dd_project
# Windows:
dd_project\Scripts\activate
# Linux/Mac:
source dd_project/bin/activate

pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
# Edit .env and fill in your WandB API key (and Supabase credentials if using the DB)
```

Get your WandB key at https://wandb.ai/authorize.

### 3. Fetch the molecule dataset

**Recommended — ChEMBL SQLite (~4.5 GB download, ~2 hours total):**

```bash
# Download ChEMBL 34
aria2c -c -x 16 -s 16 -k 1M https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_sqlite.tar.gz
tar -xzf chembl_34_sqlite.tar.gz

# Extract 200k diverse drug-like molecules
python scripts/chembl_sqlite_loader.py \
    --db_path chembl_34/chembl_34_sqlite/chembl_34.db \
    --n_molecules 200000
```

**Alternative — ChEMBL REST API (slower, ~14 hours for 100k):**

```bash
python scripts/auto_fetch_molecules.py
```

### 4. Build the feature-engineered dataset

```bash
python data/compute_features.py          # QED, SA, ADMET (~30 min)
python data/extract_bioactivity.py       # pChEMBL values for ~42k molecules
python data/merge_bioactivity_with_rl_dataset.py
```

### 5. (Optional) Ingest to database

```bash
python scripts/ingest_to_supabase.py --target supabase   # or --target local
```

### 6. Train

Open `notebooks/d3pm_ultimate_conditional_diffusion_ORGANIZED.ipynb` in Jupyter or Colab and run top-to-bottom. The notebook handles Stage 1 (unconditional, 200k) and Stage 2 (conditional, 42k) training, WandB logging, and checkpoint management.

## Dataset

| Split | Size | Description |
|-------|------|-------------|
| Full pre-training corpus | 200k | Diverse drug-like ChEMBL molecules (MW 150–500, LogP ≤ 5, Tanimoto < 0.75) |
| Bioactivity subset | ~42k (21.1%) | Molecules with pChEMBL values; used for Stage 2 conditioning |

## Key results (Stage 1 ablation)

| Model | Target QED | Mean QED | Frac > 0.7 | Uniqueness |
|-------|-----------|----------|------------|------------|
| Best-of-N (QED-only) | 0.9 | 0.463 | 9.8% | 82.4% |
| Native steering (QED-only) | 0.9 | 0.46 | 9.8% | 82.4% |
| Native steering (Multi-prop) | 0.9 | 0.363 | 2.0% | 7.0% |

The steering investigation (`notebooks/results/steering_investigation/findings.md`) shows that conditioning achieves a control slope of ~0.34 vs the ideal 1.0, and that the model has *capacity* for drug-like output (best-of-N reaches QED 0.89) but cannot summon it on demand — the empirical motivation for RL fine-tuning as the steering engine.

## Configuration

All hyperparameters live in `configs/`:

- `model_config.yaml` — architecture (hidden dim, layers, heads), diffusion schedule, training (batch size, LR, epochs)
- `rl_config.yaml` — PPO (clip ε, GAE λ, entropy coefficient), reward weights (QED 30%, SA 20%, binding 30%, toxicity 10%, novelty 10%)
- `docking_config.yaml` — docking validation settings

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `WANDB_API_KEY` | Yes (for logging) | WandB key from https://wandb.ai/authorize |
| `SUPABASE_HOST` | Only if using DB | Supabase project host |
| `SUPABASE_PORT` | Only if using DB | Default: 5432 |
| `SUPABASE_DATABASE` | Only if using DB | Default: postgres |
| `SUPABASE_USER` | Only if using DB | Database user |
| `SUPABASE_PASSWORD` | Only if using DB | Database password |
