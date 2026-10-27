# Controlling Molecular Properties in Diffusion Models: a Four-Arm Comparison

A comparative study of **which strategy best controls the properties of generated molecules**.
Four strategies are trained on the same 200k ChEMBL dataset, evaluated under one harness at one
operating point (5 seeds), and judged against a filtered-unconditional baseline that uses no control
at all.

All four models are the same backbone, **CondDiT**: a discrete (masked) diffusion transformer over
[SELFIES](https://github.com/aspuru-guzik-group/selfies), DiT-style with adaLN-Zero conditioning,
a learnable null embedding and classifier-free guidance (d=512, 6 layers, 8 heads, ~30.7M parameters).

## The four arms

| Stage | Arm | Mechanism |
|---|---|---|
| **1** | Conditional diffusion, **1 property** (QED) | conditioning |
| **2** | Conditional diffusion, **9 properties** + per-property dropout | conditioning |
| **3** | RL fine-tune (REINFORCE + KL anchor) of the Stage-2 policy | weight optimisation |
| **4** | Filtered-unconditional (trained only on the QED > 0.5 subset) | none: train on good data |

The nine conditioning properties are `qed, mw, hbd, hba, psa, num_rings, sa_score, hERG, pchembl_mean`.
Stages 1 and 4 are architecturally identical (`n_props=1`; Stage 4 only ever sees the null embedding);
Stages 2 and 3 share the 9-property architecture. Parameter counts differ by exactly the width of the
extra property projections, so the conditioning-width comparison varies one thing.

## Results

Mean ± s.d. over 5 seeds (42-46). Sampler: `cfg=1.0, temp=0.4, noise=0.5, pad_penalty=1.0,
steps=50, min_len=12` for every arm.

| | QED steering range | joint target, top-50 (lower better) | unconditional mean QED |
|---|---:|---:|---:|
| **Stage 1**: 1 property | **0.682 ± 0.011** | 0.247 ± 0.012 | 0.692 |
| **Stage 2**: 9 properties | 0.555 ± 0.004 | **0.134 ± 0.009** | 0.712 |
| **Stage 3**: RL | 0.546 ± 0.005 | 0.140 ± 0.005 | 0.687 |
| **Stage 4**: filtered | 0.577 ± 0.010 ⚠️ | 1.304 ± 0.042 | **0.772** |

1. **Conditioning width matches task width.** Narrow conditioning steers its single axis harder
   (11.2× pooled σ, exact rank p = 0.0040); broad conditioning is required to express a
   multi-property request at all (7.6× pooled σ on the joint target). Neither is "better"; they answer
   different requests.
2. **Training on good data raises the floor and buys no control.** Stage 4 has the best unconditional
   QED of any arm and is ~10× worse on the joint target.
3. **RL did not add control beyond conditioning.** Stage 3 vs Stage 2 separates on neither metric.
   This null is **bounded**: it holds at `lr=1e-5`, 1000 steps, with a KL anchor. It does not show that
   RL cannot help.

⚠️ **Stage 4's steering range (0.577) is not steering.** It is the spread of a non-monotonic response to
out-of-distribution conditioning (the model was trained always-null). Quote the full curve, never the
range alone. See [PAPER_EVIDENCE_MAP.md](PAPER_EVIDENCE_MAP.md).

**Limitations.** `hERG` and `pchembl_mean` are trained into Stages 2 and 3 but cannot be scored on novel
molecules (no live predictor), so all evaluation uses `qed`, `mw` and `sa_score`. The operating point
runs at `cfg=1.0`, i.e. guidance amplification is off.

## Repository layout

```
paper/                      LaTeX manuscript (Related Work is still a stub)
PAPER_EVIDENCE_MAP.md       every claim mapped to the committed artifact behind it
notebooks/
  dit_conditional_generation.ipynb   the live notebook: model, training, sampler, all four arms
  generate_results.py                the results harness (mirrored as notebook CELL 30)
  train_stage1.py                    Stage-1 training script
  data/                              the 200k dataset, tokenizer and vocab (Git LFS)
  results/paper/                     the authoritative four-arm results package
  results/README.md                  how to read the package
data/  scripts/  database/  Dataset construction from ChEMBL (see below)
configs/                    early-plan YAML configs (not read by the notebook)
docs/design/                design record for the Stage 3 and Stage 4 cells
archive/                    superseded code, notebooks, results and docs (see archive/README.md)
```

## Reproducing

**Training and evaluation run in Google Colab on a GPU** (developed on an A100; Drive mounted for
checkpoints). The checkpoints are not in this repository.

1. Open `notebooks/dit_conditional_generation.ipynb` on a GPU runtime. Check `print(device)` after
   CELL 15: on CPU the evaluation silently runs for hours.
2. Run CELL 1-15 (data, model, setup), then CELL 17 (sampler).
3. Train each arm: Stage 1 (CELL 31), Stage 2 (CELL 22-24), Stage 4 (CELL 32), Stage 3 (CELL 33-34).
4. Run CELL 30 (`notebooks/generate_results.py`) to regenerate `notebooks/results/paper/`. One
   measurement pass produces the per-stage folders and the `comparison/` tables together.

Install the Python dependencies with `pip install -r requirements.txt`. The SA score uses RDKit's
`sascorer` contrib module. Experiment tracking is optional and uses Weights & Biases.

### Dataset construction

The training set is 200k diverse drug-like molecules selected from ChEMBL 34 (Tanimoto < 0.75),
with RDKit-derived features, ADMETlab endpoints and pChEMBL values (21.1% coverage).
The processed files are in `notebooks/data/`. To rebuild from scratch:

```bash
# ChEMBL 34 SQLite dump (~4.5 GB)
aria2c -c -x 16 -s 16 -k 1M https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/chembl_34_sqlite.tar.gz
tar -xzf chembl_34_sqlite.tar.gz
python scripts/chembl_sqlite_loader.py --db_path chembl_34/chembl_34_sqlite/chembl_34.db --n_molecules 200000
python data/compute_features.py        # QED, SA, ...
python data/extract_bioactivity.py     # pChEMBL values
python data/merge_all_features.py      # merge with ADMETlab results
python scripts/selfies_conversion_pipeline.py
```

The scripts in `data/` were written incrementally and read intermediate files from
`chembl_curated_out/`; check each script's header for its inputs before running. The optional
PostgreSQL/Supabase ingestion (`database/`, `scripts/ingest_to_supabase.py`) needs credentials in a
local `.env` (never committed) and is not required for the experiments.

## Data and licensing

The dataset derives from ChEMBL (CC BY-SA 3.0) and ADMETlab predictions; check the upstream terms before redistributing the files in `notebooks/data/`.
