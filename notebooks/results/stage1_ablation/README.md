# Stage 1 Ablation — QED-only (1-prop) vs Multi-property (11-prop)

**Research question:** Does narrowing the conditioning vector from 11 properties to a
single property (QED) sharpen native QED steering? (i.e. is the "property-dilution"
hypothesis true?)

**Headline finding (to be confirmed by the numbers this folder will hold):**
narrowing to QED-only does **not** improve native steering. Property control is a *weak
prior* regardless of how many properties are conditioned on; strong control comes from
**best-of-N reranking** (and, in Stage 3, RL). This is reported as a negative ablation.

---

## How to generate everything

1. Open `notebooks/dit_conditional_generation.ipynb` in Colab (Drive mounted, GPU on).
2. Run the data + model + setup cells (1–15) and the sampler cell (17) so `tokenizer`,
   `sample`, `device`, `config`, `QMEAN/QSTD`, `PAD/BOS/EOS/MASKi`, `train_df`, and the
   `DiTBlock/TimestepEmbed/modulate` building blocks exist.
3. Paste and run `generate_stage1_ablation.py` (also pasted as a notebook cell in chat).
4. It writes everything to `config.model_dir/stage1_ablation/` on Drive.
5. Download that folder and drop its contents here (into `results/stage1_ablation/`).

The script auto-detects both checkpoints' architectures from their state_dicts, so it
loads the **wide-MLP QED-only** model and the **narrow-MLP 11-prop** model without manual
config. Both are evaluated with the **identical** matched sampler
(`cfg=1.0, temperature=0.4, noise=0.0, steps=50`).

---

## Folder structure (populated by the script)

```
stage1_ablation/
  config/                               EVERYTHING needed to reproduce
    model_params.json                   both models: full arch, param counts, ckpt metadata,
                                          training settings (cfg-dropout, lr, batch, epoch, file size)
    run_config.json                     seed, sampler settings, targets, N's, per-target QED z-scores
    norm_stats.json                     exact per-property normalization (mean/std) used for conditioning
    environment.json                    python / torch / rdkit / numpy versions, GPU, CUDA, git commit
  csv/
    native_steering_qedonly.csv         per-target summary (1-prop model)
    native_steering_multiprop.csv       per-target summary (11-prop model)
    native_steering_permol_qedonly.csv  every generated molecule (qed, ha, smiles)
    native_steering_permol_multiprop.csv
    bestofN_qedonly_qed0.9.csv          ranked best-of-N at target 0.9 (1-prop)
    bestofN_multiprop_qed0.9.csv        ranked best-of-N at target 0.9 (11-prop)
    comparison_by_target.csv            *** side-by-side: 1-prop vs 11-prop + delta, per target
    qc_summary.csv                      validity/uniqueness/novelty/SA/diversity, both models
  figure_data/                          the exact numbers behind every figure (one CSV per figure)
    fig1_meanQED_vs_target.csv  fig2_meanHA_vs_target.csv  fig3_frac_gt07.csv
    fig4_bestofN_qed.csv        fig5_qed_by_target.csv
  figures/                              300-dpi PNGs, paper-ready
    fig1_native_steering_meanQED.png    meanQED vs target, both models (+/- std)
    fig2_size_vs_target_meanHA.png      meanHA vs target (the "size-proxy" finding)
    fig3_frac_above_0.7.png             frac QED>0.7 vs target, both models
    fig4_bestofN_distribution.png       QED histogram of N samples at target 0.9
    fig5_qed_boxplot_by_target.png      QED distribution per target, both models
  tables/
    summary_table.csv / .md             one-row-per-(model,target) master table
    comparison_table.md                 *** side-by-side + headline comparison (paste into paper)
    headline_comparison.csv             steering spread, max meanQED, best-of-N per model
  manifest.json                         consolidated: run config + env + model summaries + QC + caveat
```

**Reproducibility:** the run is seeded (`SEED=42`), so re-running the script on the same
checkpoints reproduces every number and figure exactly. `config/` captures both models'
full parameters, the sampler settings, the normalization, and the software environment.

---

## Methodology notes (for the paper + honest caveats)

- **Native steering** sets QED to its z-scored target at conditioning index 0; for the
  11-prop model the other 10 properties are held at their dataset mean (z-score 0), so the
  probe isolates QED control.
- **Best-of-N**: generate N, rank by RDKit QED, keep top-K. Identical protocol for both.
- **Metrics**: validity (RDKit-parseable), uniqueness (distinct canonical SMILES),
  novelty (fraction not in the training set), SA score (synthesizability, lower=easier),
  internal diversity (mean pairwise 1−Tanimoto on Morgan fps), mean heavy atoms.
- **Checkpoint provenance (11-prop model):** `cond_dit_best.pt` was saved with only
  `{model, n_props}` — no epoch/val/optimizer. Its **architecture is authoritative** (inferred
  directly from the weights: n_props=11, d=512, 6 layers, 8 heads, narrow MLP, 29.88M params).
  Its **training metadata is documented from project history, not stored**: cfg-dropout 0.1,
  lr 3e-4, batch 64, **~epoch 40** (labelled "ep40" in `best_of_n_baseline_qed0.9.csv`; exact
  epoch/val not recoverable). The QED-only checkpoint, by contrast, stores epoch (86) and val
  (1.298) authoritatively.
- **Confound (state honestly):** the two checkpoints were trained at *different budgets*
  (QED-only: ~120 ep, wide property-MLP, cfg-dropout 0.3; multi-prop: ~40 ep, narrow MLP,
  cfg-dropout 0.1). So this is an **"as-trained" comparison**, not a perfectly controlled
  one. The qualitative conclusion (both steer weakly; reranking carries both) is robust to
  this confound. For a *bulletproof* controlled claim, re-train both at identical settings
  — recommended as a cheap follow-up if a reviewer pushes.
