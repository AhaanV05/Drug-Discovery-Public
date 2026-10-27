# Stages 3 and 4 — design

**Status:** implemented as notebook CELL 32–34 (§18); both arms trained and evaluated (see `notebooks/results/paper/`). Kept as the design record.
**Goal:** add the two remaining arms so the comparative study has its control and its
weight-optimisation mechanism.

| Stage | Arm | Mechanism | Cell | Output |
|---|---|---|---|---|
| **3** | RL fine-tune of the Stage-2 policy | weight optimisation | 33 + 34 | `stage3_rl/` |
| **4** | filtered-unconditional, `qed > 0.5` | none — train on good data | 32 | `stage4_filtered/` |

Train **Stage 4 first**: cheapest, and the result that most constrains the framing.

---

## Stage 4 — filtered-unconditional

Architecturally identical to Stage 1 (`n_props=1, d=512, nl=6`), so parameter counts match and the
comparison stays clean. `keep` is zero on every step, so the property path only ever sees the learnable
null and the model is genuinely unconditional.

**Decisions**

1. **`n_props=1` trained always-null, not `n_props=0`.** `props_of()` returns `[]` for `n_props <= 0`,
   which would drop the arm out of the steering figures entirely. A visibly **flat** steering curve
   beside Stages 1–2 is a better result than an absent row — it is the picture the paper wants for a
   baseline that cannot steer. It also needs no harness change.
2. **Canonical norm stats.** `stage2_norm_stats.json` is loaded, never recomputed on the filtered
   subset. Recomputing would silently put Stage 4 in a different frame from every other arm.
3. **Token cache sliced by the same boolean mask** as the dataframe — `Stage2Dataset` asserts
   positional alignment, and re-tokenising would break it.
4. **Retained fraction asserted** into a plausible band, so a mis-specified filter fails in seconds
   rather than after an hour of training.

**To state in the paper**

- The harness passes Stage 4 a z-scored QED with the mask set. That is out-of-distribution input for a
  model trained always-null. Its non-response *is* the measurement — say so rather than letting a
  reviewer find it.
- Novelty stays referenced to the **full** training set, not the filtered subset, so every arm is scored
  against the same reference. Worth one sentence, since Stage 4 trained on a different distribution.
- Its `val_realCE` is computed on the filtered val subset and is **not** comparable to Stage 1/2 val CE —
  a different denominator, not a better model. Only harness metrics compare arms.
- Filtering inherently costs training data. That is a property of the method, not a confound to correct.

---

## Stage 3 — RL fine-tune

**Trajectory REINFORCE with a KL anchor**, not reward-weighted regression. RWR distils best-of-N into
the weights, which is the same mechanism as Stage 2 + reranking applied twice; the study's spine is
*search versus weight optimisation*, and RWR collapses it. One-step policy gradient was rejected for a
sharper reason: if it fails you cannot distinguish "RL does not help" from "the estimator was broken",
and this project has already read four artifacts as findings.

**Reward.** `−|z(achieved) − z(requested)|` in z-units against `stage2_norm_stats.json` — the same
moments as the conditioning. Raw units make a QED error of 0.1 and an MW error of 0.1 Da incomparable,
and z-units make extending `REWARD_PROPS` to `mw`/`sa_score` a list edit rather than a rewrite.

**The target is sampled per batch** from the harness's 0.3–0.9 sweep. A fixed target would optimise one
point and stop being a control method — its steering range would read ≈ 0, which looks like catastrophic
failure when it actually means a different objective was optimised. One request shared across the batch
is also what makes the batch-mean baseline valid.

**Invalid molecules stay in the batch at a floor reward.** `gen_mols()` (CELL 25) drops unparseable
outputs; reusing it here would be a survivorship bug — the gradient would only ever see valid molecules,
so the policy gets no pressure to stay parseable, while the logged reward looks healthy because it is
averaged over survivors.

### Three failure modes the implementation guards

1. **`log π` must match the sampling distribution.** The sampler edits logits (mask/BOS bans, `min_len`
   EOS+PAD bans) and divides by temperature. Tracer and replay both call the single helper
   `_rl_logit_rules`; if they diverge, the gradient is biased with no visible symptom.
2. **Dropout stays off during replay** (`policy.eval()`). Sampling ran with dropout off, so replaying
   with it on would make `log π` a different distribution from the one that produced the actions — the
   same bias arriving by a different door.
3. **Invalid-molecule survivorship**, as above.

**Stated approximation.** MaskGIT's reveal *order* (confidence + Gumbel noise + `pad_penalty`) is
treated as environment, not policy; the actions are the token draws. Standard for MaskGIT-RL, and what
makes the trajectory log-prob tractable.

### β is reported as a frontier, not a point

The KL coefficient is this arm's most consequential hyperparameter, and *"RL adds control"* versus
*"RL trades diversity for control"* are different claims that only the frontier distinguishes. Picking
one β silently chooses a point on that curve and calls it the answer — the same error as reporting an
own-best operating point.

CELL 34 runs `BETAS = [0.01, 0.05, 0.15, 0.50]`, each from a **fresh** copy of Stage 2 (β must not
chain), and plots reward against KL, uniqueness and internal diversity. **The frontier is the result**;
the promoted point is just what the harness scores.

### Gate

CELL 33 ends in a ~300-step gate judged on **trend**: mean reward rising, KL bounded, validity holding.
Flat reward with climbing KL means a broken estimator, not a hard problem — do not scale up. Budget
≈ 2 s/step (50 no-grad forwards dominate, then RDKit, then 4 replayed steps with grad), so ≈ 35 min per
β at 1000 steps; four β ≈ 2.3 h.

### Checkpoint metric consistency

`load_arm()` reads `ck['best']` into the `val` column of `models.json`. Stages 1/2 store a val CE
(lower better); storing a reward there would silently mix two metrics in one column — the same class of
error as the top-10/top-50 mislabel. Stage 3 therefore computes a genuine `val_realCE` with the Stage-2
validation shape and stores that as `best`, keeping `best_reward` separate. That also yields the
RL-cost signal for free: how much reconstruction was traded for control.

### Promotion

`find_ckpt()` tries `checkpoints/best.pt` first, then falls back to `**/best.pt`. CELL 34 copies the
chosen run to `stage3_rl/checkpoints/best.pt` and records the choice in `promoted.json`. **If promotion
were skipped, the harness would silently load an arbitrary β from `frontier/`** — hence the loud print
and the recorded provenance.

---

## Integration

Both arms are already registered in CELL 30's `ARMS` list, so a checkpoint is all that is needed.

⚠️ **Adding either arm forces a full re-run of CELL 30** — every arm is re-measured in one pass, so the
committed Stage 1/2 numbers will shift within seed noise. That is by design; `comparison/` is
authoritative.

## Not done here

Training. Both cells are gated and neither has been run — no Stage 3 or Stage 4 checkpoint exists, and
no claim about either arm is supported yet.
