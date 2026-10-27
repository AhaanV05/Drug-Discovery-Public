# Why conditioning does not steer QED — failure-mode analysis

**One-line answer:** mapping QED to SELFIES/SMILES is **not impossible**. The same model already emits
QED 0.887 under best-of-N, so drug-like molecules are in its output
distribution. What is weak is **on-demand control**, and this analysis quantifies why.

Every number below is derived (read-only) from the saved Stage-1 ablation package — **no model was run**.

**Provenance of the analysed data:** seed 42, MaskGIT sampler
cfg=1.0 / temp=0.4 / noise=0.0 / steps=50,
256 molecules per target per model. Note **cfg=1.0 = no
classifier-free-guidance amplification** — steering was measured on the raw conditional.

## Headline numbers

| metric | QED-only | Multi-prop |
|---|---:|---:|
| steering spread (max-min mean QED) | 0.231 | 0.128 |
| slope: achieved mean QED vs target (ideal 1.0) | 0.343 | -0.033 |
| corr(target, QED) | +0.441 | -0.042 |
| partial corr(target, QED given size) | +0.302 | +0.119 |
| corr(target, size) — the size lever | -0.382 | -0.673 |
| generated mean QED (data prior = 0.66) | 0.357 | 0.442 |
| best-of-N pool top QED | 0.887 | 0.867 |
| best-of-N top-50 median QED | 0.746 | 0.371 |

## Findings

**1. Steering is weak and asymmetric.** Achieved mean QED tracks the target with slope
0.34 (QED-only) — far below the ideal 1.0 — while the
Multi-prop model has essentially zero net response (slope -0.03,
corr -0.04). The low end is roughly hit
(QED-only @0.3 -> 0.27); the high end is badly missed
(@0.9 -> 0.46).

**2. Two gaps, not one.** The conditional distribution *does* shift (Cohen d[0.9 vs 0.3] =
1.17 for QED-only), but it is (a) **compressed** — sliding only
~0.03 QED per 0.1 of requested change — and (b) **anchored low**:
generated QED averages 0.36, well below the training prior
0.66. So there is a *generation-quality* gap on top of the *control* gap.

**3. It leans on a molecular-size crutch.** For QED-only, ~37%
of the (weak) target->QED effect is mediated by molecule size, though real structure-level control remains
(partial corr +0.30). The Multi-prop model drives size
even harder (corr(target,size) -0.67) and **over-shrinks into fragments
at high targets**, cancelling its QED gains — net effect ~0.

**4. It moves several QED components, not just size.** RDKit decomposition of the generated molecules
(parse rate 100%) shows the QED-only model shifts a *broad* set of Bickerton
components with target — strongest: HBA -0.75, PSA -0.71, ALERTS -0.55, ALOGP +0.48, AROM +0.41. So the model learned a genuine but weak
multi-lever QED response; the limitation is **magnitude**, not absence.

**5. Capacity exists; only control is missing.** The same model reaches QED
0.89 (pool top) / 0.75
(top-50 median) under best-of-N, versus a native mean of 0.46 at
target 0.9. The drug-like molecules are *in the distribution* — conditioning just does not summon them on demand.

## Conclusion

Conditioning-via-denoising trains the property as a near-redundant side-channel of a reconstruction loss,
so the model develops only a weak, partly size-mediated control knob — and it was measured with guidance
off (cfg=1.0). The mapping is real and present in the model; the fix is any mechanism that supplies a
**direct property gradient**: RL fine-tuning (Phase 3), classifier/predictor guidance at sampling, or
selection (best-of-N). This is the empirical motivation for making RL the steering engine.

## Figures
- `figures/fig1_steering_response.png` — achieved vs requested QED, plus the residual.
- `figures/fig2_prior_regression.png` — per-target QED distributions vs the data prior.
- `figures/fig3_size_proxy.png` — QED-vs-size, the size lever, and the Multi-prop fragment collapse.
- `figures/fig4_qed_components.png` — which QED desirability components respond to target.
- `figures/fig5_capacity_vs_control.png` — best-of-N capacity vs native on-demand control.
