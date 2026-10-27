# =============================================================================
# RESULTS PACKAGE — controllable molecular generation, comparative study
#
#   Stage 1  conditional diffusion, single property (QED)
#   Stage 2  conditional diffusion, 9 properties + per-property dropout
#   Stage 3  RL fine-tune of the Stage-2 policy          (auto-detected)
#   Stage 4  filtered-unconditional baseline, QED > 0.5  (auto-detected)
#
# Untrained stages are skipped. Register a new arm with one line in ARMS and
# every table, figure and statistic below picks it up.
#
# OUTPUT LAYOUT
#   results/
#     stage1_qedonly/   csv · figures · tables · molecules   <- this arm alone
#     stage2_multiprop/ ...
#     stage3_rl/        ...
#     stage4_filtered/  ...
#     comparison/       csv · figures · tables               <- all arms together
#     config/           models · norm_stats · environment
#     manifest.json
#
# ONE MEASUREMENT PASS, TWO VIEWS. Per-stage and comparison outputs are written
# from the same generated molecules, so a per-stage number can never disagree
# with the comparison table. Do not regenerate one without the other.
#
# Every comparative number is measured over N_SEEDS draws and reported as
# mean +/- s.d. A single-draw arm cannot be compared against a replicated one.
#
# RUNTIME  ~13 min for 2 arms at N_SEEDS=5 on an A100. GPU required.
# =============================================================================
import os, sys, json, time, itertools
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------- config ----
SEED    = 42
N_SEEDS = 5
SEEDS   = [SEED + i for i in range(N_SEEDS)]

# Operating point, identical for every arm. cfg=1.0 => guidance amplification
# is OFF: all reported control comes from the raw conditional.
CFG, TEMP, NOISE, STEPS, MIN_LEN, PAD_PEN = 1.0, 0.4, 0.5, 50, 12, 1.0

N_STEER, N_JOINT, N_QC = 256, 512, 512
QED_TARGETS  = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MW_TARGETS   = [250, 300, 350, 400, 450]
JOINT_TARGET = {'qed': 0.8, 'mw': 350.0, 'sa_score': 2.5}
PROPS9 = ['qed','mw','hbd','hba','psa','num_rings','sa_score','hERG','pchembl_mean']

# (label, folder under config.model_dir). The checkpoint is located inside the
# folder whatever its internal layout -- see find_ckpt.
ARMS = [
    ('Stage 1 (QED-only)',   'stage1_qedonly'),
    ('Stage 2 (9-property)', 'stage2_multiprop'),
    ('Stage 3 (RL)',         'stage3_rl'),
    ('Stage 4 (filtered)',   'stage4_filtered'),
]

ROOT = config.model_dir / 'results'
CMP  = ROOT / 'comparison'
for sub in ('csv','figures','tables'):
    (CMP/sub).mkdir(parents=True, exist_ok=True)
(ROOT/'config').mkdir(parents=True, exist_ok=True)
print(f"-> writing to {ROOT}\n")
t_start = time.time()

# --------------------------------------------------------------- scorers ----
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    sa_of = lambda m: sascorer.calculateScore(m)
except Exception:
    sa_of = lambda m: float('nan')

S2_NORM = json.load(open(config.model_dir/'stage2_norm_stats.json'))

# ------------------------------------------------------------ model load ----
# PREFLIGHT. Printed before anything else so that a failure below is
# self-diagnosing rather than requiring a separate investigation.
print("model_dir:", config.model_dir)
if not config.model_dir.exists():
    raise SystemExit(
        "model_dir does not exist. Drive is probably not mounted:\n"
        "    from google.colab import drive; drive.mount('/content/drive')")
print("contents:", sorted(x.name for x in config.model_dir.iterdir()), "\n")

def find_ckpt(folder):
    """Locate an arm's checkpoint regardless of the folder's internal layout.

    Renaming folders in the Drive web UI does not always preserve the inner
    structure, so try the expected location first, then the folder root, then
    anywhere beneath it. Returns (path, note) or (None, why-not).
    """
    base = config.model_dir / folder
    if not base.exists():
        return None, "folder not present"
    for pat in ('checkpoints/best.pt', 'best.pt', '**/best.pt', '**/*.pt'):
        hits = sorted(base.glob(pat))
        if hits:
            rel = hits[0].relative_to(base)
            return hits[0], ('' if str(rel) == 'checkpoints/best.pt' else f' [{rel}]')
    inner = sorted(x.name for x in base.iterdir())[:6]
    return None, f"no .pt inside; folder holds {inner}"

def load_arm(full):
    ck = torch.load(full, map_location=device)
    sd = ck['model'] if 'model' in ck else ck
    n_in = sd['pm.0.weight'].shape[1] if 'pm.0.weight' in sd else 1
    m = CondDiT(tokenizer.vocab_size, 128, n_props=max(n_in,1), d=512, nh=8, nl=6,
                T=100, pad_id=PAD, bos_id=BOS, eos_id=EOS, mask_id=MASKi).to(device)
    m.load_state_dict(sd, strict=False); m.eval()
    m.n_props = n_in
    m.meta = dict(epoch=ck.get('epoch','n/s'), val=ck.get('best','n/s'),
                  n_parameters=sum(p.numel() for p in m.parameters()))
    return m

MODELS, FOLDER, META, MISSING = {}, {}, {}, []
print("arms:")
for label, folder in ARMS:
    ckpt, note = find_ckpt(folder)
    if ckpt is None:
        MISSING.append((label, folder, note))
        print(f"   -  {label:24s} {folder}/ -- {note}"); continue
    m = load_arm(ckpt)
    MODELS[label] = m; FOLDER[label] = folder
    META[label] = dict(tag=label, folder=folder,
                       path=str(ckpt.relative_to(config.model_dir)),
                       n_props=m.n_props, **m.meta)
    for sub in ('csv','figures','tables','molecules'):
        (ROOT/folder/sub).mkdir(parents=True, exist_ok=True)
    print(f"   ok {label:24s} n_props={m.n_props} "
          f"params={m.meta['n_parameters']:,} epoch={m.meta['epoch']}{note}")
# A comparison needs at least two arms. Loading only one is not a smaller run --
# it is a different, useless one, and it wastes ~10 GPU-minutes before you notice.
# Colab's Drive mount frequently serves a stale directory listing after folders are
# renamed in the web UI, which is the usual cause; remount and re-run.
if len(MODELS) < 2:
    raise SystemExit(
        "\nSTOPPING: only %d arm(s) loaded -- a comparison needs at least 2.\n\n"
        "Not loaded:\n  %s\n\n"
        "If a folder above appears in the 'contents:' listing printed at the top,\n"
        "the Drive mount is stale -- run:\n"
        "    from google.colab import drive\n"
        "    drive.flush_and_unmount(); drive.mount('/content/drive')\n"
        "and re-run. If it is absent, the folder is named differently on Drive;\n"
        "fix the name in ARMS.\n"
        % (len(MODELS), "\n  ".join(f"{l}  ({f}/): {n}" for l, f, n in MISSING)))
if MISSING:
    print("\n   note: %d arm(s) not yet trained -- %s"
          % (len(MISSING), ", ".join(l for l, _, _ in MISSING)))
json.dump(META, open(ROOT/'config'/'models.json','w'), indent=1, default=str)

# --------------------------------------------------------------- helpers ----
def props_of(model):
    """Which properties this arm's conditioning vector can express."""
    if model.n_props <= 0:  return []
    return ['qed'] if model.n_props == 1 else PROPS9[:model.n_props]

def cond_for(model, targets):
    n = max(model.n_props, 1)
    P, M = torch.zeros(1, n, device=device), torch.zeros(1, n, device=device)
    for j, c in enumerate(props_of(model)):
        if c in targets:
            P[0, j] = (float(targets[c]) - S2_NORM[c]['mean']) / S2_NORM[c]['std']
            M[0, j] = 1.0
    return P, M

def gen(model, targets, n, noise=NOISE, pad_penalty=PAD_PEN, temp=TEMP):
    """Generate and describe. Returns a molecule-level DataFrame."""
    pv, cmv = cond_for(model, targets)
    tk = sample(model, pv, n=n, steps=STEPS, cfg=CFG, temperature=temp, noise=noise,
                min_len=MIN_LEN, cmask=cmv, pad_penalty=pad_penalty)
    rows = []
    for t in tk:
        smi = tokenizer.decode(t.cpu().tolist())
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None: continue
        rows.append(dict(smiles=Chem.MolToSmiles(mol), qed=QED.qed(mol),
                         mw=Descriptors.MolWt(mol), sa=sa_of(mol),
                         ha=mol.GetNumHeavyAtoms()))
    d = pd.DataFrame(rows); d.attrs['n_requested'] = n
    return d

def joint_rank(df, target=JOINT_TARGET):
    parts = [(df[c] - target[k]).abs() / S2_NORM[k]['std']
             for k, c in (('qed','qed'), ('mw','mw'), ('sa_score','sa'))]
    d = df.copy(); d['dist'] = np.mean(parts, axis=0)
    return d.sort_values('dist').reset_index(drop=True)

train_smiles = set(train_df.smiles)
def qc_of(df):
    smi = df.smiles.tolist()
    if not smi: return {}
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024)
           for s in smi[:256]]
    sims = [DataStructs.TanimotoSimilarity(a, b) for a, b in itertools.combinations(fps, 2)]
    return dict(uniqueness=len(set(smi))/len(smi),
                novelty=sum(s not in train_smiles for s in smi)/len(smi),
                meanQED=df.qed.mean(), meanSA=df.sa.mean(),
                frac_SA_lt45=(df.sa < 4.5).mean(), meanMW=df.mw.mean(),
                meanHA=df.ha.mean(),
                internal_diversity=1 - np.mean(sims) if sims else np.nan)

def agg(df, keys, cols):
    g = df.groupby(keys)[cols].agg(['mean','std']).reset_index()
    g.columns = keys + [f'{a}_{b}' for a, b in g.columns[len(keys):]]
    return g

def pm(g, cols, keep):
    d = g[keep].copy()
    for c in cols:
        d[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(g[f'{c}_mean'], g[f'{c}_std'])]
    return d

def md(path, title, frame, note=''):
    path.write_text(f"## {title}\n\n" + (note + "\n\n" if note else "")
                    + frame.to_markdown(index=False) + "\n", encoding='utf-8')

# Molecule-level output is kept for the reference seed only: enough to inspect
# and to draw example structures, without writing 40k rows per section.
KEEP = {}

# ================================ MEASURE ===================================
print(f"\n[1/5] QED targeting ({N_SEEDS} seeds)")
rows = []
for sd_ in SEEDS:
    torch.manual_seed(sd_); np.random.seed(sd_)
    for label, model in MODELS.items():
        for tq in QED_TARGETS:
            d = gen(model, {'qed': tq}, N_STEER)
            rows.append(dict(seed=sd_, model=label, target_qed=tq,
                             validity=len(d)/N_STEER, meanQED=d.qed.mean(),
                             stdQED=d.qed.std(), meanMW=d.mw.mean(),
                             meanSA=d.sa.mean(), meanHA=d.ha.mean(),
                             frac_gt07=(d.qed > 0.7).mean()))
            if sd_ == SEED:
                KEEP[('qed', label, tq)] = d.assign(target_qed=tq)
    print(f"   seed {sd_} done")
qed_raw = pd.DataFrame(rows)
qed_sum = agg(qed_raw, ['model','target_qed'],
              ['meanQED','stdQED','meanMW','meanSA','meanHA','frac_gt07','validity'])
spread = (qed_raw.groupby(['seed','model']).meanQED
          .agg(lambda s: s.max()-s.min()).rename('steering_range').reset_index())
spread_sum = (spread.groupby('model').steering_range
              .agg(['mean','std','min','max','count']).reset_index())

print("\n[2/5] molecular-weight targeting")
mw_arms = [k for k, m in MODELS.items() if 'mw' in props_of(m)]
rows = []
for sd_ in SEEDS:
    torch.manual_seed(sd_); np.random.seed(sd_)
    for label in mw_arms:
        for tm in MW_TARGETS:
            d = gen(MODELS[label], {'mw': float(tm)}, N_STEER)
            rows.append(dict(seed=sd_, model=label, target_mw=tm, meanMW=d.mw.mean(),
                             stdMW=d.mw.std(), abs_error=abs(d.mw.mean()-tm),
                             meanQED=d.qed.mean()))
            if sd_ == SEED:
                KEEP[('mw', label, tm)] = d.assign(target_mw=tm)
    print(f"   seed {sd_} done")
mw_raw = pd.DataFrame(rows) if rows else pd.DataFrame()
mw_sum = (agg(mw_raw, ['model','target_mw'], ['meanMW','abs_error','meanQED'])
          if len(mw_raw) else pd.DataFrame())

print(f"\n[3/5] joint target {JOINT_TARGET}")
rows = []
for sd_ in SEEDS:
    torch.manual_seed(sd_); np.random.seed(sd_)
    for label, model in MODELS.items():
        tgt = {k: v for k, v in JOINT_TARGET.items() if k in props_of(model)}
        d = joint_rank(gen(model, tgt, N_JOINT))
        rows.append(dict(seed=sd_, model=label,
                         expressible=len(tgt), top10=d.head(10).dist.mean(),
                         top50=d.head(50).dist.mean(),
                         top10_qed=d.head(10).qed.mean(),
                         top10_mw=d.head(10).mw.mean(),
                         top10_sa=d.head(10).sa.mean()))
        if sd_ == SEED:
            KEEP[('joint', label, None)] = d
    print(f"   seed {sd_} done")
joint_raw = pd.DataFrame(rows)
joint_sum = agg(joint_raw, ['model'],
                ['top10','top50','top10_qed','top10_mw','top10_sa'])

print("\n[4/5] unconditional generation quality")
rows = []
for sd_ in SEEDS:
    torch.manual_seed(sd_); np.random.seed(sd_)
    for label, model in MODELS.items():
        d = gen(model, {}, N_QC)
        rows.append(dict(seed=sd_, model=label, validity=len(d)/N_QC, **qc_of(d)))
        if sd_ == SEED:
            KEEP[('qc', label, None)] = d
    print(f"   seed {sd_} done")
qc_raw = pd.DataFrame(rows)
QC_COLS = ['validity','uniqueness','novelty','meanQED','meanSA','frac_SA_lt45',
           'meanMW','meanHA','internal_diversity']
qc_sum = agg(qc_raw, ['model'], QC_COLS)

print("\n[5/5] sampler characterization (reference arm, single seed)")
ref = 'Stage 2 (9-property)' if 'Stage 2 (9-property)' in MODELS else list(MODELS)[0]
torch.manual_seed(SEED); np.random.seed(SEED)
rows = []
for temp, ns in itertools.product([0.4, 0.7, 1.0], [0.0, 0.5, 1.0]):
    d = gen(MODELS[ref], {'qed': 0.9}, N_QC, noise=ns, temp=temp)
    rows.append(dict(temperature=temp, noise=ns, validity=len(d)/N_QC, **qc_of(d)))
front = pd.DataFrame(rows)
rows = []
for pp in [0.0, 0.5, 1.0, 2.0]:
    du = gen(MODELS[ref], {}, N_QC, pad_penalty=pp)
    dq = gen(MODELS[ref], {'qed': 0.9}, N_STEER, pad_penalty=pp)
    rows.append(dict(pad_penalty=pp, uncond_meanMW=du.mw.mean(),
                     uncond_meanHA=du.ha.mean(), uncond_meanQED=du.qed.mean(),
                     meanQED_at_09=dq.qed.mean()))
pps = pd.DataFrame(rows)

# ============================ COMPARISON VIEW ===============================
print("\nwriting comparison/")
for name, frame in [('qed_targeting_raw', qed_raw), ('qed_targeting_summary', qed_sum),
                    ('steering_range_raw', spread), ('steering_range', spread_sum),
                    ('joint_target_raw', joint_raw), ('joint_target_summary', joint_sum),
                    ('qc_raw', qc_raw), ('qc_summary', qc_sum),
                    ('mw_targeting_raw', mw_raw), ('mw_targeting_summary', mw_sum),
                    ('sampler_frontier', front), ('sampler_pad_penalty', pps)]:
    if len(frame): frame.to_csv(CMP/'csv'/f'{name}.csv', index=False)

md(CMP/'tables'/'steering_range.md', f'QED steering range ({N_SEEDS} seeds)',
   spread_sum.round(4),
   'Range = max - min achieved mean QED across requested targets 0.3-0.9.')
md(CMP/'tables'/'joint_target.md', f'Multi-objective control ({N_SEEDS} seeds)',
   pm(joint_sum, ['top10','top50'], ['model']),
   f'Target {JOINT_TARGET}; mean |z-distance| over a pool of {N_JOINT}. Arms receive only '
   'the components their conditioning vector can express. Top-1 omitted: as a single '
   'extreme order statistic its seed-to-seed CV approaches 90% and it does not separate arms.')
md(CMP/'tables'/'qc.md', f'Unconditional generation quality ({N_SEEDS} seeds)',
   pm(qc_sum, ['validity','uniqueness','novelty','meanQED','meanSA','internal_diversity'],
      ['model']))
md(CMP/'tables'/'qed_targeting.md', f'QED targeting by request ({N_SEEDS} seeds)',
   pm(qed_sum, ['meanQED','meanSA'], ['model','target_qed']))
if len(mw_sum):
    md(CMP/'tables'/'mw_targeting.md', f'Molecular-weight targeting ({N_SEEDS} seeds)',
       pm(mw_sum, ['meanMW','abs_error'], ['model','target_mw']))

plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'font.size':9,
                     'axes.grid':True,'grid.alpha':0.3})
COL = {'Stage 1 (QED-only)':'#4C9F70', 'Stage 2 (9-property)':'#2E86AB',
       'Stage 3 (RL)':'#D1495B', 'Stage 4 (filtered)':'#888888'}
col = lambda m: COL.get(m, '#999')

fig, ax = plt.subplots(figsize=(5.5, 4))
for label in MODELS:
    s = qed_sum[qed_sum.model == label].sort_values('target_qed')
    ax.plot(s.target_qed, s.meanQED_mean, marker='o', ms=3.5, label=label, color=col(label))
    ax.fill_between(s.target_qed, s.meanQED_mean-s.meanQED_std,
                    s.meanQED_mean+s.meanQED_std, alpha=0.18, color=col(label))
ax.plot(QED_TARGETS, QED_TARGETS, ls='--', lw=1, c='k', alpha=0.4, label='ideal')
ax.set_xlabel('requested QED'); ax.set_ylabel('achieved mean QED')
ax.set_title(f'QED targeting, all arms ({N_SEEDS} seeds, ±1 s.d.)', fontsize=10)
ax.legend(fontsize=7); fig.tight_layout()
fig.savefig(CMP/'figures'/'fig1_qed_targeting_all_arms.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
s = spread_sum.sort_values('mean', ascending=False)
ax.bar(range(len(s)), s['mean'], yerr=s['std'], capsize=4, color=[col(m) for m in s.model])
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.model, fontsize=7, rotation=15, ha='right')
ax.set_ylabel('QED steering range'); ax.set_title(f'Steering range ({N_SEEDS} seeds)', fontsize=10)
fig.tight_layout(); fig.savefig(CMP/'figures'/'fig2_steering_range.png'); plt.close(fig)

if len(mw_sum):
    fig, ax = plt.subplots(figsize=(5, 4))
    for label in mw_arms:
        s = mw_sum[mw_sum.model == label].sort_values('target_mw')
        ax.errorbar(s.target_mw, s.meanMW_mean, yerr=s.meanMW_std, marker='o', ms=4,
                    capsize=3, label=label, color=col(label))
    ax.plot(MW_TARGETS, MW_TARGETS, ls='--', lw=1, c='k', alpha=0.4, label='ideal')
    ax.set_xlabel('requested MW (Da)'); ax.set_ylabel('achieved mean MW (Da)')
    ax.set_title(f'Molecular-weight targeting ({N_SEEDS} seeds)', fontsize=10)
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(CMP/'figures'/'fig3_mw_targeting.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
s = joint_sum.sort_values('top50_mean')
ax.bar(range(len(s)), s.top50_mean, yerr=s.top50_std, capsize=4, color=[col(m) for m in s.model])
ax.set_xticks(range(len(s))); ax.set_xticklabels(s.model, fontsize=7, rotation=15, ha='right')
ax.set_ylabel('joint-target distance (lower better)')
ax.set_title(f'Multi-objective control, top-50 ({N_SEEDS} seeds)', fontsize=10)
fig.tight_layout(); fig.savefig(CMP/'figures'/'fig4_joint_target.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
sc = ax.scatter(front.internal_diversity, front.meanQED, c=front.uniqueness,
                s=55, cmap='viridis', vmin=0, vmax=1)
for _, r in front.iterrows():
    ax.annotate(f"{r.temperature:g}/{r.noise:g}", (r.internal_diversity, r.meanQED),
                fontsize=6, xytext=(3,3), textcoords='offset points')
ax.set_xlabel('internal diversity'); ax.set_ylabel('mean QED at target 0.9')
ax.set_title(f'Sampler frontier — {ref} (label: temp/noise)', fontsize=10)
plt.colorbar(sc, label='uniqueness'); fig.tight_layout()
fig.savefig(CMP/'figures'/'fig5_sampler_frontier.png'); plt.close(fig)

# ============================== PER-STAGE VIEW ==============================
# Same measurements, sliced to one arm. Written from the same pass as the
# comparison above, so the numbers cannot drift apart.
print("writing per-stage folders")
for label, model in MODELS.items():
    D = ROOT / FOLDER[label]
    c = col(label)

    qed_raw[qed_raw.model == label].to_csv(D/'csv'/'qed_targeting_raw.csv', index=False)
    q = qed_sum[qed_sum.model == label].sort_values('target_qed')
    q.to_csv(D/'csv'/'qed_targeting_summary.csv', index=False)
    qc_raw[qc_raw.model == label].to_csv(D/'csv'/'qc_raw.csv', index=False)
    qc_sum[qc_sum.model == label].to_csv(D/'csv'/'qc_summary.csv', index=False)
    joint_raw[joint_raw.model == label].to_csv(D/'csv'/'joint_target_raw.csv', index=False)
    spread[spread.model == label].to_csv(D/'csv'/'steering_range_raw.csv', index=False)
    if len(mw_raw) and label in mw_arms:
        mw_raw[mw_raw.model == label].to_csv(D/'csv'/'mw_targeting_raw.csv', index=False)
        mw_sum[mw_sum.model == label].to_csv(D/'csv'/'mw_targeting_summary.csv', index=False)

    # molecules (reference seed) -- example structures for the paper
    mols = pd.concat([v for (s, l, _), v in KEEP.items() if l == label and s == 'qed'],
                     ignore_index=True)
    mols.to_csv(D/'molecules'/'qed_targeting_molecules.csv', index=False)
    KEEP[('qc', label, None)].to_csv(D/'molecules'/'unconditional_sample.csv', index=False)
    jm = KEEP[('joint', label, None)]
    jm.head(50).to_csv(D/'molecules'/'joint_target_top50.csv', index=False)

    sr = spread_sum[spread_sum.model == label].iloc[0]
    js = joint_sum[joint_sum.model == label].iloc[0]
    md(D/'tables'/'summary.md', f'{label} — summary ({N_SEEDS} seeds)',
       pd.DataFrame([{'metric':'QED steering range',
                      'value':f"{sr['mean']:.3f} ± {sr['std']:.3f}"},
                     {'metric':'joint-target top-10',
                      'value':f"{js.top10_mean:.3f} ± {js.top10_std:.3f}"},
                     {'metric':'joint-target top-50',
                      'value':f"{js.top50_mean:.3f} ± {js.top50_std:.3f}"},
                     {'metric':'properties expressible',
                      'value':', '.join(props_of(model)) or 'none (unconditional)'},
                     {'metric':'parameters', 'value':f"{model.meta['n_parameters']:,}"},
                     {'metric':'checkpoint epoch', 'value':str(model.meta['epoch'])}]))
    md(D/'tables'/'qed_targeting.md', f'{label} — QED targeting',
       pm(q, ['meanQED','meanSA','meanMW'], ['target_qed']))
    md(D/'tables'/'qc.md', f'{label} — generation quality',
       pm(qc_sum[qc_sum.model == label],
          ['validity','uniqueness','novelty','meanQED','meanSA','internal_diversity'],
          ['model']))
    if len(mw_sum) and label in mw_arms:
        md(D/'tables'/'mw_targeting.md', f'{label} — molecular-weight targeting',
           pm(mw_sum[mw_sum.model == label], ['meanMW','abs_error'], ['target_mw']))

    # fig1 - targeting curve
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(q.target_qed, q.meanQED_mean, marker='o', ms=4, color=c)
    ax.fill_between(q.target_qed, q.meanQED_mean-q.meanQED_std,
                    q.meanQED_mean+q.meanQED_std, alpha=0.2, color=c)
    ax.plot(QED_TARGETS, QED_TARGETS, ls='--', lw=1, c='k', alpha=0.4)
    ax.set_xlabel('requested QED'); ax.set_ylabel('achieved mean QED')
    ax.set_title(f'{label}\nQED targeting ({N_SEEDS} seeds, ±1 s.d.)', fontsize=9)
    fig.tight_layout(); fig.savefig(D/'figures'/'fig1_qed_targeting.png'); plt.close(fig)

    # fig2 - achieved distribution per target (spread, not just the mean)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.boxplot([KEEP[('qed', label, t)].qed.values for t in QED_TARGETS],
               labels=[str(t) for t in QED_TARGETS], showfliers=False,
               boxprops=dict(color=c), medianprops=dict(color='k'))
    ax.set_xlabel('requested QED'); ax.set_ylabel('achieved QED')
    ax.set_title(f'{label}\nachieved distribution per request (seed {SEED})', fontsize=9)
    fig.tight_layout(); fig.savefig(D/'figures'/'fig2_qed_distribution.png'); plt.close(fig)

    # fig3 - joint-target ranked pool
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(range(1, len(jm)+1), jm.dist.values, lw=1.2, color=c)
    ax.axvline(50, ls=':', c='k', alpha=0.5)
    ax.annotate('top-50', (50, ax.get_ylim()[1]*0.9), fontsize=7, xytext=(4,0),
                textcoords='offset points')
    ax.set_xlabel('candidate rank'); ax.set_ylabel('distance to joint target')
    ax.set_title(f'{label}\nranked candidate pool (seed {SEED})', fontsize=9)
    fig.tight_layout(); fig.savefig(D/'figures'/'fig3_joint_ranked.png'); plt.close(fig)

    # fig4 - molecular-weight targeting, if this arm can express it
    if len(mw_sum) and label in mw_arms:
        s = mw_sum[mw_sum.model == label].sort_values('target_mw')
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        ax.errorbar(s.target_mw, s.meanMW_mean, yerr=s.meanMW_std, marker='o',
                    ms=4, capsize=3, color=c)
        ax.plot(MW_TARGETS, MW_TARGETS, ls='--', lw=1, c='k', alpha=0.4)
        ax.set_xlabel('requested MW (Da)'); ax.set_ylabel('achieved mean MW (Da)')
        ax.set_title(f'{label}\nmolecular-weight targeting', fontsize=9)
        fig.tight_layout(); fig.savefig(D/'figures'/'fig4_mw_targeting.png'); plt.close(fig)

    print(f"   {label} -> {FOLDER[label]}/")

# =============================== MANIFEST ===================================
json.dump(dict(seed=SEED, seeds=SEEDS, n_seeds=N_SEEDS,
               sampler=dict(cfg=CFG, temperature=TEMP, noise=NOISE, steps=STEPS,
                            min_len=MIN_LEN, pad_penalty=PAD_PEN),
               n_steer=N_STEER, n_joint=N_JOINT, n_qc=N_QC,
               qed_targets=QED_TARGETS, mw_targets=MW_TARGETS,
               joint_target=JOINT_TARGET, conditioning_props=PROPS9,
               arms=list(MODELS), folders=FOLDER, models=META,
               layout='per-stage folders + comparison/, written from one measurement pass',
               runtime_sec=round(time.time()-t_start, 1)),
          open(ROOT/'manifest.json','w'), indent=1, default=str)
json.dump(dict(python=sys.version.split()[0], torch=torch.__version__,
               numpy=np.__version__, pandas=pd.__version__,
               rdkit=Chem.rdBase.rdkitVersion,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
               generated=time.strftime('%Y-%m-%d %H:%M:%S')),
          open(ROOT/'config'/'environment.json','w'), indent=1)
json.dump(S2_NORM, open(ROOT/'config'/'norm_stats.json','w'), indent=1)

print(f"\ndone in {time.time()-t_start:.0f}s -> {ROOT}")
print("   per-stage: " + ", ".join(FOLDER[k] for k in MODELS))
print("   comparison/ holds all arms together")
print("   download the whole results/ folder into notebooks/results/paper/")
