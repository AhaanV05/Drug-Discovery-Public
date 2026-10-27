# ============================================================================
# STAGE 1 ABLATION — QED-only (1-prop) vs Multi-property (11-prop)
# Fully reproducible, reviewer-grade package: matched sampler, fixed seed, both
# models' full params, every number behind every figure, side-by-side comparison,
# best-of-N, QC, and environment capture.
#
# RUN INSIDE dit_conditional_generation.ipynb AFTER cells 1-15 + 17, so these
# globals exist: tokenizer, sample, device, config, QMEAN, QSTD, PAD, BOS, EOS,
# MASKi, train_df, train_dataset, and DiTBlock / TimestepEmbed / modulate (cell 14).
#
# Nothing is fabricated — every number comes from real generation on your models.
# ============================================================================
import os, sys, json, math, time, platform, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rdkit import Chem, rdBase
from rdkit.Chem import QED, RDConfig, AllChem, DataStructs

# ---- CONFIG -----------------------------------------------------------------
QEDONLY_CKPT = CKPT_DIR / 'best.pt'                       # 1-prop, wide MLP
MULTI_CKPT   = config.model_dir / 'cond_dit_best.pt'      # 11-prop, narrow MLP
ABL          = config.model_dir / 'stage1_ablation'

SEED      = 42                                             # <-- makes the whole run reproducible
CFG, TEMP, NOISE, STEPS = 1.0, 0.4, 0.0, 50               # matched operating point
TARGETS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_STEER   = 256
N_BESTN   = 512
KEEP_BESTN= 50
BESTN_TGT = 0.9
ASSUMED_HEADS = 8                                          # nh isn't stored in weights; cell-14 default

# Training settings NOT stored in the checkpoint — documented from project history.
# (epoch IS read authoritatively from each checkpoint; these are the rest.)
TRAIN_SETTINGS = {
    'QED-only':   dict(cfg_dropout=0.3, lr=3e-4, batch=64,
                       note='wide property-MLP retrain (this session)'),
    'Multi-prop': dict(cfg_dropout=0.1, lr=3e-4, batch=64,
                       note='original narrow property-MLP model'),
}

for sub in ('csv', 'figures', 'tables', 'config', 'figure_data'):
    (ABL / sub).mkdir(parents=True, exist_ok=True)
print(f"📁 ablation output → {ABL}")

# ---- reproducibility seed ---------------------------------------------------
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ---- SA score + novelty reference ------------------------------------------
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
_train_canon = set()
for s in train_df['canonical_smiles'].tolist():
    m = Chem.MolFromSmiles(str(s))
    if m: _train_canon.add(Chem.MolToSmiles(m))
print(f"   novelty reference: {len(_train_canon):,} training molecules")

# ---- Flexible CondDiT (reuses cell-14 DiTBlock/TimestepEmbed/modulate) -------
class FlexCondDiT(nn.Module):
    def __init__(self, V, max_len, n_props, d=512, nh=8, nl=6, T=100, dr=0.1,
                 pm_mult=4, pad_id=0, bos_id=1, eos_id=2, mask_id=3):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(max_len, d); self.te = TimestepEmbed(d)
        self.pm = nn.Sequential(nn.Linear(n_props, pm_mult*d), nn.SiLU(), nn.Linear(pm_mult*d, d))
        self.null = nn.Parameter(torch.zeros(n_props))
        self.blocks = nn.ModuleList([DiTBlock(d, nh, dr) for _ in range(nl)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.af = nn.Sequential(nn.SiLU(), nn.Linear(d, 2*d))
        self.head = nn.Linear(d, V)
        self.T=T; self.max_len=max_len; self.pad_id=pad_id; self.bos_id=bos_id
        self.eos_id=eos_id; self.mask_id=mask_id; self.vocab_size=V; self.n_props=n_props
        s = torch.arange(T+1)/T; ab = torch.cos((s+0.008)/1.008 * math.pi/2)**2
        self.register_buffer('alpha_bars', (ab/ab[0])[1:])
    def forward(self, xt, t, props, use_cond, pad_mask):
        B, L = xt.shape
        pos = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, -1)
        h = self.tok(xt) + self.pos(pos)
        uc = use_cond.view(B, 1).float()
        p = uc * props + (1 - uc) * self.null.unsqueeze(0)
        c = self.te(t) + self.pm(p)
        for blk in self.blocks: h = blk(h, c, pad_mask)
        sh, sc = self.af(c).chunk(2, -1); h = modulate(self.nf(h), sh, sc)
        return self.head(h)

def load_any(path, tag):
    ck = torch.load(path, map_location=device)
    sd = ck['model'] if 'model' in ck else ck
    d       = sd['tok.weight'].shape[1]
    V       = sd['tok.weight'].shape[0]
    max_len = sd['pos.weight'].shape[0]
    n_props = sd['null'].shape[0]
    pm_mult = sd['pm.0.weight'].shape[0] // d
    nl      = len({k.split('.')[1] for k in sd if k.startswith('blocks.')})
    T       = sd['alpha_bars'].shape[0] if 'alpha_bars' in sd else 100
    m = FlexCondDiT(V, max_len, n_props, d=d, nh=ASSUMED_HEADS, nl=nl, T=T, pm_mult=pm_mult,
                    pad_id=PAD, bos_id=BOS, eos_id=EOS, mask_id=MASKi).to(device)
    m.load_state_dict(sd); m.eval()
    n_param = sum(p.numel() for p in m.parameters())
    # full checkpoint metadata (everything except the big state_dict tensors)
    ck_meta = {k: (str(v) if not torch.is_tensor(v) else f'<tensor {tuple(v.shape)}>')
               for k, v in ck.items() if k != 'model'}
    params = dict(tag=tag, path=str(path), vocab_size=V, d_model=d, n_layers=nl,
                  n_heads=ASSUMED_HEADS, n_props=n_props, pm_mult=pm_mult, T=T,
                  max_len=max_len, n_parameters=n_param,
                  checkpoint_epoch=ck.get('epoch', '?'), checkpoint_best_val=ck.get('best', '?'),
                  checkpoint_keys=list(ck.keys()), checkpoint_meta=ck_meta,
                  file_size_mb=round(Path(path).stat().st_size/1e6, 2),
                  **TRAIN_SETTINGS.get(tag, {}))
    print(f"   ✅ {tag}: n_props={n_props} d={d} nl={nl} pm_mult={pm_mult} params={n_param:,} "
          f"(epoch {params['checkpoint_epoch']}, val {params['checkpoint_best_val']})")
    return m, params

# ---- conditioning: QED at idx 0, other props held at dataset mean (z=0) ------
def cond_vec(model, target_qed):
    v = torch.zeros(model.n_props, device=device)
    v[0] = (target_qed - QMEAN) / QSTD
    return v

def decode_mol(tk):
    m = Chem.MolFromSmiles(tokenizer.decode(tk.cpu().tolist()))
    if m is None: return None
    try: return dict(qed=QED.qed(m), ha=m.GetNumHeavyAtoms(), smiles=Chem.MolToSmiles(m))
    except Exception: return None

@torch.no_grad()
def steering(model, tag):
    summ, permol = [], []
    for tq in TARGETS:
        recs = [r for r in (decode_mol(tk) for tk in
                sample(model, cond_vec(model, tq), n=N_STEER, cfg=CFG, steps=STEPS,
                       temperature=TEMP, noise=NOISE)) if r]
        q  = np.array([r['qed'] for r in recs]) if recs else np.array([0.0])
        ha = np.array([r['ha']  for r in recs]) if recs else np.array([0.0])
        uniq = len({r['smiles'] for r in recs}) / max(len(recs), 1)
        summ.append(dict(model=tag, target=tq, n_valid=len(recs),
                         meanQED=float(q.mean()), stdQED=float(q.std()), medianQED=float(np.median(q)),
                         meanHA=float(ha.mean()), frac_gt07=float((q > 0.7).mean()), uniqueness=uniq))
        for r in recs: permol.append(dict(model=tag, target=tq, **r))
        print(f"      {tag:10s} t{tq:.1f}: meanQED={q.mean():.3f} meanHA={ha.mean():.1f} "
              f">0.7={100*(q>0.7).mean():.1f}% valid={len(recs)}")
    return pd.DataFrame(summ), pd.DataFrame(permol)

@torch.no_grad()
def best_of_n(model, tag):
    recs = [r for r in (decode_mol(tk) for tk in
            sample(model, cond_vec(model, BESTN_TGT), n=N_BESTN, cfg=CFG, steps=STEPS,
                   temperature=TEMP, noise=NOISE)) if r]
    df = pd.DataFrame(recs).sort_values('qed', ascending=False).reset_index(drop=True)
    df.insert(0, 'model', tag)
    print(f"      {tag:10s} best-of-{N_BESTN}@{BESTN_TGT}: best={df['qed'].iloc[0]:.3f} "
          f"median(top{KEEP_BESTN})={df['qed'].head(KEEP_BESTN).median():.3f}")
    return df

def internal_diversity(smiles, cap=200):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
           for m in (Chem.MolFromSmiles(s) for s in smiles[:cap]) if m]
    if len(fps) < 2: return 0.0
    sims = []
    for i in range(len(fps)): sims += list(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return float(1 - np.mean(sims)) if sims else 0.0

def qc_row(tag, bestn_df, permol):
    smis = permol[permol.model == tag]['smiles'].tolist()
    canon = [s for s in smis if Chem.MolFromSmiles(s)]
    novel = [s for s in canon if s not in _train_canon]
    sa = []
    for s in canon[:500]:
        m = Chem.MolFromSmiles(s)
        if m:
            try: sa.append(sascorer.calculateScore(m))
            except Exception: pass
    return dict(model=tag, n=len(smis), validity=len(canon)/max(len(smis),1),
                uniqueness=len(set(canon))/max(len(canon),1), novelty=len(novel)/max(len(canon),1),
                mean_SA=float(np.mean(sa)) if sa else float('nan'),
                frac_SA_lt45=float(np.mean([x<4.5 for x in sa])) if sa else float('nan'),
                internal_diversity=internal_diversity(canon),
                bestN_top_qed=float(bestn_df['qed'].iloc[0]),
                bestN_median_topK=float(bestn_df['qed'].head(KEEP_BESTN).median()))

# ============================================================================
# RUN
# ============================================================================
t0 = time.time()
print("\n[1/5] loading models…")
models, params = {}, {}
m1, p1 = load_any(QEDONLY_CKPT, 'QED-only'); models['QED-only'] = m1; params['QED-only'] = p1
if Path(MULTI_CKPT).exists():
    m11, p11 = load_any(MULTI_CKPT, 'Multi-prop'); models['Multi-prop'] = m11; params['Multi-prop'] = p11
else:
    print(f"   ⚠️ multi-prop checkpoint not found at {MULTI_CKPT} — running QED-only only.")

print("\n[2/5] native steering…")
ss, sp = [], []
for tag, m in models.items():
    s, pm = steering(m, tag); ss.append(s); sp.append(pm)
steer_summ = pd.concat(ss, ignore_index=True); steer_permol = pd.concat(sp, ignore_index=True)

print("\n[3/5] best-of-N…")
bestn = {tag: best_of_n(m, tag) for tag, m in models.items()}

print("\n[4/5] QC…")
qc = pd.DataFrame([qc_row(tag, bestn[tag], steer_permol) for tag in models])

print("\n[5/5] comparison + saving everything…")
tag2slug = {'QED-only': 'qedonly', 'Multi-prop': 'multiprop'}

# ---- per-model CSVs ---------------------------------------------------------
for tag in models:
    slug = tag2slug[tag]
    steer_summ[steer_summ.model==tag].to_csv(ABL/'csv'/f'native_steering_{slug}.csv', index=False)
    steer_permol[steer_permol.model==tag].to_csv(ABL/'csv'/f'native_steering_permol_{slug}.csv', index=False)
    bestn[tag].to_csv(ABL/'csv'/f'bestofN_{slug}_qed{BESTN_TGT}.csv', index=False)
qc.to_csv(ABL/'csv'/'qc_summary.csv', index=False)

# ---- side-by-side COMPARISON (the headline deliverable) ---------------------
comparison = None
if len(models) == 2:
    a = steer_summ[steer_summ.model=='QED-only'].set_index('target')
    b = steer_summ[steer_summ.model=='Multi-prop'].set_index('target')
    comparison = pd.DataFrame({
        'target': a.index,
        'QEDonly_meanQED':  a['meanQED'].values,
        'Multiprop_meanQED':b['meanQED'].values,
        'delta_meanQED':    (b['meanQED']-a['meanQED']).values,
        'QEDonly_meanHA':   a['meanHA'].values,
        'Multiprop_meanHA': b['meanHA'].values,
        'QEDonly_frac>0.7': a['frac_gt07'].values,
        'Multiprop_frac>0.7':b['frac_gt07'].values,
    })
    comparison.to_csv(ABL/'csv'/'comparison_by_target.csv', index=False)
    # headline summary (spread + best-of-N) per model
    head = []
    for tag in models:
        d = steer_summ[steer_summ.model==tag]
        head.append(dict(model=tag,
                         steering_spread=float(d.meanQED.max()-d.meanQED.min()),
                         max_meanQED=float(d.meanQED.max()),
                         bestN_top=float(bestn[tag]['qed'].iloc[0]),
                         bestN_median_topK=float(bestn[tag]['qed'].head(KEEP_BESTN).median())))
    head_df = pd.DataFrame(head)
    head_df.to_csv(ABL/'tables'/'headline_comparison.csv', index=False)
    with open(ABL/'tables'/'comparison_table.md', 'w') as f:
        f.write("## Native steering — side by side\n\n")
        f.write(comparison.round(3).to_markdown(index=False))
        f.write("\n\n## Headline comparison\n\n")
        f.write(head_df.round(3).to_markdown(index=False))

steer_summ.to_csv(ABL/'tables'/'summary_table.csv', index=False)
with open(ABL/'tables'/'summary_table.md', 'w') as f:
    f.write(steer_summ.round(3).to_markdown(index=False))

# ---- per-figure backing data (every number behind every graph) --------------
steer_summ.pivot(index='target', columns='model', values=['meanQED','stdQED']).to_csv(ABL/'figure_data'/'fig1_meanQED_vs_target.csv')
steer_summ.pivot(index='target', columns='model', values='meanHA').to_csv(ABL/'figure_data'/'fig2_meanHA_vs_target.csv')
steer_summ.pivot(index='target', columns='model', values='frac_gt07').to_csv(ABL/'figure_data'/'fig3_frac_gt07.csv')
pd.concat([bestn[t][['model','qed']] for t in models]).to_csv(ABL/'figure_data'/'fig4_bestofN_qed.csv', index=False)
steer_permol[['model','target','qed']].to_csv(ABL/'figure_data'/'fig5_qed_by_target.csv', index=False)

# ---- FIGURES ----------------------------------------------------------------
COL = {'QED-only':'#d62728', 'Multi-prop':'#1f77b4'}
plt.figure(figsize=(6,4.2))
for tag in models:
    d = steer_summ[steer_summ.model==tag]
    plt.errorbar(d.target, d.meanQED, yerr=d.stdQED, marker='o', capsize=3, label=tag, color=COL[tag])
plt.plot([0.3,0.9],[0.3,0.9],'k--',alpha=.4,label='ideal (y=x)')
plt.axhline(0.657,color='gray',ls=':',alpha=.6,label='data mean QED')
plt.xlabel('target QED'); plt.ylabel('achieved meanQED'); plt.title('Native QED steering')
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(ABL/'figures'/'fig1_native_steering_meanQED.png', dpi=300); plt.close()

plt.figure(figsize=(6,4.2))
for tag in models:
    d = steer_summ[steer_summ.model==tag]
    plt.plot(d.target, d.meanHA, marker='s', label=tag, color=COL[tag])
plt.axhline(24.1,color='gray',ls=':',alpha=.6,label='data mean HA')
plt.xlabel('target QED'); plt.ylabel('mean heavy atoms'); plt.title('Molecule size vs QED target')
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(ABL/'figures'/'fig2_size_vs_target_meanHA.png', dpi=300); plt.close()

plt.figure(figsize=(6,4.2)); w=0.35; x=np.arange(len(TARGETS))
for i,tag in enumerate(models):
    d = steer_summ[steer_summ.model==tag].sort_values('target')
    plt.bar(x+(i-0.5)*w, d.frac_gt07.values, w, label=tag, color=COL[tag])
plt.xticks(x,[f'{t:.1f}' for t in TARGETS]); plt.xlabel('target QED'); plt.ylabel('fraction QED > 0.7')
plt.title('High-QED hit rate'); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(ABL/'figures'/'fig3_frac_above_0.7.png', dpi=300); plt.close()

plt.figure(figsize=(6,4.2))
for tag in models: plt.hist(bestn[tag]['qed'], bins=30, alpha=.5, label=tag, color=COL[tag])
plt.axvline(0.657,color='gray',ls=':',alpha=.6,label='data mean')
plt.xlabel('QED'); plt.ylabel('count'); plt.title(f'Best-of-{N_BESTN} pool @ target {BESTN_TGT}')
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(ABL/'figures'/'fig4_bestofN_distribution.png', dpi=300); plt.close()

plt.figure(figsize=(7,4.2)); tags=list(models); data,positions,colors=[],[],[]
for ti,tq in enumerate(TARGETS):
    for mi,tag in enumerate(tags):
        data.append(steer_permol[(steer_permol.model==tag)&(steer_permol.target==tq)]['qed'].values)
        colors.append(COL[tag]); positions.append(ti*(len(tags)+1)+mi)
bp = plt.boxplot(data, positions=positions, widths=.8, patch_artist=True, showfliers=False)
for patch,c in zip(bp['boxes'],colors): patch.set_facecolor(c); patch.set_alpha(.6)
plt.xticks([ti*(len(tags)+1)+0.5 for ti in range(len(TARGETS))],[f'{t:.1f}' for t in TARGETS])
plt.xlabel('target QED'); plt.ylabel('QED'); plt.title('QED distribution by target')
plt.legend([Patch(facecolor=COL[t],alpha=.6) for t in tags], tags, fontsize=8)
plt.tight_layout(); plt.savefig(ABL/'figures'/'fig5_qed_boxplot_by_target.png', dpi=300); plt.close()

# ---- CONFIG / PARAMS / ENVIRONMENT (full reproducibility) -------------------
json.dump(params, open(ABL/'config'/'model_params.json','w'), indent=2, default=str)

run_config = dict(seed=SEED, sampler=dict(cfg=CFG, temperature=TEMP, noise=NOISE, steps=STEPS),
                  targets=TARGETS, n_steer=N_STEER, n_bestN=N_BESTN, keep_bestN=KEEP_BESTN,
                  bestN_target=BESTN_TGT,
                  conditioning='QED at index 0 (z-scored); for 11-prop the other 10 held at dataset mean (z=0)',
                  qed_norm=dict(mean=QMEAN, std=QSTD),
                  qed_zscore_per_target={str(t): (t-QMEAN)/QSTD for t in TARGETS})
json.dump(run_config, open(ABL/'config'/'run_config.json','w'), indent=2, default=str)

# full norm_stats (documents exactly how every property was normalized)
try: json.dump(train_dataset.norm_stats, open(ABL/'config'/'norm_stats.json','w'), indent=2, default=str)
except Exception as e: print("   (norm_stats not saved:", e, ")")

def _git():
    try: return subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return None
env = dict(python=sys.version.split()[0], platform=platform.platform(),
           torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
           rdkit=rdBase.rdkitVersion, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           git_commit=_git())
json.dump(env, open(ABL/'config'/'environment.json','w'), indent=2, default=str)

manifest = dict(run_config=run_config, environment=env,
                models={t: {k: params[t][k] for k in
                            ('n_props','d_model','n_layers','n_heads','pm_mult','T','max_len',
                             'n_parameters','checkpoint_epoch','checkpoint_best_val','cfg_dropout')}
                        for t in models},
                qc=qc.to_dict('records'), runtime_sec=round(time.time()-t0,1),
                caveat="As-trained comparison: checkpoints differ in epochs/MLP-width/cfg-dropout. "
                       "Qualitative conclusion robust; retrain matched for a bulletproof claim.")
json.dump(manifest, open(ABL/'manifest.json','w'), indent=2, default=str)

# ---- console summary --------------------------------------------------------
print("\n================  MODEL PARAMS  ================")
for tag in models:
    p = params[tag]
    print(f"  {tag:10s}: n_props={p['n_props']} d={p['d_model']} layers={p['n_layers']} "
          f"heads={p['n_heads']} pm_mult={p['pm_mult']} params={p['n_parameters']:,} "
          f"epoch={p['checkpoint_epoch']} cfg_dropout={p.get('cfg_dropout','?')}")
print("\n================  QC SUMMARY  ==================")
print(qc.round(3).to_string(index=False))
if comparison is not None:
    print("\n================  COMPARISON (native steering)  ================")
    print(comparison.round(3).to_string(index=False))
    print("\n  headline:"); print(head_df.round(3).to_string(index=False))
print(f"\n✅ done in {manifest['runtime_sec']}s — everything under {ABL}")
print("   download that folder into  notebooks/results/stage1_ablation/  for the paper.")
