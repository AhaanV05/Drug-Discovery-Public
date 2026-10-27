# =============================================================================
# CELL 31 · PHASE 1 RE-RUN — QED-only trained WITH the PAD fix
#
# WHY: the original Phase-1 ablation compared a 1-prop and an 11-prop model, but
# BOTH were trained before the PAD-masking fix, so neither had length control.
# Three of its conclusions turned out to be artifacts (dilution refutation,
# "conditioning is weak", noise=0.0).
#
# This trains a QED-only model under the SAME recipe as CELL 24 — identical
# dropout, lr, batch, optimiser, dataset, and the fixed q_sample — differing
# ONLY in n_props (1 vs 9). That yields a properly matched 1-vs-9 ablation,
# which is a better controlled experiment than Phase 1 ever was (its 11-prop
# comparator was ep~40 with a narrow property-MLP — always a confound).
#
# Prereqs: CELL 1-15, 17, 22, 23.
# GATE: leave FAST_LOOP_Q1=True for a ~3 min sanity check, then set False.
# =============================================================================
import time, math, json

EPOCHS_Q1    = 200
LR_Q1        = 3e-4
PROP_DROP    = 0.3        # identical to CELL 24
ALLNULL_PROB = 0.10       # identical to CELL 24
FAST_LOOP_Q1 = True       # <<< gate switch
FAST_STEPS   = 300

Q1_DIR  = config.model_dir / 'stage1_qedonly'
Q1_CKPT = Q1_DIR / 'checkpoints'; Q1_CKPT.mkdir(parents=True, exist_ok=True)
amp = torch.cuda.is_available()

# --- datasets: reuse Stage2Dataset, slice to QED (index 0 of Stage2Dataset.PROPS) ---
QED_IDX = Stage2Dataset.PROPS.index('qed')
train_dsq = Stage2Dataset(train_df, train_dataset._tokens)
val_dsq   = Stage2Dataset(val_df,   val_dataset._tokens, norm_stats=train_dsq.norm_stats)
train_loaderq = DataLoader(train_dsq, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
val_loaderq   = DataLoader(val_dsq,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

ditq = CondDiT(tokenizer.vocab_size, 128, n_props=1, d=512, nh=8, nl=6, T=100,
               pad_id=PAD, bos_id=BOS, eos_id=EOS, mask_id=MASKi).to(device)
optq = torch.optim.AdamW(ditq.parameters(), lr=LR_Q1, weight_decay=0.01)
print(f"QED-only POST-FIX CondDiT: {sum(p.numel() for p in ditq.parameters()):,} params")
print(f"  → {Q1_DIR}")

@torch.no_grad()
def _valq(m, nb=16):
    m.eval(); ce=cn=acc=ec=en=0
    for i,b in enumerate(val_loaderq):
        if i>=nb: break
        x0=b['tokens'].to(device)
        props=b['props'][:, QED_IDX:QED_IDX+1].to(device)
        pmask=b['pmask'][:, QED_IDX:QED_IDX+1].to(device)
        t=torch.randint(0,m.T,(x0.size(0),),device=device); xt,im=m.q_sample(x0,t)
        logits=m(xt,t,props,pmask,pad_mask=(xt==PAD))
        real=im&(x0!=PAD)&(x0!=BOS)&(x0!=EOS)     # same metric as CELL 24 → comparable
        if real.any():
            ce+=F.cross_entropy(logits[real].float(),x0[real],reduction='sum').item()
            acc+=(logits[real].argmax(-1)==x0[real]).sum().item(); cn+=real.sum().item()
        em=im&(x0==EOS)
        if em.any(): ec+=(logits[em].argmax(-1)==x0[em]).sum().item(); en+=em.sum().item()
    m.train(); return ce/max(cn,1), acc/max(cn,1), ec/max(en,1)

def _train_epoch_q(max_steps=None):
    ditq.train(); step=0
    for b in train_loaderq:
        x0=b['tokens'].to(device); B=x0.size(0)
        props=b['props'][:, QED_IDX:QED_IDX+1].to(device)
        pmask=b['pmask'][:, QED_IDX:QED_IDX+1].to(device)
        keep=(torch.rand(B,1,device=device)>PROP_DROP).float()*pmask
        allnull=(torch.rand(B,device=device)<ALLNULL_PROB).float().unsqueeze(1)
        keep=keep*(1-allnull)
        t=torch.randint(0,ditq.T,(B,),device=device); xt,im=ditq.q_sample(x0,t)
        optq.zero_grad()
        with torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            logits=ditq(xt,t,props,keep,pad_mask=(xt==PAD))
            if not im.any(): continue
            loss=F.cross_entropy(logits[im].float(),x0[im])
        loss.backward(); torch.nn.utils.clip_grad_norm_(ditq.parameters(),1.0); optq.step()
        step+=1
        if max_steps and step>=max_steps: break
    return step

if FAST_LOOP_Q1:
    print("=== GATE: fast-loop ===")
    t0=time.time(); n=_train_epoch_q(max_steps=FAST_STEPS)
    vce,vacc,veos=_valq(ditq)
    print(f"after {n} steps ({time.time()-t0:.0f}s): "
          f"val_realCE={vce:.3f}  ppl={math.exp(vce):.1f}  acc={vacc:.3f}  eos={veos:.3f}")
    print("PASS if val_realCE is falling from ~4.9 and eos is above ~0.2.")
    print("NOTE: expect the same inverted signature as the 9-prop post-fix run —")
    print("      eos starts LOW (~0.2-0.4) because the PAD length-hint is gone. That is correct.")
    print("If PASS -> set FAST_LOOP_Q1=False and re-run for the full train.")
else:
    wb=init_wandb("qedonly_padfix_d512_l6",
                  {'epochs':EPOCHS_Q1,'lr':LR_Q1,'prop_drop':PROP_DROP,
                   'allnull':ALLNULL_PROB,'n_props':1,'pad_fix':True})
    best=float('inf'); history=[]
    def _payloadq(ep):
        return {'model':ditq.state_dict(),'opt':optq.state_dict(),'epoch':ep,'best':best,
                'history':history,'n_props':1,'props':['qed'],'pad_fix':True}
    print(f"{'ep':>3}|{'val_realCE':>10}|{'ppl':>5}|{'acc':>5}|{'eos':>5}|{'sec':>4}")
    for ep in range(1,EPOCHS_Q1+1):
        t0=time.time(); _train_epoch_q()
        vce,vacc,veos=_valq(ditq); sec=time.time()-t0
        rec={'epoch':ep,'val_realCE':vce,'ppl':math.exp(vce),'acc':vacc,'eos_acc':veos,'sec':sec}
        history.append(rec)
        print(f"{ep:>3}|{vce:>10.3f}|{math.exp(vce):>5.1f}|{vacc:>5.3f}|{veos:>5.3f}|{sec:>4.0f}")
        torch.save(_payloadq(ep), Q1_CKPT/'latest.pt')
        if ep%10==0: torch.save(_payloadq(ep), Q1_CKPT/f'epoch_{ep:03d}.pt')
        if vce<best:
            best=vce; torch.save(_payloadq(ep), Q1_CKPT/'best.pt'); print(f"   best {vce:.3f} saved")
        json.dump(history, open(Q1_DIR/'history.json','w'), indent=1)
        if wb: wb.log(rec)
    if wb: wb.finish()
    print("QED-only post-fix training complete ->", Q1_CKPT)
    print("Now re-run CELL 30 — it will pick up 'Ph1 QED-only POST-FIX' automatically.")
