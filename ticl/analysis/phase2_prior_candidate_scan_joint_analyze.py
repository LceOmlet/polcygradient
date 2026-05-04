
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict


def finite(x):
    try:
        y=float(x)
    except Exception:
        return None
    return y if math.isfinite(y) else None

def get(d,path,default=None):
    cur=d
    for p in path.split('.'):
        if isinstance(cur,dict) and p in cur: cur=cur[p]
        else: return default
    return cur

def stat(vals):
    vals=sorted(v for v in (finite(v) for v in vals) if v is not None)
    if not vals: return {'n':0,'mean':None,'q10':None,'q50':None,'q90':None,'min':None,'max':None}
    def q(f):
        if len(vals)==1: return vals[0]
        pos=f*(len(vals)-1); lo=math.floor(pos); hi=math.ceil(pos)
        if lo==hi: return vals[lo]
        return vals[lo]*(hi-pos)+vals[hi]*(pos-lo)
    return {'n':len(vals),'mean':sum(vals)/len(vals),'q10':q(.1),'q50':q(.5),'q90':q(.9),'min':vals[0],'max':vals[-1]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows-jsonl', required=True)
    ap.add_argument('--gym-summary-json', default='/home/chen/RLPFN/artifacts/phase2_prior_gym_action_causal_sensitivity_all_gain07_terminal_ablation_0428/summary.json')
    ap.add_argument('--output-dir', required=True)
    args=ap.parse_args()
    rows=[]
    p=Path(args.rows_jsonl)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip(): rows.append(json.loads(line))
    outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    gym=json.loads(Path(args.gym_summary_json).read_text())
    metric_paths={
        'return_std':'discounted_effective_return.std',
        'return_q50':'discounted_effective_return.q50',
        'reward_sens':'per_action_reward_sensitivity.q50',
        'reward_sens_effdim':'per_action_reward_sensitivity.effective_dim_fraction',
        'state_sens':'per_action_state_sensitivity.q50',
        'state_sens_effdim':'per_action_state_sensitivity.effective_dim_fraction',
        'obs_std':'obs_value_std_all',
        'obs_dim_std_q50':'obs_per_dim_std.q50',
        'obs_dim_std_effdim':'obs_per_dim_std.effective_dim_fraction',
        'obs_rank':'obs_cov.effective_rank_fraction',
        'obs_top_eig':'obs_cov.top_eigen_share',
        'drift_mean':'step_obs_drift_l2.mean',
        'drift_q50':'step_obs_drift_l2.q50',
        'reward_std':'reward_effective.std',
        'reward_q50':'reward_effective.q50',
        'terminal_rate':'terminal_event_rate',
        'id_h2':'horizon_2_bias_corrected_identity_score',
        'id_h5':'horizon_5_bias_corrected_identity_score',
    }
    # gym q50 references
    refs={}
    for m,path in metric_paths.items():
        if m in ('terminal_rate','id_h2','id_h5'): continue
        vals={}
        for gid,gd in gym['gym_references'].items():
            cur=gd
            ok=True
            for pp in path.split('.'):
                if isinstance(cur,dict) and pp in cur: cur=cur[pp]
                else: ok=False; break
            if ok:
                vals[gid]=cur.get('q50') if isinstance(cur,dict) and 'q50' in cur else cur
        refs[m]=vals
    thresholds={
        'return_std_gym_min': min(refs['return_std'].values()),
        'reward_sens_gym_min': min(refs['reward_sens'].values()),
        'state_sens_gym_min': min(refs['state_sens'].values()),
        'obs_std_gym_min': min(refs['obs_std'].values()),
        'obs_rank_gym_min': min(refs['obs_rank'].values()),
        'obs_top_eig_gym_max': max(refs['obs_top_eig'].values()),
        'reward_std_gym_min': min(refs['reward_std'].values()),
        'drift_mean_gym_min': min(refs['drift_mean'].values()),
    }
    by=defaultdict(list)
    flat=[]
    for r in rows:
        rec={'case':r.get('case'), 'seed':r.get('frozen_h_seed')}
        for m,path in metric_paths.items(): rec[m]=get(r,path)
        snap=r.get('sampled_env_snapshot') or {}
        for k,v in snap.items():
            if isinstance(v,(int,float,bool)): rec['snap_'+k]=float(v)
        by[str(r.get('case'))].append(rec)
        flat.append(rec)
    case_rows=[]
    for case,rs in sorted(by.items()):
        row={'case':case,'row_count':len(rs)}
        for m in metric_paths:
            st=stat([r.get(m) for r in rs])
            for sk,sv in st.items(): row[f'{m}_{sk}']=sv
        # feasibility counts
        def c(pred): return sum(1 for r in rs if pred(r))
        row['count_return_ge_gym_min']=c(lambda r: finite(r.get('return_std')) is not None and r['return_std']>=thresholds['return_std_gym_min'])
        row['count_return_action_state_ge_gym_min']=c(lambda r: all(finite(r.get(k)) is not None for k in ['return_std','reward_sens','state_sens']) and r['return_std']>=thresholds['return_std_gym_min'] and r['reward_sens']>=thresholds['reward_sens_gym_min'] and r['state_sens']>=thresholds['state_sens_gym_min'])
        row['count_full_loose_gymlike']=c(lambda r: all(finite(r.get(k)) is not None for k in ['return_std','reward_sens','state_sens','obs_std','obs_rank','obs_top_eig','reward_std','drift_mean']) and r['return_std']>=thresholds['return_std_gym_min'] and r['reward_sens']>=thresholds['reward_sens_gym_min'] and r['state_sens']>=thresholds['state_sens_gym_min'] and r['obs_std']>=thresholds['obs_std_gym_min'] and r['obs_rank']>=thresholds['obs_rank_gym_min'] and r['obs_top_eig']<=thresholds['obs_top_eig_gym_max'] and r['reward_std']>=thresholds['reward_std_gym_min'] and r['drift_mean']>=thresholds['drift_mean_gym_min'])
        case_rows.append(row)
    if case_rows:
        with (outdir/'case_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
            w.writeheader(); w.writerows(case_rows)
    # paired deltas vs baseline for cases present on same seeds
    pairs=defaultdict(dict)
    for r in flat: pairs[str(r['seed'])][str(r['case'])]=r
    cases=[c for c in sorted(by) if c!='baseline']
    delta_rows=[]
    for case in cases:
        paired=[d for d in pairs.values() if 'baseline' in d and case in d]
        row={'case':case,'paired_n':len(paired)}
        for m in metric_paths:
            deltas=[]; ratios=[]
            for d in paired:
                a=finite(d['baseline'].get(m)); b=finite(d[case].get(m))
                if a is not None and b is not None:
                    deltas.append(b-a)
                    if abs(a)>1e-12: ratios.append(b/a)
            sd=stat(deltas); sr=stat(ratios)
            row[f'{m}_delta_mean']=sd['mean']; row[f'{m}_delta_q50']=sd['q50']; row[f'{m}_ratio_q50']=sr['q50']
        delta_rows.append(row)
    if delta_rows:
        with (outdir/'paired_delta_vs_baseline.csv').open('w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            w.writeheader(); w.writerows(delta_rows)
    # top env/case candidates by loose score
    def ratio(x,t,higher=True):
        x=finite(x)
        if x is None or t is None or t==0: return 0.0
        if higher: return min(x/t,1.0)
        return min(t/x,1.0) if x>0 else 1.0
    for r in flat:
        r['score']=sum([
            ratio(r.get('return_std'), thresholds['return_std_gym_min']),
            ratio(r.get('reward_sens'), thresholds['reward_sens_gym_min']),
            ratio(r.get('state_sens'), thresholds['state_sens_gym_min']),
            ratio(r.get('obs_std'), thresholds['obs_std_gym_min']),
            ratio(r.get('obs_rank'), thresholds['obs_rank_gym_min']),
            ratio(r.get('obs_top_eig'), thresholds['obs_top_eig_gym_max'], False),
            ratio(r.get('reward_std'), thresholds['reward_std_gym_min']),
            ratio(r.get('drift_mean'), thresholds['drift_mean_gym_min']),
        ])/8
    top=sorted(flat,key=lambda r:r['score'], reverse=True)[:50]
    if top:
        keys=['score','case','seed','return_std','return_q50','reward_sens','reward_sens_effdim','state_sens','state_sens_effdim','obs_std','obs_dim_std_q50','obs_dim_std_effdim','obs_rank','obs_top_eig','drift_mean','drift_q50','reward_std','reward_q50','terminal_rate','id_h2','id_h5','snap_alpha','snap_noise_std','snap_init_state_std','snap_init_std','snap_state_output_scale','snap_state_full_rms_target','snap_terminal_reset_count_target']
        with (outdir/'top_candidates.csv').open('w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows([{k:r.get(k) for k in keys} for r in top])
    summary={'rows':len(rows),'case_count':len(by),'thresholds':thresholds,'case_summary':case_rows,'paths':{'case_summary':str(outdir/'case_summary.csv'),'paired_delta':str(outdir/'paired_delta_vs_baseline.csv'),'top_candidates':str(outdir/'top_candidates.csv')}}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
