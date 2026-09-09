"""Merge disjoint retry evidence; replay the combined portfolio chronologically."""
import gzip
import json
import sys
from pathlib import Path
import pandas as pd
from backtest_tide_alt_entries import stats
from tide_replay_support import chronological_portfolio

def load(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())

def aggregate(archives):
    original=archives[0]['summary']
    events,candidates,coverage,errors={},{},{},{}
    for archive in archives:
        s=archive['summary']
        assert s['source_sha256']==original['source_sha256']
        for key in ['start','end','params']:
            assert s[key]==original[key]
        for key in ['risk','cost','max_notional']:
            assert s['settings'][key]==original['settings'][key]
        failed={e['symbol'] for e in s['errors']}
        for symbol in set(s['symbols'])-failed:
            errors.pop(symbol,None)
        errors.update({e['symbol']:e for e in s['errors']})
        for row in archive['events']:
            key=row['event_id']
            if key in events: assert events[key]==row, f'Changed event: {key}'
            events[key]=row
        for row in archive['candidates']:
            key=(row['event_id'],row['arm'])
            if key in candidates: assert candidates[key]==row, f'Changed candidate: {key}'
            candidates[key]=row
        for row in s['coverage']:
            key=(row['symbol'],row['interval'])
            if key in coverage: assert coverage[key]==row, f'Changed market data: {key}'
            coverage[key]=row
    out={k:v for k,v in original.items() if k not in ['arms','diagnostics','raw_event_counts','confirmation_diagnostics','paired_candidates','errors','coverage']}
    out.update(errors=list(errors.values()),coverage=list(coverage.values()),symbols_requested=len(original['symbols']),symbols_completed=len(original['symbols'])-len(errors),status='PARTIAL_DATA' if errors else 'COMPLETE',arms={})
    t=pd.DataFrame(list(candidates.values()));e=pd.DataFrame(list(events.values()))
    out['raw_event_counts']=e.status.value_counts().to_dict()
    out['confirmation_diagnostics']={name:int(e[name].fillna(False).sum()) for name in ['raw_pass','next_pass','combined_pass','confirmation_pass']}
    output=Path('reports/alt_entries_20260909');output.mkdir(parents=True,exist_ok=True)
    for arm in ['early','confirmed']:
        subset=t[t.arm.eq(arm)]
        accepted,rejected=chronological_portfolio(subset,respect_equity=True)
        accepted.to_csv(output/f'{arm}_accepted.csv',index=False)
        rejected.to_csv(output/f'{arm}_rejected.csv',index=False)
        candidate_stats=stats(subset)
        for key in ['return_pct','realized_max_drawdown']:candidate_stats.pop(key,None)
        out['arms'][arm]=dict(**stats(accepted),ending_equity=1000+float(accepted.pnl_usdt.sum()),candidates=len(subset),portfolio_rejected=len(rejected),reject_reasons=rejected.reject_reason.value_counts().to_dict(),all_candidate_diagnostic=candidate_stats,first_entry=str(accepted.entry_time.min()),last_entry=str(accepted.entry_time.max()),top5_pnl=float(accepted.nlargest(5,'pnl_usdt').pnl_usdt.sum()),by_month={str(k):{'trades':len(g),'net_pnl':float(g.pnl_usdt.sum())} for k,g in accepted.groupby(accepted.exit_time.dt.strftime('%Y-%m'))},by_symbol={str(k):stats(g) for k,g in accepted.groupby('symbol')})
    pairs=t.pivot(index='event_id',columns='arm',values='net_R').dropna()
    out['posthoc_confirmed_subset_NOT_TRADABLE_FILTER']={'pairs':len(pairs),'early_mean_R':float(pairs.early.mean()),'confirmed_mean_R':float(pairs.confirmed.mean())}
    out['provenance']={'source_commit':'0fa5da378a142d520a4ff12237dc2fdccedbd9e3','deployments':['43f7d3c6-0b56-4535-943b-38ced1f9dfb1','d8f07492-e23f-4f0d-b587-4a97c3b3fdc8'],'method':'Merge candidate events then rerun portfolio; never add separate portfolio PnLs.'}
    (output/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({k:({a:{i:j for i,j in b.items() if i!='by_symbol'} for a,b in v.items()} if k=='arms' else v) for k,v in out.items() if k not in ['coverage','symbols','limitations','settings','source_sha256']},indent=2))

if __name__=='__main__':aggregate([load(p) for p in sys.argv[1:]])
