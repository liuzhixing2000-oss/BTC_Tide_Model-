"""Frozen, separate pre-trigger filters; retrospective portfolio experiment."""
import gzip,json,hashlib
from pathlib import Path
import pandas as pd
from tide_replay_support import chronological_portfolio
from backtest_tide_alt_entries import stats

def run(source,features):
    c=pd.DataFrame([x for x in source['candidates'] if x['arm']=='confirmed'])
    f=pd.DataFrame(features)
    assert len(c)==191 and set(c.event_id)==set(f.event_id)
    c=c.merge(f[['event_id','cutoff','volume_ratio_6h','both_htf_down']],on='event_id',validate='one_to_one')
    c['entry_time']=pd.to_datetime(c.entry_time,utc=True)
    assert (pd.to_datetime(c.cutoff,utc=True)<c.entry_time).all()
    assert c[['volume_ratio_6h','both_htf_down']].notna().all().all()
    c['day']=c.entry_time.dt.strftime('%Y-%m-%d')
    # Natural round cutoff frozen for this experiment, not an optimized quantile.
    masks={'baseline':pd.Series(True,index=c.index),'volume_half':c.volume_ratio_6h<=.5,'dual_down':c.both_htf_down.eq(1)}
    midpoint=pd.Timestamp('2026-07-21',tz='UTC')
    busiest=list(c.day.value_counts().head(2).index)
    segments={'full':pd.Series(True,index=c.index),'first50':c.entry_time<midpoint,'second50':c.entry_time>=midpoint,'without_two_busiest_days':~c.day.isin(busiest)}
    out={'status':'COMPLETE','rules':{'volume_half':'Pre-trigger last6h average volume <=0.5 times preceding18h average volume','dual_down':'Pre-trigger1h AND4h SMA50<SMA200','exit':'Original structural stop and fixed4h exit; no weak-exit changes'},'busiest_days':busiest,'segments':{},'limitations':['Retrospective; thresholds/hypotheses informed by this previously inspected period. No segment is clean OOS.','0.5 volume cutoff fixed before this experiment; no search or combination of filters.','Each segment starts with1000U independently. Daily/capital/position rules unchanged.','Current eligible altcoin pool has selection/survivorship bias. Simplified confirmed entry, not full live production.','No floating drawdown, funding, liquidation or tick execution simulation.','Two-busiest-day removal is a sensitivity diagnostic, not proof against shared market beta.']}
    for segment,scope in segments.items():
        results={};accepted={}
        for arm,mask in masks.items():
            subset=c[scope&mask].copy();a,r=chronological_portfolio(subset,respect_equity=True)
            accepted[arm]=a
            results[arm]={'candidates':len(subset),'portfolio_rejected':len(r),**stats(a)}
        base=accepted['baseline'];top=set(base.nlargest(5,'pnl_usdt').event_id)
        for arm in ['volume_half','dual_down']:
            permitted=set(c.loc[masks[arm],'event_id']);filtered=base[~base.event_id.isin(permitted)]
            same=base[base.event_id.isin(permitted)]
            results[arm]['vs_baseline']={'net_delta':results[arm]['net_pnl']-results['baseline']['net_pnl'],'baseline_trades_filtered':len(filtered),'baseline_winner_profit_filtered':float(filtered.loc[filtered.pnl_usdt>0,'pnl_usdt'].sum()),'baseline_loser_loss_avoided':float(-filtered.loc[filtered.pnl_usdt<0,'pnl_usdt'].sum()),'baseline_top5_filtered':list(sorted(top-permitted)),'baseline_top5_profit_filtered':float(filtered.loc[filtered.event_id.isin(top),'pnl_usdt'].sum()),'retained_baseline_pnl':float(same.pnl_usdt.sum()),'new_admitted_ids':list(sorted(set(accepted[arm].event_id)-set(base.event_id)))}
        out['segments'][segment]=results
        if segment=='full':out['full_accepted']={k:v.to_dict('records') for k,v in accepted.items()}
    return out

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    source=json.loads(gzip.decompress((root/'reports/alt_entries_20260909/retry.json.gz').read_bytes()))
    path=root/'reports/preconditions_20260910/features.json';features=json.loads(path.read_text())
    out=run(source,features);out['feature_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    p=root/'reports/filter_experiment_20260911';p.mkdir(parents=True,exist_ok=True)
    (p/'results.json').write_text(json.dumps(out,default=str,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in out.items() if k!='full_accepted'},indent=2))
