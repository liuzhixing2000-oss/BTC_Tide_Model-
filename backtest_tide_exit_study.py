"""Frozen confirmed entries: fixed 4h, weak-at-1h, and strong-at-4h exits."""
import argparse
import base64
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tide_replay_support import chronological_portfolio
from backtest_tide_alt_entries import stats

BAR=pd.Timedelta(minutes=15)
COST=.001
ARMS=['baseline','weak_1h','extend_8h']

def result(candidate,frame,k,price,reason,arm,at_open=False,activated=False):
    c=dict(candidate)
    c.update(arm=arm,exit=float(price),exit_reason=reason,
             exit_time=frame.iloc[k].open_time+(pd.Timedelta(0) if at_open else BAR),
             bars_held=k if at_open else k+1,activated=activated)
    c['gross_return']=price/c['entry']-1
    c['net_return']=c['gross_return']-COST
    c['pnl_usdt']=c['notional']*c['net_return']
    c['net_R']=c['net_return']/c['stop_pct']
    # Do not carry stale MFE/MAE from the old exit window.
    c.pop('bar_mfe_upper_bound',None);c.pop('bar_mae_lower_bound',None)
    return c

def legacy(candidate,frame):
    for k in range(16):
        row=frame.iloc[k]
        if row.low<=candidate['stop']:
            return result(candidate,frame,k,min(float(row.open),candidate['stop']),'structural_stop','legacy')
    return result(candidate,frame,15,float(frame.iloc[15].close),'fixed_4h','legacy')

def simulate(candidate,frame,arm):
    """Decision at a completed bar; fill next open. Stop always has priority."""
    if arm not in ARMS: raise ValueError(arm)
    activated=False
    for k in range(32):
        row=frame.iloc[k]
        if row.low<=candidate['stop']:
            return result(candidate,frame,k,min(float(row.open),candidate['stop']),'structural_stop',arm,activated=activated)
        net_r=(float(row.close)/candidate['entry']-1-COST)/candidate['stop_pct']
        reason=None
        if arm=='weak_1h' and k==3:
            mfe_r=(float(frame.iloc[:4].high.max())/candidate['entry']-1)/candidate['stop_pct']
            if mfe_r<.5 and net_r<=0:
                activated=True;reason='weak_1h'
        if k==15:
            strong=net_r>=.5 and row.close>frame.iloc[k-4].close
            if arm=='extend_8h' and strong:
                activated=True
            else:
                reason='fixed_4h'
        if arm=='extend_8h' and activated and k>15:
            if row.close<=frame.iloc[k-3:k+1].close.mean(): reason='lost_1h_mean'
            elif k==31: reason='max_8h'
        if reason:
            # If the next open gaps below the structural stop, the open is still
            # the attainable exit price. Do not assume a fill at the old stop.
            return result(candidate,frame,k+1,float(frame.iloc[k+1].open),reason,arm,at_open=True,activated=activated)
    raise AssertionError('No exit')

def validate_window(candidate,frame,end):
    if len(frame)!=33: raise ValueError(f'Expected 33 bars, got {len(frame)}')
    if frame.iloc[0].open_time!=pd.Timestamp(candidate['entry_time']): raise ValueError('Wrong entry time')
    if not frame.open_time.diff().iloc[1:].eq(BAR).all(): raise ValueError('Data gap')
    if frame.iloc[-1].open_time+BAR>end: raise ValueError('Right censored')
    prices=frame[['open','high','low','close']]
    if not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any(): raise ValueError('Invalid prices')
    if not np.isclose(frame.iloc[0].open,candidate['entry'],rtol=1e-12,atol=0): raise ValueError('Entry changed')
    old=legacy(candidate,frame)
    for key in ['exit','pnl_usdt','net_R']:
        if not np.isclose(old[key],candidate[key],rtol=1e-10,atol=1e-10): raise ValueError(f'Legacy mismatch {key}')
    if str(old['exit_time'])!=str(pd.Timestamp(candidate['exit_time'])) or old['exit_reason']!=candidate['exit_reason']:
        raise ValueError('Legacy exit time/reason mismatch')
    n=int(candidate['bars_held']);path=frame.iloc[:n]
    for key,value in [('bar_mfe_upper_bound',path.high.max()/candidate['entry']-1),('bar_mae_lower_bound',path.low.min()/candidate['entry']-1)]:
        if not np.isclose(value,candidate[key],rtol=1e-10,atol=1e-10): raise ValueError(f'Legacy path summary mismatch {key}')

def paired_summary(base,other):
    b=pd.DataFrame(base).set_index('event_id');o=pd.DataFrame(other).set_index('event_id').loc[b.index]
    dp=o.pnl_usdt-b.pnl_usdt;dr=o.net_R-b.net_R
    day=pd.to_datetime(b.entry_time,utc=True).dt.strftime('%Y-%m-%d')
    groups=pd.DataFrame({'delta':dr,'day':day}).groupby('day').delta.agg(['sum','size'])
    rng=np.random.default_rng(20260910);draws=rng.integers(0,len(groups),(3000,len(groups)))
    means=groups['sum'].to_numpy()[draws].sum(axis=1)/groups['size'].to_numpy()[draws].sum(axis=1)
    changed=np.abs(dp)>1e-9
    winners=b.pnl_usdt>0;losers=~winners
    return dict(n=len(b),changed=int(changed.sum()),activated=int(o.activated.sum()),delta_pnl=float(dp.sum()),
        mean_delta_R=float(dr.mean()),entry_day_cluster_bootstrap_95CI=list(np.quantile(means,[.025,.975])),
        improved=int((dp>1e-9).sum()),worsened=int((dp< -1e-9).sum()),
        baseline_loser_delta=float(dp[losers].sum()),baseline_winner_delta=float(dp[winners].sum()),
        winners_to_losses=int(((b.pnl_usdt>0)&(o.pnl_usdt<=0)).sum()),losses_to_winners=int(((b.pnl_usdt<=0)&(o.pnl_usdt>0)).sum()),
        top5_baseline_winner_delta=float(dp.loc[b.nlargest(5,'pnl_usdt').index].sum()),
        changed_events=[dict(event_id=i,baseline_pnl=float(b.loc[i,'pnl_usdt']),variant_pnl=float(o.loc[i,'pnl_usdt']),delta=float(dp.loc[i]),reason=o.loc[i,'exit_reason']) for i in dp.index[changed]])

def analyse(source,rows,errors,coverage):
    original=pd.DataFrame([c for c in source['candidates'] if c['arm']=='confirmed'])
    old_accepted,_=chronological_portfolio(original,respect_equity=True)
    original_ids=set(old_accepted.event_id)
    arms={name:[c for c in rows if c['arm']==name] for name in ARMS}
    out=dict(status='INCOMPLETE' if errors else 'COMPLETE',errors=errors,coverage=coverage,
        requested_candidates=len(original),completed_candidates=len(arms['baseline']),
        original_admitted=len(original_ids),start=source['summary']['start'],end=source['summary']['end'],
        parameters={'weak_check_bars':4,'weak_mfe_R_below':.5,'weak_net_R_at_most':0,'extension_check_bars':16,'extension_net_R_at_least':.5,'max_hold_bars':32,'cost_round_trip':COST},
        original_legacy_portfolio=stats(old_accepted),portfolios={},paired_all={},paired_original_cohort={},halves={},
        limitations=['Exploratory retrospective study; this same period was already inspected, so halves are not clean OOS.',
        'Original current eligible universe has survivorship/selection lookahead risk. Frozen simplified confirmation policy, not full production strategy.',
        'Stop path uses 15m OHLC; intrabar stop exits timestamped at bar end. No floating drawdown, funding or liquidation simulation.',
        'Data retrieved again; each old entry/exit/PnL/MFE/MAE summary must reproduce, but full historical input hashes are unavailable for these shorter windows.',
        'All discretionary and timed exits execute next open; the old 4h-close baseline is audited separately.',
        'Bootstrap clusters by entry UTC day, does not remove all cross-day dependence; intervals are exploratory.',
        'Independent portfolio replay and fixed original-admitted cohort answer different questions; candidate PnL is not an account return.'])
    midpoint=pd.Timestamp(source['summary']['start'])+pd.Timedelta(days=50)
    for name,records in arms.items():
        accepted,rejected=chronological_portfolio(pd.DataFrame(records),respect_equity=True)
        out['portfolios'][name]=dict(**stats(accepted),rejected=len(rejected),exit_reasons=accepted.exit_reason.value_counts().to_dict(),
            by_month={str(k):dict(n=len(g),pnl=float(g.pnl_usdt.sum())) for k,g in accepted.groupby(accepted.exit_time.dt.strftime('%Y-%m'))})
        out['halves'][name]={label:stats(accepted.loc[mask]) for label,mask in [('first',accepted.entry_time<midpoint),('second',accepted.entry_time>=midpoint)]}
        out.setdefault('accepted',{})[name]=accepted.to_dict('records')
        out.setdefault('rejected',{})[name]=rejected.to_dict('records')
        if name!='baseline':
            out['paired_all'][name]=paired_summary(arms['baseline'],records)
            out['paired_original_cohort'][name]=paired_summary([c for c in arms['baseline'] if c['event_id'] in original_ids],[c for c in records if c['event_id'] in original_ids])
    return out

def main():
    import requests
    p=argparse.ArgumentParser();p.add_argument('--offline');args=p.parse_args()
    root=Path(__file__).resolve().parent
    source=json.loads(gzip.decompress((root/'reports/alt_entries_20260909/retry.json.gz').read_bytes()))
    candidates=[c for c in source['candidates'] if c['arm']=='confirmed']
    end=pd.Timestamp(source['summary']['end'])
    def worker(c):
        try:
            start=pd.Timestamp(c['entry_time']);finish=start+33*BAR
            if args.offline:
                data=pd.DataFrame(json.loads(Path(args.offline).read_text())['windows'][c['event_id']])
            else:
                for attempt in range(3):
                    try:
                        r=requests.get('https://api.bybit.com/v5/market/kline',params=dict(category='linear',symbol=c['symbol'],interval='15',start=int(start.timestamp()*1000),end=int(finish.timestamp()*1000)-1,limit=1000),timeout=30)
                        r.raise_for_status();j=r.json()
                        if j.get('retCode')!=0: raise RuntimeError(str(j))
                        data=pd.DataFrame(j['result']['list'],columns=['ts','open','high','low','close','volume','turnover'])
                        data['open_time']=pd.to_datetime(pd.to_numeric(data.ts),unit='ms',utc=True)
                        data=data.drop(columns='ts');break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(attempt+1)
            data['open_time']=pd.to_datetime(data.open_time,utc=True)
            for col in ['open','high','low','close','volume','turnover']:data[col]=pd.to_numeric(data[col])
            data=data.sort_values('open_time').reset_index(drop=True)
            validate_window(c,data,end)
            rows=[simulate(c,data,arm) for arm in ARMS]
            cv=dict(event_id=c['event_id'],bars=len(data),sha256=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest())
            print('EXIT_OK '+c['event_id'],flush=True)
            return rows,data.to_dict('records'),cv,None
        except Exception as ex:
            error=dict(event_id=c['event_id'],error=str(ex));print('EXIT_FAILED '+json.dumps(error),flush=True)
            return [],None,None,error
    rows=[];windows={};coverage=[];errors=[]
    print('EXIT_STUDY_START '+json.dumps({'candidates':len(candidates),'symbols':len({c['symbol'] for c in candidates})}),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for c,(tr,window,cv,error) in zip(candidates,pool.map(worker,candidates)):
            if error:errors.append(error)
            else:rows.extend(tr);windows[c['event_id']]=window;coverage.append(cv)
    if not rows:raise RuntimeError('No validated windows')
    summary=analyse(source,rows,errors,coverage)
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    archive=dict(summary=summary,windows=windows,candidates=rows)
    packed=gzip.compress(json.dumps(archive,default=str,allow_nan=False).encode(),mtime=0)
    b64=base64.b64encode(packed).decode();chunks=[b64[i:i+6000] for i in range(0,len(b64),6000)]
    for i,chunk in enumerate(chunks):
        print('EXIT_ARCHIVE '+json.dumps(dict(index=i,total=len(chunks),data=chunk)),flush=True);time.sleep(.1)
    headline={k:v for k,v in summary.items() if k not in ['coverage','accepted','rejected']}
    for scope in ['paired_all','paired_original_cohort']:
        headline[scope]={k:{a:b for a,b in v.items() if a!='changed_events'} for k,v in headline[scope].items()}
    print('EXIT_REPORT '+json.dumps(headline,default=str,allow_nan=False),flush=True)
    print('EXIT_ARCHIVE_END '+json.dumps(dict(chunks=len(chunks),sha256=hashlib.sha256(packed).hexdigest())),flush=True)
    return 2 if errors else 0

if __name__=='__main__':raise SystemExit(main())
