"""Descriptive pre-trigger context study. No strategy fitting or threshold search."""
import base64,gzip,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from tide_replay_support import chronological_portfolio

FEATURES=['return_6h','return_24h','return_72h','consecutive_down_1h',
 'distance_1h_ma50','distance_1h_ma200','distance_4h_ma50','distance_4h_ma200',
 'distance_prior_24h_low','atr_pct','atr_ratio','range_ratio_6h','volume_ratio_6h',
 'down_volume_ratio_12h','rsi14','rsi_change_6h','macd_hist_atr','rsi_divergence','macd_divergence','both_htf_down']

def clean(rows,interval,cutoff):
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume','turnover'])
    for c in d.columns:d[c]=pd.to_numeric(d[c])
    d['time']=pd.to_datetime(d.ts,unit='ms',utc=True)
    d=d[d.time+pd.Timedelta(minutes=interval)<=cutoff].sort_values('time').reset_index(drop=True)
    if len(d)<220:raise ValueError('Insufficient warmup')
    if not d.time.diff().iloc[1:].eq(pd.Timedelta(minutes=interval)).all():raise ValueError('Gap')
    if d.iloc[-1].time+pd.Timedelta(minutes=interval)!=cutoff.floor(f'{interval}min'):raise ValueError('Stale cutoff')
    if not np.isfinite(d[['open','high','low','close','volume']].to_numpy()).all():raise ValueError('Invalid data')
    if (d[['open','high','low','close']]<=0).any().any():raise ValueError('Nonpositive price')
    return d

def indicators(d,h,f):
    c=d.close;delta=c.diff();gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=(100-100/(1+gain/loss)).where(loss!=0,100).where((gain+loss)!=0,50)
    tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean();hist=macd-macd.ewm(span=9,adjust=False).mean()
    recent=d.iloc[-12:];older=d.iloc[-24:-12];i=recent.low.idxmin();j=older.low.idxmin();lower=d.loc[i,'low']<d.loc[j,'low']
    ndown=0
    for value in h.close.diff().iloc[1:].iloc[::-1]:
        if value>=0:break
        ndown+=1
    downv=d.volume.where(delta<0,0)
    safe=lambda a,b:float(a/b) if b>0 else None
    out={f'return_{n}h':float(c.iloc[-1]/c.iloc[-1-4*n]-1) for n in [6,24,72]}
    out.update(consecutive_down_1h=ndown,distance_prior_24h_low=float(c.iloc[-1]/d.low.iloc[-97:-1].min()-1),
        atr_pct=float(atr.iloc[-1]/c.iloc[-1]),atr_ratio=safe(atr.iloc[-1],atr.iloc[-57:-1].mean()),
        range_ratio_6h=safe(d.high.iloc[-24:].max()-d.low.iloc[-24:].min(),d.high.iloc[-48:-24].max()-d.low.iloc[-48:-24].min()),
        volume_ratio_6h=safe(d.volume.iloc[-24:].mean(),d.volume.iloc[-96:-24].mean()),
        down_volume_ratio_12h=safe(downv.iloc[-48:].sum(),downv.iloc[-96:-48].sum()),
        rsi14=float(rsi.iloc[-1]),rsi_change_6h=float(rsi.iloc[-1]-rsi.iloc[-25]),macd_hist_atr=safe(hist.iloc[-1],atr.iloc[-1]),
        rsi_divergence=int(lower and rsi.iloc[i]>rsi.iloc[j]),macd_divergence=int(lower and hist.iloc[i]>hist.iloc[j]))
    for label,frame in [('1h',h),('4h',f)]:
        for n in [50,200]:out[f'distance_{label}_ma{n}']=float(frame.close.iloc[-1]/frame.close.iloc[-n:].mean()-1)
    out['both_htf_down']=int(h.close.iloc[-50:].mean()<h.close.iloc[-200:].mean() and f.close.iloc[-50:].mean()<f.close.iloc[-200:].mean())
    if any(v is not None and not np.isfinite(v) for v in out.values()):raise ValueError('Invalid indicator')
    return out

def effect(d,feature):
    q=d[[feature,'net_R','pnl_usdt']].dropna()
    if len(q)<3 or q[feature].nunique()<2:return None
    return float(q[feature].rank().corr(q.net_R.rank()))

def perf(d):
    p=d.pnl_usdt;loss=-p[p<0].sum()
    return dict(n=len(d),win_rate=float((p>0).mean()) if len(d) else None,mean_R=float(d.net_R.mean()) if len(d) else None,
        median_R=float(d.net_R.median()) if len(d) else None,profit_factor=float(p[p>0].sum()/loss) if loss else None)

def analysis(records):
    d=pd.DataFrame(records);mid=pd.Timestamp('2026-07-21',tz='UTC');d['first']=pd.to_datetime(d.entry_time,utc=True)<mid
    out={}
    for scope,s in [('all_candidates',d),('original_admitted',d[d.admitted])]:
        first=s[s['first']];second=s[~s['first']];trim=s.drop(s.nlargest(5,'net_R').index)
        details={}
        for feature in FEATURES:
            valid=s.dropna(subset=[feature]);a=first[feature].dropna()
            if feature in ['rsi_divergence','macd_divergence','both_htf_down']:
                cuts=None;bucket=lambda x:x.map({0:'no',1:'yes'})
            elif len(a) and a.nunique()>2:
                cuts=list(np.unique(a.quantile([1/3,2/3]).to_numpy()));bucket=lambda x:pd.cut(x,[-np.inf]+cuts+[np.inf],labels=False)
            else:
                cuts=None;bucket=lambda x:x
            b=bucket(valid[feature]);groups={}
            for k,g in valid.groupby(b,observed=True):
                groups[str(k)]={'all':perf(g),'first':perf(g[g['first']]),'second':perf(g[~g['first']]),'without_top5':perf(g.loc[g.index.intersection(trim.index)])}
            correlations={'all':effect(s,feature),'first':effect(first,feature),'second':effect(second,feature),'without_top5':effect(trim,feature)}
            vals=list(correlations.values());same=all(x is not None and x>0 for x in vals) or all(x is not None and x<0 for x in vals)
            details[feature]=dict(missing=int(s[feature].isna().sum()),winner_median=float(s.loc[s.pnl_usdt>0,feature].median()),loser_median=float(s.loc[s.pnl_usdt<=0,feature].median()),
                rank_correlation=correlations,all_directions_agree=same,first_half_tercile_cuts=cuts,groups=groups)
        out[scope]=dict(n=len(s),first_n=len(first),second_n=len(second),features=details)
    return out

def main():
    import requests
    root=Path(__file__).resolve().parent
    source=json.loads(gzip.decompress((root/'reports/alt_entries_20260909/retry.json.gz').read_bytes()))
    cand=[c for c in source['candidates'] if c['arm']=='confirmed'];admitted,_=chronological_portfolio(pd.DataFrame(cand),respect_equity=True);ids=set(admitted.event_id)
    def worker(c):
        try:
            cutoff=pd.Timestamp(c['raw_time'])-pd.Timedelta(minutes=15);frames=[];raws={}
            for interval,count in [(15,672),(60,320),(240,250)]:
                end=cutoff.floor(f'{interval}min');start=end-pd.Timedelta(minutes=interval*count)
                for attempt in range(3):
                    try:
                        r=requests.get('https://api.bybit.com/v5/market/kline',params=dict(category='linear',symbol=c['symbol'],interval=str(interval),start=int(start.timestamp()*1000),end=int(end.timestamp()*1000)-1,limit=1000),timeout=30);r.raise_for_status();j=r.json()
                        if j.get('retCode')!=0:raise RuntimeError(str(j))
                        rows=j['result']['list'];frame=clean(rows,interval,cutoff)
                        if len(frame)!=count:raise ValueError(f'Missing {interval} history {len(frame)}/{count}')
                        frames.append(frame);raws[str(interval)]=rows;break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(attempt+1)
            record=dict(event_id=c['event_id'],symbol=c['symbol'],entry_time=c['entry_time'],cutoff=str(cutoff),net_R=c['net_R'],pnl_usdt=c['pnl_usdt'],admitted=c['event_id'] in ids,**indicators(*frames))
            print('PRE_OK '+c['event_id'],flush=True);return record,raws,None
        except Exception as ex:
            error=dict(event_id=c['event_id'],error=str(ex));print('PRE_FAILED '+json.dumps(error),flush=True);return None,None,error
    records=[];histories={};errors=[]
    print('PRE_START 191 frozen candidates; 20 prespecified descriptive features; pre-trigger cutoff.',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for c,(record,raw,error) in zip(cand,pool.map(worker,cand)):
            if error:errors.append(error)
            else:records.append(record);histories[c['event_id']]=raw
    summary=dict(status='INCOMPLETE' if errors else 'COMPLETE',errors=errors,requested=len(cand),completed=len(records),analyses=analysis(records),
        limitations=['Exploratory associations on previously inspected data, not OOS, causal effects or deployable filters.',
        '20 features examined; apparent patterns can occur by chance. No p-value significance claims or optimized entry thresholds.',
        'Cutoff is original signal bar OPEN; excludes original signal candle, confirmation candle and all subsequent prices.',
        'RSI/MACD divergence is a prespecified two-block lower-low/higher-indicator proxy, not retrospectively identified swing pivots.',
        'Bin cuts use first50 days, applied unchanged to second50; both periods previously seen, so not untouched validation.',
        'Candidate trades overlap and returns are correlated. Do not add candidate PnL as account returns.',
        'Current eligible pool has survivorship/lookahead selection bias; simplified confirmation and fixed4h outcomes, not full live strategy.'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    packed=gzip.compress(json.dumps(dict(summary=summary,records=records,histories=histories),allow_nan=False).encode(),mtime=0)
    enc=base64.b64encode(packed).decode();chunks=[enc[i:i+6000] for i in range(0,len(enc),6000)]
    for i,chunk in enumerate(chunks):print('PRE_ARCHIVE '+json.dumps(dict(index=i,total=len(chunks),data=chunk)),flush=True);time.sleep(.1)
    print('PRE_REPORT '+json.dumps({k:v for k,v in summary.items() if k!='analyses'}),flush=True)
    print('PRE_ARCHIVE_END '+json.dumps(dict(chunks=len(chunks),sha256=hashlib.sha256(packed).hexdigest())),flush=True)
    return 2 if errors else 0

if __name__=='__main__':raise SystemExit(main())
