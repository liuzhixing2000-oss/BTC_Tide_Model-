"""BTC background and causal altcoin levels for frozen Tide candidates."""
import base64,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd

def frame(rows,minutes,cutoff):
    d=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume','turnover'])
    for col in d:d[col]=pd.to_numeric(d[col])
    d['time']=pd.to_datetime(d.ts,unit='ms',utc=True)
    d=d[d.time+pd.Timedelta(minutes=minutes)<=cutoff].sort_values('time').reset_index(drop=True)
    need=200 if minutes in [60,240] else 97
    if len(d)<need:raise ValueError(f'Insufficient {minutes} bars')
    if not d.time.diff().iloc[1:].eq(pd.Timedelta(minutes=minutes)).all():raise ValueError('Input gap')
    if d.iloc[-1].time+pd.Timedelta(minutes=minutes)!=cutoff.floor(f'{minutes}min'):raise ValueError('Stale cutoff')
    if not np.isfinite(d[['open','high','low','close']].to_numpy()).all():raise ValueError('Invalid prices')
    return d

def trend(d):
    ma50=d.close.iloc[-50:].mean();ma200=d.close.iloc[-200:].mean()
    return dict(down=bool(ma50<ma200),price_above_ma200=bool(d.close.iloc[-1]>ma200),ma50_slope_12h=float(ma50/d.close.iloc[-53:-3].mean()-1))

def context(d,f,cutoff):
    price=float(d.close.iloc[-1]);prev=f.close.shift()
    tr=pd.concat([f.high-f.low,(f.high-prev).abs(),(f.low-prev).abs()],axis=1).max(axis=1)
    atr=float(tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean().iloc[-1])
    if atr<=0:raise ValueError('Invalid ATR')
    day=cutoff.floor('D');week=day-pd.Timedelta(days=day.weekday())
    yesterday=f[(f.time>=day-pd.Timedelta(days=1))&(f.time<day)]
    lastweek=f[(f.time>=week-pd.Timedelta(days=7))&(f.time<week)]
    # Both ranges end before the recent24h test interval starts.
    old7=f[(f.time>=cutoff.floor('4h')-pd.Timedelta(days=8))&(f.time<cutoff.floor('4h')-pd.Timedelta(days=1))]
    if len(yesterday)!=6 or len(lastweek)!=42 or len(old7)!=42:raise ValueError('Incomplete level history')
    recent_low=float(d.low.iloc[-96:].min());oldlow=float(old7.low.min());weeklow=float(lastweek.low.min())
    out={'price':price,'return_6h':float(price/d.close.iloc[-25]-1),'return_24h':float(price/d.close.iloc[-97]-1),
       'return_7d':float(f.close.iloc[-1]/f.close.iloc[-43]-1),'return_30d':float(f.close.iloc[-1]/f.close.iloc[-181]-1),
       'distance_30d_high':float(price/f.high.iloc[-180:].max()-1),'atr4h':atr,
       'previous_day_low':float(yesterday.low.min()),'previous_week_low':weeklow,'old7d_low':oldlow,
       'previous_day_low_dist_atr':float((price-yesterday.low.min())/atr),
       'previous_week_low_dist_atr':float((price-weeklow)/atr),'old7d_low_dist_atr':float((price-oldlow)/atr),
       'near_previous_week_low':bool(abs(price-weeklow)<=atr),
       'reclaimed_old7d_low':bool(recent_low<oldlow and price>oldlow),
       'ma50_slope_12h':trend(f)['ma50_slope_12h'],
       'down_4h':trend(f)['down'],'above_4h_ma200':trend(f)['price_above_ma200']}
    return out

NUMERIC=['btc_return_6h','btc_return_24h','btc_return_7d','btc_ma50_slope_12h',
 'alt_return_7d','alt_return_30d','alt_distance_30d_high','alt_ma50_slope_12h',
 'alt_previous_day_low_dist_atr','alt_previous_week_low_dist_atr','alt_old7d_low_dist_atr',
 'relative_6h','relative_24h']
BINARY=['btc_down_1h','btc_down_4h','btc_reclaimed_old7d_low','alt_down_4h',
 'alt_near_previous_week_low','alt_reclaimed_old7d_low']

def group_stats(d):
    big=d.net_R>=2
    return {'n':len(d),'big_winners':int(big.sum()),'big_winner_rate':float(big.mean()) if len(d) else None,
      'win_rate':float((d.net_R>0).mean()) if len(d) else None,'mean_R':float(d.net_R.mean()) if len(d) else None}

def analyse(records):
    d=pd.DataFrame(records);out={}
    for scope,s in [('all191',d),('original134',d[d.admitted])]:
        groups={name:s[mask] for name,mask in [('big',s.net_R>=2),('ordinary_win',(s.net_R>0)&(s.net_R<2)),('loss',s.net_R<=0)]}
        out[scope]={'groups':{k:group_stats(v) for k,v in groups.items()},'medians':{},'conditions':{},'btc_regimes':{str(k):group_stats(g) for k,g in s.groupby('btc_regime')}}
        for col in NUMERIC:out[scope]['medians'][col]={k:float(g[col].median()) for k,g in groups.items()}
        for col in BINARY:out[scope]['conditions'][col]={str(k):group_stats(g) for k,g in s.groupby(col)}
        out[scope]['big_winner_records']=s[s.net_R>=2].to_dict('records')
    return out

def main():
    import requests
    root=Path(__file__).resolve().parent
    alt_path=root/'reports/market_context_20260913/alt_context.json'
    alts=json.loads(alt_path.read_text());cuts=[pd.Timestamp(x['cutoff']) for x in alts]
    end=max(cuts);raw={}
    for interval,warmup in [(15,2),(60,12),(240,45)]:
        start=min(cuts)-pd.Timedelta(days=warmup);cursor=int(end.timestamp()*1000)-1;rows=[]
        while cursor>int(start.timestamp()*1000):
            for attempt in range(3):
                try:
                    r=requests.get('https://api.bybit.com/v5/market/kline',params=dict(category='linear',symbol='BTCUSDT',interval=str(interval),start=int(start.timestamp()*1000),end=cursor,limit=1000),timeout=30);r.raise_for_status();j=r.json()
                    if j.get('retCode')!=0:raise RuntimeError(str(j))
                    batch=j['result']['list'];break
                except Exception:
                    if attempt==2:raise
                    time.sleep(attempt+1)
            if not batch:break
            rows.extend(batch);cursor=min(int(x[0]) for x in batch)-1
            if len(batch)<1000:break
        raw[str(interval)]=rows;print(f'BTC_LOADED {interval} {len(rows)}',flush=True)
    records=[];errors=[]
    for a in alts:
        try:
            cutoff=pd.Timestamp(a['cutoff']);d=frame(raw['15'],15,cutoff);h=frame(raw['60'],60,cutoff);f=frame(raw['240'],240,cutoff)
            bc=context(d,f,cutoff);bc['down_1h']=trend(h)['down']
            r={**a,**{'btc_'+k:v for k,v in bc.items()}}
            r['btc_regime']=('1h_down' if r['btc_down_1h'] else '1h_up')+'/'+('4h_down' if r['btc_down_4h'] else '4h_up')
            for window in ['6h','24h']:r['relative_'+window]=(1+r['alt_return_'+window])/(1+r['btc_return_'+window])-1
            records.append(r)
        except Exception as ex:errors.append({'event_id':a['event_id'],'error':str(ex)})
    summary={'status':'INCOMPLETE' if errors else 'COMPLETE','requested':len(alts),'completed':len(records),'errors':errors,
      'analysis':analyse(records),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'alt_file_sha256':hashlib.sha256(alt_path.read_bytes()).hexdigest(),
      'limitations':['Exploratory on previously inspected100days; only8 >=2R candidates,4 originally admitted. Not OOS or causal inference.',
      'BTC is background only, never a traded candidate. Relative returns and key levels use pre-trigger completed candles.',
      'Up/down labels mean SMA50 versus SMA200, not a complete higher-high/lower-low chart classification.',
      'Previous day/week use UTC calendar; old7d level excludes recent24h. Near means within1 pre-trigger4hATR, not demonstrated support.',
      '30day returns use last completed4h bar,6h/24h use15m. Key level price is last completed15m close.',
      'Shared dates/symbols/regime and many examined conditions limit effective sample size. No threshold search or live changes.']}
    packed=gzip.compress(json.dumps({'summary':summary,'records':records,'btc_rows':raw},allow_nan=False).encode(),mtime=0)
    enc=base64.b64encode(packed).decode();chunks=[enc[i:i+6000] for i in range(0,len(enc),6000)]
    for i,c in enumerate(chunks):print('MARKET_ARCHIVE '+json.dumps({'index':i,'total':len(chunks),'data':c}),flush=True);time.sleep(.1)
    print('MARKET_REPORT '+json.dumps({k:v for k,v in summary.items() if k!='analysis'}),flush=True)
    print('MARKET_ARCHIVE_END '+json.dumps({'chunks':len(chunks),'sha256':hashlib.sha256(packed).hexdigest()}),flush=True)
    return 2 if errors else 0

if __name__=='__main__':raise SystemExit(main())
