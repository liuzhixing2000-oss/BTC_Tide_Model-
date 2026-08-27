#!/usr/bin/env python3
"""Six-month replay of the CURRENT Tide V10.9 policy.

Purpose
-------
Estimate how the present Tide policy would have behaved over ~180 days using:
- current V10.9 signal construction / scoring / grading code;
- current frozen bundle/watchlist;
- A+/A/A-/B+ only;
- structure stop + fixed 4h exit;
- 1000 USDT account, 1R=20 USDT;
- score risk tiers 5/10/15/20 USDT;
- max open planned risk 60 USDT, max daily realised loss 40 USDT;
- fee/slippage exactly as the engine uses: FEE_SLIPPAGE per completed trade.

IMPORTANT: this is a POLICY REPLAY, not a clean OOS proof. The current bundle and
its historical symbol score were selected using information available later than
some bars in the replay window. Output explicitly labels this survivorship/lookahead
limitation. Forward ledger data remains the decisive OOS test going forward.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BYBIT='https://api.bybit.com/v5/market/kline'
ACCOUNT_START=1000.0
ONE_R=20.0
MAX_OPEN_RISK=60.0
MAX_DAILY_LOSS=40.0
MAX_POSITIONS=12
MAX_MARGIN=800.0
DISPLAY_LEVERAGE=50.0
MIN_NOTIONAL=100.0
MAX_NOTIONAL=2500.0


def risk_usdt(score: float) -> float:
    if score >= 90: return 20.0
    if score >= 80: return 15.0
    if score >= 70: return 10.0
    return 5.0


def fetch_range(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Bybit inverse pagination, then return ascending completed candles."""
    rows=[]; cursor_end=int(end.timestamp()*1000); start_ms=int(start.timestamp()*1000)
    while cursor_end > start_ms:
        r=requests.get(BYBIT,params={'category':'linear','symbol':symbol,'interval':interval,'start':start_ms,'end':cursor_end,'limit':1000},timeout=30)
        r.raise_for_status(); j=r.json()
        if j.get('retCode')!=0: raise RuntimeError(j)
        batch=j.get('result',{}).get('list',[])
        if not batch: break
        rows.extend(batch)
        oldest=min(int(x[0]) for x in batch)
        if oldest <= start_ms or len(batch)<1000: break
        cursor_end=oldest-1; time.sleep(.03)
    if not rows:return pd.DataFrame(columns=['open_time','open','high','low','close','volume','turnover'])
    df=pd.DataFrame(rows,columns=['ts','open','high','low','close','volume','turnover'])
    df['ts']=pd.to_numeric(df.ts);df=df.drop_duplicates('ts').sort_values('ts')
    df=df[(df.ts>=start_ms)&(df.ts<int(end.timestamp()*1000))]
    df['open_time']=pd.to_datetime(df.ts,unit='ms',utc=True)
    for c in ['open','high','low','close','volume','turnover']:df[c]=pd.to_numeric(df[c],errors='coerce')
    return df[['open_time','open','high','low','close','volume','turnover']].reset_index(drop=True)


def load_engine(repo: Path):
    # Engine runtime paths may point into TIDE_DATA_DIR. Use the checked-in bundle.
    os.environ.setdefault('TIDE_DATA_DIR',str((repo/'v10_bundle').resolve()))
    spec=importlib.util.spec_from_file_location('tide_engine',repo/'crypto_tide_engine_v10_9_dynamic_risk.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def htf_snapshot(engine, df: pd.DataFrame, hours: int, signal_close: pd.Timestamp) -> dict:
    if hasattr(engine,'_htf_trend_snapshot'):
        return engine._htf_trend_snapshot(df,hours,signal_close)
    w=df.copy();w['open_time']=pd.to_datetime(w.open_time,utc=True);w=w[w.open_time+pd.Timedelta(hours=hours)<=signal_close]
    if len(w)<205:return {'trend':'MIXED','strength':50.0}
    c=w.close.astype(float);e50=c.ewm(span=50,adjust=False).mean();e200=c.ewm(span=200,adjust=False).mean();s=e50.pct_change(3)
    cc=float(c.iloc[-1]);a=float(e50.iloc[-1]);b=float(e200.iloc[-1]);ss=float(s.iloc[-1])
    trend='UP' if cc>a>b and ss>0 else ('DOWN' if cc<a<b and ss<0 else 'MIXED')
    strength=float(np.clip(50+20*np.tanh(ss*200)+15*np.tanh((cc/a-1)*50)+15*np.tanh((a/b-1)*30),0,100))
    return {'trend':trend,'strength':strength}


def historical_score(row: pd.Series) -> float | None:
    try:
        x=float(row.get('score',np.nan));return None if not np.isfinite(x) else x
    except Exception:return None


def candidate_trades(engine, symbol, hist_score, df15, df1h, df4h, entry_params, test_start, test_end):
    frame=engine.model_frame(df15,df1h,entry_params)
    if frame.empty or 'signal' not in frame.columns:return []
    out=[]
    signal_idx=np.flatnonzero(frame['signal'].fillna(False).astype(bool).to_numpy())
    for i in signal_idx:
        if i+16>=len(frame):continue
        latest=frame.iloc[i]
        signal_close=pd.Timestamp(latest['open_time'])+pd.Timedelta(minutes=15)
        if not(test_start<=signal_close.to_pydatetime()<test_end):continue
        score=float(engine.signal_score(latest,hist_score))
        # Historical replay uses CAUTION=70 as the neutral market requirement.
        assessment=engine.production_entry_assessment(latest,score,{'minimum_signal_score':70.0})
        h1=htf_snapshot(engine,df1h,1,signal_close);h4=htf_snapshot(engine,df4h,4,signal_close)
        htf={'h1_trend':h1.get('trend','MIXED'),'h1_strength':float(h1.get('strength',50.0)),
             'h4_trend':h4.get('trend','MIXED'),'h4_strength':float(h4.get('strength',50.0))}
        grade=engine.tide_grade(latest,score,assessment,htf)
        if grade.get('grade') not in {'A+','A','A-','B+'}:continue
        risk=assessment['risk'];entry=float(risk['entry_price']);stop=float(risk['hard_stop']);stop_pct=max((entry-stop)/entry,1e-9)
        planned=risk_usdt(score);notional=float(np.clip(planned/stop_pct,MIN_NOTIONAL,MAX_NOTIONAL))
        # If capped notional cannot consume planned R, actual risk is lower.
        actual_risk=notional*stop_pct
        exit_idx=i+16;exit_price=float(frame.iloc[exit_idx]['close']);reason='fixed_4h';hit_idx=None
        for j in range(i+1,min(i+17,len(frame))):
            if float(frame.iloc[j]['low'])<=stop:
                hit_idx=j;exit_idx=j;exit_price=stop;reason='hard_structure_stop';break
        gross=exit_price/entry-1;net=gross-float(engine.FEE_SLIPPAGE);pnl=notional*net
        window=frame.iloc[i+1:exit_idx+1]
        mfe=(float(window.high.max())/entry-1) if not window.empty else 0.0
        mae=(float(window.low.min())/entry-1) if not window.empty else 0.0
        out.append({'symbol':symbol,'entry_time':signal_close,'exit_time':pd.Timestamp(frame.iloc[exit_idx]['open_time'])+pd.Timedelta(minutes=15),
                    'grade':grade['grade'],'route':'BPLUS' if grade['grade']=='B+' else ('NEXT_RESCUE' if any('NEXT RESCUE' in str(x) for x in grade.get('reasons',[])) else 'CORE'),
                    'signal_score':score,'raw_quality':float(latest.get('raw_quality_score',np.nan)),'next_quality':float(assessment['next_quality']),
                    'combined':float(assessment['combined_quality']),'confirmation_tests':int(assessment['confirmation_tests']),
                    'volume_multiple':float(latest.get('volume_multiple',np.nan)),'h4_trend':htf['h4_trend'],'h1_trend':htf['h1_trend'],'h1_strength':htf['h1_strength'],
                    'entry':entry,'stop':stop,'stop_pct':stop_pct,'planned_risk_usdt':planned,'actual_risk_usdt':actual_risk,'notional':notional,
                    'margin_at_50x':notional/DISPLAY_LEVERAGE,'exit':exit_price,'exit_reason':reason,'gross_return':gross,'net_return':net,'pnl_usdt':pnl,
                    'mfe':mfe,'mae':mae,'bars_held':exit_idx-i})
    return out


def portfolio(candidates: pd.DataFrame):
    if candidates.empty:return candidates.copy(),pd.DataFrame()
    c=candidates.sort_values(['entry_time','grade','signal_score'],ascending=[True,True,False]).copy()
    active=[];accepted=[];rejected=[];equity=ACCOUNT_START;daily_loss={}
    grade_priority={'A+':0,'A':1,'A-':2,'B+':3}
    c['_gp']=c.grade.map(grade_priority).fillna(9);c=c.sort_values(['entry_time','_gp','signal_score'],ascending=[True,True,False])
    for row in c.to_dict('records'):
        t=pd.Timestamp(row['entry_time']);active=[a for a in active if pd.Timestamp(a['exit_time'])>t]
        day=t.tz_convert('Australia/Sydney').date().isoformat();loss=daily_loss.get(day,0.0)
        reasons=[]
        if len(active)>=MAX_POSITIONS:reasons.append('max_positions')
        if sum(float(a['actual_risk_usdt']) for a in active)+row['actual_risk_usdt']>MAX_OPEN_RISK+1e-9:reasons.append('max_open_risk')
        if sum(float(a['margin_at_50x']) for a in active)+row['margin_at_50x']>MAX_MARGIN+1e-9:reasons.append('max_margin')
        if loss>=MAX_DAILY_LOSS-1e-9:reasons.append('daily_loss_limit')
        if reasons:row['admitted']=False;row['reject_reason']='|'.join(reasons);rejected.append(row);continue
        row['admitted']=True;row['reject_reason']='';row['equity_before']=equity;equity+=row['pnl_usdt'];row['equity_after']=equity
        if row['pnl_usdt']<0:daily_loss[day]=daily_loss.get(day,0.0)-row['pnl_usdt']
        accepted.append(row);active.append(row)
    a=pd.DataFrame(accepted);r=pd.DataFrame(rejected)
    return a,r


def summary(trades: pd.DataFrame,start,end,candidates,rejected):
    if trades.empty:return {'start':str(start),'end':str(end),'trades':0,'ending_equity':ACCOUNT_START,'net_pnl':0.0,'return_pct':0.0}
    pnl=trades.pnl_usdt.astype(float);wins=pnl>0;eq=ACCOUNT_START+pnl.cumsum();peak=eq.cummax();dd=eq/peak-1
    gp=pnl[pnl>0].sum();gl=-pnl[pnl<0].sum();pf=float(gp/gl) if gl>0 else math.inf
    return {'start':str(start),'end':str(end),'candidates':int(len(candidates)),'admitted':int(len(trades)),'rejected_by_portfolio':int(len(rejected)),
            'ending_equity':float(eq.iloc[-1]),'net_pnl':float(pnl.sum()),'return_pct':float(eq.iloc[-1]/ACCOUNT_START-1),'win_rate':float(wins.mean()),
            'profit_factor':pf,'avg_pnl_usdt':float(pnl.mean()),'median_pnl_usdt':float(pnl.median()),'max_drawdown':float(dd.min()),
            'stop_rate':float((trades.exit_reason=='hard_structure_stop').mean()),'avg_net_return':float(trades.net_return.mean())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--days',type=int,default=180);ap.add_argument('--end',default=None);ap.add_argument('--out',default='reports/v10_9_six_month');ap.add_argument('--max-symbols',type=int,default=0);args=ap.parse_args()
    repo=Path(__file__).resolve().parent;out=Path(args.out);out.mkdir(parents=True,exist_ok=True);engine=load_engine(repo)
    end=datetime.fromisoformat(args.end.replace('Z','+00:00')) if args.end else datetime.now(timezone.utc);end=end.astimezone(timezone.utc);start=end-timedelta(days=args.days)
    # Extra warm-up is essential for EMA200 / raw lookbacks.
    fetch15=start-timedelta(days=15);fetch1=start-timedelta(days=25);fetch4=start-timedelta(days=45)
    stage=pd.read_csv(repo/'v10_bundle'/'stage2_full_results.csv');eligible=stage[stage.eligible.astype(str).str.lower().eq('true')].copy()
    if args.max_symbols>0:eligible=eligible.head(args.max_symbols)
    entry_cfg=json.loads((repo/'v10_bundle'/'entry_parameter_config.json').read_text());entry_params=entry_cfg.get('parameters') or entry_cfg.get('entry_parameters') or getattr(engine,'ACTIVE_ENTRY_PARAMS',None)
    if not entry_params:entry_params={'lookback_bars':24,'volume_lookback':24,'volume_multiplier':1.5,'lower_wick_threshold':0.35,'cooldown_bars':24}
    print(f'V10.9 six-month replay {start.isoformat()} -> {end.isoformat()} | symbols={len(eligible)}')
    alltr=[];errors=[]
    for n,row in enumerate(eligible.itertuples(index=False),1):
        sym=str(row.symbol);print(f'[{n}/{len(eligible)}] {sym}',flush=True)
        try:
            d15=fetch_range(sym,'15',fetch15,end);d1=fetch_range(sym,'60',fetch1,end);d4=fetch_range(sym,'240',fetch4,end)
            if len(d15)<500 or len(d1)<220 or len(d4)<205:raise RuntimeError(f'insufficient data 15={len(d15)} 1h={len(d1)} 4h={len(d4)}')
            hs=historical_score(pd.Series(row._asdict()));tr=candidate_trades(engine,sym,hs,d15,d1,d4,entry_params,start,end);alltr.extend(tr);print('  candidates',len(tr))
        except Exception as ex:errors.append({'symbol':sym,'error':repr(ex)});print('  ERROR',repr(ex))
        time.sleep(.05)
    cand=pd.DataFrame(alltr);accepted,rejected=portfolio(cand);s=summary(accepted,start,end,cand,rejected)
    if not cand.empty:cand.to_csv(out/'candidate_trades.csv',index=False)
    if not accepted.empty:accepted.to_csv(out/'portfolio_trades.csv',index=False)
    if not rejected.empty:rejected.to_csv(out/'portfolio_rejections.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'errors.csv',index=False)
    by_grade=[]
    if not accepted.empty:
        for g,x in accepted.groupby('grade'):
            by_grade.append({'grade':g,'trades':len(x),'win_rate':float((x.pnl_usdt>0).mean()),'net_pnl':float(x.pnl_usdt.sum()),'avg_net_return':float(x.net_return.mean()),'stop_rate':float((x.exit_reason=='hard_structure_stop').mean())})
    pd.DataFrame(by_grade).to_csv(out/'by_grade.csv',index=False)
    (out/'summary.json').write_text(json.dumps(s,indent=2,default=str))
    lines=['# Tide V10.9 — current-policy six-month replay','',f"Window: {start.isoformat()} → {end.isoformat()}",f"Current eligible symbols replayed: {len(eligible)}",'',
           '## Portfolio assumptions','', '- Starting account: **1000 USDT**','- 1R: **20 USDT (2%)**','- Score risk tiers: <70=5U, 70–79=10U, 80–89=15U, >=90=20U','- Max simultaneous planned risk: **60U**','- Daily realised-loss gate: **40U**','- Maximum positions: **12**','- Exit: current structure stop or fixed 4h','- Costs: engine `FEE_SLIPPAGE` deducted once per completed trade','',
           '## Headline','',f"- Candidate signals: **{s.get('candidates',0)}**",f"- Admitted trades: **{s.get('admitted',s.get('trades',0))}**",f"- Ending equity: **{s.get('ending_equity',ACCOUNT_START):.2f} USDT**",f"- Net P/L: **{s.get('net_pnl',0):+.2f} USDT**",f"- Return: **{s.get('return_pct',0):+.2%}**",f"- Win rate: **{s.get('win_rate',0):.1%}**",f"- Profit factor: **{s.get('profit_factor',0):.2f}**",f"- Max drawdown: **{s.get('max_drawdown',0):.2%}**",f"- Stop rate: **{s.get('stop_rate',0):.1%}**",'',
           '## Interpretation warning','', '**This is not a clean out-of-sample validation.** It replays today’s frozen V10.9 policy and today’s eligible-symbol bundle backward through the last six months. Because the current bundle was selected with later information, symbol selection / historical score contain survivorship-lookahead. Use this to understand policy behaviour and account risk, not to claim a proven live edge. The new forward ledger is the clean OOS test.','']
    if by_grade:
        lines+=['## By grade','']+[f"- {x['grade']}: n={x['trades']}, win={x['win_rate']:.1%}, P/L={x['net_pnl']:+.2f}U, avg return={x['avg_net_return']:+.3%}, stop={x['stop_rate']:.1%}" for x in by_grade]
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()
