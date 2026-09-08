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
import argparse, importlib.util, json, math, os, sys, tempfile, shutil, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from tide_replay_support import simulate_live_exit, chronological_portfolio

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
    df=df[(df.ts>=start_ms)&(df.ts+int(interval)*60000<=int(end.timestamp()*1000))]
    df['open_time']=pd.to_datetime(df.ts,unit='ms',utc=True)
    for c in ['open','high','low','close','volume','turnover']:df[c]=pd.to_numeric(df[c],errors='coerce')
    return df[['open_time','open','high','low','close','volume','turnover']].reset_index(drop=True)


def load_engine(repo: Path):
    # Load through the same account-profile wrapper as production, in scratch.
    import crypto_tide_v10_9_dynamic_risk_live as launcher
    runtime = Path(tempfile.mkdtemp(prefix='tide-replay-'))
    for name in ['entry_parameter_config.json','online_learning.json','market_regime.json']:
        shutil.copy2(repo/'v10_bundle'/name, runtime/name)
    return launcher.load_model(repo/'crypto_tide_engine_v10_9_dynamic_risk.py', runtime)


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
    # Match live has_scored_setup, including the existing Rescue/B+ paths.
    signal_idx=np.flatnonzero(frame['combined_setup_score'].notna().to_numpy())
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
        sizing=engine.dynamic_position_size(score,risk)
        planned=sizing['planned_loss_usdt'];notional=sizing['suggested_notional_usdt']
        actual_risk=sizing['effective_planned_loss_usdt']
        result=simulate_live_exit(engine,frame,i,risk)
        if result is None:continue
        exit_idx,exit_price,reason,_=result
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
                    'margin':sizing['suggested_margin_usdt'],'leverage':sizing['suggested_leverage'],'exit':exit_price,'exit_reason':reason,'gross_return':gross,'net_return':net,'pnl_usdt':pnl,
                    'mfe':mfe,'mae':mae,'bars_held':exit_idx-i})
    return out


def portfolio(candidates: pd.DataFrame, engine=None):
    if engine is None:
        return chronological_portfolio(candidates)
    return chronological_portfolio(candidates, capital=engine.PORTFOLIO_CAPITAL_USDT,
        max_positions=engine.PORTFOLIO_MAX_POSITIONS,
        max_open_risk=engine.PORTFOLIO_CAPITAL_USDT*engine.MAX_TOTAL_OPEN_RISK_PCT,
        max_margin=engine.PORTFOLIO_CAPITAL_USDT*engine.PORTFOLIO_MAX_MARGIN_UTILISATION,
        max_daily_loss=engine.PORTFOLIO_CAPITAL_USDT*engine.MAX_DAILY_LOSS_PCT)


def summary(trades: pd.DataFrame,start,end,candidates,rejected):
    if trades.empty:return {'start':str(start),'end':str(end),'trades':0,'ending_equity':ACCOUNT_START,'net_pnl':0.0,'return_pct':0.0}
    pnl=trades.pnl_usdt.astype(float);wins=pnl>0;eq=ACCOUNT_START+pnl.cumsum();peak=eq.cummax().clip(lower=ACCOUNT_START);dd=eq/peak-1
    gp=pnl[pnl>0].sum();gl=-pnl[pnl<0].sum();pf=float(gp/gl) if gl>0 else math.inf
    return {'start':str(start),'end':str(end),'candidates':int(len(candidates)),'admitted':int(len(trades)),'rejected_by_portfolio':int(len(rejected)),
            'ending_equity':float(eq.iloc[-1]),'net_pnl':float(pnl.sum()),'return_pct':float(eq.iloc[-1]/ACCOUNT_START-1),'win_rate':float(wins.mean()),
            'profit_factor':pf,'avg_pnl_usdt':float(pnl.mean()),'median_pnl_usdt':float(pnl.median()),'max_drawdown':float(dd.min()),
            'stop_rate':float(trades.exit_reason.astype(str).str.contains('stop').mean()),'avg_net_return':float(trades.net_return.mean())}


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
    cand=pd.DataFrame(alltr);accepted,rejected=portfolio(cand,engine);core=accepted[accepted.tracking_class.eq('production')] if not accepted.empty else accepted;s=summary(core,start,end,cand,rejected)
    s.update(replay_version='2026-09-08-corrected',symbols_requested=len(eligible),symbols_failed=len(errors),symbols_completed=len(eligible)-len(errors),coverage_status='COMPLETE_REQUESTED_UNIVERSE' if not errors else 'INCOMPLETE',market_gate_assumption=70,tie_order='exits_then_symbol',oos=False)
    diagnostics=[]
    if not cand.empty:
        diag=cand.copy();diag['unit_r_net']=diag['net_return']/diag['stop_pct']
        diag['score_band']=pd.cut(diag.signal_score,[-np.inf,70,80,90,np.inf],right=False,labels=['<70','70-79','80-89','90+'])
        diag['time_half']=np.where(pd.to_datetime(diag.entry_time,utc=True)<pd.Timestamp(start)+(pd.Timestamp(end)-pd.Timestamp(start))/2,'first','second')
        for columns in [['grade'],['score_band'],['grade','time_half']]:
            for key,group in diag.groupby(columns,observed=True):
                values=group.unit_r_net.dropna();trimmed=values.sort_values(ascending=False).iloc[3:]
                diagnostics.append(dict(group=str(key),dimension='/'.join(columns),n=len(values),mean_R=float(values.mean()),median_R=float(values.median()),mean_R_without_top3=float(trimmed.mean()) if len(trimmed) else None,evidence='retrospective_candidates_not_independent_OOS'))
    pd.DataFrame(diagnostics).to_csv(out/'score_grade_diagnostics.csv',index=False)
    if not cand.empty:cand.to_csv(out/'candidate_trades.csv',index=False)
    if not accepted.empty:accepted.to_csv(out/'portfolio_trades.csv',index=False)
    if not rejected.empty:rejected.to_csv(out/'portfolio_rejections.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'errors.csv',index=False)
    by_grade=[]
    if not accepted.empty:
        for g,x in accepted.groupby('grade'):
            by_grade.append({'grade':g,'trades':len(x),'win_rate':float((x.pnl_usdt>0).mean()),'net_pnl':float(x.pnl_usdt.sum()),'avg_net_return':float(x.net_return.mean()),'stop_rate':float(x.exit_reason.astype(str).str.contains('stop').mean())})
    pd.DataFrame(by_grade).to_csv(out/'by_grade.csv',index=False)
    s['grade_score_diagnostics']=diagnostics
    print('TIDE_REPLAY_REPORT '+json.dumps(s,default=str),flush=True)
    (out/'summary.json').write_text(json.dumps(s,indent=2,default=str))
    lines=['# Tide V10.9 — corrected historical policy replay (not OOS)','',f"Window: {start.isoformat()} → {end.isoformat()}",f"Current eligible symbols replayed: {len(eligible)}",'',
           '## Portfolio assumptions','', '- Starting account: **1000 USDT**','- 1R: **20 USDT (2%)**','- Score risk tiers: <70=5U, 70–79=10U, 80–89=15U, >=90=20U','- Max simultaneous planned risk: **60U**','- Daily realised-loss gate: **40U**','- Maximum positions: **12**','- Exit: actual production exit function (including structural stop updates), or fixed 4h','- B+ is separate shadow tracking and contributes zero account P/L','- Simultaneous events: exits first, then entries in symbol order','- Historical market gate remains CAUTION=70; production environment overrides and websocket arrival order are not reconstructed','- Costs: engine `FEE_SLIPPAGE` deducted once per completed trade',f'- Data coverage: {len(eligible)-len(errors)}/{len(eligible)} symbols completed; {len(errors)} failed','',
           '## Headline','',f"- Candidate signals: **{s.get('candidates',0)}**",f"- Admitted trades: **{s.get('admitted',s.get('trades',0))}**",f"- Ending equity: **{s.get('ending_equity',ACCOUNT_START):.2f} USDT**",f"- Net P/L: **{s.get('net_pnl',0):+.2f} USDT**",f"- Return: **{s.get('return_pct',0):+.2%}**",f"- Win rate: **{s.get('win_rate',0):.1%}**",f"- Profit factor: **{s.get('profit_factor',0):.2f}**",f"- Max drawdown: **{s.get('max_drawdown',0):.2%}**",f"- Stop rate: **{s.get('stop_rate',0):.1%}**",'',
           '## Interpretation warning','', '**This is not a clean out-of-sample validation.** It replays today’s frozen V10.9 policy and today’s eligible-symbol bundle backward through the last six months. Because the current bundle was selected with later information, symbol selection / historical score contain survivorship-lookahead. Use this to understand policy behaviour and account risk, not to claim a proven live edge. Only complete, reconciled forward records can support a prospective test. Historical replay uses CAUTION=70, frozen bundle settings and deterministic callback ordering; it is not an exact reproduction of the live service.','']
    if by_grade:
        lines+=['## By grade','']+[f"- {x['grade']}: n={x['trades']}, win={x['win_rate']:.1%}, P/L={x['net_pnl']:+.2f}U, avg return={x['avg_net_return']:+.3%}, stop={x['stop_rate']:.1%}" for x in by_grade]
    (out/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()

