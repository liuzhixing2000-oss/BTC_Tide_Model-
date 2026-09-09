#!/usr/bin/env python3
"""Controlled raw-close versus confirmation-close experiment; no production writes."""
import argparse
import base64
import gzip
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tide_replay_support import chronological_portfolio

PARAMS = dict(lookback_bars=24, volume_lookback=24, volume_multiplier=1.5,
              lower_wick_threshold=0.35, cooldown_bars=24)
BAR = pd.Timedelta(minutes=15)


def raw_quality_at_close(row):
    """Only original-bar data; never read next candle for shared eligibility."""
    wick = np.clip((row.lower_wick_ratio-.35)/(.95-.35), 0, 1)
    volume = np.clip((row.volume_multiple-1.5)/(4.-1.5), 0, 1)
    close_position = np.clip((row.close-row.low)/max(row.high-row.low, 1e-12),0,1)
    return float(100*(.45*wick+.35*volume+.20*close_position))


def confirmation_pass(row):
    # Deliberately no historical score, grade, or market-score filter in either arm.
    return (row.raw_quality_score >= 58 and row.confirmation_quality_score >= 95
            and row.combined_setup_score >= 70 and row.secondary_confirmation_tests >= 2)


def make_trade(frame, raw_i, entry_i, symbol, arm, stop, risk, cost, max_notional):
    """Next-open fill. Fixed original structural stop, 16 bars from each fill.

    No take-profit or trailing stop; this is a controlled exit, not live exit parity.
    Stop gaps fill at the worse open; intrabar stops are timestamped at bar end.
    """
    entry = float(frame.iloc[entry_i].open)
    if entry <= stop or not np.isfinite(entry):
        return None
    distance = (entry - stop) / entry
    notional = min(risk / distance, max_notional)
    exit_i = entry_i + 15
    price = float(frame.iloc[exit_i].close)
    reason = 'fixed_4h'
    for j in range(entry_i, exit_i + 1):
        row = frame.iloc[j]
        if row.low <= stop:
            exit_i, price, reason = j, min(float(row.open), stop), 'structural_stop'
            break
    net = price / entry - 1 - cost
    path = frame.iloc[entry_i:exit_i+1]
    return dict(symbol=symbol, arm=arm, grade='EXPERIMENT',
                event_id=f'{symbol}:{frame.iloc[raw_i].open_time.isoformat()}',
                raw_time=frame.iloc[raw_i].open_time + BAR,
                entry_time=frame.iloc[entry_i].open_time,
                exit_time=frame.iloc[exit_i].open_time + BAR,
                entry=entry, stop=stop, stop_pct=distance, exit=price,
                exit_reason=reason, notional=notional, margin=notional/50,
                actual_risk_usdt=notional*distance, planned_risk_usdt=risk,
                gross_return=price/entry-1, net_return=net, pnl_usdt=notional*net,
                net_R=net/distance, cost_R=cost/distance,
                bars_held=exit_i-entry_i+1,
                bar_mfe_upper_bound=float(path.high.max())/entry-1,
                bar_mae_lower_bound=float(path.low.min())/entry-1)


def candidates(frame, symbol, start, end, risk=10., cost=.001, max_notional=2500.):
    trades, events = [], []
    frame = frame.reset_index(drop=True)
    for i in np.flatnonzero(frame.raw_signal.fillna(False).to_numpy()):
        raw = frame.iloc[i]
        if not start <= raw.open_time + BAR < end:
            continue
        event = dict(event_id=f'{symbol}:{raw.open_time.isoformat()}', symbol=symbol,
                     raw_time=raw.open_time+BAR, status='complete', confirmation_pass=False)
        events.append(event)
        event['raw_at_close'] = raw_quality_at_close(raw)
        if not np.isfinite(event['raw_at_close']) or event['raw_at_close'] < 58:
            event['status'] = 'raw_below_58'; continue
        # Equal observable horizon for both arms; exclude only data incompleteness,
        # never confirmation failure or outcome. Do not fetch beyond --end.
        if i+17 >= len(frame) or frame.iloc[i+17].open_time+BAR > end:
            event['status'] = 'right_censored'; continue
        window = frame.iloc[i:i+18]
        if not window.open_time.diff().iloc[1:].eq(BAR).all():
            event['status'] = 'data_gap'; continue
        if not np.isfinite(window[['open','high','low','close']].to_numpy()).all():
            event['status'] = 'invalid_prices'; continue
        stop = float(raw.low - .5*raw.atr14)
        if not np.isfinite(stop) or stop <= 0:
            event['status'] = 'invalid_stop'; continue
        passed = bool(confirmation_pass(frame.iloc[i+1]))
        event['confirmation_pass'] = passed
        for name, col, threshold in [('raw_pass','raw_quality_score',58),('next_pass','confirmation_quality_score',95),('combined_pass','combined_setup_score',70)]:
            event[name] = bool(frame.iloc[i+1][col] >= threshold)
            event[col] = float(frame.iloc[i+1][col])
        for arm, entry_i in [('early', i+1), ('confirmed', i+2)]:
            if arm == 'confirmed' and not passed:
                continue
            trade = make_trade(frame, i, entry_i, symbol, arm, stop, risk, cost, max_notional)
            event[arm+'_status'] = 'candidate' if trade else 'open_at_or_below_stop'
            if trade:
                trade['confirmation_eventually_passed'] = passed  # diagnostics ONLY
                trades.append(trade)
    return trades, events


def stats(frame):
    if frame.empty:
        return dict(trades=0, net_pnl=0., profit_factor=None, win_rate=None,
                    mean_R=None, realized_max_drawdown=None)
    frame=frame.sort_values(['exit_time','symbol'])
    p=frame.pnl_usdt; equity=1000+p.cumsum(); peak=equity.cummax().clip(lower=1000)
    loss=-p[p<0].sum()
    return dict(trades=len(frame), net_pnl=float(p.sum()), return_pct=float(p.sum()/1000),
                profit_factor=float(p[p>0].sum()/loss) if loss else None,
                win_rate=float((p>0).mean()), mean_R=float(frame.net_R.mean()),
                mean_stop_pct=float(frame.stop_pct.mean()), mean_cost_R=float(frame.cost_R.mean()),
                realized_max_drawdown=float((equity/peak-1).min()))


def main():
    from backtest_v10_9_six_month import fetch_range, load_engine
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', default='ALT_BUNDLE')
    parser.add_argument('--days', type=int, default=100)
    parser.add_argument('--end', required=True, help='Explicit UTC cutoff, e.g. 2026-09-09T00:00:00Z')
    parser.add_argument('--risk', type=float, default=10.)
    parser.add_argument('--cost', type=float, default=.001, help='Round-trip return fraction')
    parser.add_argument('--max-notional', type=float, default=2500.)
    parser.add_argument('--data-dir', help='Offline SYMBOL_15.csv and SYMBOL_60.csv directory')
    parser.add_argument('--out', default='reports/tide_early_entry')
    args=parser.parse_args()
    end=pd.Timestamp(args.end)
    if end.tzinfo is None: parser.error('--end must include a timezone')
    end=end.tz_convert('UTC'); start=end-pd.Timedelta(days=args.days)
    if args.days<=0 or args.risk<=0 or args.cost<0 or args.max_notional<=0:
        parser.error('Invalid days/risk/cost/notional')
    repo=Path(__file__).resolve().parent
    if args.symbols == 'ALT_BUNDLE':
        stage=pd.read_csv(repo/'v10_bundle'/'stage2_full_results.csv')
        symbols=stage.loc[stage.eligible.astype(str).str.lower().eq('true'),'symbol'].astype(str).tolist()
        symbols=sorted(set(symbols)-{'BTCUSDT','ETHUSDT','SOLUSDT'})
    else:
        symbols=list(dict.fromkeys(s.strip().upper() for s in args.symbols.split(',') if s.strip()))
    print('TIDE_UNIVERSE '+json.dumps(symbols),flush=True)
    if not symbols: parser.error('No symbols')
    repo=Path(__file__).resolve().parent; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    engine=load_engine(repo)
    engine.RAW_QUALITY_WEIGHT=.75; engine.CONFIRMATION_QUALITY_WEIGHT=.25
    all_trades, all_events, errors, coverage = [], [], [], []
    def process_symbol(symbol):
        local_coverage=[]
        print(f'Loading {symbol}', flush=True)
        try:
            data=[]
            for interval, warmup in [('15',15),('60',25)]:
                if args.data_dir:
                    d=pd.read_csv(Path(args.data_dir)/f'{symbol}_{interval}.csv')
                else:
                    d=fetch_range(symbol,interval,(start-pd.Timedelta(days=warmup)).to_pydatetime(),end.to_pydatetime())
                d['open_time']=pd.to_datetime(d.open_time,utc=True)
                d=d.sort_values('open_time').drop_duplicates('open_time')
                d=d[(d.open_time>=start-pd.Timedelta(days=warmup)) &
                    (d.open_time+pd.Timedelta(minutes=int(interval))<=end)].copy()
                for col in ['open','high','low','close','volume']:
                    d[col]=pd.to_numeric(d[col],errors='raise')
                if len(d)<205: raise ValueError(f'Insufficient {interval} data')
                if not d.open_time.diff().iloc[1:].eq(pd.Timedelta(minutes=int(interval))).all():
                    raise ValueError(f'Gaps in {interval} input; repair data before replay')
                if d.open_time.iloc[0]>start-pd.Timedelta(minutes=int(interval)*205):
                    raise ValueError(f'Insufficient {interval} pre-window warmup')
                if d.open_time.iloc[-1]+pd.Timedelta(minutes=int(interval))<end.floor(f'{interval}min'):
                    raise ValueError(f'{interval} data does not reach cutoff')
                csv=d.to_csv(index=False)
                (out/f'{symbol}_{interval}.csv').write_text(csv)
                local_coverage.append(dict(symbol=symbol,interval=interval,bars=len(d),
                                     sha256=hashlib.sha256(csv.encode()).hexdigest()))
                data.append(d)
            frame=engine.model_frame(data[0],data[1],PARAMS)
            t,e=candidates(frame,symbol,start,end,args.risk,args.cost,args.max_notional)
            print(f'COMPLETED {symbol} raw={len(e)} candidates={len(t)}',flush=True)
            return t,e,local_coverage,None
        except Exception as exc:
            error=dict(symbol=symbol,error=str(exc))
            print(f'FAILED {symbol}: {exc}',flush=True)
            return [],[],local_coverage,error
    # Four bounded data workers; no live state or Telegram writes.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for tr,ev,cv,error in pool.map(process_symbol,symbols):
            all_trades.extend(tr); all_events.extend(ev); coverage.extend(cv)
            if error: errors.append(error)
    t=pd.DataFrame(all_trades); events=pd.DataFrame(all_events)
    t.to_csv(out/'candidates.csv',index=False); events.to_csv(out/'raw_events.csv',index=False)
    summary=dict(version=3,start=str(start),end=str(end),symbols=symbols,errors=errors,
                 coverage=coverage,status='INCOMPLETE' if errors else 'COMPLETE',
                 settings=vars(args),params=PARAMS,arms={},diagnostics={},
                 limitations=['Retrospective, not OOS; current eligible bundle selected using later information: survivorship/lookahead risk.',
                 'ALT_BUNDLE excludes BTC/ETH/SOL. Partial coverage excluded from both arms equally; no invented candles.',
                 'Fixed 10U planned risk; additional affordability gate rejects entries exceeding remaining realized equity.',
                 'Both arms require original Raw>=58 computed at raw close; no future confirmation used for early selection.',
                 'Confirmation arm additionally requires Next95/Combined70/2of3; not full live scoring or Rescue.',
                 'Both arms use fixed original low minus 0.5 raw ATR stop and 16 bars, no live trailing stop.',
                 'Next-open fills; round-trip costs fixed, funding excluded; no intrabar price path.',
                 'Stop exits timestamped at bar close; drawdown uses realized equity, not floating P/L.',
                 '50x margin convention; no exchange liquidation simulation; not an execution recommendation.'])
    for arm in ['early','confirmed']:
        subset=t[t.arm.eq(arm)].copy() if not t.empty else t.copy()
        accepted,rejected=chronological_portfolio(subset, respect_equity=True)
        accepted.to_csv(out/f'{arm}_accepted.csv',index=False)
        rejected.to_csv(out/f'{arm}_rejected.csv',index=False)
        summary['arms'][arm]=dict(**stats(accepted),portfolio_rejected=len(rejected))
        summary['arms'][arm]['by_symbol'] = {str(k): stats(g) for k,g in accepted.groupby('symbol')} if not accepted.empty else {}
        if not subset.empty:
            summary['diagnostics'][arm]={str(k):stats(g) for k,g in subset.groupby('confirmation_eventually_passed')}
    summary['raw_event_counts']=events.status.value_counts().to_dict() if not events.empty else {}
    summary['confirmation_diagnostics'] = {name: int(events[name].fillna(False).sum()) if name in events else 0 for name in ['raw_pass','next_pass','combined_pass','confirmation_pass']}
    summary['paired_candidates'] = {}
    if not t.empty:
        pairs=t[t.arm.isin(['early','delayed'])].pivot(index='event_id',columns='arm',values='net_R')
        if {'early','delayed'}.issubset(pairs.columns):
            pairs=pairs.dropna(subset=['early','delayed'])
            delta=pairs['delayed']-pairs['early']
            summary['paired_candidates']=dict(n=len(pairs),mean_delayed_minus_early_R=float(delta.mean()) if len(delta) else None)
    summary['source_sha256']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
        ['backtest_tide_alt_entries.py','crypto_tide_engine_v10_9_dynamic_risk.py','tide_replay_support.py','v10_bundle/stage2_full_results.csv']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    # Compressed evidence chunks avoid platform log-rate and line-length limits.
    payload=json.dumps(dict(summary=summary,events=all_events,candidates=all_trades),default=str,allow_nan=False).encode()
    packed=gzip.compress(payload,mtime=0); encoded=base64.b64encode(packed).decode()
    chunks=[encoded[i:i+6000] for i in range(0,len(encoded),6000)]
    for i,chunk in enumerate(chunks):
        print('TIDE_ALT_ARCHIVE '+json.dumps(dict(index=i,total=len(chunks),data=chunk)),flush=True)
        time.sleep(.1)
    headline={k:v for k,v in summary.items() if k not in ['coverage','source_sha256','symbols']}
    headline['symbols_requested']=len(symbols); headline['symbols_completed']=len(symbols)-len(errors)
    for arm in headline['arms']:
        headline['arms'][arm]=dict(headline['arms'][arm]); headline['arms'][arm].pop('by_symbol',None)
    print('TIDE_ALT_REPORT '+json.dumps(headline,allow_nan=False),flush=True)
    print('TIDE_ALT_ARCHIVE_END '+json.dumps(dict(chunks=len(chunks),sha256=hashlib.sha256(packed).hexdigest())),flush=True)
    return 2 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
