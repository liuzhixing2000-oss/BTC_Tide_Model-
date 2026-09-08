"""Reconcile model journals with alert records; never send historical messages.

Derived files preserve provenance. Missing history is reported, not filled with
market replay. A journal row is a model outcome, not an exchange execution.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pandas as pd


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def timestamp(value):
    try:
        x = pd.to_datetime(value, utc=True, errors='raise')
        return x.isoformat() if pd.notna(x) else None
    except (TypeError, ValueError):
        return None


def metrics(rows):
    returns = [number(r.get('net_return')) for r in rows]
    returns = [x for x in returns if x is not None]
    pnls, rs = [], []
    for r in rows:
        net, notional, risk = (number(r.get(k)) for k in ['net_return','suggested_notional','effective_planned_loss_usdt'])
        if net is not None and notional is not None and notional > 0:
            pnls.append(net*notional)
            if risk is not None and risk > 0: rs.append(net*notional/risk)
    loss = -sum(x for x in pnls if x < 0)
    return dict(n=len(rows), returns_usable=len(returns), pnl_usable=len(pnls),
                net_model_pnl_usdt=sum(pnls) if pnls else None,
                mean_net_return=sum(returns)/len(returns) if returns else None,
                mean_R=sum(rs)/len(rs) if rs else None,
                win_rate=sum(x>0 for x in returns)/len(returns) if returns else None,
                profit_factor=sum(x for x in pnls if x>0)/loss if loss>0 else None)


def build_audit(runtime, commit=None):
    runtime = Path(runtime)
    output = runtime/'forward_audit'
    output.mkdir(parents=True, exist_ok=True)
    journal_path = runtime/'live_trade_journal.csv'
    ledger_path = runtime/'forward_ledger'/'tide_forward_events.jsonl'
    errors, rows, ledger = [], [], []
    if journal_path.exists():
        with journal_path.open(newline='', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f), 2):
                if None in row or not row.get('symbol') or timestamp(row.get('entry_time')) is None or timestamp(row.get('exit_time')) is None:
                    errors.append({'source':'journal','line':i,'error':'invalid row'});continue
                row['_source_line'] = i
                rows.append(row)
    if ledger_path.exists():
        with ledger_path.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                try: ledger.append(json.loads(line))
                except (ValueError,TypeError): errors.append({'source':'ledger','line':i,'error':'invalid JSON'})

    unique, duplicate_count, conflicts = {}, 0, 0
    for row in rows:
        key = (row['symbol'],timestamp(row['entry_time']),row.get('tracking_class','production'))
        if key in unique:
            duplicate_count += 1
            if any(str(unique[key].get(k)) != str(row.get(k)) for k in ['exit_time','exit_price','net_return']): conflicts += 1
        else: unique[key] = row
    rows = list(unique.values())
    signals = [r for r in ledger if r.get('event_type')=='signal']
    exits = [r for r in ledger if r.get('event_type')=='exit']
    matched_signals, matched_exits = set(), set()
    recovered = []
    for row in rows:
        entry = pd.Timestamp(timestamp(row['entry_time']))
        # Legacy journal entry_time is the 15m opening time. New rows carry
        # an explicit decision timestamp; preserve the old field unchanged.
        decision = timestamp(row.get('entry_decision_utc')) or (entry+pd.Timedelta(minutes=15)).isoformat()
        matches = [i for i,r in enumerate(signals) if r.get('symbol')==row['symbol'] and timestamp(r.get('candle_close_utc'))==decision]
        matched_signals.update(matches)
        journal_exit = pd.Timestamp(timestamp(row['exit_time']))
        exit_close = timestamp(row.get('exit_decision_utc')) or (journal_exit+pd.Timedelta(minutes=15)).isoformat()
        em = [i for i,r in enumerate(exits) if i not in matched_exits and r.get('symbol')==row['symbol'] and timestamp(r.get('recorded_at_utc')) and abs((pd.Timestamp(r['recorded_at_utc'])-pd.Timestamp(exit_close)).total_seconds())<=300 and number(r.get('exit_price')) is not None and number(row.get('exit_price')) is not None and math.isclose(number(r['exit_price']),number(row['exit_price']),rel_tol=1e-6)]
        matched_exits.update(em[:1])
        row['entry_decision_utc'] = decision
        row['exit_decision_utc'] = exit_close
        row['signal_record_matched'] = bool(matches)
        row['exit_alert_record_matched'] = bool(em)
        row['timestamp_basis'] = 'explicit_or_legacy_open_plus_15m'
        row['audit_trade_id'] = hashlib.sha256(f"{row['symbol']}:{decision}:{row.get('tracking_class','production')}".encode()).hexdigest()[:24]
        if not em:
            recovered.append(dict(event_type='reconstructed_exit',audit_trade_id=row['audit_trade_id'],symbol=row['symbol'],
                entry_decision_utc=decision,exit_decision_utc=exit_close,exit_price=number(row.get('exit_price')),
                net_return=number(row.get('net_return')),exit_reason=row.get('exit_reason'),
                provenance='live_trade_journal.csv',source_line=row['_source_line'],telegram_delivery='not_evidenced'))

    groups = {}
    for row in rows:
        tracking = row.get('tracking_class','production')
        score = number(row.get('entry_signal_score'))
        band = 'unknown' if score is None else '<70' if score<70 else '70-79' if score<80 else '80-89' if score<90 else '90+'
        for key in [f'class:{tracking}',f'{tracking}/grade:{row.get("tide_grade","unknown")}',f'{tracking}/score:{band}']:
            groups.setdefault(key,[]).append(row)
    times = [r['entry_decision_utc'] for r in rows]
    summary = dict(audit_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),commit=commit,
        runtime_dir=str(runtime),data_mount_verified=os.path.ismount('/data'),
        journal_exists=journal_path.exists(),ledger_exists=ledger_path.exists(),
        journal_rows=len(rows),duplicate_journal_rows=duplicate_count,conflicting_duplicate_rows=conflicts,
        signal_records=len(signals),exit_alert_records=len(exits),
        matched_signal_records=len(matched_signals),matched_exit_records=len(matched_exits),
        reconstructed_exit_records=len(recovered),unmatched_signal_records=len(signals)-len(matched_signals),
        invalid_rows=len(errors),first_entry_utc=min(times) if times else None,last_entry_utc=max(times) if times else None,
        evidence_status='PARTIAL_MODEL_RECORDS_NOT_EXCHANGE_FILLS',
        limitations=['Completeness before earliest retained record is unknown.','Ledger persistence is not proof of Telegram delivery.','Legacy rows do not identify code/config revision.','Unmatched signals may be open, blocked or missing outcomes.','No historical market outcomes are invented.'],
        groups={key:metrics(value) for key,value in groups.items()})
    def atomic(name, text):
        tmp=output/(name+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(output/name)
    atomic('summary.json',json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2))
    atomic('reconciled_trades.jsonl',''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    atomic('reconstructed_exits.jsonl',''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in recovered))
    atomic('parse_errors.json',json.dumps(errors,indent=2))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('runtime');args=parser.parse_args()
    print(json.dumps(build_audit(args.runtime),ensure_ascii=False))
