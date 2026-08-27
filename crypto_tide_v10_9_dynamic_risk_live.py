#!/usr/bin/env python3
"""Crypto Tide V10.9 All-Signals Quality Display Live Engine.

Adds a persistent forward-test ledger on the Railway volume. Every Tide SIGNAL
and Tide EXIT Telegram payload is written to JSONL/CSV before transmission, so
redeployments or Railway log-retention limits cannot erase the research record.
A daily Telegram document backup is also sent for an off-platform copy.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, re, shutil, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

REQUIRED_BUNDLE_FILES = [
    'manifest.json','stage2_full_results.csv','exit_config.json',
    'entry_parameter_config.json','online_learning.json','market_regime.json'
]
LEDGER_TZ = ZoneInfo('Australia/Sydney')
LEDGER_VERSION = 'v1'

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model-file',type=Path,default=Path('crypto_tide_engine_v10_9_dynamic_risk.py'))
    p.add_argument('--bundle-dir',type=Path,default=Path('v10_bundle'))
    p.add_argument('--runtime-dir',type=Path,default=Path(os.getenv('V10_RUNTIME_DIR','/data/tide_v10_9_runtime')))
    p.add_argument('--bundle-check-seconds',type=int,default=int(os.getenv('V10_BUNDLE_CHECK_SECONDS','300')))
    return p.parse_args()

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def validate_bundle(bundle_dir):
    if not bundle_dir.exists(): raise FileNotFoundError(f'Bundle directory not found: {bundle_dir}')
    for name in REQUIRED_BUNDLE_FILES:
        if not (bundle_dir/name).exists(): raise FileNotFoundError(f'Required bundle file missing: {bundle_dir/name}')
    manifest=json.loads((bundle_dir/'manifest.json').read_text(encoding='utf-8'))
    for name,meta in manifest.get('files',{}).items():
        path=bundle_dir/name
        if not path.exists(): raise FileNotFoundError(f'Manifest references missing file: {path}')
        expected=meta.get('sha256')
        if expected and sha256(path)!=expected: raise RuntimeError(f'Bundle checksum mismatch: {name}')
    stage2=pd.read_csv(bundle_dir/'stage2_full_results.csv')
    if stage2.empty or not {'symbol','eligible'}.issubset(stage2.columns):
        raise RuntimeError('Invalid stage2_full_results.csv')
    manifest.setdefault('stage2_rows',len(stage2))
    return manifest

def fingerprint(bundle_dir):
    h=hashlib.sha256()
    for name in sorted(REQUIRED_BUNDLE_FILES):
        p=bundle_dir/name; s=p.stat()
        h.update(name.encode()); h.update(str(s.st_size).encode()); h.update(str(s.st_mtime_ns).encode())
    return h.hexdigest()

def prepare_runtime(bundle_dir,runtime_dir):
    runtime_dir.mkdir(parents=True,exist_ok=True)
    for p in bundle_dir.iterdir():
        if p.is_file() and p.name!='manifest.json': shutil.copy2(p,runtime_dir/p.name)

def load_model(model_file,runtime_dir):
    if not model_file.exists(): raise FileNotFoundError(f'Strategy module missing: {model_file}')
    os.environ['TIDE_DATA_DIR']=str(runtime_dir.resolve())
    spec=importlib.util.spec_from_file_location('crypto_tide_v10_4_runtime',model_file)
    if spec is None or spec.loader is None: raise RuntimeError(f'Unable to import {model_file}')
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def watch_bundle(bundle_dir,initial,seconds,log):
    while True:
        time.sleep(max(60,seconds))
        try:
            if fingerprint(bundle_dir)!=initial:
                log('V10.9 bundle changed; restarting to load new research bundle.')
                os.execv(sys.executable,[sys.executable,*sys.argv])
        except Exception as exc: log('V10.6 bundle watcher error',repr(exc))

def _rx(pattern,text,cast=str,default=None):
    m=re.search(pattern,text,re.I|re.M)
    if not m:return default
    try:return cast(m.group(1))
    except Exception:return default

def _parse_event(text):
    now=datetime.now(timezone.utc).isoformat()
    signal=re.search(r'Tide SIGNAL:\s*([A-Z0-9]+USDT)',text)
    exit_m=re.search(r'Tide (?:B\+ SECONDARY |WATCH |SYSTEM )?EXIT:\s*([A-Z0-9]+USDT)',text,re.I)
    research=re.search(r'Tide RESEARCH SIGNAL:\s*([A-Z0-9]+USDT)',text,re.I)
    if signal:
        symbol=signal.group(1); event_type='signal'
    elif exit_m:
        symbol=exit_m.group(1); event_type='exit'
    elif research:
        symbol=research.group(1); event_type='research_signal'
    else:
        return None
    record={
        'ledger_version':LEDGER_VERSION,'recorded_at_utc':now,'event_type':event_type,'symbol':symbol,
        'raw_message':text,
        'grade':_rx(r'TIDE GRADE:\s*(A\+|A-|A|B\+|B|C)',text),
        'entry_route':_rx(r'Entry route:\s*([^\n]+)',text),
        'default_system_trade':_rx(r'Default system trade:\s*([^\n]+)',text),
        'production_position_opened':_rx(r'Production position opened:\s*([^\n]+)',text),
        'secondary_bplus_tracking':_rx(r'Secondary B\+ tracking:\s*([^\n]+)',text),
        'candle_close_utc':_rx(r'Candle close UTC:\s*([^\n]+)',text),
        'reference_price':_rx(r'Reference/current price:\s*([-+0-9.eE]+)',text,float),
        'h4_trend':_rx(r'4H trend:\s*([^\n]+)',text),
        'h4_strength':_rx(r'4H strength:\s*([-+0-9.eE]+)',text,float),
        'h1_trend':_rx(r'1H trend:\s*([^\n]+)',text),
        'h1_strength':_rx(r'1H strength:\s*([-+0-9.eE]+)',text,float),
        'raw_quality':_rx(r'Raw Tide:\s*([-+0-9.eE]+)',text,float),
        'next_quality':_rx(r'Next candle:\s*([-+0-9.eE]+)',text,float),
        'combined_setup':_rx(r'Combined setup:\s*([-+0-9.eE]+)',text,float),
        'confirmation_tests':_rx(r'Confirmation:\s*(\d+)/3',text,int),
        'signal_score':_rx(r'Signal score:\s*([-+0-9.eE]+)',text,float),
        'volume_multiple':_rx(r'Volume multiple:\s*([-+0-9.eE]+)x',text,float),
        'lower_wick_ratio':_rx(r'Lower wick ratio:\s*([-+0-9.eE]+)',text,float),
        'stop':_rx(r'^Stop:\s*([-+0-9.eE]+)',text,float),
        'stop_risk_pct':_rx(r'Stop risk:\s*([-+0-9.eE]+)%',text,lambda x:float(x)/100),
        'backtest_expectancy_reference':_rx(r'Backtest expectancy reference:\s*([-+0-9.eE]+)%',text,lambda x:float(x)/100),
        'exit_reason':_rx(r'Exit reason:\s*([^\n]+)',text),
        'exit_price':_rx(r'Reference exit:\s*([-+0-9.eE]+)',text,float),
        'net_return':_rx(r'Estimated net return:\s*([-+0-9.eE]+)%',text,lambda x:float(x)/100),
        'bars_held':_rx(r'Bars held:\s*(\d+)',text,int),
        'mfe':_rx(r'Maximum favourable excursion:\s*([-+0-9.eE]+)%',text,lambda x:float(x)/100),
        'mae':_rx(r'Maximum adverse excursion:\s*([-+0-9.eE]+)%',text,lambda x:float(x)/100),
    }
    key_time=record.get('candle_close_utc') or now
    record['event_key']=f"{event_type}:{symbol}:{key_time}:{record.get('grade') or record.get('exit_reason') or ''}"
    return record

def install_forward_ledger(model,runtime_dir):
    ledger_dir=runtime_dir/'forward_ledger'; ledger_dir.mkdir(parents=True,exist_ok=True)
    jsonl=ledger_dir/'tide_forward_events.jsonl'; csv_path=ledger_dir/'tide_forward_events.csv'
    lock=threading.Lock(); seen=set()
    if jsonl.exists():
        try:
            for line in jsonl.read_text(encoding='utf-8').splitlines():
                try: seen.add(json.loads(line).get('event_key'))
                except Exception: pass
        except Exception: pass
    original_send=model.send_tg
    def persist(record):
        if not record:return
        with lock:
            key=record.get('event_key')
            if key in seen:return
            seen.add(key)
            with jsonl.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
            flat={k:v for k,v in record.items() if k!='raw_message'}
            frame=pd.DataFrame([flat]); frame.to_csv(csv_path,mode='a',header=not csv_path.exists(),index=False)
            print('FORWARD_LEDGER',json.dumps({k:flat.get(k) for k in ('event_type','symbol','grade','candle_close_utc','reference_price','signal_score','exit_reason','net_return')},default=str),flush=True)
    def wrapped_send(text):
        try:persist(_parse_event(str(text)))
        except Exception as exc:print('FORWARD_LEDGER_ERROR',repr(exc),flush=True)
        return original_send(text)
    model.send_tg=wrapped_send
    def backup_loop():
        last_date=None
        while True:
            try:
                local=datetime.now(LEDGER_TZ)
                if local.hour==8 and local.minute<15 and local.date()!=last_date:
                    if csv_path.exists() and hasattr(model,'send_tg_document'):
                        model.send_tg_document(csv_path,f'Tide forward ledger backup — {local.date().isoformat()}')
                    if jsonl.exists() and hasattr(model,'send_tg_document'):
                        model.send_tg_document(jsonl,f'Tide forward raw ledger — {local.date().isoformat()}')
                    last_date=local.date()
            except Exception as exc:print('FORWARD_LEDGER_BACKUP_ERROR',repr(exc),flush=True)
            time.sleep(300)
    threading.Thread(target=backup_loop,daemon=True).start()
    print('Forward ledger enabled:',jsonl,flush=True)
    print('Forward ledger CSV:',csv_path,flush=True)
    return jsonl,csv_path

def main():
    a=parse_args()
    print('='*100,flush=True); print('CRYPTO TIDE V10.9 B+ MINIMUM TELEGRAM LIVE ENGINE',flush=True); print('='*100,flush=True)
    print('Startup UTC:',datetime.now(timezone.utc).isoformat(),flush=True)
    manifest=validate_bundle(a.bundle_dir); fp=fingerprint(a.bundle_dir); prepare_runtime(a.bundle_dir,a.runtime_dir)
    model=load_model(a.model_file,a.runtime_dir)
    install_forward_ledger(model,a.runtime_dir)
    token,chat_id=model.telegram_credentials()
    print('Telegram token configured:',bool(token),flush=True); print('Telegram chat ID configured:',bool(chat_id),flush=True)
    if not token or not chat_id: raise RuntimeError('Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to Railway Variables.')
    stage2=pd.read_csv(model.STAGE2_CSV)
    stage2['eligible']=stage2['eligible'].astype(str).str.lower().eq('true')
    selected=model.select_current_watchlist(stage2)
    version=manifest.get('bundle_version','unknown'); generated=manifest.get('generated_at_utc','unknown')
    eligible=manifest.get('eligible_symbols',int(stage2['eligible'].sum()))
    model.send_tg(
        '🟢 Crypto Tide V10.9 Stable Online\n\n'
        f'Bundle version: {version}\nBundle generated: {generated}\n'
        f'Research-eligible symbols: {eligible}\nRealtime symbols selected: {len(selected)}\n'
        f'Maximum positions: {model.PORTFOLIO_MAX_POSITIONS}\n'
        f'Reference capital: {model.PORTFOLIO_CAPITAL_USDT:.0f} USDT\n'
        f'Base margin: {model.BASE_MARGIN_USDT:.0f} USDT\n'
        f'Displayed leverage: {model.DEFAULT_LEVERAGE:.0f}x\n'
        'Telegram mode: B+ OR HIGHER ONLY\n'
        'Risk mode: DYNAMIC — grade is independent of stop distance\n'
        f'Production Next threshold: {model.PRODUCTION_MIN_NEXT_QUALITY:.0f}\n'
        f'Production Combined threshold: {model.PRODUCTION_MIN_COMBINED_QUALITY:.0f}\n'
        f'Production Signal threshold: {model.PRODUCTION_MIN_SIGNAL_SCORE:.0f}\n'
        f'Minimum confirmation tests: {model.PRODUCTION_MIN_CONFIRMATION_TESTS}/3\n'
        'Hard-stop gate: OFF — structural stop distance controls position size\n'
        f'Structure stop buffer: {model.PRODUCTION_STRUCTURE_BUFFER_ATR:.2f} ATR\n'
        f'Fixed hold: {model.PRODUCTION_FIXED_HOLD_BARS * 0.25:.2f} hours\n'
        f'Production exit: {model.PRODUCTION_EXIT_METHOD}\n'
        'Forward ledger: ENABLED — persistent Railway volume + daily Telegram backup\n'
        'No Stage1 or Stage2 research scan was run during startup.'
    )
    print('Bundle version:',version,flush=True); print('Bundle generated:',generated,flush=True)
    print('Realtime symbols selected:',len(selected),flush=True); print('Research scan skipped: yes',flush=True)
    threading.Thread(target=watch_bundle,args=(a.bundle_dir,fp,a.bundle_check_seconds,model.log),daemon=True).start()
    model.start_monitor(selected)

if __name__=='__main__': main()
