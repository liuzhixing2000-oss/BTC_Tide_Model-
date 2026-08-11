#!/usr/bin/env python3
"""Crypto Tide V10.8.2.1.1 All-Signals Quality Display Live Engine."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shutil, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REQUIRED_BUNDLE_FILES = [
    'manifest.json','stage2_full_results.csv','exit_config.json',
    'entry_parameter_config.json','online_learning.json','market_regime.json'
]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model-file',type=Path,default=Path('crypto_tide_engine_v10_8_2_1.py'))
    p.add_argument('--bundle-dir',type=Path,default=Path('v10_bundle'))
    p.add_argument('--runtime-dir',type=Path,default=Path(os.getenv('V10_RUNTIME_DIR','/tmp/tide_v10_8_2_1_runtime')))
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
    for p in runtime_dir.iterdir():
        if p.is_file(): p.unlink()
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
                log('V10.8.2.1 bundle changed; restarting to load new research bundle.')
                os.execv(sys.executable,[sys.executable,*sys.argv])
        except Exception as exc: log('V10.6 bundle watcher error',repr(exc))

def main():
    a=parse_args()
    print('='*100,flush=True); print('CRYPTO TIDE V10.8.2.1 WATCH-TRACKING LIVE ENGINE',flush=True); print('='*100,flush=True)
    print('Startup UTC:',datetime.now(timezone.utc).isoformat(),flush=True)
    manifest=validate_bundle(a.bundle_dir); fp=fingerprint(a.bundle_dir); prepare_runtime(a.bundle_dir,a.runtime_dir)
    model=load_model(a.model_file,a.runtime_dir)
    token,chat_id=model.telegram_credentials()
    print('Telegram token configured:',bool(token),flush=True); print('Telegram chat ID configured:',bool(chat_id),flush=True)
    if not token or not chat_id: raise RuntimeError('Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to Railway Variables.')
    stage2=pd.read_csv(model.STAGE2_CSV)
    stage2['eligible']=stage2['eligible'].astype(str).str.lower().eq('true')
    selected=model.select_current_watchlist(stage2)
    version=manifest.get('bundle_version','unknown'); generated=manifest.get('generated_at_utc','unknown')
    eligible=manifest.get('eligible_symbols',int(stage2['eligible'].sum()))
    model.send_tg(
        '🟢 Crypto Tide V10.8.2.1 Stable Online\n\n'
        f'Bundle version: {version}\nBundle generated: {generated}\n'
        f'Research-eligible symbols: {eligible}\nRealtime symbols selected: {len(selected)}\n'
        f'Maximum positions: {model.PORTFOLIO_MAX_POSITIONS}\n'
        f'Reference capital: {model.PORTFOLIO_CAPITAL_USDT:.0f} USDT\n'
        f'Base margin: {model.BASE_MARGIN_USDT:.0f} USDT\n'
        f'Displayed leverage: {model.DEFAULT_LEVERAGE:.0f}x\n'
        'Telegram mode: ALL SIGNALS + A+/A/A-/B/C EXPECTANCY GRADES\n'
        f'Production Next threshold: {model.PRODUCTION_MIN_NEXT_QUALITY:.0f}\n'
        f'Production Combined threshold: {model.PRODUCTION_MIN_COMBINED_QUALITY:.0f}\n'
        f'Production Signal threshold: {model.PRODUCTION_MIN_SIGNAL_SCORE:.0f}\n'
        f'Minimum confirmation tests: '
        f'{model.PRODUCTION_MIN_CONFIRMATION_TESTS}/3\n'
        f'Maximum production hard-stop risk: '
        f'{model.PRODUCTION_MAX_HARD_STOP_RISK_PCT:.2%}\n'
        f'Structure stop buffer: '
        f'{model.PRODUCTION_STRUCTURE_BUFFER_ATR:.2f} ATR\n'
        f'Fixed hold: '
        f'{model.PRODUCTION_FIXED_HOLD_BARS * 0.25:.2f} hours\n'
        f'Production exit: {model.PRODUCTION_EXIT_METHOD}\n'
        'No Stage1 or Stage2 research scan was run during startup.'
    )
    print('Bundle version:',version,flush=True); print('Bundle generated:',generated,flush=True)
    print('Realtime symbols selected:',len(selected),flush=True); print('Research scan skipped: yes',flush=True)
    threading.Thread(target=watch_bundle,args=(a.bundle_dir,fp,a.bundle_check_seconds,model.log),daemon=True).start()
    model.start_monitor(selected)

if __name__=='__main__': main()
