#!/usr/bin/env python3
"""Railway one-shot runner for the current Tide V10.9 six-month policy replay.

Recommended use: create a SECOND Railway service from the same GitHub repo.
Do not replace the live Tide service start command.

Environment variables:
  BACKTEST_DAYS=180
  BACKTEST_MAX_SYMBOLS=0
  BACKTEST_END=             # optional ISO8601 end time
  BACKTEST_OUT=/data/v10_9_six_month
  TELEGRAM_BOT_TOKEN=       # optional, sends ZIP + summary when finished
  TELEGRAM_CHAT_ID=         # optional
"""
from __future__ import annotations
import os, shutil, subprocess, sys
from pathlib import Path
import requests


def send_document(path: Path, caption: str) -> None:
    token=os.getenv('TELEGRAM_BOT_TOKEN','').strip(); chat=os.getenv('TELEGRAM_CHAT_ID','').strip()
    if not token or not chat or not path.exists(): return
    try:
        with path.open('rb') as f:
            requests.post(f'https://api.telegram.org/bot{token}/sendDocument',data={'chat_id':chat,'caption':caption[:1000]},files={'document':(path.name,f)},timeout=120).raise_for_status()
    except Exception as exc:
        print('Telegram document send failed:',repr(exc),flush=True)


def send_text(text: str) -> None:
    token=os.getenv('TELEGRAM_BOT_TOKEN','').strip(); chat=os.getenv('TELEGRAM_CHAT_ID','').strip()
    if not token or not chat:return
    try:requests.post(f'https://api.telegram.org/bot{token}/sendMessage',json={'chat_id':chat,'text':text[:4000]},timeout=30).raise_for_status()
    except Exception as exc:print('Telegram text send failed:',repr(exc),flush=True)


def main():
    days=int(os.getenv('BACKTEST_DAYS','180')); max_symbols=int(os.getenv('BACKTEST_MAX_SYMBOLS','0'))
    out=Path(os.getenv('BACKTEST_OUT','/data/v10_9_six_month')); out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'backtest_v10_9_six_month.py','--days',str(days),'--max-symbols',str(max_symbols),'--out',str(out)]
    end=os.getenv('BACKTEST_END','').strip()
    if end:cmd += ['--end',end]
    print('='*90,flush=True);print('RAILWAY TIDE V10.9 SIX-MONTH BACKTEST',flush=True);print('Command:',' '.join(cmd),flush=True);print('='*90,flush=True)
    send_text(f'🧪 Tide V10.9 backtest started\nDays: {days}\nMax symbols: {max_symbols or "ALL"}')
    rc=subprocess.run(cmd).returncode
    if rc!=0:
        send_text(f'❌ Tide V10.9 backtest failed (exit {rc}). Check Railway logs.')
        raise SystemExit(rc)
    zip_base=str(out.parent/(out.name+'_results'))
    zip_path=Path(shutil.make_archive(zip_base,'zip',root_dir=out))
    summary=out/'SUMMARY.md'
    if summary.exists():
        txt=summary.read_text(encoding='utf-8',errors='replace')
        print('\n'+txt,flush=True)
        send_text('✅ Tide V10.9 six-month backtest completed.\n\n'+txt[:3500])
    send_document(zip_path,f'Tide V10.9 {days}-day backtest results')
    print('RESULT_ZIP:',zip_path,flush=True)
    print('Backtest finished successfully. This is a one-shot worker and may now stop.',flush=True)

if __name__=='__main__':main()
