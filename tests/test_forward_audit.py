import csv
import json
import tempfile
from pathlib import Path
import unittest
from tide_forward_audit import build_audit


class ForwardAuditTests(unittest.TestCase):
    def test_recover_missing_exit_without_claiming_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'forward_ledger').mkdir()
            row=dict(symbol='BTCUSDT',entry_time='2026-09-08T00:00:00Z',exit_time='2026-09-08T04:00:00Z',tracking_class='production',tide_grade='A',entry_signal_score=82,net_return=0.01,suggested_notional=1000,effective_planned_loss_usdt=20,exit_price=101,exit_reason='fixed_4h')
            with (root/'live_trade_journal.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=row);writer.writeheader();writer.writerow(row);writer.writerow(row)
            event=dict(event_type='signal',symbol='BTCUSDT',candle_close_utc='2026-09-08T00:15:00Z')
            (root/'forward_ledger'/'tide_forward_events.jsonl').write_text(json.dumps(event)+'\n')
            report=build_audit(root)
            self.assertEqual(report['journal_rows'],1)
            self.assertEqual(report['duplicate_journal_rows'],1)
            self.assertEqual(report['matched_signal_records'],1)
            self.assertEqual(report['reconstructed_exit_records'],1)
            self.assertEqual(report['groups']['class:production']['net_model_pnl_usdt'],10)
            recovered=json.loads((root/'forward_audit'/'reconstructed_exits.jsonl').read_text())
            self.assertEqual(recovered['telegram_delivery'],'not_evidenced')
            self.assertEqual(build_audit(root)['journal_rows'],1)
            self.assertEqual((root/'forward_ledger'/'tide_forward_events.jsonl').read_text(),json.dumps(event)+'\n')

    def test_empty_history_is_not_evidence_of_zero_trades(self):
        with tempfile.TemporaryDirectory() as root:
            report=build_audit(root)
            self.assertFalse(report['journal_exists'])
            self.assertEqual(report['evidence_status'],'PARTIAL_MODEL_RECORDS_NOT_EXCHANGE_FILLS')

if __name__=='__main__':unittest.main()
