"""Exercise actual exit code with persistence and messaging replaced by collectors."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load_function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)
    return namespace[name]

class ExitAlertTests(unittest.TestCase):
    def test_exit_is_saved_and_rendered_for_each_tracking_class(self):
        for tracking, grade, title in [('production', 'A+', 'SYSTEM EXIT'), ('watch', 'B+', 'B+ SECONDARY EXIT'), ('watch', 'B', 'WATCH EXIT')]:
            with self.subTest(tracking=tracking, grade=grade):
                sent, saved, journal = [], [], []
                namespace = dict(FEE_SLIPPAGE=0.001, active_positions={}, ACTIVE_POSITIONS_FILE='unused', save_json=lambda *args: saved.append(args), append_live_trade=journal.append, send_tg=sent.append)
                close = load_function('crypto_tide_engine_v10_9_dynamic_risk.py', 'close_research_position', namespace)
                position = dict(entry_price=100, bars_held=16, tracking_class=tracking, tide_grade=grade, method='structure_fixed4h', signal_low=98, hard_stop=97, highest_price=105, maximum_favourable_excursion_pct=0.05, maximum_adverse_excursion_pct=-0.02)
                close('BTCUSDT', position, dict(open_time=datetime(2026,9,8,tzinfo=timezone.utc)), 102, 'fixed_4h')
                self.assertEqual(position['status'], 'closed')
                self.assertEqual(len(saved), 1)
                self.assertEqual(len(journal), 1)
                self.assertEqual(len(sent), 1)
                self.assertIn(title, sent[0])
                self.assertIn('Maximum favourable excursion: 5.00%', sent[0])
                self.assertIn('Maximum adverse excursion: -2.00%', sent[0])
                self.assertIn('Estimated net return: 1.90%', sent[0])

if __name__ == '__main__':
    unittest.main()
