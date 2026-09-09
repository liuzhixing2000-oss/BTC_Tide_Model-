import unittest
import pandas as pd
from test_tide_early_entry import EarlyEntryTests
from backtest_tide_alt_entries import candidates
from tide_replay_support import chronological_portfolio

class AltEntriesTests(EarlyEntryTests):
    def run_frame(self,f):
        return candidates(f,'TEST',f.open_time.iloc[0],f.open_time.iloc[-1]+pd.Timedelta(minutes=15))
    def test_failed_confirmation_keeps_early(self):
        f=self.frame(); f.loc[1,'confirmation_quality_score']=40
        t,e=self.run_frame(f)
        self.assertEqual([r['arm'] for r in t],['early'])
        self.assertFalse(e[0]['confirmation_pass'])
    def test_next_open_and_common_stop(self):
        f=self.frame(); f.loc[1,'open']=101; f.loc[2,'open']=102
        t,e=self.run_frame(f)
        self.assertEqual([r['entry'] for r in t],[101,102])
        self.assertEqual([r['stop'] for r in t],[98,98])
    def test_no_entries_when_remaining_equity_insufficient(self):
        f=self.frame(); t,e=self.run_frame(f)
        a,b=dict(t[0]),dict(t[0]); a['pnl_usdt']=-995
        b['entry_time']=a['exit_time']+pd.Timedelta(minutes=15)
        b['exit_time']=b['entry_time']+pd.Timedelta(minutes=15)
        accepted,rejected=chronological_portfolio(pd.DataFrame([a,b]),respect_equity=True,max_daily_loss=2000)
        self.assertEqual(len(accepted),1)
        self.assertIn('insufficient_remaining_equity',rejected.iloc[0].reject_reason)

if __name__=='__main__': unittest.main()
