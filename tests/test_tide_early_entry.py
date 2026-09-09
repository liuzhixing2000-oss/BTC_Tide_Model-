import unittest
import pandas as pd
from backtest_tide_early_entry import candidates, make_trade, raw_quality_at_close

class EarlyEntryTests(unittest.TestCase):
    def frame(self):
        f=pd.DataFrame(dict(open_time=pd.date_range('2026-01-01',periods=25,freq='15min',tz='UTC'),
            open=[100.]*25,high=[103.]*25,low=[99.]*25,close=[102.]*25,
            lower_wick_ratio=[.9]*25,volume_multiple=[4.]*25,atr14=[2.]*25,raw_signal=[True]+[False]*24,
            raw_quality_score=[80.]*25,confirmation_quality_score=[98.]*25,
            combined_setup_score=[85.]*25,secondary_confirmation_tests=[3]*25))
        return f
    def run_frame(self,f):
        return candidates(f,'TEST',f.open_time.iloc[0],f.open_time.iloc[-1]+pd.Timedelta(minutes=15))
    def test_failed_confirmation_keeps_early(self):
        f=self.frame(); f.loc[1,'confirmation_quality_score']=40
        t,e=self.run_frame(f)
        self.assertEqual([x['arm'] for x in t],['early','delayed'])
        self.assertFalse(e[0]['confirmation_pass'])
    def test_future_confirmation_does_not_change_early_entry(self):
        f=self.frame(); a,_=self.run_frame(f)
        f.loc[1,'confirmation_quality_score']=40; b,_=self.run_frame(f)
        for key in ['entry','stop','notional','entry_time','pnl_usdt']:
            self.assertEqual(a[0][key],b[0][key])
    def test_next_open_and_common_stop(self):
        f=self.frame(); f.loc[1,'open']=101; f.loc[2,'open']=102
        t,_=self.run_frame(f)
        self.assertEqual([x['entry'] for x in t],[101,102,102])
        self.assertEqual([x['stop'] for x in t],[98,98,98])
    def test_shared_raw_gate_uses_original_bar(self):
        f=self.frame(); f.loc[0,'volume_multiple']=1.6; f.loc[0,'lower_wick_ratio']=.36
        self.assertLess(raw_quality_at_close(f.iloc[0]),58)
        t,e=self.run_frame(f)
        self.assertEqual(t,[]); self.assertEqual(e[0]['status'],'raw_below_58')
    def test_gap_stop_fills_at_worse_open(self):
        f=self.frame(); f.loc[3,['open','low']]=[95,94]
        t=make_trade(f,0,1,'TEST','early',98,10,.001,2500)
        self.assertEqual(t['exit'],95)
        self.assertLess(t['net_R'],-1)
    def test_common_censoring_and_missing_bars(self):
        f=self.frame().iloc[:17]; t,e=self.run_frame(f)
        self.assertEqual(t,[]); self.assertEqual(e[0]['status'],'right_censored')
        f=self.frame().drop(index=5); t,e=self.run_frame(f)
        self.assertEqual(t,[]); self.assertEqual(e[0]['status'],'data_gap')

if __name__=='__main__': unittest.main()
