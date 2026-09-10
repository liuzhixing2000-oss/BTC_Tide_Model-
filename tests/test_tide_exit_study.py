import unittest
import pandas as pd
from backtest_tide_exit_study import simulate,validate_window,legacy

class ExitRulesTest(unittest.TestCase):
    def setUp(self):
        self.c=dict(symbol='ALTUSDT',event_id='x',arm='confirmed',grade='EXPERIMENT',entry=100.,stop=90.,stop_pct=.1,notional=100.,margin=2.,actual_risk_usdt=10.,planned_risk_usdt=10.,entry_time='2026-06-01 00:00:00+00:00')
        self.f=pd.DataFrame(dict(open_time=pd.date_range(self.c['entry_time'],periods=33,freq='15min'),open=100.,high=101.,low=99.,close=100.,volume=1.,turnover=100.))
    def strong(self):
        self.f.loc[12:15,['open','close']]=106.
        self.f.loc[12:15,'high']=107.
    def test_weak_decides_on_close_fills_next_open(self):
        self.f.loc[4,'open']=97.
        t=simulate(self.c,self.f,'weak_1h')
        self.assertEqual(t['exit'],97.)
        self.assertEqual(t['exit_time'],self.f.iloc[4].open_time)
        self.assertEqual(t['exit_reason'],'weak_1h')
    def test_stop_precedes_weak_decision(self):
        self.f.loc[3,'low']=89.
        t=simulate(self.c,self.f,'weak_1h')
        self.assertEqual(t['exit_reason'],'structural_stop')
        self.assertFalse(t['activated'])
    def test_reached_half_R_is_not_weak(self):
        self.f.loc[2,'high']=106.
        self.assertEqual(simulate(self.c,self.f,'weak_1h')['bars_held'],16)
    def test_extension_does_not_widen_stop(self):
        self.strong();self.f.loc[16,['open','low']]=[89.,88.]
        t=simulate(self.c,self.f,'extend_8h')
        self.assertTrue(t['activated']);self.assertEqual(t['exit'],89.)
        self.assertEqual(t['stop'],90.)
    def test_extension_reversal_fills_next_open(self):
        self.strong();self.f.loc[16,'close']=105.;self.f.loc[16,'high']=106.;self.f.loc[17,'open']=94.
        t=simulate(self.c,self.f,'extend_8h')
        self.assertEqual(t['exit_reason'],'lost_1h_mean');self.assertEqual(t['exit'],94.)
    def test_maximum_eight_hours(self):
        self.strong()
        for k in range(16,33):
            self.f.loc[k,['open','close']]=107.+k-16
            self.f.loc[k,'high']=108.+k-16
        t=simulate(self.c,self.f,'extend_8h')
        self.assertEqual(t['bars_held'],32);self.assertEqual(t['exit_reason'],'max_8h')
    def test_baseline_uses_next_open(self):
        self.f.loc[16,'open']=98.
        self.assertEqual(simulate(self.c,self.f,'baseline')['exit'],98.)
        self.assertEqual(legacy(self.c,self.f)['exit'],100.)
    def test_gap_rejected_before_any_simulation(self):
        self.f.loc[5,'open_time']+=pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError,'Data gap'):
            validate_window(self.c,self.f,pd.Timestamp('2026-06-02',tz='UTC'))

if __name__=='__main__':unittest.main()
