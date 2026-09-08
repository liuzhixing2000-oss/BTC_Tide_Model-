import unittest
from types import SimpleNamespace
import threading
import numpy as np
import pandas as pd
from test_exit_alert import load_function
from tide_replay_support import chronological_portfolio, simulate_live_exit


def candidate(symbol, entry, exit, pnl=-20, grade='A', risk=20, margin=50):
    return dict(symbol=symbol,entry_time=entry,exit_time=exit,grade=grade,
                signal_score=90,pnl_usdt=pnl,actual_risk_usdt=risk,notional=1000,margin=margin)


class ReplayTests(unittest.TestCase):
    def test_future_loss_does_not_block_earlier_entry(self):
        rows=[candidate('A','2026-09-08T00:00Z','2026-09-08T04:00Z',-40),candidate('B','2026-09-08T01:00Z','2026-09-08T05:00Z',20),candidate('C','2026-09-08T04:15Z','2026-09-08T08:15Z',20)]
        accepted,rejected=chronological_portfolio(pd.DataFrame(rows))
        self.assertEqual(set(accepted.symbol),{'A','B'})
        self.assertEqual(rejected.iloc[0].symbol,'C')
        self.assertIn('maximum_daily_loss',rejected.iloc[0].reject_reason)
        self.assertEqual(accepted.loc[accepted.symbol.eq('B'),'equity_before'].iloc[0],1000)
        self.assertEqual(accepted.iloc[-1].equity_after,980)

    def test_loss_belongs_to_exit_day(self):
        rows=[candidate('A','2026-09-08T13:00Z','2026-09-08T15:00Z',-40),candidate('B','2026-09-08T15:15Z','2026-09-08T16:00Z')]
        accepted,rejected=chronological_portfolio(pd.DataFrame(rows))
        self.assertEqual(len(accepted),1)
        self.assertEqual(rejected.iloc[0].symbol,'B')

    def test_shadow_does_not_consume_core_budget(self):
        rows=[candidate('A','2026-09-08T00:00Z','2026-09-08T04:00Z',-40,'B+',60,800),candidate('B','2026-09-08T01:00Z','2026-09-08T05:00Z',20,'A',60,800)]
        accepted,rejected=chronological_portfolio(pd.DataFrame(rows),max_positions=1)
        self.assertTrue(rejected.empty)
        self.assertEqual(accepted.pnl_usdt.sum(),20)
        self.assertEqual(accepted.iloc[0].tracking_class,'watch')

    def test_same_symbol_cannot_overlap(self):
        rows=[candidate('A','2026-09-08T00:00Z','2026-09-08T04:00Z',grade='B+'),candidate('A','2026-09-08T01:00Z','2026-09-08T05:00Z')]
        accepted,rejected=chronological_portfolio(pd.DataFrame(rows))
        self.assertEqual(len(accepted),1)
        self.assertIn('existing_active_position',rejected.iloc[0].reject_reason)

    def test_shared_live_exit_trails_and_does_not_see_future(self):
        namespace=dict(pd=pd,np=np,DEFAULT_EXIT_METHOD='structure_fixed4h',EXIT_MAX_BARS=96,
                       strategy_spec=lambda method:SimpleNamespace(exit_mode='fixed4h'))
        engine=SimpleNamespace(realtime_exit_update=load_function('crypto_tide_engine_v10_9_dynamic_risk.py','realtime_exit_update',namespace))
        frame=pd.DataFrame(dict(open_time=pd.date_range('2026-09-08',periods=20,freq='15min',tz='UTC'),open=105.,high=110.,low=[104,103,102,103,104,101]+[104]*14,close=105.))
        risk=dict(entry_price=105.,atr=1.,hard_stop=95.)
        result=simulate_live_exit(engine,frame,0,risk)
        self.assertEqual(result[0],5)
        self.assertAlmostEqual(result[1],101.8)
        self.assertEqual(result[2],'dynamic_structure_stop')
        frame.loc[10,'low']=1
        self.assertEqual(simulate_live_exit(engine,frame,0,risk)[:3],result[:3])

    def test_gaps_are_censored(self):
        namespace=dict(pd=pd,np=np,DEFAULT_EXIT_METHOD='structure_fixed4h',EXIT_MAX_BARS=96,
                       strategy_spec=lambda method:SimpleNamespace(exit_mode='fixed4h'))
        engine=SimpleNamespace(realtime_exit_update=load_function('crypto_tide_engine_v10_9_dynamic_risk.py','realtime_exit_update',namespace))
        frame=pd.DataFrame(dict(open_time=pd.date_range('2026-09-08',periods=20,freq='30min',tz='UTC'),open=105.,high=110.,low=104.,close=105.))
        self.assertIsNone(simulate_live_exit(engine,frame,0,dict(entry_price=105,atr=1,hard_stop=95)))

if __name__=='__main__':unittest.main()
