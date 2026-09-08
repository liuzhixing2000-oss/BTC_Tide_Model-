"""Import the real engine with a network-disabled exchange adapter in tests only."""
import os
import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from backtest_v10_9_six_month import load_engine, candidate_trades


class NoNetwork:
    def __init__(self,*args,**kwargs): pass
    def __getattr__(self,name): raise AssertionError('Network call forbidden in test: '+name)


class IntegrationTests(unittest.TestCase):
    def test_launcher_profile_and_live_candidate_path(self):
        adapter=types.ModuleType('pybit.unified_trading');adapter.HTTP=NoNetwork;adapter.WebSocket=NoNetwork
        with patch.dict(sys.modules,{'pybit':types.ModuleType('pybit'),'pybit.unified_trading':adapter}), patch.dict(os.environ,{},clear=True):
            engine=load_engine(Path(__file__).resolve().parents[1])
        self.assertEqual(engine.PORTFOLIO_CAPITAL_USDT,1000)
        sizing=engine.dynamic_position_size(90,dict(risk_pct=0.4))
        self.assertEqual(sizing['suggested_notional_usdt'],50)
        self.assertEqual(sizing['effective_planned_loss_usdt'],20)
        self.assertLess(sizing['suggested_leverage'],50)
        frame=pd.DataFrame(dict(open_time=pd.date_range('2026-09-08',periods=18,freq='15min',tz='UTC'),open=100.,high=101.,low=99.,close=100.,atr14=1.,signal=False,combined_setup_score=float('nan'),raw_quality_score=0.,confirmation_quality_score=96.,secondary_confirmation_tests=3,volume_multiple=2.))
        frame.loc[0,'combined_setup_score']=66.
        engine.model_frame=lambda *args: frame
        engine.signal_score=lambda *args: 70.
        trades=candidate_trades(engine,'BTCUSDT',None,frame,pd.DataFrame(),pd.DataFrame(),{},pd.Timestamp('2026-09-08',tz='UTC'),pd.Timestamp('2026-09-09',tz='UTC'))
        self.assertEqual(len(trades),1)
        self.assertEqual(trades[0]['route'],'NEXT_RESCUE')
        self.assertEqual(trades[0]['bars_held'],16)
        self.assertEqual(trades[0]['exit_reason'],'fixed_4h')
        self.assertAlmostEqual(trades[0]['pnl_usdt'],-trades[0]['notional']*engine.FEE_SLIPPAGE)

if __name__=='__main__':unittest.main()
