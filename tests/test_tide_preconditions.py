import unittest
import pandas as pd
import numpy as np
from study_tide_preconditions import clean,indicators,FEATURES

class PreconditionsTest(unittest.TestCase):
    def rows(self,interval,n=672):
        end=pd.Timestamp('2026-06-01',tz='UTC')
        times=pd.date_range(end-pd.Timedelta(minutes=interval*n),periods=n,freq=f'{interval}min')
        return [[int(t.timestamp()*1000),100.,101.,99.,100.,10.,1000.] for t in times]
    def test_trigger_and_future_bars_excluded(self):
        cutoff=pd.Timestamp('2026-06-01',tz='UTC');rows=self.rows(15)
        before=clean(rows,15,cutoff)
        after=clean(rows+[[int(cutoff.timestamp()*1000),100,10000,1,9999,1e9,1e9]],15,cutoff)
        pd.testing.assert_frame_equal(before,after)
    def test_gap_is_rejected(self):
        rows=self.rows(15);rows.pop(300)
        with self.assertRaisesRegex(ValueError,'Gap'):clean(rows,15,pd.Timestamp('2026-06-01',tz='UTC'))
    def test_flat_market_features_are_finite(self):
        cutoff=pd.Timestamp('2026-06-01',tz='UTC')
        frames=[clean(self.rows(i),i,cutoff) for i in [15,60,240]]
        f=indicators(*frames)
        self.assertEqual(set(f),set(FEATURES));self.assertEqual(f['rsi14'],50.)
        self.assertEqual(f['rsi_divergence'],0);self.assertEqual(f['macd_divergence'],0)
        self.assertEqual(f['volume_ratio_6h'],1.);self.assertIsNone(f['down_volume_ratio_12h'])
    def test_return_uses_only_prior_completed_closes(self):
        cutoff=pd.Timestamp('2026-06-01',tz='UTC');frames=[clean(self.rows(i),i,cutoff) for i in [15,60,240]]
        frames[0].loc[frames[0].index[-1],'close']=102.
        self.assertAlmostEqual(indicators(*frames)['return_6h'],.02)

if __name__=='__main__':unittest.main()
