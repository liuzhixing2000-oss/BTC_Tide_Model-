# Frozen market context study

191 frozen altcoin candidates; BTC background only. Big winners net R >= 2; original134 reported separately. All features use completed candles before raw trigger open. No production changes.

BTC SMA50/200 regimes on1h/4h,6h/24h/7d returns; altcoin7d/30d returns and30day high distance; relative6h/24h returns. Key levels: previous UTC day low, previous complete Monday-Sunday UTC week low, and7days ending at least24h before cutoff. Signed distance divided by pre-trigger4hATR; near previous-week low means absolute distance<=1ATR. Reclaim means recent24h low below old7day low and final close above it.

Exploratory previously inspected100days;8 candidate big winners and4 admitted. Comparisons are descriptive, overlapping dates and selection bias prevent treating this as independent confirmation. No threshold tuning.

Validation: future bar injection has no effect; stale cutoff rejected; archived altcoin context recalculation checked.
