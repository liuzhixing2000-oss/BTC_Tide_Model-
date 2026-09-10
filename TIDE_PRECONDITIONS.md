# Pre-trigger context study, locked 2026-09-10

Same191 frozen confirmed candidates, original fixed4h net outcomes. Primary descriptive analysis on all candidates; original134 admitted trades checked separately. No deployment to production or threshold optimization.

Cutoff is the OPEN of the original raw Tide signal candle (30minutes before confirmed entry). Only bars whose close is <= this cutoff are included. Thus indicators precede both the raw trigger and its confirmation, and never use the subsequent trend.

Twenty predefined features:

- Prior6h,24h,72h close returns on15m bars; consecutive negative1h close changes.
- Last completed1h/4h close distance to respective SMA50/SMA200.
- Last15m close distance to the minimum low of96 preceding15m bars, excluding the last bar.
- Wilder-style EWM ATR14 / last close; ATR14 relative to mean ATR over preceding56 bars.
- Latest6h high-low range / preceding6h range.
- Latest6h average volume / preceding18h average volume.
- Down-close-bar volume latest12h / preceding12h; missing if denominator zero. This is candle volume, NOT orderflow delta or verified seller-initiated volume.
- RSI14, RSI change over6h, MACD(12,26,9) histogram / ATR14.
- RSI/MACD divergence proxy: minimum low in most recent12 bars lower than minimum in preceding12 bars, while indicator at the later low is higher. No future pivot confirmation. These are two-block proxies, not general chart divergence.
- Both1h and4h SMA50<SMA200.

History:672 bars15m,320 bars1h,250 bars4h per event; incomplete histories are errors/excluded, not filled. Archive raw API rows for exact offline feature reproduction.

Descriptive outputs: winning/losing medians, Spearman correlation with netR, first50/second50 direction, exclusion of largest5 winners, and bins defined using first-half terciles then applied unchanged to second-half. Binary features are yes/no. The same inspected period is NOT clean OOS. No statistical-significance claim from20 feature comparisons. Bin results are not executable portfolio returns.

Run `python study_tide_preconditions.py`; PRE_ARCHIVE prints gzip/base64 evidence chunks, PRE_REPORT coverage, PRE_ARCHIVE_END SHA256. Existing frozen evidence is in reports/alt_entries_20260909/retry.json.gz.
