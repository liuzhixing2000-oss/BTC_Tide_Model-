# Frozen exit study — 2026-09-10

Exploratory, not OOS. Uses the 191 confirmed candidates frozen in the prior 232-altcoin experiment. Same original entry, stop, position size, cost and capital constraints. No new grade, Rescue, threshold search or production change.

Rules locked before retrieving paths:

1. Baseline: exit after 16 completed 15m bars, at the following open.
2. Weak at 1h: after 4 bars, maximum high since entry has not reached +0.5 gross R AND current close after 0.1% round-trip cost is <=0R; exit at next open. Otherwise baseline exit.
3. Strong at 4h: after 16 bars, current close after cost is >=+0.5R AND above the close 4 bars ago; extend holding. Starting with bar17, exit next open if close <= mean of the latest 4 completed closes, otherwise exit after 32 bars at next open. Original structural stop remains unchanged.

An intrabar structural stop precedes a close-based decision. Gaps fill at the worse open. Stops are conservatively timestamped at bar end. The old 4h-close execution is independently reproduced as a reference; the three compared arms all use next-open timing for timed/conditional exits.

Retrieve exactly 33 completed bars for each frozen event. Validate contiguous coverage, unchanged entry and reproduction of old exit time, price, reason, net R, net PnL and MFE/MAE summaries. Store all 33-bar windows for offline replay. Do not invent missing candles or treat a mismatched original outcome as valid.

Compare all 191 candidates paired by event ID, the same original admitted134-event cohort, and separate full chronological portfolio replays. Report saved loss versus lost winner profit, top-five winner effect, activations, changed outcomes, month/half stability, and paired mean deltaR with a fixed-seed 3000-resample entry-day cluster bootstrap. This interval is exploratory; it is not protection against selection bias or all temporal dependence.

The June1–September9 period was already inspected. No split within it is clean OOS. A favorable result would justify future validation, not production deployment.

Run: `python backtest_tide_exit_study.py`

Offline reproduction: `python backtest_tide_exit_study.py --offline path/to/uncompressed_archive.json`

Evidence prints as EXIT_ARCHIVE gzip/base64 chunks, followed by EXIT_REPORT and EXIT_ARCHIVE_END with SHA256.
