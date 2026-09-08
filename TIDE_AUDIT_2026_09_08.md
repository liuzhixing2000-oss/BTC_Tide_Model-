# Tide V10.9 reliability and evidence audit

Scope: issue #3. No score thresholds, grading definitions or account-risk tiers were optimized.

## Production changes

- Fix the exit message percentage formatting exception (hotfix c6da8abd).
- Keep intrabar candle caches updated, but compute signals only on confirmed 15m closes, as pre-alerts are disabled.
- Emit each log message as one string to reduce interleaved output.
- Preserve the existing trade CSV schema. Add a separate structured closed-trade stream with decision timestamps and code revision.
- At startup, reconcile retained model trade journal and alert ledger into separate derived audit files. Never replay historical Telegram messages or modify original ledger files.

## Corrected historical replay

- Load the same account-profile wrapper as production; use its actual sizing and leverage function.
- Use all scored candidates, matching the current live decision path, including Rescue and B+; this deliberately does not change live confirmation or cooldown policy.
- Execute the actual production exit function against isolated state and only available candles. Missing exit-path candles are censored.
- Realize P/L on exit time and attribute daily losses to the Sydney exit date. Include initial capital in the drawdown peak.
- Prevent overlapping positions in the same symbol. B+ occupies independent zero-capital shadow tracking, as on the live service.
- Provide prespecified grade/score/time-half diagnostics in uniform R without selecting a new threshold.

## Limits

This is retrospective policy replay, not a clean OOS test. The current frozen universe and historical scores contain later information. Historical market gating remains explicitly fixed at CAUTION=70; historical service overrides and exact websocket callback arrival order are unavailable. Equal-time exits precede entries in symbol order. Fees use the existing engine constant; OHLC stop fills are estimates. A+/B+ marketing labels and historical expectancy constants have not been recalibrated.

Forward reconstruction only uses retained journal rows. It does not infer missing historical trades from K-lines. A recorded alert is not proof of successful Telegram delivery, and a model trade is not an exchange fill. Startup reports verify whether /data is a mount; merely writing to a /data path does not prove persistence.

## Validation

`python -m unittest discover -s tests -v`

Ten tests cover all exit-message tracking classes, journal reconciliation and provenance, future-loss admission, midnight settlement, same-symbol overlap, B+ separation, shared trailing-stop execution, missing candles, and real launcher/sizing/Rescue integration with exchange calls disabled.

Workflow: `.github/workflows/tide_audit_replay.yml`, frozen 180-day window ending 2026-09-08 04:00 UTC. Read data coverage before interpreting any result. Results are uploaded as GitHub Actions artifacts.
