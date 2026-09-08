"""Replay adapters: execute the production exit function without I/O or alerts."""
from types import FunctionType
import threading
import pandas as pd


def simulate_live_exit(engine, frame, entry_index, risk):
    """Bind the actual live exit code to isolated state; never touch live globals.

    The function receives only candles available at the simulated close. No
    independent implementation of trailing stops or fixed-time exits is used.
    """
    symbol = '__REPLAY__'
    position = dict(status='active', method='structure_fixed4h',
                    entry_price=risk['entry_price'], entry_atr=risk['atr'],
                    hard_stop=risk['hard_stop'], current_stop=None, bars_held=0,
                    highest_price=risk['entry_price'], lowest_price=risk['entry_price'])
    captured = []
    def close(symbol, position, latest, price, reason):
        captured.append((float(price), reason, position.copy()))
        position['status'] = 'closed'
    namespace = dict(engine.realtime_exit_update.__globals__)
    namespace.update(active_positions={symbol: position}, market_data={},
                     ACTIVE_POSITIONS_FILE=None,
                     position_lock=threading.Lock(), data_lock=threading.Lock(),
                     close_research_position=close, save_json=lambda *args: None)
    update = FunctionType(engine.realtime_exit_update.__code__, namespace)
    for j in range(entry_index + 1, min(entry_index + 17, len(frame))):
        if pd.Timestamp(frame.iloc[j]['open_time']) - pd.Timestamp(frame.iloc[j-1]['open_time']) != pd.Timedelta(minutes=15):
            return None  # Missing bars make the stop path unknowable.
        namespace['market_data'][symbol] = {'15': frame.iloc[max(0,j-4):j+1]}
        update(symbol, frame.iloc[j])
        if captured:
            price, reason, state = captured[0]
            return j, price, reason, state
    return None  # Right-censored: do not invent a completed trade.


def chronological_portfolio(candidates, *, capital=1000.0, max_positions=12,
                            max_open_risk=60.0, max_margin=800.0, max_daily_loss=40.0):
    """Settle on exit, then admit entries. B+ is zero-capital shadow tracking.

    Equal-time exits precede entries, with a deterministic symbol order. Actual
    websocket callback ordering is unavailable in historical OHLC data.
    """
    if candidates.empty:
        return candidates.copy(), pd.DataFrame()
    c = candidates.copy()
    for col in ['entry_time', 'exit_time']:
        c[col] = pd.to_datetime(c[col], utc=True)
    if (c.exit_time <= c.entry_time).any():
        raise ValueError('Exit must occur after entry')
    c = c.sort_values(['entry_time', 'symbol'], kind='stable')
    active, accepted, rejected, daily_loss = [], [], [], {}
    equity = capital

    def settle(until):
        nonlocal equity, active
        due = sorted([a for a in active if a['exit_time'] <= until], key=lambda a: (a['exit_time'], a['symbol']))
        active = [a for a in active if a['exit_time'] > until]
        for a in due:
            a['equity_before_exit'] = equity
            equity += a['pnl_usdt']
            a['equity_after'] = equity
            day = a['exit_time'].tz_convert('Australia/Sydney').date().isoformat()
            daily_loss[day] = daily_loss.get(day, 0.0) + max(0.0, -a['pnl_usdt'])

    for row in c.to_dict('records'):
        t = row['entry_time']
        settle(t)
        shadow = row['grade'] == 'B+'
        row['tracking_class'] = 'watch' if shadow else 'production'
        row['shadow_pnl_usdt'] = row['pnl_usdt'] if shadow else 0.0
        if shadow:
            row.update(pnl_usdt=0.0, actual_risk_usdt=0.0, notional=0.0, margin=0.0)
        core = [a for a in active if a['tracking_class'] == 'production']
        reasons = []
        if any(a['symbol'] == row['symbol'] for a in active):
            reasons.append('existing_active_position')
        day = t.tz_convert('Australia/Sydney').date().isoformat()
        if not shadow:
            if len(core) >= max_positions: reasons.append('maximum_position_count')
            if sum(a['actual_risk_usdt'] for a in core) + row['actual_risk_usdt'] > max_open_risk + 1e-9: reasons.append('maximum_total_open_risk')
            if sum(a['margin'] for a in core) + row['margin'] > max_margin + 1e-9: reasons.append('maximum_margin_utilisation')
            if daily_loss.get(day,0.0) >= max_daily_loss - 1e-9: reasons.append('maximum_daily_loss')
        row.update(admitted=not reasons, reject_reason='|'.join(reasons), equity_before=equity)
        if reasons:
            rejected.append(row)
        else:
            accepted.append(row)
            active.append(row)
    settle(pd.Timestamp.max.tz_localize('UTC'))
    return pd.DataFrame(accepted).sort_values(['exit_time','symbol']) if accepted else pd.DataFrame(), pd.DataFrame(rejected)
