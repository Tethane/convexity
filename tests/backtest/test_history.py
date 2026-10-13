"""Provider protocol fixtures test mechanics; prices are never production fallback."""
import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from types import SimpleNamespace as NS

import pytest

from convexity.backtest.data import Session, SessionCalendar
from convexity.backtest.history import HistoryError, HistoryRequest, IBKRHistoricalData, _digest
from convexity.brokers.ibkr import PaperAccountConfig
from convexity.runtime import StrategyMetadata, StrategyContract, BarSubscription

CONFIG = PaperAccountConfig('DU123456', 7497, True)


class Client:
    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate
        self.disconnected = False

    async def connectAsync(self, *args, **kwargs):
        assert kwargs['readonly'] is True

    def managedAccounts(self):
        return [CONFIG.account]

    async def reqContractDetailsAsync(self, contract):
        contract.conId = {'SPY': 1, 'QQQ': 2}[contract.symbol]
        return [NS(contract=contract)]

    async def reqHistoricalDataAsync(self, contract, **kwargs):
        self.calls.append((contract, kwargs))
        end = kwargs['endDateTime']
        amount, unit = kwargs['durationStr'].split()
        start = end.date() - timedelta(days=int(amount) - 1) if unit == 'D' else (end - timedelta(seconds=int(amount))).date()
        calendar = SessionCalendar.from_exchange('XNYS', start, end.date())
        rows = []
        for session in calendar.sessions:
            if kwargs['barSizeSetting'] == '1 day':
                times = [session.label]
            else:
                step = 60 if kwargs['barSizeSetting'] == '1 min' else int(kwargs['barSizeSetting'].split()[0])
                lower = max(session.open, end - timedelta(seconds=int(amount))) if unit == 'S' else session.open
                upper = min(session.close, end)
                times = [lower + timedelta(seconds=i) for i in range(0, int((upper-lower).total_seconds()), step)]
            rows.extend(NS(date=t, open=100, high=102, low=99, close=101, volume=1234) for t in times)
        return self.mutate(rows) if self.mutate else rows

    def isConnected(self):
        return not self.disconnected

    def disconnect(self):
        self.disconnected = True


def provider(tmp_path, client):
    return IBKRHistoricalData(CONFIG, cache_dir=tmp_path, client_factory=lambda: client, request_interval=0)


def test_daily_multi_symbol_cache_and_refresh_preserves_snapshots(tmp_path):
    request = HistoryRequest(('SPY', 'QQQ'), '2024-11-27', '2024-11-29')
    client = Client()
    source = provider(tmp_path, client)
    result = asyncio.run(source.fetch(request))
    assert len(result.data.bars) == 4  # Thanksgiving excluded, early close retained.
    assert result.data.price_basis == 'split_adjusted_price_return'
    assert result.data.bars[-1].end.hour == 18
    assert client.disconnected and len(client.calls) == 2
    assert result.manifest['contracts']['SPY']['conId'] == 1
    assert result.manifest['requests'][0]['raw_rows']
    cached = asyncio.run(source.fetch(request))
    assert cached.cache_hit and cached.data.identity == result.data.identity
    assert len(client.calls) == 2
    refreshed = asyncio.run(provider(tmp_path, Client()).fetch(request, refresh=True))
    assert refreshed.cache_path != result.cache_path
    assert result.cache_path.exists() and refreshed.cache_path.exists()


def test_daily_chunks_whole_requested_year(tmp_path):
    client = Client()
    result = asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2024-01-01', '2024-12-31')))
    assert len(client.calls) == 1
    assert len(result.data.bars) == 252
    assert result.data.bars[0].start.date() == date(2024, 1, 2)
    assert result.data.bars[-1].end.date() == date(2024, 12, 31)


def test_minute_sessions_dst_and_early_close(tmp_path):
    client = Client()
    result = asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2024-11-27', '2024-11-29', '1 min')))
    assert len(result.data.bars) == 390 + 210
    assert len(client.calls) == 1
    assert result.data.bars[-1].end == datetime(2024, 11, 29, 18, tzinfo=timezone.utc)
    assert all(c[1]['whatToShow'] == 'TRADES' and c[1]['formatDate'] == 2 for c in client.calls)


@pytest.mark.parametrize('bar_size', ['1 day', '1 min'])
def test_missing_final_bar_fails_without_cache(tmp_path, bar_size):
    client = Client(lambda rows: rows[:-1])
    with pytest.raises(HistoryError, match='incomplete'):
        asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29', bar_size)))
    assert client.disconnected
    assert not list(tmp_path.rglob('*.json'))


def test_conflicting_duplicate_rejected(tmp_path):
    def mutate(rows):
        rows.append(NS(**(vars(rows[-1]) | {'close': 100})))
        return rows
    with pytest.raises(HistoryError, match='Conflicting duplicate'):
        asyncio.run(provider(tmp_path, Client(mutate)).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')))


def test_daily_wire_string_supported(tmp_path):
    def mutate(rows):
        for row in rows:
            row.date = row.date.strftime('%Y%m%d')
        return rows
    result = asyncio.run(provider(tmp_path, Client(mutate)).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')))
    assert len(result.data.bars) == 1


def test_naive_intraday_rejected(tmp_path):
    def mutate(rows):
        rows[0].date = rows[0].date.replace(tzinfo=None)
        return rows
    with pytest.raises(HistoryError, match='timezone-aware'):
        asyncio.run(provider(tmp_path, Client(mutate)).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29', '1 min')))


def test_corrupt_cache_fails_closed_and_explicit_refresh_repairs(tmp_path):
    source = provider(tmp_path, Client())
    request = HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')
    result = asyncio.run(source.fetch(request))
    payload = json.loads(result.cache_path.read_text())
    payload['bars'][0]['close'] = '99'
    result.cache_path.write_text(json.dumps(payload))
    with pytest.raises(HistoryError, match='cache is invalid'):
        asyncio.run(source.fetch(request))
    refreshed = asyncio.run(provider(tmp_path, Client()).fetch(request, refresh=True))
    assert refreshed.data.bars[0].close == 101


def test_cache_with_valid_hash_still_validates_coverage(tmp_path):
    source = provider(tmp_path, Client())
    request = HistoryRequest(('SPY',), '2024-11-27', '2024-11-29')
    result = asyncio.run(source.fetch(request))
    payload = json.loads(result.cache_path.read_text())
    payload['bars'] = payload['bars'][:-1]
    from convexity.backtest.data import HistoricalData
    payload['manifest']['data_identity'] = HistoricalData(result.data.bars[:-1], price_basis=result.data.price_basis).identity
    identity = _digest(payload)
    (tmp_path / 'snapshots' / (identity + '.json')).write_text(json.dumps(payload))
    source._index(request).write_text(json.dumps({'sha256': identity}))
    with pytest.raises(HistoryError, match='cache is invalid'):
        asyncio.run(source.fetch(request))


def test_future_request_never_connects(tmp_path):
    client = Client()
    with pytest.raises(HistoryError, match='future session'):
        asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2030-01-02', '2030-01-03')))
    assert not client.calls and not client.disconnected


def test_empty_holiday_period_never_connects(tmp_path):
    with pytest.raises(HistoryError, match='no exchange sessions'):
        asyncio.run(provider(tmp_path, Client()).fetch(HistoryRequest(('SPY',), '2024-11-28', '2024-11-28')))


def test_wrong_contract_fails_before_history(tmp_path):
    class Wrong(Client):
        async def reqContractDetailsAsync(self, contract):
            result = await super().reqContractDetailsAsync(contract)
            result[0].contract.currency = 'CAD'
            return result
    client = Wrong()
    with pytest.raises(HistoryError, match='differs'):
        asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')))
    assert not client.calls and client.disconnected


@pytest.mark.parametrize('kwargs', [dict(symbols='SPY'), dict(symbols=('SPY', 'SPY')), dict(bar_size='0 sec'), dict(start='2025-01-01'), dict(calendar='XLON')])
def test_bad_request(kwargs):
    with pytest.raises(ValueError):
        HistoryRequest(**(dict(symbols=('SPY',), start='2024-01-01', end='2024-01-05') | kwargs))


def test_fetch_and_cached_research_are_identical(tmp_path):
    from convexity.backtest.research import Research
    from convexity.runtime import Strategy, SubmitOrder
    class Buy(Strategy):
        metadata = StrategyMetadata('buy')
        contract = StrategyContract(('SPY',))
        def on_start(self, context):
            return [SubmitOrder('SPY', 'BUY', 1)]
    client = Client()
    work = Research('2024-11-27', '2024-11-29')
    source = provider(tmp_path, client)
    first = work.run(Buy, provider=source, cache=None)
    second = work.run(Buy, provider=source, cache=None)
    assert first.to_dict() == second.to_dict()
    assert len(client.calls) == 1


def test_cached_calendar_cannot_silently_shorten_period(tmp_path):
    source = provider(tmp_path, Client())
    request = HistoryRequest(('SPY',), '2024-11-27', '2024-11-29')
    result = asyncio.run(source.fetch(request))
    payload = json.loads(result.cache_path.read_text())
    payload['bars'] = payload['bars'][:-1]
    payload['sessions'] = payload['sessions'][:-1]
    from convexity.backtest.data import HistoricalData
    payload['manifest']['data_identity'] = HistoricalData(result.data.bars[:-1], price_basis=result.data.price_basis).identity
    identity = _digest(payload)
    (tmp_path / 'snapshots' / (identity + '.json')).write_text(json.dumps(payload))
    source._index(request).write_text(json.dumps({'sha256': identity}))
    with pytest.raises(HistoryError, match='cache is invalid'):
        asyncio.run(source.fetch(request))


def test_conflicting_overlap_between_chunks_rejected(tmp_path):
    class Overlap(Client):
        prior = None
        async def reqHistoricalDataAsync(self, contract, **kwargs):
            rows = await super().reqHistoricalDataAsync(contract, **kwargs)
            if self.prior is not None:
                rows.insert(0, NS(**(vars(self.prior) | {'close': 100})))
            self.prior = rows[-1]
            return rows
    with pytest.raises(HistoryError, match='revised overlapping history'):
        asyncio.run(provider(tmp_path, Overlap()).fetch(HistoryRequest(('SPY',), '2023-01-01', '2024-12-31')))


def test_gateway_failure_disconnects_and_never_creates_cache(tmp_path):
    class Failure(Client):
        async def reqHistoricalDataAsync(self, contract, **kwargs):
            raise TimeoutError('test broker timeout')
    client = Failure()
    with pytest.raises(HistoryError, match='No backtest data was fabricated'):
        asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')))
    assert client.disconnected
    assert not list(tmp_path.rglob('*.json'))


def test_cancelled_acquisition_disconnects_without_cache(tmp_path):
    class Cancelled(Client):
        async def reqHistoricalDataAsync(self, contract, **kwargs):
            raise asyncio.CancelledError
    client = Cancelled()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')))
    assert client.disconnected
    assert not list(tmp_path.rglob('*.json'))


@pytest.mark.parametrize(('size', 'counts'), [('7 mins', (56, 30)), ('17 mins', (23, 13)),
                                            ('1 hour', (7, 4)), ('2.5 hours', (3, 2))])
def test_custom_intraday_size_complete_sessions_and_cache(tmp_path, size, counts):
    client = Client()
    source = provider(tmp_path, client)
    request = HistoryRequest(('SPY', 'QQQ'), '2024-11-27', '2024-11-29', size)
    result = asyncio.run(source.fetch(request))
    assert len(result.data.bars) == sum(counts) * 2
    assert all(call[1]['barSizeSetting'] == '1 min' for call in client.calls)
    for symbol in request.symbols:
        bars = [bar for bar in result.data.bars if bar.symbol == symbol]
        assert sum(bar.volume for bar in bars) == 600 * 1234
        assert bars[-1].end == datetime(2024, 11, 29, 18, tzinfo=timezone.utc)
    assert result.manifest['source_bar_size'] == '1 min'
    cached = asyncio.run(source.fetch(request))
    assert cached.cache_hit and cached.data.identity == result.data.identity
    assert len(client.calls) == 2


def test_aggregation_preserves_ohlcv_with_out_of_order_source(tmp_path):
    def mutate(rows):
        for i, row in enumerate(rows):
            row.open, row.high, row.low, row.close, row.volume = 100+i, 102+i, 99+i, 101+i, i+1
        return list(reversed(rows))
    result = asyncio.run(provider(tmp_path, Client(mutate)).fetch(
        HistoryRequest(('SPY',), '2024-11-29', '2024-11-29', '7 mins')))
    bar = result.data.bars[0]
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (100, 108, 99, 107, 28)
    assert bar.end - bar.start == timedelta(minutes=7)


def test_larger_bars_do_not_hide_missing_source_minutes(tmp_path):
    def mutate(rows):
        del rows[2]
        return rows
    with pytest.raises(HistoryError, match='incomplete 1 min'):
        asyncio.run(provider(tmp_path, Client(mutate)).fetch(
            HistoryRequest(('SPY',), '2024-11-29', '2024-11-29', '1 hour')))
    assert not list(tmp_path.rglob('*.json'))


@pytest.mark.parametrize(('size', 'source', 'count'), [('7 secs', '1 secs', 18),
                                                     ('90 secs', '30 secs', 2),
                                                     ('45 seconds', '15 secs', 3)])
def test_custom_seconds_use_divisible_source_and_clip_close(tmp_path, monkeypatch, size, source, count):
    start = datetime(2024, 11, 29, 14, 30, tzinfo=timezone.utc)
    calendar = SessionCalendar((Session(start.date(), start, start + timedelta(minutes=2)),))
    monkeypatch.setattr(SessionCalendar, 'from_exchange', lambda *args: calendar)
    client = Client()
    result = asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), start.date(), start.date(), size)))
    assert len(result.data.bars) == count
    assert client.calls[0][1]['barSizeSetting'] == source
    assert client.calls[0][1]['durationStr'] == '120 S'
    assert result.data.bars[-1].end == calendar.sessions[0].close


@pytest.mark.parametrize(('size', 'count'), [('2 days', 3), ('1 week', 2), ('1 month', 2), ('2 years', 1)])
def test_multisession_history_aggregates_daily_source(tmp_path, size, count):
    client = Client()
    result = asyncio.run(provider(tmp_path, client).fetch(
        HistoryRequest(('SPY',), '2024-11-27', '2024-12-04', size)))
    assert len(result.data.bars) == count
    assert sum(bar.volume for bar in result.data.bars) == 5 * 1234
    assert client.calls[0][1]['barSizeSetting'] == '1 day'


def test_weekly_research_uses_daily_execution_and_completed_observations(tmp_path):
    from convexity.backtest.research import Research
    from convexity.runtime import Bars, Strategy, SubmitOrder
    seen, initial = [], []
    class Weekly(Strategy):
        metadata = StrategyMetadata('weekly')
        contract = StrategyContract(('SPY',), (BarSubscription(('SPY',), '1 week'),), warmup_sessions=3)
        def on_start(self, context):
            initial.extend(context.history.get('SPY', ()))
        def on_event(self, event, context):
            if isinstance(event, Bars):
                seen.append((event.time, len(context.history['SPY'])))
                if len(seen) == 1:
                    return [SubmitOrder('SPY', 'BUY', 1)]
    client = Client()
    result = Research('2024-11-27', '2024-12-06').run(Weekly, provider=provider(tmp_path, client), cache=None)
    assert result.status == 'completed'
    assert [time.date() for time, _ in seen] == [date(2024, 11, 29), date(2024, 12, 6)]
    assert initial and all(bar.end.date() < date(2024, 11, 27) for bar in initial)
    assert seen[0][1] == len(initial) + 1 and seen[1][1] == len(initial) + 2
    assert len(result.fills) == 1
    assert result.fills[0].occurred_at.date() == date(2024, 12, 2)
    assert all(call[1]['barSizeSetting'] == '1 day' for call in client.calls)


def test_second_chunks_pacing_and_aggregation_across_chunk_boundary(tmp_path, monkeypatch):
    start = datetime(2024, 11, 29, 14, 30, tzinfo=timezone.utc)
    calendar = SessionCalendar((Session(start.date(), start, start + timedelta(minutes=31)),))
    monkeypatch.setattr(SessionCalendar, 'from_exchange', lambda *args: calendar)
    sleeps = []
    async def sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr(asyncio, 'sleep', sleep)
    client = Client()
    result = asyncio.run(provider(tmp_path, client).fetch(HistoryRequest(('SPY',), start.date(), start.date(), '7 secs')))
    assert [call[1]['durationStr'] for call in client.calls] == ['1800 S', '60 S']
    assert sleeps == [10.1]
    assert len(result.data.bars) == 266
    assert sum(bar.volume for bar in result.data.bars) == 1860 * 1234
    assert result.data.bars[257].start == start + timedelta(seconds=1799)
    assert result.data.bars[257].end == start + timedelta(seconds=1806)


def test_custom_interval_tracks_dst_session_open(tmp_path):
    result = asyncio.run(provider(tmp_path, Client()).fetch(
        HistoryRequest(('SPY',), '2024-11-01', '2024-11-04', '11 mins')))
    assert len(result.data.bars) == 72
    assert result.data.bars[0].start.hour == 13
    assert result.data.bars[36].start.hour == 14
    assert all(bar.start.date() == bar.end.date() for bar in result.data.bars)


def test_weekly_signal_day_order_expires_on_next_session(tmp_path):
    from convexity.backtest.research import Research
    from convexity.runtime import Bars, Strategy, SubmitOrder
    class Weekly(Strategy):
        metadata = StrategyMetadata('weekly')
        contract = StrategyContract(('SPY',), (BarSubscription(('SPY',), '1 week'),))
        def on_event(self, event, context):
            if isinstance(event, Bars) and not context.orders:
                return [SubmitOrder('SPY', 'BUY', 1, order_type='limit', limit_price=Decimal(1), time_in_force='DAY')]
    result = Research('2024-11-27', '2024-12-06').run(Weekly, provider=provider(tmp_path, Client()), cache=None)
    assert result.status == 'completed'
    assert not result.fills
    assert result.orders[0].status == 'expired'
    assert result.orders[0].expires_at is not None
    assert result.orders[0].expires_at.date() == date(2024, 12, 2)


def test_offline_resample_preserves_warmup_and_split_boundaries(tmp_path):
    result = asyncio.run(provider(tmp_path, Client()).fetch(
        HistoryRequest(('SPY',), '2024-11-25', '2024-12-06')))
    boundary = datetime(2024, 11, 27, 14, 30, tzinfo=timezone.utc)
    observations = result.data.resample('1 week', result.calendar, boundaries=(boundary,))
    assert [(bar.start.date(), bar.end.date()) for bar in observations.bars] == [
        (date(2024, 11, 25), date(2024, 11, 26)),
        (date(2024, 11, 27), date(2024, 11, 29)),
        (date(2024, 12, 2), date(2024, 12, 6))]
    missing = replace(result.data, bars=result.data.bars[:1] + result.data.bars[2:])
    with pytest.raises(HistoryError, match='Incomplete source coverage'):
        missing.resample('1 week', result.calendar)


def test_minute_chunks_batch_sessions_with_bounded_source_rows():
    from convexity.backtest.history import _chunks
    calendar = SessionCalendar.from_exchange('XNYS', '2022-01-01', '2025-12-31')
    chunks = list(_chunks(calendar.sessions, '1 min'))
    assert tuple(session for group, _ in chunks for session in group) == calendar.sessions
    assert len(chunks) < len(calendar.sessions) / 4
    for group, duration in chunks:
        assert sum((session.close - session.open).total_seconds() / 60 for session in group) <= 2000
        assert int(duration.split()[0]) <= 365


def test_daily_chunks_bound_calendar_duration():
    from convexity.backtest.history import _chunks
    calendar = SessionCalendar.from_exchange('XNYS', '2022-01-01', '2025-12-31')
    chunks = list(_chunks(calendar.sessions, '1 day'))
    assert len(chunks) == 4
    assert tuple(session for group, _ in chunks for session in group) == calendar.sessions
    assert all(int(duration.split()[0]) <= 365 for _, duration in chunks)


def test_historical_connection_needs_no_paper_account_or_file(tmp_path):
    from convexity.backtest.history import HistoricalConnectionConfig
    class ReadOnlyClient(Client):
        async def connectAsync(self, *args, **kwargs):
            assert kwargs['readonly'] is True
            assert 'account' not in kwargs
        def managedAccounts(self):
            raise AssertionError('Historical data does not inspect trading accounts')
    source = IBKRHistoricalData(HistoricalConnectionConfig(), cache_dir=tmp_path,
                                client_factory=ReadOnlyClient, request_interval=0)
    result = asyncio.run(source.fetch(HistoryRequest(('SPY',), '2024-11-29', '2024-11-29')))
    assert len(result.data.bars) == 1


def test_historical_connection_reads_environment(monkeypatch):
    from convexity.backtest.history import HistoricalConnectionConfig
    monkeypatch.setenv('IBKR_HOST', 'localhost')
    monkeypatch.setenv('IBKR_PORT', '4002')
    monkeypatch.setenv('IBKR_CLIENT_ID', '19')
    config = HistoricalConnectionConfig.from_env()
    assert (config.host, config.port, config.client_id) == ('localhost', 4002, 19)
