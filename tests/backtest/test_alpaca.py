"""Offline wire fixtures for Alpaca acquisition, credentials, and provider routing."""
import asyncio
from datetime import timedelta
from decimal import Decimal
from email.message import Message
import json
from urllib.error import HTTPError

import pytest

from convexity import AlpacaCredentials, AlpacaHistoricalData, FallbackHistoricalProvider, Research
from convexity.backtest import HistoryError, HistoryRequest, SessionCalendar
from tests.backtest.test_cache import Provider
from tests.backtest.test_research import BuyAtStart


def request(symbols=('SPY',), size='1 min'):
    return HistoryRequest(symbols, '2024-11-27', '2024-11-29', size)


def rows():
    calendar = SessionCalendar.from_exchange('XNYS', '2024-11-27', '2024-11-29')
    result = []
    for session in calendar.sessions:
        # Include pre/post-market bars; only regular session minutes survive.
        for offset in range(-1, int((session.close - session.open).total_seconds() / 60) + 1):
            result.append(dict(t=(session.open + timedelta(minutes=offset)).isoformat(),
                               o='100.123456789', h='102', l='99', c='101', v=100))
    return result


def provider(monkeypatch, pages):
    result = AlpacaHistoricalData(AlpacaCredentials('test-key', 'test-secret'), request_interval=0)
    calls = []
    def page(params, credentials):
        calls.append(dict(params))
        return pages.pop(0)
    monkeypatch.setattr(result, '_request_page', page)
    return result, calls


def test_pagination_regular_hours_early_close_and_decimal_prices(monkeypatch):
    raw = rows()
    source, calls = provider(monkeypatch, [dict(bars={'SPY': raw[:7]}, next_page_token='next'),
                                         dict(bars={'SPY': raw[7:]}, next_page_token=None)])
    result = asyncio.run(source.fetch(request()))
    assert len(result.data.bars) == 600
    assert result.data.bars[-1].end.hour == 18
    assert result.data.bars[0].open == Decimal('100.123456789')
    assert len(calls) == 2 and calls[1]['page_token'] == 'next'
    assert calls[0]['feed'] == 'sip' and calls[0]['adjustment'] == 'split'
    assert calls[0]['limit'] == 10000 and calls[0]['asof'] == '-'
    assert 'test-key' not in json.dumps(result.manifest)
    assert 'test-secret' not in repr(source._credentials)


def test_daily_aggregates_regular_minutes(monkeypatch):
    source, _ = provider(monkeypatch, [dict(bars={'SPY': rows()}, next_page_token=None)])
    result = asyncio.run(source.fetch(request(size='1 day')))
    assert [b.volume for b in result.data.bars] == [39000, 21000]
    assert result.data.bars[0].open == Decimal('100.123456789')


@pytest.mark.parametrize('mutation', ['missing', 'conflict', 'naive', 'volume', 'alignment', 'empty', 'other_symbol'])
def test_bad_data_is_rejected(monkeypatch, mutation):
    raw = rows()
    symbols = {'SPY': raw}
    if mutation == 'missing': del raw[4]
    if mutation == 'conflict': raw.append(dict(raw[4], c='100'))
    if mutation == 'naive': raw[4]['t'] = '2024-11-27T14:33:00'
    if mutation == 'volume': raw[4]['v'] = '0.1'
    if mutation == 'alignment': raw[4]['t'] = '2024-11-27T14:33:01Z'
    if mutation == 'empty': symbols = {}
    if mutation == 'other_symbol': symbols = {'QQQ': raw}
    source, _ = provider(monkeypatch, [dict(bars=symbols, next_page_token=None)])
    with pytest.raises(HistoryError):
        asyncio.run(source.fetch(request()))


def test_missing_symbol_panel_is_history_error(monkeypatch):
    source, _ = provider(monkeypatch, [dict(bars={'SPY': rows(), 'QQQ': rows()[:-3]}, next_page_token=None)])
    with pytest.raises(HistoryError):
        asyncio.run(source.fetch(request(('SPY', 'QQQ'))))


def test_repeated_token_and_subminute_fail(monkeypatch):
    source, calls = provider(monkeypatch, [dict(bars={}, next_page_token='same')] * 2)
    with pytest.raises(HistoryError, match='pagination'):
        asyncio.run(source.fetch(request()))
    with pytest.raises(HistoryError, match='sub-minute'):
        asyncio.run(source.fetch(request(size='30 secs')))
    assert len(calls) == 2


@pytest.mark.parametrize('status, attempts', [(401, 1), (403, 1), (429, 3), (500, 3)])
def test_http_errors_bounded_and_secret_safe(monkeypatch, status, attempts):
    source = AlpacaHistoricalData(AlpacaCredentials('test-key', 'test-secret'), request_interval=0)
    calls = []
    def fail(*args):
        calls.append(None)
        headers = Message()
        headers['Retry-After'] = '0'
        raise HTTPError('https://example.invalid/test-secret', status, 'test-key', headers, None)
    monkeypatch.setattr(source, '_request_page', fail)
    with pytest.raises(HistoryError, match=f'HTTP {status}') as error:
        asyncio.run(source.fetch(request()))
    assert len(calls) == attempts
    assert 'test-key' not in str(error.value) and 'test-secret' not in str(error.value)
    assert error.value.__suppress_context__


def test_total_timeout_and_cancellation(monkeypatch):
    source = AlpacaHistoricalData(AlpacaCredentials('key', 'secret'), total_timeout=0.01)
    async def slow(*args):
        await asyncio.sleep(10)
    monkeypatch.setattr(source, '_page', slow)
    with pytest.raises(HistoryError, match='total timeout'):
        asyncio.run(source.fetch(request()))
    async def cancel(*args):
        raise asyncio.CancelledError
    monkeypatch.setattr(source, '_page', cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(source.fetch(request()))


def test_credentials_file_environment_and_permissions(tmp_path, monkeypatch):
    monkeypatch.delenv('APCA_API_KEY_ID', raising=False)
    monkeypatch.delenv('APCA_API_SECRET_KEY', raising=False)
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    directory = tmp_path / 'convexity'
    directory.mkdir()
    path = directory / 'alpaca.json'
    path.write_text(json.dumps(dict(api_key='file-key', secret_key='file-secret')))
    path.chmod(0o600)
    assert AlpacaCredentials.load().api_key == 'file-key'
    path.chmod(0o644)
    with pytest.raises(HistoryError, match='600'): AlpacaCredentials.load()
    monkeypatch.setenv('APCA_API_KEY_ID', 'env-key')
    with pytest.raises(HistoryError, match='both'): AlpacaCredentials.load()
    monkeypatch.setenv('APCA_API_SECRET_KEY', 'env-secret')
    assert AlpacaCredentials.load().api_key == 'env-key'
    monkeypatch.delenv('APCA_API_KEY_ID')
    monkeypatch.delenv('APCA_API_SECRET_KEY')
    path.chmod(0o600)
    path.write_text('{"secret_key": "bad-secret"')
    with pytest.raises(HistoryError) as error: AlpacaCredentials.load()
    assert 'bad-secret' not in str(error.value)


def test_fallback_cache_does_not_poison_primary_and_can_recover(tmp_path):
    class Primary(Provider):
        source = 'primary'
        unavailable = True
        async def fetch(self, request, *, refresh=False):
            if self.unavailable: raise HistoryError('offline')
            return await super().fetch(request, refresh=refresh)
    primary, backup = Primary(), Provider()
    router = FallbackHistoricalProvider(primary, backup, cache=tmp_path / 'history.sqlite3')
    with pytest.warns(RuntimeWarning, match='Falling back'):
        result = asyncio.run(router.fetch(request(size='1 day')))
    assert result.data.source == backup.source
    assert result.manifest['fallback']['selected'] == 'IBKR'
    with pytest.warns(RuntimeWarning):
        assert asyncio.run(router.fetch(request(size='1 day'))).cache_hit
    assert len(backup.requests) == 1
    primary.unavailable = False
    result = asyncio.run(router.fetch(request(size='1 day')))
    assert result.data.source == primary.source and 'fallback' not in result.manifest
    assert asyncio.run(router.fetch(request(size='1 day'))).cache_hit
    assert len(primary.requests) == 1
    asyncio.run(router.fetch(request(size='1 day'), refresh=True))
    assert primary.requests[-1][1] is True


def test_default_research_selects_alpaca_and_replays_offline(tmp_path, monkeypatch):
    import convexity.backtest.providers as module
    primary, backup = Provider(), Provider()
    monkeypatch.setattr(module, 'AlpacaHistoricalData', lambda: primary)
    monkeypatch.setattr(module, 'IBKRHistoricalData', lambda: backup)
    path = tmp_path / 'history.sqlite3'
    first = Research('2024-01-08', '2024-01-10').run(BuyAtStart, cache=path)
    second = Research('2024-01-08', '2024-01-10').run(BuyAtStart, cache=path)
    assert first.to_dict() == second.to_dict()
    assert len(primary.requests) == 1 and not backup.requests


def test_http_transport_headers_decimal_parsing_and_no_redirect(monkeypatch):
    import io
    from urllib.parse import parse_qs, urlsplit
    import convexity.backtest.alpaca as module
    captured = {}
    class Opener:
        def open(self, request, *, timeout):
            captured.update(request=request, timeout=timeout)
            return io.StringIO('{"bars": {}, "value": 100.123456789}')
    def opener(handler):
        assert handler.redirect_request(None, None, 302, None, None, 'https://evil.invalid') is None
        return Opener()
    monkeypatch.setattr(module, 'build_opener', opener)
    source = AlpacaHistoricalData(request_timeout=4)
    payload = source._request_page({'symbols': 'SPY,QQQ', 'feed': 'sip'}, AlpacaCredentials('key', 'secret'))
    sent = captured['request']
    assert urlsplit(sent.full_url).hostname == 'data.alpaca.markets'
    assert parse_qs(urlsplit(sent.full_url).query)['symbols'] == ['SPY,QQQ']
    assert sent.get_header('Apca-api-key-id') == 'key'
    assert sent.get_header('Apca-api-secret-key') == 'secret'
    assert 'secret' not in sent.full_url and captured['timeout'] == 4
    assert payload['value'] == Decimal('100.123456789')


def test_setup_script_stores_owner_only_without_printing_keys(tmp_path, monkeypatch, capsys):
    from scripts import configure_alpaca
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    monkeypatch.setattr(configure_alpaca.sys.stdin, 'isatty', lambda: True)
    values = iter(['hidden-key', 'hidden-secret'])
    monkeypatch.setattr(configure_alpaca.getpass, 'getpass', lambda prompt: next(values))
    configure_alpaca.main()
    path = tmp_path / 'convexity' / 'alpaca.json'
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text()) == dict(api_key='hidden-key', secret_key='hidden-secret')
    output = capsys.readouterr().out
    assert 'hidden-key' not in output and 'hidden-secret' not in output


def test_alpaca_cache_loads_without_credentials_or_network(tmp_path, monkeypatch):
    from convexity.backtest import CachedHistoricalProvider
    source, calls = provider(monkeypatch, [dict(bars={'SPY': rows()}, next_page_token=None)])
    path = tmp_path / 'history.sqlite3'
    first = asyncio.run(CachedHistoricalProvider(source, path).fetch(request()))
    offline = AlpacaHistoricalData()
    def forbidden():
        raise AssertionError('Covered requests must not load credentials')
    monkeypatch.setattr(AlpacaCredentials, 'load', forbidden)
    second = asyncio.run(CachedHistoricalProvider(offline, path).fetch(request()))
    assert second.cache_hit and len(calls) == 1
    assert first.data == second.data and first.manifest == second.manifest
