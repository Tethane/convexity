"""Daily point-in-time research loading into the existing event replay engine."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from convexity.runtime import Bars, Instrument, Timer
from convexity.runtime.contracts import (
    BarSubscription,
    EventSubscription,
    event_schedule,
)
from .data import HistoricalData, SessionCalendar
from .bar_size import parse_bar_size
from .exchange import ExecutionModel
from .history import HistoricalDataset, HistoryError, HistoryRequest, validate_coverage
from .runner import Backtest, serialize


def _coverage_failure(error):
    """Only date-local coverage failures justify smaller acquisition requests."""
    message = str(error).lower()
    infrastructure = (
        "credential",
        "api key",
        "secret",
        "permission",
        "entitlement",
        "unauthorized",
        "forbidden",
        "http",
        "network",
        "timeout",
        "pacing",
        "rate limit",
        "connection",
        "corrupt",
        "checksum",
        "cache is invalid",
        "different",
        "unexpected",
        "malformed",
    )
    return (
        isinstance(error, HistoryError)
        and not any(word in message for word in infrastructure)
        and any(
            word in message
            for word in (
                "incomplete",
                "missing",
                "empty",
                "no bars",
                "no data",
                "coverage gap",
            )
        )
    )


def _runs(sessions, labels):
    result, group = [], []
    for session in sessions:
        if str(session.label) in labels:
            group.append(session)
        elif group:
            result.append(tuple(group))
            group = []
    if group:
        result.append(tuple(group))
    return result


def _cached_runs(provider, request, sessions):
    """Locate cache keys only; normal fetch validates payloads and provenance.

    Prefer one provider namespace and never read or manufacture bars here.
    Missing rows remain missing. A corrupt cache is reported by ordinary fetch.
    """
    import sqlite3
    from contextlib import closing
    from .cache import CachedHistoricalProvider
    from .providers import FallbackHistoricalProvider

    candidates = (
        (provider.primary, provider.fallback)
        if isinstance(provider, FallbackHistoricalProvider)
        else (provider,)
    )
    for candidate in candidates:
        if (
            not isinstance(candidate, CachedHistoricalProvider)
            or not candidate.path.is_file()
        ):
            continue
        with closing(sqlite3.connect(candidate.path)) as db:
            rows = db.execute(
                "SELECT session FROM history_sessions WHERE namespace=? AND calendar=? "
                "AND resolution=? AND symbol=? AND session BETWEEN ? AND ? "
                "AND (? IS NULL OR price_basis=?)",
                (
                    candidate.namespace,
                    request.calendar,
                    request.bar_size,
                    request.symbols[0],
                    str(request.start),
                    str(request.end),
                    candidate.price_basis,
                    candidate.price_basis,
                ),
            ).fetchall()
        if rows:
            return candidate, _runs(sessions, {row[0] for row in rows})
    return None, []


async def _daily_segments(provider, request, sessions, *, refresh, policy):
    """Preserve usable observations around gaps without unbounded provider retries."""
    import sqlite3

    successful, failures = [], []

    def failure(part, error):
        failures.append(
            dict(
                start=str(part.start),
                end=str(part.end),
                reason=f"{type(error).__name__}: {error}",
            )
        )

    def part_request(group):
        return HistoryRequest(
            request.symbols,
            group[0].label,
            group[-1].label,
            request.bar_size,
            request.calendar,
        )

    async def fetch(source, part, group, refresh_part):
        dataset = await source.fetch(part, refresh=refresh_part)
        expected = SessionCalendar(tuple(group), request.calendar)
        if dataset.calendar.sessions != expected.sessions:
            raise HistoryError("Provider returned a different session calendar.")
        if {bar.symbol for bar in dataset.data.bars} != set(part.symbols):
            raise HistoryError("Provider returned a different historical ticker.")
        if getattr(dataset.data, "sparse", False) and policy == "exclude":
            valid = {(session.open, session.close) for session in group}
            if any((bar.start, bar.end) not in valid for bar in dataset.data.bars):
                raise HistoryError("Provider returned unexpected daily intervals.")
        else:
            validate_coverage(dataset.data, expected, part)
        successful.append((dataset, part))

    try:
        await fetch(provider, request, sessions, refresh)
        return successful, failures
    except (HistoryError, RuntimeError, ValueError, OSError) as original:
        if policy == "fail":
            raise
        failure(request, original)
        retry_coverage = _coverage_failure(original)

    recovered = set()
    if not refresh:
        try:
            cached, groups = _cached_runs(provider, request, sessions)
            for group in groups:
                part = part_request(group)
                try:
                    await fetch(cached, part, group, False)
                    recovered.update(str(session.label) for session in group)
                except (HistoryError, RuntimeError, ValueError, OSError) as error:
                    failure(part, error)
        except (sqlite3.Error, OSError) as error:
            failure(request, error)
    if not retry_coverage:
        return successful, failures
    if len(sessions) == 1 and not recovered:
        return successful, failures
    pending = _runs(sessions, {str(s.label) for s in sessions} - recovered)
    # The original interval already failed. Split once before acquiring it again.
    if not recovered and len(pending) == 1 and len(pending[0]) > 1:
        group = pending.pop()
        middle = len(group) // 2
        pending.extend((group[:middle], group[middle:]))
    attempts = 0
    while pending:
        group = pending.pop(0)
        part = part_request(group)
        if attempts >= 64:
            failure(
                part,
                HistoryError(
                    "Coverage salvage request budget exhausted; interval remains unavailable."
                ),
            )
            continue
        attempts += 1
        try:
            await fetch(provider, part, group, refresh)
        except (HistoryError, RuntimeError, ValueError, OSError) as error:
            failure(part, error)
            if _coverage_failure(error) and len(group) > 1:
                middle = len(group) // 2
                pending.extend((group[:middle], group[middle:]))
    return successful, failures


async def run_universe_research(
    research,
    instances,
    *,
    provider,
    cache,
    execution,
    initial_cash,
    base_currency,
    fx_rates,
    limits,
    output,
    refresh,
    metadata,
    missing_data,
    corporate_actions=(),
):
    """Fetch verified ticker intervals; never expose future membership to callbacks."""
    from convexity.data.universe import UniverseStore
    from .cache import CachedHistoricalProvider
    from .providers import FallbackHistoricalProvider

    if missing_data not in ("exclude", "fail"):
        raise ValueError("missing_data must be 'exclude' or 'fail'.")
    declarations = tuple(
        dict.fromkeys(u for s in instances.values() for u in s.contract.universes)
    )
    subscriptions = tuple(
        sub for s in instances.values() for sub in s.contract.all_subscriptions
    )
    calendars = {u.calendar for u in declarations} | {
        sub.calendar for sub in subscriptions if isinstance(sub, BarSubscription)
    }
    calendars |= {
        schedule.calendar
        for sub in subscriptions
        if (schedule := event_schedule(sub)) is not None
    }
    if len(calendars) != 1:
        raise ValueError("Dynamic universe research requires one exchange calendar.")
    calendar_name = next(iter(calendars))
    calendar = SessionCalendar.from_exchange(
        calendar_name, research.start, research.end
    )
    if not calendar.sessions:
        raise ValueError("Requested dates contain no exchange trading sessions.")
    first, last = calendar.sessions[0].open, calendar.sessions[-1].close
    warmup_count = max(s.contract.warmup_sessions for s in instances.values())
    history_start = research._calendar_start(
        calendar_name, calendar.sessions[0].label, max(1, warmup_count)
    )
    history_calendar = SessionCalendar.from_exchange(
        calendar_name, history_start, research.end
    )
    history_first = history_calendar.sessions[0].open
    timer_owners = {}
    for owner, strategy in instances.items():
        for timer in research._timers(strategy.contract.all_subscriptions, calendar):
            timer_owners.setdefault(timer, []).append(owner)
    timers = tuple(sorted(timer_owners, key=lambda t: (t.time, t.name)))
    if any(
        session.open < timer.time < session.close
        for timer in timers
        for session in calendar.sessions
    ):
        raise ValueError(
            "Dynamic daily universe research supports session-open/close decisions; intraday execution requires an explicit intraday data replay."
        )
    if any(
        isinstance(sub, BarSubscription)
        and parse_bar_size(sub.bar_size).label != "1 day"
        for sub in subscriptions
    ):
        raise ValueError("Dynamic universe observations currently require daily bars.")
    from convexity.runtime import FillEvent, OrderUpdate, Rejection, SubscriptionUpdate

    if any(
        isinstance(sub, EventSubscription)
        and sub.event_type
        not in (Timer, FillEvent, OrderUpdate, Rejection, SubscriptionUpdate)
        for sub in subscriptions
    ):
        raise ValueError("No historical source for the declared event subscription.")
    if provider is None:
        from .alpaca_daily import AlpacaDailyHistoricalData

        provider = FallbackHistoricalProvider(
            primary=AlpacaDailyHistoricalData(), cache=cache
        )
    if cache is not None and not isinstance(
        provider, (CachedHistoricalProvider, FallbackHistoricalProvider)
    ):
        provider = CachedHistoricalProvider(provider, cache)
    jobs, owner_ids, failures, provenance = set(), {}, [], []
    for owner, strategy in instances.items():
        ids = set()
        for declaration in strategy.contract.universes:
            with UniverseStore(declaration.store_path) as store:
                members = store.historical_members(
                    (declaration.name,),
                    first,
                    last + timedelta(microseconds=1),
                    allow_reconstructed=declaration.allow_reconstructed,
                )
                for member in members:
                    ids.add(member.security_id)
                    mappings = store.symbol_history(
                        member.security_id,
                        history_first,
                        last + timedelta(microseconds=1),
                        allow_reconstructed=declaration.allow_reconstructed,
                    )
                    if not mappings:
                        failures.append(
                            dict(
                                security_id=member.security_id,
                                symbol=member.symbol,
                                start=str(history_start),
                                end=str(research.end),
                                reason="No verified historical ticker mapping",
                            )
                        )
                    for mapping in mappings:
                        jobs.add(
                            (
                                member.security_id,
                                mapping.symbol,
                                mapping.start,
                                mapping.end,
                            )
                        )
                provenance.append(
                    dict(
                        owner=owner,
                        universe=declaration.name,
                        reconstructed=declaration.allow_reconstructed,
                        store=declaration.store_path,
                    )
                )
        # Fixed symbols such as a benchmark share the same daily replay.
        fixed = set(strategy.contract.symbols) | set(
            strategy.contract.available_observed_symbols
        )
        for symbol in fixed:
            jobs.add((symbol, symbol, history_first, last + timedelta(microseconds=1)))
        owner_ids[owner] = ids | fixed
    bars, bases, manifests = {}, set(), []
    native_daily = False
    for identity, symbol, start, end in sorted(
        jobs, key=lambda row: (row[0], row[2], row[1])
    ):
        sessions = tuple(
            s for s in history_calendar.sessions if start <= s.open and s.close < end
        )
        if not sessions:
            continue
        request = HistoryRequest(
            (symbol,), sessions[0].label, sessions[-1].label, calendar=calendar_name
        )
        try:
            segments, errors = await _daily_segments(
                provider, request, sessions, refresh=refresh, policy=missing_data
            )
        except (HistoryError, RuntimeError, ValueError, OSError) as exc:
            raise HistoryError(
                f"Missing historical data for {identity} ({symbol}): {exc}"
            ) from exc
        failures.extend(
            dict(security_id=identity, symbol=symbol, **error) for error in errors
        )
        for dataset, part in segments:
            mapped_bars = {}
            for bar in dataset.data.bars:
                if not start <= bar.start or not bar.end < end:
                    raise HistoryError(
                        "Provider returned observations outside the historical ticker interval."
                    )
                mapped = replace(bar, symbol=identity)
                key = identity, bar.end
                if key in bars and bars[key] != mapped:
                    raise HistoryError(
                        "Conflicting historical ticker data for a stable security identity."
                    )
                mapped_bars[key] = mapped
            bars.update(mapped_bars)
            native_daily |= (
                "native-daily" in dataset.data.source
                or dataset.manifest.get("aggregation") == "native_daily"
            )
            bases.add(dataset.data.price_basis)
            manifests.append(
                dict(
                    security_id=identity,
                    symbol=symbol,
                    request=part.to_dict(),
                    manifest=dataset.manifest,
                )
            )
    if len(bases) > 1:
        raise HistoryError(
            "Cannot combine raw and adjusted historical price conventions."
        )
    data = (
        HistoricalData(
            tuple(bars.values()),
            source="historical-universe-provider-native-daily"
            if native_daily
            else "historical-universe-provider",
            price_basis=next(iter(bases), "unadjusted_no_corporate_actions"),
            sparse=True,
        )
        if bars
        else None
    )
    if native_daily and any(
        timer.time != session.open
        for timer in timers
        for session in calendar.sessions
        if session.open <= timer.time <= session.close
    ):
        raise ValueError(
            "Native daily universe research requires session-open decisions; bars are published at the next session open."
        )
    observations, warmup = {}, {}

    for owner, strategy in instances.items():
        count = strategy.contract.warmup_sessions
        cutoff = (
            history_calendar.sessions[-len(calendar.sessions) - count].open
            if count
            else first
        )
        observed = (
            owner_ids[owner]
            if strategy.contract.universes
            else set(strategy.contract.available_observed_symbols)
        )
        panels, prior = {}, []
        for bar in bars.values():
            if bar.symbol not in observed:
                continue
            if cutoff <= bar.start and bar.end <= first:
                prior.append(bar)
            elif first <= bar.start and bar.end <= last:
                publication = bar.end
                if native_daily:
                    publication = next(
                        (
                            session.open
                            for session in calendar.sessions
                            if session.open > bar.end
                        ),
                        None,
                    )
                    if publication is None:
                        continue
                panels.setdefault(publication, {})[bar.symbol] = bar
        observations[owner] = tuple(
            Bars(instant, values, delayed=native_daily)
            for instant, values in sorted(panels.items())
        )
        warmup[owner] = tuple(prior)
    # Coverage describes true eligible IDs separately from observed daily prices.
    coverage = []
    for session in calendar.sessions:
        observable = {
            identity for identity, instant in bars if instant == session.close
        }
        for declaration in declarations:
            snapshot = declaration.resolve(session.close)
            eligible = set(snapshot.security_ids)
            missing = sorted(eligible - observable)
            coverage.append(
                dict(
                    date=str(session.label),
                    universe=declaration.name,
                    eligible=len(eligible),
                    observable=len(eligible & observable),
                    ratio=len(eligible & observable) / len(eligible)
                    if eligible
                    else None,
                    missing=missing,
                    warnings=list(snapshot.warnings),
                )
            )
            if missing and missing_data == "fail":
                raise HistoryError(
                    f"Incomplete {declaration.name} coverage on {session.label}: {', '.join(missing)}"
                )
    run_metadata = dict(metadata or {})
    run_metadata["research"] = dict(
        start=str(research.start),
        end=str(research.end),
        contracts={owner: serialize(s.contract) for owner, s in instances.items()},
        history=manifests,
        provenance=provenance,
        execution_bar_size="1 day",
        observation_availability="next session open"
        if native_daily
        else "completed regular session",
        execution_assumptions="Daily OHLC approximates regular execution; native Alpaca daily volume includes extended hours and may overstate regular-session participation capacity.",
        coverage=coverage,
        provider_failures=failures,
        missing_data_policy=missing_data,
        universe="Decision-time eligible universe is distinct from observable prices and selected candidates.",
        limitations="Historical data may omit delisted securities; incomplete coverage is not survivorship-bias-free. Imported reconstructed membership may lack contemporaneous publication evidence and pre-entry ticker history.",
    )
    model = (
        ExecutionModel.preset(execution) if isinstance(execution, str) else execution
    )
    identities = {bar.symbol for bar in bars.values()}
    instrument_definitions = {}
    for strategy in instances.values():
        for instrument in strategy.contract.resolved_instruments:
            if (
                instrument.symbol in instrument_definitions
                and instrument_definitions[instrument.symbol] != instrument
            ):
                raise ValueError(
                    f"Conflicting instrument definitions for {instrument.symbol}."
                )
            instrument_definitions[instrument.symbol] = instrument
    replay = Backtest(
        data=data,
        strategies={owner: (lambda s=s: s) for owner, s in instances.items()},
        execution=model,
        initial_cash=initial_cash,
        base_currency=base_currency,
        fx_rates=fx_rates,
        limits=limits,
        instruments=tuple(
            instrument_definitions.get(s, Instrument(s)) for s in sorted(identities)
        ),
        calendar=history_calendar,
        timers=timers,
        timer_owners={t: tuple(v) for t, v in timer_owners.items()},
        start=first,
        end=last,
        metadata=run_metadata,
        publish_bars=False,
        preload_marks=True,
        strategy_warmup=warmup,
        observations_by_owner=observations,
        corporate_actions=corporate_actions,
    )
    result = replay.run()
    # Include decisions/rejections as evidence, retaining normal execution reports.
    run_metadata["research"]["portfolio_plans"] = {
        owner: serialize(getattr(s, "plans", ())) for owner, s in instances.items()
    }
    run_metadata["research"]["portfolio_rejections"] = {
        owner: list(getattr(s, "rejections", ())) for owner, s in instances.items()
    }
    if output is not None:
        if data is None:
            result.export(output)
        else:
            research._export(
                HistoricalDataset(
                    data, history_calendar, run_metadata["research"], Path(cache or ".")
                ),
                result,
                Path(output),
            )
    return result.raise_for_failure()
