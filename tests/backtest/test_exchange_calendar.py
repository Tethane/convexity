"""Exchange-derived production sessions: holidays, early close and UTC DST changes.

NYSE 2026 calendar: https://www.nyse.com/trade/hours-calendars
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from convexity.backtest import SessionCalendar

pytest.importorskip("exchange_calendars")


def test_thanksgiving_holiday_and_black_friday_early_close():
    calendar = SessionCalendar.from_exchange("XNYS", "2026-11-26", "2026-11-27")
    assert [session.label for session in calendar.sessions] == [date(2026, 11, 27)]
    (session,) = calendar.sessions
    assert session.open == datetime(2026, 11, 27, 14, 30, tzinfo=timezone.utc)
    assert session.close == datetime(2026, 11, 27, 18, tzinfo=timezone.utc)
    assert calendar.daily("exit", offset=-timedelta(minutes=15))[0].time == datetime(
        2026, 11, 27, 17, 45, tzinfo=timezone.utc
    )
    assert "exchange_calendars-" in calendar.name


def test_us_session_clock_changes_with_daylight_saving():
    calendar = SessionCalendar.from_exchange("XNYS", date(2026, 3, 6), date(2026, 3, 9))
    before, after = calendar.sessions
    assert (before.open.hour, before.close.hour) == (14, 21)
    assert (after.open.hour, after.close.hour) == (13, 20)
    calendar = SessionCalendar.from_exchange("XNYS", "2026-10-30", "2026-11-02")
    before, after = calendar.sessions
    assert (before.open.hour, before.close.hour) == (13, 20)
    assert (after.open.hour, after.close.hour) == (14, 21)


def test_full_year_monthly_schedule_uses_exchange_sessions():
    calendar = SessionCalendar.from_exchange("XNYS", "2026-01-01", "2026-12-31")
    timers = calendar.monthly("rebalance", at="open")
    assert len(timers) == 12
    assert timers[0].time.date() == date(2026, 1, 2)
    assert timers[1].time.date() == date(2026, 2, 2)
    assert not {date(2026, 1, 1), date(2026, 6, 19), date(2026, 7, 3)} & {
        session.label for session in calendar.sessions
    }


def test_one_day_and_holiday_only_ranges_are_supported():
    assert (
        len(SessionCalendar.from_exchange("XNYS", "2026-11-27", "2026-11-27").sessions)
        == 1
    )
    assert (
        SessionCalendar.from_exchange("XNYS", "2026-11-26", "2026-11-26").sessions == ()
    )


def test_calendar_with_intraday_break_requires_explicit_supported_model():
    with pytest.raises(ValueError, match="intraday breaks"):
        SessionCalendar.from_exchange("XHKG", "2026-10-09", "2026-10-09")
