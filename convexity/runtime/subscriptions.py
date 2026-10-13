"""Owner-scoped active subscription state shared by simulation and paper trading."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime, time, timedelta
import hashlib
import json
from types import MappingProxyType, SimpleNamespace

from .contracts import (BarSubscription, ClockSubscription, EventSubscription,
                        StrategyContract, filter_context, select_event)
from .calendar_schedule import SessionSchedule
from .types import Bars, Context, Subscribe, SubscriptionUpdate, Timer, Unsubscribe, aware


def _canonical(value):
    if isinstance(value, type):
        return {"event_type": value.__module__ + "." + value.__qualname__}
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__module__ + "." + type(value).__qualname__,
                "fields": {name: _canonical(item) for name, item in asdict(value).items()}}
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"seconds": value.total_seconds()}
    return value


def subscription_id(subscription) -> str:
    """A stable default identity for an immutable subscription declaration."""
    if not isinstance(subscription, (BarSubscription, ClockSubscription, EventSubscription, SessionSchedule)):
        raise TypeError("Use a declared bar, clock, session, or event subscription.")
    payload = json.dumps(_canonical(subscription), sort_keys=True, separators=(",", ":"))
    return "sub-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class SubscriptionBook:
    """An immutable capability pool with mutable, checkpointable activation state.

    Activation never replays past events. Initial declarations may see explicitly
    seeded warmup; dynamically activated observations begin after activation.
    """
    def __init__(self, contract: StrategyContract | None, owner: str, restored: dict | None = None):
        self.contract, self.owner = contract, owner
        self._timer_cache = {}
        self._pool = {subscription_id(sub): sub for sub in contract.all_subscriptions} if contract else {}
        self._records = {subscription_id(sub): {"capability": subscription_id(sub), "active": True, "since": None}
                         for sub in contract.subscriptions} if contract else {}
        self._fingerprint = hashlib.sha256(json.dumps({"pool": sorted(self._pool),
            "initial": sorted(self._records)}, sort_keys=True).encode()).hexdigest()
        if restored is not None:
            if restored.get("contract") != self._fingerprint:
                raise ValueError("Persisted subscriptions belong to a different strategy capability contract.")
            records = restored.get("records")
            if not isinstance(records, dict):
                raise ValueError("Invalid persisted subscription state.")
            self._records = {}
            for identity, row in records.items():
                if not identity or row.get("capability") not in self._pool or type(row.get("active")) is not bool:
                    raise ValueError("Invalid persisted subscription identity/capability.")
                since = datetime.fromisoformat(row["since"]) if row.get("since") is not None else None
                if since is not None:
                    aware(since)
                self._records[identity] = {"capability": row["capability"], "active": row["active"], "since": since}

    @property
    def active(self):
        return MappingProxyType({identity: self._pool[row["capability"]]
                                 for identity, row in self._records.items() if row["active"]})

    def dump(self) -> dict:
        return {"contract": self._fingerprint, "records": {
            identity: {**row, "since": row["since"].isoformat() if row["since"] is not None else None}
            for identity, row in self._records.items()}}

    def apply(self, command: Subscribe | Unsubscribe, instant: datetime) -> SubscriptionUpdate:
        aware(instant)
        identity = command.subscription_id
        previous = self._records.get(identity)
        if isinstance(command, Subscribe):
            try:
                capability = subscription_id(command.subscription)
            except (ValueError, TypeError) as error:
                return SubscriptionUpdate(instant, self.owner, identity, None, "rejected",
                                          f"Unsupported subscription {type(command.subscription).__name__}: {error}")
            if capability not in self._pool:
                return SubscriptionUpdate(instant, self.owner, identity, command.subscription, "rejected",
                                          "Subscription is outside the strategy's declared capability pool.")
            if previous is not None and previous["capability"] != capability:
                return SubscriptionUpdate(instant, self.owner, identity, command.subscription, "rejected",
                                          "Subscription identity already belongs to another declaration.")
            if previous is None or not previous["active"]:
                self._records[identity] = {"capability": capability, "active": True, "since": instant}
            return SubscriptionUpdate(instant, self.owner, identity, command.subscription, "active")
        if not isinstance(command, Unsubscribe):
            raise TypeError("Subscription routing requires Subscribe or Unsubscribe.")
        if previous is None:
            return SubscriptionUpdate(instant, self.owner, identity, None, "rejected", "Unknown subscription identity.")
        previous["active"] = False
        return SubscriptionUpdate(instant, self.owner, identity, self._pool[previous["capability"]], "inactive")

    def _active_rows(self):
        return [(self._pool[row["capability"]], row["since"]) for row in self._records.values() if row["active"]]

    def _proxy(self):
        assert self.contract is not None
        return SimpleNamespace(contract=self.contract)

    def select(self, event):
        if isinstance(event, SubscriptionUpdate):
            return event if event.strategy_id == self.owner else None
        if self.contract is None:
            return event
        active = [(sub, since) for sub, since in self._active_rows() if since is None or event.time > since or (event.time == since and not isinstance(event, (Bars, Timer)))]
        if isinstance(event, Bars):
            symbols = {symbol for sub, _ in active if isinstance(sub, BarSubscription) for symbol in sub.symbols}
            bars = {symbol: bar for symbol, bar in event.bars.items() if symbol in symbols}
            return Bars(event.time, bars) if bars else None
        if isinstance(event, Timer):
            candidates = [sub for sub, _ in active if isinstance(sub, (ClockSubscription, SessionSchedule)) and sub.name == event.name]
            # Source runners scope declared schedules to owners. Disambiguate
            # same-name capabilities at different times when selecting activation.
            same_name = [sub for sub in self._pool.values() if isinstance(sub, (ClockSubscription, SessionSchedule)) and sub.name == event.name]
            if len(same_name) > 1:
                candidates = [sub for sub in candidates if self._timer_matches(sub, event)]
            return event if candidates else None
        subscribed = any(isinstance(sub, EventSubscription) and isinstance(event, sub.event_type) for sub, _ in active)
        return select_event(self._proxy(), event, subscriptions=tuple(self.active.values())) if subscribed else None

    def _timer_matches(self, sub, event: Timer) -> bool:
        for year in range(max(1, event.time.year - 1), min(9999, event.time.year + 1) + 1):
            key = (sub, year)
            if key not in self._timer_cache:
                self._timer_cache[key] = frozenset(sub.materialize(year).timers)
            if event in self._timer_cache[key]:
                return True
        return False

    def filter_context(self, context: Context) -> Context:
        if self.contract is None:
            return replace(context, subscriptions=self.active)
        context = filter_context(self._proxy(), context, subscriptions=tuple(self.active.values()))
        by_symbol = {}
        for sub, since in self._active_rows():
            if isinstance(sub, BarSubscription):
                for symbol in sub.symbols:
                    by_symbol.setdefault(symbol, []).append(since)
        history = {symbol: tuple(bar for bar in context.history.get(symbol, ())
                                 if any(since is None or bar.end > since for since in activation))
                   for symbol, activation in by_symbol.items()}
        return replace(context, history=history, subscriptions=self.active)
