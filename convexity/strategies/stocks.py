"""Deterministic stock rules; research examples, not profitability claims."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
import math

from convexity.runtime import Bars, Context, Event, Strategy, SubmitOrder, Timer
from convexity.runtime.types import decimal


def rebalance_orders(context: Context, targets: Mapping[str, int]) -> tuple[SubmitOrder, ...]:
    """Plan target minus owned positions and pending orders, reductions first.

    Targets are whole shares. Omitted symbols are unchanged. A symbol with a
    working order waits for its terminal outcome before changing direction.
    Reversals close first and open on a later invocation. Calling again after
    routing these commands does not duplicate pending quantities.
    """
    reductions, additions = [], []
    for symbol, target in sorted(targets.items()):
        if symbol not in context.marks or type(target) is not int:
            raise ValueError("Targets require observed symbols and whole-share quantities.")
        current = context.position(symbol)
        pending = context.pending(symbol)
        delta = target - current - pending
        if not delta or any(not o.terminal and o.request.symbol == symbol for o in context.orders):
            continue
        if current and (target == 0 or (current > 0) != (target > 0)):
            delta = -current
        command = SubmitOrder(symbol, "BUY" if delta > 0 else "SELL", abs(delta))
        if abs(current + delta) < abs(current):
            reductions.append(command)
        else:
            additions.append(command)
    # Wait for reductions on other symbols too, so their reserved capacity is
    # actually released before allocating replacement positions.
    if reductions:
        return tuple(reductions)
    if any(not o.terminal for o in context.orders):
        return ()
    return tuple(additions)


class TargetStrategy(Strategy):
    """Adapt a user function (including convexity.math models) to callbacks.

    The function receives only observed Context and returns absolute share
    targets on a named timer. Returning None skips that decision. New targets
    are pursued on subsequent bars; execution and risk remain runtime concerns.
    """
    def __init__(self, target: Callable[[Context], Mapping[str, int] | None], *, timer: str = "rebalance"):
        if not callable(target) or not timer:
            raise ValueError("Supply a target function and timer name.")
        self.target = target
        self.timer = timer
        self.targets: dict[str, int] = {}
        self._last_timer = None

    def dump_state(self):
        return {"version": 1, "timer": self.timer, "targets": self.targets,
                "last_timer": self._last_timer.isoformat() if self._last_timer else None}

    def load_state(self, state):
        if state is None:
            return
        if state.get("version") != 1 or state.get("timer") != self.timer:
            raise ValueError("TargetStrategy checkpoint version/timer differs from configuration.")
        targets = state["targets"]
        if not isinstance(targets, dict) or any(not isinstance(s, str) or type(q) is not int for s, q in targets.items()):
            raise ValueError("TargetStrategy checkpoint requires symbol-keyed whole-share targets.")
        self.targets = dict(targets)
        self._last_timer = datetime.fromisoformat(state["last_timer"]) if state["last_timer"] else None
        if self._last_timer is not None and self._last_timer.utcoffset() is None:
            raise ValueError("TargetStrategy checkpoint timer must be timezone-aware.")

    def on_event(self, event: Event, context: Context):
        if isinstance(event, Timer) and event.name == self.timer:
            if self._last_timer == event.time:
                return None
            self._last_timer = event.time
            targets = self.target(context)
            if targets is None:
                self.targets = {}
                return None
            self.targets = dict(targets)
        elif not isinstance(event, Bars):
            return None
        return rebalance_orders(context, self.targets)


@dataclass(frozen=True)
class StockRule:
    universe: tuple[str, ...]
    lookback: int = 3
    leg_weight: Decimal = Decimal(".30")
    timer: str = "rebalance"
    shortable: tuple[str, ...] | None = None
    exit_timer: str | None = None
    exit_retry_seconds: float = 20

    def __post_init__(self):
        object.__setattr__(self, "universe", tuple(self.universe))
        object.__setattr__(self, "leg_weight", decimal(self.leg_weight))
        if len(self.universe) < 2 or len(set(self.universe)) != len(self.universe) or not all(self.universe):
            raise ValueError("Supply at least two distinct stock symbols.")
        if type(self.lookback) is not int or self.lookback < 2 or not 0 < self.leg_weight <= Decimal(".5") or not self.timer:
            raise ValueError("lookback >= 2, 0 < leg_weight <= .5, and a timer are required.")
        if self.shortable is not None:
            object.__setattr__(self, "shortable", tuple(self.shortable))
            if not set(self.shortable) <= set(self.universe):
                raise ValueError("shortable must be a subset of the universe.")
        if (type(self.exit_retry_seconds) not in (int, float) or not math.isfinite(self.exit_retry_seconds)
                or not 0 < self.exit_retry_seconds <= 300):
            raise ValueError("exit_retry_seconds must be finite and in (0, 300].")
        if self.exit_timer is not None and (not self.exit_timer or self.exit_timer == self.timer):
            raise ValueError("An explicit exit timer must differ from the rebalance timer.")


class MonthlyLongShort(TargetStrategy):
    """Long highest close-to-close return, short lowest; tie means skip.

    An unavailable short skips the entire new rebalance. Symbols break interior
    rank ties deterministically. Sizing rounds toward zero and uses current
    account equity; enforce a separate strategy budget with RiskLimits.
    """
    def __init__(self, config: StockRule):
        self.config = config
        self.last_decision = "waiting for schedule"
        super().__init__(self._rank, timer=config.timer)

    def on_event(self, event: Event, context: Context):
        if isinstance(event, Timer) and event.name == self.config.exit_timer:
            self.targets = dict.fromkeys(self.config.universe, 0)
            self.last_decision = "explicit exit timer: target flat"
            working = tuple(o for o in context.orders if not o.terminal)
            if working:
                from convexity.runtime import CancelOrder
                cancellations = []
                for order in working:
                    current = context.position(order.request.symbol)
                    remaining = order.remaining * (1 if order.request.side == "BUY" else -1)
                    reducing = current * remaining < 0 and abs(remaining) <= abs(current)
                    age = (context.time - order.accepted_at).total_seconds()
                    if not order.cancel_pending and (not reducing or age >= self.config.exit_retry_seconds):
                        cancellations.append(CancelOrder(order.order_id))
                return tuple(cancellations)
            return rebalance_orders(context, self.targets)
        return super().on_event(event, context)

    def _rank(self, context: Context) -> Mapping[str, int] | None:
        rule = self.config
        if any(len(context.history.get(s, ())) < rule.lookback for s in rule.universe):
            self.last_decision = "skipped: insufficient observed history"
            return None
        scores = sorted((context.history[s][-1].close / context.history[s][-rule.lookback].close - 1, s)
                        for s in rule.universe)
        if scores[-1][0] == scores[0][0] or context.equity <= 0:
            self.last_decision = "skipped: tied ranks or nonpositive equity"
            return None
        winner, loser = scores[-1][1], scores[0][1]
        if rule.shortable is not None and loser not in rule.shortable:
            self.last_decision = f"skipped: short unavailable for {loser}"
            return None
        targets = dict.fromkeys(rule.universe, 0)
        def quantity(symbol):
            equity = (context.account.equity_in(context.instrument_currencies.get(symbol, "USD"))
                      if context.account is not None else context.equity)
            return int(equity * rule.leg_weight / context.marks[symbol])
        targets[winner] = quantity(winner)
        targets[loser] = -quantity(loser)
        self.last_decision = f"long {winner}; short {loser}"
        return targets
