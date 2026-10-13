"""Paper debit-vertical execution with leg-level journaling and no legging."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Callable

from convexity.trading import Execution, OrderJournal, OrderRecord, StockOrder, VerticalOrder
from . import ibkr
from .ibkr import PaperAccountConfig, PaperAccountError, Quote
from .ibkr_options import OptionQuoteRequest
from .ibkr_trading import PaperBroker


class VerticalBroker(PaperBroker):
    """One verified connection handles stocks and standard debit verticals.

    Explicit dollar budgets cover conservative spread-width exposure, including
    pending entries. BUY opens a debit vertical; SELL closes owned legs only.
    Only leg executions enter the journal: BAG summaries are not extra fills.
    """
    def __init__(self, config: PaperAccountConfig, journal: OrderJournal, *,
                 strategy_budgets: Mapping[str, Decimal], max_contracts: int = 1,
                 on_update=None):
        super().__init__(config, journal, on_update)
        self.strategy_budgets = {k: Decimal(str(v)) for k, v in strategy_budgets.items()}
        if not self.strategy_budgets or any(not k or not v.is_finite() or v <= 0 for k, v in self.strategy_budgets.items()):
            raise ValueError("Supply positive per-strategy dollar budgets.")
        if type(max_contracts) is not int or max_contracts <= 0:
            raise ValueError("max_contracts must be positive.")
        self.max_contracts = max_contracts
        self._submission_lock = asyncio.Lock()

    def _on_fill(self, trade: Any, fill: Any) -> None:
        try:
            e = fill.execution
            row = self._record_for(e.acctNumber, e.clientId, e.orderId, e.permId, e.orderRef)
            if row is None:
                return
            if isinstance(row.request, StockOrder):
                super()._on_fill(trade, fill)
                return
            if fill.contract.secType == "BAG":
                return
            if fill.contract.secType != "OPT":
                raise ValueError("Vertical execution must identify an option leg.")
            if e.side not in ("BOT", "SLD", "BUY", "SELL"):
                raise ValueError("Unknown option execution side.")
            side = "BUY" if e.side in ("BOT", "BUY") else "SELL"
            if side != row.request.leg_side(fill.contract.conId):
                raise ValueError("Option leg side does not match vertical intent.")
            self.journal.execution(Execution(e.execId, row.request.client_order_id, Decimal(str(e.shares)),
                Decimal(str(e.price)), side, e.time.isoformat(), contract_id=fill.contract.conId))
            self._publish(row.request.client_order_id)
        except Exception as exc:
            self._fault = exc
            self._changed.set()

    def _check_submission_state(self) -> None:
        for record in self.journal.orders():
            if record.account != self.config.account:
                continue
            if record.replace_pending or record.filled_quantity > record.executed_quantity or (record.uncertain and not record.terminal):
                raise PaperAccountError("Unresolved order or missing leg executions; reconcile before submitting.")

    async def _qualified_legs(self, request: VerticalOrder):
        from ib_async import Contract
        legs = []
        for identity in (request.long_leg, request.short_leg):
            details = await self._ib.reqContractDetailsAsync(Contract(conId=identity.contract_id, exchange="SMART"))
            if len(details) != 1:
                raise PaperAccountError("Vertical leg did not resolve uniquely.")
            contract = details[0].contract
            expected = (identity.contract_id, identity.symbol, identity.expiry.strftime("%Y%m%d"),
                        identity.strike, identity.right, str(identity.multiplier), identity.trading_class)
            actual = (contract.conId, contract.symbol, contract.lastTradeDateOrContractMonth[:8],
                      Decimal(str(contract.strike)), contract.right, contract.multiplier, contract.tradingClass)
            if contract.secType != "OPT" or contract.currency != "USD" or actual != expected:
                raise PaperAccountError("Qualified vertical leg differs from the requested identity.")
            contract.exchange = "SMART"
            legs.append(contract)
        return tuple(legs)

    async def vertical_quotes(self, request: VerticalOrder) -> tuple[Quote, Quote]:
        """Read fresh live leg snapshots on the verified execution connection."""
        self._check()
        receipts: dict[int, dict] = {}
        def ticks(tickers):
            for ticker in tickers:
                if ticker.contract.conId not in receipts:
                    continue
                for tick in ticker.ticks:
                    if tick.tickType in (1, 66):
                        receipts[ticker.contract.conId]["bid"] = tick.time
                    elif tick.tickType in (2, 67):
                        receipts[ticker.contract.conId]["ask"] = tick.time
        async with asyncio.timeout(self.config.timeout):
            legs = await self._qualified_legs(request)
            receipts = {c.conId: {} for c in legs}
            self._ib.pendingTickersEvent += ticks
            try:
                self._ib.reqMarketDataType(1)
                values = await self._ib.reqTickersAsync(*legs, regulatorySnapshot=False)
                if len(values) != 2:
                    raise PaperAccountError("Expected two vertical leg snapshots.")
                quotes = []
                for identity, contract, ticker in zip((request.long_leg, request.short_leg), legs, values):
                    quote_request = OptionQuoteRequest(identity.symbol, identity.expiry, identity.strike,
                                                       identity.right, identity.trading_class)
                    quotes.append(ibkr._stock_quote(contract, quote_request, ticker, receipts[contract.conId], True, []))
                return quotes[0], quotes[1]
            finally:
                self._ib.pendingTickersEvent -= ticks

    async def submit(self, request: StockOrder | VerticalOrder, *, submission_guard: Callable[[], None] | None = None) -> OrderRecord:
        async with self._submission_lock:
            if isinstance(request, StockOrder):
                return await super().submit(request, submission_guard=submission_guard)
            return await self._submit_vertical(request, submission_guard=submission_guard)

    async def close_at_limit(self, request: VerticalOrder) -> OrderRecord:
        """Explicitly close owned balanced legs at a supplied minimum credit.

        This operator action works during a market-data outage. It does not
        infer a price or assert quote freshness. All ownership, identity,
        unresolved-order and quantity checks still apply; it cannot open legs.
        """
        if not isinstance(request, VerticalOrder) or request.side != "SELL":
            raise ValueError("Explicit close requires a SELL vertical with a minimum credit limit.")
        async with self._submission_lock:
            return await self._submit_vertical(request, quoted=False)

    async def _submit_vertical(self, request: VerticalOrder, *, quoted: bool = True,
                               submission_guard: Callable[[], None] | None = None) -> OrderRecord:
        from ib_async import Contract, ComboLeg, LimitOrder
        self._check()
        try:
            existing = self.journal.get(request.client_order_id)
        except KeyError:
            existing = None
        if existing is not None:
            if existing.request != request or existing.account != self.config.account:
                raise ValueError("Order ID already belongs to another intent/account.")
            return existing
        self._check_submission_state()
        budget = self.strategy_budgets.get(request.strategy_id)
        if budget is None or request.quantity > self.max_contracts:
            raise PaperAccountError("Vertical strategy budget or contract-count limit is missing/exceeded.")
        if request.limit_price % Decimal(".01"):
            raise ValueError("Vertical limits must be in whole cents; broker validates contract-specific increments.")
        async with asyncio.timeout(self.config.timeout * 2):
            legs = await self._qualified_legs(request)
            quotes = await self.vertical_quotes(request) if quoted else ()
            if any(q.status != "received" for q in quotes):
                raise PaperAccountError("Both option legs require fresh live two-sided quotes.")
            actual = {p.contract.contract_id: p.quantity for p in await self.positions()}
            owned = [self.journal.attributed_position(self.config.account, request.strategy_id, leg.conId) for leg in legs]
            for leg in legs:
                total = sum((self.journal.attributed_position(self.config.account, owner, leg.conId)
                             for owner in {r.request.strategy_id for r in self.journal.orders()}), Decimal(0))
                if actual.get(leg.conId, Decimal(0)) != total:
                    raise PaperAccountError("Option positions differ from journal ownership; possible manual activity/assignment. Reconcile.")
            rows = [r for r in self.journal.orders() if r.account == self.config.account
                    and isinstance(r.request, VerticalOrder) and r.request.strategy_id == request.strategy_id]
            requested_legs = {leg.conId for leg in legs}
            if any(not r.terminal and isinstance(r.request, VerticalOrder) and requested_legs.intersection(
                    (r.request.long_leg.contract_id, r.request.short_leg.contract_id)) for r in rows):
                raise PaperAccountError("Finish or cancel working orders on these option legs before another intent.")
            if request.side == "SELL":
                if owned[0] != -owned[1] or owned[0] < request.quantity or owned[1] > -request.quantity:
                    raise PaperAccountError("Closing vertical exceeds owned long/short legs.")
            else:
                if any(owned):
                    raise PaperAccountError("Existing vertical legs must be closed before a new entry.")
                risk = Decimal(0)
                pairs = set()
                for row in rows:
                    old = row.request
                    assert isinstance(old, VerticalOrder)
                    key = (old.long_leg.contract_id, old.short_leg.contract_id)
                    if key in pairs:
                        continue
                    pairs.add(key)
                    quantity = max(abs(self.journal.attributed_position(self.config.account, request.strategy_id, c)) for c in key)
                    risk += quantity * old.width * old.long_leg.multiplier
                risk += sum((r.remaining_quantity * r.request.width * 100 for r in rows
                             if not r.terminal and isinstance(r.request, VerticalOrder) and r.request.side == "BUY"), Decimal(0))
                if risk + request.quantity * request.width * 100 > budget:
                    raise PaperAccountError("Conservative vertical width exposure exceeds strategy budget.")
                summary = await self._ib.accountSummaryAsync(self.config.account)
                funds = {(v.currency, Decimal(v.value)) for v in summary
                         if v.account == self.config.account and v.tag == "AvailableFunds" and v.currency != "BASE"}
                if len(funds) != 1:
                    raise PaperAccountError("A unique broker AvailableFunds balance is required.")
                base, available = funds.pop()
                if base != "USD":
                    rates = {Decimal(v.value) for v in summary if v.account == self.config.account
                             and v.tag.removeprefix("$LEDGER-") == "ExchangeRate" and v.currency == "USD"}
                    if len(rates) != 1:
                        raise PaperAccountError("Broker USD exchange rate is required for available-funds conversion.")
                    rate = rates.pop()
                    if not rate.is_finite() or rate <= 0:
                        raise PaperAccountError("Broker USD exchange rate must be finite and positive.")
                    available /= rate
                pending = sum((r.request.quantity * (r.request.limit_price if isinstance(r.request, VerticalOrder) else r.request.risk_price) * (100 if isinstance(r.request, VerticalOrder) else 1)
                               for r in self.journal.orders() if r.account == self.config.account and not r.terminal), Decimal(0))
                if available is None or not available.is_finite() or request.maximum_debit + pending > available:
                    raise PaperAccountError("Broker available funds cannot cover entry plus pending commitments.")
            now = datetime.now(timezone.utc)
            if any(receipt is None or not 0 <= (now - receipt).total_seconds() <= q.max_age
                   for q in quotes for receipt in (q.bid_received_at, q.ask_received_at)):
                raise PaperAccountError("Option quotes expired during preflight; no order sent.")
            self._check()
            self._check_submission_state()
            bag = Contract(secType="BAG", symbol=request.symbol, currency="USD", exchange="SMART",
                comboLegs=[ComboLeg(conId=legs[0].conId, ratio=1, action="BUY", exchange="SMART"),
                           ComboLeg(conId=legs[1].conId, ratio=1, action="SELL", exchange="SMART")])
            broker_id = self._ib.client.getReqId()
            order = LimitOrder(request.side, request.quantity, float(request.limit_price), orderId=broker_id,
                account=self.config.account, orderRef=request.client_order_id, tif="DAY", transmit=True)
            if submission_guard is not None:
                submission_guard()
            if not self.journal.intent(request, self.config.account, broker_id, self.config.client_id, 0):
                return self.journal.get(request.client_order_id)
            if not quoted:
                self.journal.diagnostic(request.client_order_id,
                    "Explicit owned-position close at caller-supplied minimum credit; market quotes not required.")
            try:
                trade = self._ib.placeOrder(bag, order)
                self._trades[request.client_order_id] = trade
            except BaseException:
                self.journal.uncertain(request.client_order_id, "Combo submission outcome unknown; no automatic retry or legging.")
                raise
            self._publish(request.client_order_id)
            return self.journal.get(request.client_order_id)
