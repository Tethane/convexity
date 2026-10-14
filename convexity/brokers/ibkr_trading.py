"""Persistent IBKR paper session for explicit stock limit orders.

This is the execution path a future Runtime will call. No scheduler or strategy
logic lives here, and no order is retried automatically.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from collections.abc import Callable
from dataclasses import asdict, replace
import copy
import json
import hashlib
from typing import Any

from convexity.trading import Execution, OrderJournal, OrderRecord, StockOrder
from .ibkr import (
    PaperAccountConfig,
    PaperAccountError,
    PaperAccountSnapshot,
    Position,
    Quote,
    StockQuoteRequest,
    _connect_paper,
    _contract,
    _new_client,
)

__all__ = ["PaperBroker"]


class PaperBroker:
    """Async session; only orders owned by this journal/account/client are mutable.

    Close disconnects without implicit cancellation/liquidation. Use cancel()
    explicitly, and keep the journal open until the session is closed.
    """

    def __init__(
        self,
        config: PaperAccountConfig,
        journal: OrderJournal,
        on_update: Callable[[OrderRecord], None] | None = None,
    ):
        self.config = config
        self.journal = journal
        self.on_update = on_update
        self._ib: Any = None
        self._changed = asyncio.Event()
        self._fault: Exception | None = None
        self._trades: dict[str, Any] = {}
        self._handlers: list[tuple[Any, Any]] = []
        self._observed_open_orders: set[str] = set()

    @property
    def capabilities(self):
        from convexity.runtime.capabilities import PAPER_CAPABILITIES

        return PAPER_CAPABILITIES

    async def __aenter__(self):
        from ib_async import ExecutionFilter

        if self._ib is not None:
            raise RuntimeError("Use a new PaperBroker for each session.")
        self._ib = _new_client()
        try:
            async with asyncio.timeout(self.config.timeout):
                await _connect_paper(self._ib, self.config)
                # _connect_paper disables startup order fetches. Explicitly fetch
                # only after verifying the expected account. readonly is not a
                # server-side trading lock; all mutations below are explicit.
                for event, handler in (
                    (self._ib.openOrderEvent, self._on_open_trade),
                    (self._ib.orderStatusEvent, self._on_trade),
                    (self._ib.execDetailsEvent, self._on_fill),
                    (self._ib.commissionReportEvent, self._on_commission),
                    (self._ib.errorEvent, self._on_error),
                    (self._ib.disconnectedEvent, self._on_disconnect),
                ):
                    event += handler
                    self._handlers.append((event, handler))
                for record in self.journal.orders():
                    if record.account == self.config.account and not record.terminal:
                        self.journal.uncertain(
                            record.request.client_order_id,
                            "Session reopened; broker state must be observed before assuming the order is working.",
                        )
                for trade in await self._ib.reqOpenOrdersAsync():
                    self._on_open_trade(trade)
                for trade in await self._ib.reqCompletedOrdersAsync(apiOnly=True):
                    self._on_completed_trade(trade)
                for fill in await self._ib.reqExecutionsAsync(
                    ExecutionFilter(acctCode=self.config.account)
                ):
                    self._on_fill(None, fill)
                self._check()
                for record in self.journal.orders():
                    if (
                        record.account == self.config.account
                        and record.filled_quantity > record.executed_quantity
                    ):
                        self.journal.uncertain(
                            record.request.client_order_id,
                            "Broker reports fills missing from the execution journal; recover execution details before new submissions.",
                        )
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_):
        for event, handler in self._handlers:
            event -= handler
        self._handlers.clear()
        if self._ib is not None:
            self._ib.disconnect()

    def _check(self) -> None:
        if self._fault is not None:
            raise PaperAccountError(
                "Order recording failed; stop submitting and inspect the journal."
            ) from self._fault
        if self._ib is None or not self._ib.isConnected():
            raise PaperAccountError("Paper broker is not connected.")

    def _check_submission_state(self) -> None:
        for record in self.journal.orders():
            if record.account != self.config.account:
                continue
            if (
                not record.terminal
                and record.group_id
                and (self.journal.group(record.group_id) or {}).get("transport_state")
                != "sent"
            ):
                raise PaperAccountError(
                    "A staged order group has unresolved transmission; reconcile every member before submitting."
                )
            if record.filled_quantity > record.executed_quantity:
                raise PaperAccountError(
                    "Reported fills are missing execution details; reconcile the journal before submitting a new intent."
                )
            if record.replace_pending or (record.uncertain and not record.terminal):
                raise PaperAccountError(
                    "An order has an unresolved outcome; reconcile before submitting a new intent."
                )

    def _record_for(
        self, account: str, client_id: int, broker_id: int, perm_id: int, ref: str
    ) -> OrderRecord | None:
        if account != self.config.account:
            return None
        for row in self.journal.orders():
            if row.account != account:
                continue
            if perm_id and row.permanent_id == perm_id:
                return row
            if (
                row.request.client_order_id == ref
                and row.broker_client_id == client_id
                and row.broker_order_id == broker_id
            ):
                return row
        return None

    def _publish(self, local_id: str) -> None:
        self._changed.set()
        if self.on_update is not None:
            self.on_update(self.journal.get(local_id))

    def _on_completed_trade(self, trade: Any) -> None:
        # ib_async completedOrder constructs an OrderStatus with no cumulative
        # quantity. Filled confirms the whole intent, but executions are still
        # required for attribution; startup/submission gates check that gap.
        self._on_trade(
            trade, completed=True, observed_terms=hasattr(trade.order, "orderType")
        )

    def _on_open_trade(self, trade: Any) -> None:
        self._on_trade(trade, observed_terms=True)

    def _native_group_key(self, group_id):
        return (
            "cx-"
            + hashlib.sha256(
                json.dumps(
                    (self.config.account, self.config.client_id, group_id)
                ).encode()
            ).hexdigest()[:40]
        )

    def _reconcile_group(self, group_id):
        group = self.journal.group(group_id)
        if group is None or group["transport_state"] == "sent":
            return
        rows = [row for row in self.journal.orders() if row.group_id == group_id]
        # A host-assigned permanent ID plus an observed working/terminal order
        # for EVERY member proves server acceptance after an ambiguous send.
        # A local staged Trade or mere orderStatus echo cannot release the gate.
        for row in rows:
            trade = self._trades.get(row.request.client_order_id)
            if (
                row.request.client_order_id not in self._observed_open_orders
                or trade is None
                or not row.permanent_id
                or not row.acknowledged
                or row.replace_pending
                or not isinstance(row.request, StockOrder)
                or not self._terms_match(trade.order, row.request)
            ):
                return
            parent_id = (
                self.journal.get(row.parent_order_id).broker_order_id
                if row.parent_order_id
                else 0
            )
            if getattr(trade.order, "parentId", 0) != parent_id:
                return
            if (row.group_kind == "oco" or row.parent_order_id) and (
                getattr(trade.order, "ocaGroup", "") != self._native_group_key(group_id)
                or getattr(trade.order, "ocaType", 0) != 1
            ):
                return
        self.journal.group_transport(group_id, "sent")
        for row in rows:
            status = self._trades[row.request.client_order_id].orderStatus.status
            self.journal.status(
                row.request.client_order_id,
                status,
                row.filled_quantity,
                row.permanent_id,
            )
            self._publish(row.request.client_order_id)

    @staticmethod
    def _terms_match(order, request: StockOrder) -> bool:
        return (
            getattr(order, "orderType", None)
            == {"limit": "LMT", "stop": "STP", "stop_limit": "STP LMT"}[
                request.order_type
            ]
            and getattr(order, "action", None) == request.side
            and Decimal(str(getattr(order, "totalQuantity", -1))) == request.quantity
            and getattr(order, "tif", None) == request.time_in_force
            and (
                request.limit_price is None
                or Decimal(str(getattr(order, "lmtPrice", -1))) == request.limit_price
            )
            and (
                request.stop_price is None
                or Decimal(str(getattr(order, "auxPrice", -1))) == request.stop_price
            )
        )

    def _on_trade(
        self, trade: Any, *, completed: bool = False, observed_terms: bool = False
    ) -> None:
        try:
            o, s = trade.order, trade.orderStatus
            row = self._record_for(
                o.account, o.clientId, o.orderId, o.permId or s.permId, o.orderRef
            )
            if row is None:
                return
            local_id = row.request.client_order_id
            if (
                observed_terms
                and getattr(getattr(trade, "contract", None), "conId", row.contract_id)
                != row.contract_id
            ):
                self.journal.uncertain(
                    local_id,
                    "Observed broker contract differs from the durable order identity.",
                )
                self._publish(local_id)
                return
            self._trades[local_id] = trade
            # ib_async emits synthetic ValidationError/Cancelled statuses while
            # processing error messages. The matching error handler decides their
            # meaning; these are not independent broker order-status confirmations.
            log = getattr(trade, "log", [])
            if log and getattr(log[-1], "errorCode", 0):
                return
            if (
                observed_terms
                and row.pending_request is not None
                and self._terms_match(o, row.pending_request)
            ):
                self.journal.replacement_observed(local_id, row.pending_request)
                row = self.journal.get(local_id)
            if (
                observed_terms
                and not row.replace_pending
                and isinstance(row.request, StockOrder)
                and self._terms_match(o, row.request)
            ):
                self.journal.terms_confirmed(local_id)
            filled = Decimal(str(s.filled))
            if completed and s.status == "Filled" and filled == 0:
                filled = Decimal(row.request.quantity)
            self.journal.status(local_id, s.status, filled, o.permId or s.permId)
            if (
                observed_terms
                and isinstance(row.request, StockOrder)
                and not row.replace_pending
                and not self._terms_match(o, row.request)
            ):
                self.journal.uncertain(
                    local_id,
                    "Observed broker terms differ from accepted local terms; reconcile before new commands.",
                )
            if (
                row.group_id
                and (self.journal.group(row.group_id) or {}).get("transport_state")
                != "sent"
            ):
                self.journal.uncertain(
                    local_id,
                    "Group transmission has not been durably completed; do not infer group acceptance from one member.",
                )
            if observed_terms:
                self._observed_open_orders.add(local_id)
                if row.group_id:
                    self._reconcile_group(row.group_id)
            self._publish(local_id)
        except Exception as exc:
            self._fault = exc
            self._changed.set()

    def _on_fill(self, trade: Any, fill: Any) -> None:
        try:
            e = fill.execution
            row = self._record_for(
                e.acctNumber, e.clientId, e.orderId, e.permId, e.orderRef
            )
            if row is None or not isinstance(row.request, StockOrder):
                return
            if e.side not in ("BOT", "SLD", "BUY", "SELL"):
                raise ValueError("Unknown execution side.")
            side = "BUY" if e.side in ("BOT", "BUY") else "SELL"
            if fill.contract.conId != row.contract_id or side != row.request.side:
                raise ValueError("Execution does not match order instrument/side.")
            self.journal.execution(
                Execution(
                    e.execId,
                    row.request.client_order_id,
                    Decimal(str(e.shares)),
                    Decimal(str(e.price)),
                    side,
                    e.time.isoformat(),
                )
            )
            self._publish(row.request.client_order_id)
        except Exception as exc:
            self._fault = exc
            self._changed.set()

    def _on_commission(self, trade: Any, fill: Any, report: Any) -> None:
        try:
            if fill.execution.acctNumber == self.config.account:
                amount = Decimal(str(report.commission))
                if amount.is_finite() and abs(amount) < Decimal("1e100"):
                    self.journal.commission(
                        self.config.account, report.execId, amount, report.currency
                    )
                    e = fill.execution
                    row = self._record_for(
                        e.acctNumber, e.clientId, e.orderId, e.permId, e.orderRef
                    )
                    if row is not None:
                        self._publish(row.request.client_order_id)
        except Exception as exc:
            self._fault = exc
            self._changed.set()

    def _on_error(self, req_id: int, code: int, message: str, contract: Any) -> None:
        try:
            for row in self.journal.orders():
                if (row.account, row.broker_client_id, row.broker_order_id) != (
                    self.config.account,
                    self.config.client_id,
                    req_id,
                ):
                    continue
                local_id = row.request.client_order_id
                self.journal.diagnostic(
                    local_id,
                    f"IBKR {code}: {message.replace(self.config.account, '[ACCOUNT]')}",
                )
                if code in (201, 10329, 104, 105, 329) and row.replace_pending:
                    self.journal.replacement_rejected(local_id)
                elif code in (201, 10329):
                    self.journal.status(local_id, "Rejected", row.filled_quantity)
                elif code == 202:
                    self.journal.status(local_id, "Cancelled", row.filled_quantity)
                elif code in (10147, 10148):
                    self.journal.cancel_rejected(local_id)
                self._publish(local_id)
        except Exception as exc:
            self._fault = exc
            self._changed.set()

    def _on_disconnect(self) -> None:
        try:
            for row in self.journal.orders():
                if row.account == self.config.account and not row.terminal:
                    self.journal.uncertain(
                        row.request.client_order_id,
                        "Connection lost; outcome unresolved. No automatic resubmission.",
                    )
            self._changed.set()
        except Exception as exc:
            self._fault = exc
            self._changed.set()

    async def submit(
        self, request: StockOrder, *, submission_guard: Callable[[], None] | None = None
    ) -> OrderRecord:
        """Persist intent before sending. A repeated local ID never sends twice."""
        from ib_async import Stock

        self._check()
        try:
            existing = self.journal.get(request.client_order_id)
        except KeyError:
            existing = None
        if existing is not None:
            if existing.request != request or existing.account != self.config.account:
                raise ValueError(
                    "client_order_id already belongs to a different intent/account."
                )
            return existing
        self._check_submission_state()
        async with asyncio.timeout(self.config.timeout):
            stock = Stock(
                request.symbol,
                request.exchange,
                request.currency,
                primaryExchange=request.primary_exchange,
            )
            details = await self._ib.reqContractDetailsAsync(stock)
        if len(details) != 1 or details[0].contract.secType != "STK":
            raise PaperAccountError("Stock must resolve uniquely before submitting.")
        detail = details[0]
        stock = detail.contract
        stock.exchange = request.exchange
        if stock.conId <= 0 or request.exchange not in detail.validExchanges.split(","):
            raise PaperAccountError(
                "Requested session route is not supported for this stock."
            )
        tick = Decimal(str(detail.minTick))
        self._validate_tick(request, tick)
        broker_id = self._ib.client.getReqId()
        order = self._native_order(request, broker_id)
        # Another coroutine may have submitted while contract resolution awaited.
        self._check()
        self._check_submission_state()
        if submission_guard is not None:
            submission_guard()
        if stock.symbol != request.symbol or stock.currency != request.currency:
            raise PaperAccountError(
                "Resolved stock identity differs from the requested symbol/currency."
            )
        if not self.journal.intent(
            request, self.config.account, broker_id, self.config.client_id, stock.conId
        ):
            return self.journal.get(request.client_order_id)
        try:
            trade = self._ib.placeOrder(stock, order)
            self._trades[request.client_order_id] = trade
        except BaseException:
            self.journal.uncertain(
                request.client_order_id,
                "Submission raised; broker acceptance is unknown. Do not retry with a new ID.",
            )
            raise
        self._publish(request.client_order_id)
        return self.journal.get(request.client_order_id)

    @staticmethod
    def _validate_tick(request, tick):
        if (
            not tick.is_finite()
            or tick <= 0
            or any(
                price is not None and price % tick
                for price in (request.limit_price, request.stop_price)
            )
        ):
            raise ValueError(
                f"Order prices must follow the broker's minimum tick ({tick})."
            )

    def _native_order(self, request, broker_id):
        from ib_async import Order

        kwargs: dict[str, Any] = dict(
            orderId=broker_id,
            account=self.config.account,
            orderRef=request.client_order_id,
            action=request.side,
            totalQuantity=request.quantity,
            tif=request.time_in_force,
            outsideRth=request.session == "extended",
            transmit=True,
            orderType={"limit": "LMT", "stop": "STP", "stop_limit": "STP LMT"}[
                request.order_type
            ],
        )
        if request.limit_price is not None:
            kwargs["lmtPrice"] = float(request.limit_price)
        if request.stop_price is not None:
            kwargs["auxPrice"] = float(request.stop_price)
        return Order(**kwargs)

    async def replace(
        self,
        local_id: str,
        request: StockOrder,
        *,
        command_id: str,
        submission_guard: Callable[[], None] | None = None,
    ) -> OrderRecord:
        """Modify the same broker order ID; only openOrder terms confirm acceptance."""
        self._check()
        row = self.journal.get(local_id)
        if (row.account, row.broker_client_id) != (
            self.config.account,
            self.config.client_id,
        ):
            raise PaperAccountError("Cannot replace another account/client's order.")
        if not isinstance(row.request, StockOrder):
            raise ValueError("Replacement currently supports stock orders only.")
        if (
            replace(
                request,
                quantity=row.request.quantity,
                limit_price=row.request.limit_price,
                stop_price=row.request.stop_price,
            )
            != row.request
        ):
            raise ValueError(
                "Replacement cannot change order identity, side, type, session or time in force."
            )
        if row.group_kind == "bracket" and request.quantity != row.request.quantity:
            raise ValueError(
                "Bracket member quantity replacement would leave stale sibling exits; cancel the group instead."
            )
        trade = self._trades.get(local_id)
        if (
            trade is None
            or trade.order.account != self.config.account
            or trade.order.orderId != row.broker_order_id
        ):
            raise PaperAccountError(
                "Order must be correlated in this session before replacement."
            )
        # Qualification validates price increments while leaving the observed
        # Trade untouched; ib_async reuses that object for status echoes.
        async with asyncio.timeout(self.config.timeout):
            details = await self._ib.reqContractDetailsAsync(trade.contract)
        if len(details) != 1 or details[0].contract.conId != row.contract_id:
            raise PaperAccountError("Replacement contract identity changed.")
        self._validate_tick(request, Decimal(str(details[0].minTick)))
        if submission_guard:
            submission_guard()
        if not self.journal.replace_requested(local_id, request, command_id):
            return self.journal.get(local_id)
        native = copy.copy(trade.order)
        native.totalQuantity = request.quantity
        if request.limit_price is not None:
            native.lmtPrice = float(request.limit_price)
        if request.stop_price is not None:
            native.auxPrice = float(request.stop_price)
        try:
            self._ib.placeOrder(trade.contract, native)
        except BaseException:
            self.journal.uncertain(
                local_id,
                "Replacement transport failed; old or new terms may be active. No automatic retry.",
            )
            raise
        self._publish(local_id)
        return self.journal.get(local_id)

    async def submit_group(
        self,
        group_id: str,
        kind: str,
        requests: tuple[StockOrder, ...],
        *,
        submission_guard: Callable[[], None] | None = None,
    ) -> tuple[OrderRecord, ...]:
        """Persist every group member before native parent/OCA staged transmission."""
        from ib_async import Stock

        self._check()
        if (
            kind not in ("bracket", "oco")
            or len(requests) < 2
            or (kind == "bracket" and len(requests) != 3)
        ):
            raise ValueError(
                "Use a three-member bracket or an OCO with at least two members."
            )
        if (
            len({r.client_order_id for r in requests}) != len(requests)
            or len({r.strategy_id for r in requests}) != 1
        ):
            raise ValueError("Groups require unique member IDs and one strategy owner.")
        if kind == "bracket" and any(
            (r.symbol, r.quantity, r.currency, r.side)
            != (
                requests[0].symbol,
                requests[0].quantity,
                requests[0].currency,
                "SELL" if requests[0].side == "BUY" else "BUY",
            )
            for r in requests[1:]
        ):
            raise ValueError(
                "Bracket exits must oppose the full parent quantity on the same stock."
            )
        existing = self.journal.group(group_id)
        if existing:
            payload = json.dumps(
                [asdict(request) for request in requests], default=str, sort_keys=True
            )
            if (existing["account"], existing["kind"], existing["intent"]) != (
                self.config.account,
                kind,
                payload,
            ):
                raise ValueError(
                    "Group ID already belongs to different intent/account."
                )
            return tuple(self.journal.get(r.client_order_id) for r in requests)
        self._check_submission_state()
        if any(
            row.request.client_order_id in {r.client_order_id for r in requests}
            for row in self.journal.orders()
        ):
            raise ValueError("A group member ID already belongs to another order.")
        contracts = []
        for request in requests:
            async with asyncio.timeout(self.config.timeout):
                details = await self._ib.reqContractDetailsAsync(
                    Stock(
                        request.symbol,
                        request.exchange,
                        request.currency,
                        primaryExchange=request.primary_exchange,
                    )
                )
            if (
                len(details) != 1
                or details[0].contract.secType != "STK"
                or details[0].contract.symbol != request.symbol
                or details[0].contract.currency != request.currency
            ):
                raise PaperAccountError(
                    "Group member stock must resolve uniquely to the requested identity."
                )
            self._validate_tick(request, Decimal(str(details[0].minTick)))
            if request.exchange not in details[0].validExchanges.split(","):
                raise PaperAccountError("Group member route is unavailable.")
            stock = details[0].contract
            stock.exchange = request.exchange
            contracts.append(stock)
        self._check()
        self._check_submission_state()
        if submission_guard:
            submission_guard()
        ids = tuple(self._ib.client.getReqId() for _ in requests)
        if not self.journal.group_intent(
            group_id,
            kind,
            self.config.account,
            requests,
            ids,
            self.config.client_id,
            tuple(c.conId for c in contracts),
        ):
            return tuple(self.journal.get(r.client_order_id) for r in requests)
        try:
            for index, (request, stock, broker_id) in enumerate(
                zip(requests, contracts, ids, strict=True)
            ):
                order = self._native_order(request, broker_id)
                order.transmit = index == len(requests) - 1
                if kind == "bracket" and index:
                    order.parentId = ids[0]
                if kind == "oco" or index:
                    order.ocaGroup = self._native_group_key(group_id)
                    order.ocaType = 1
                self._trades[request.client_order_id] = self._ib.placeOrder(
                    stock, order
                )
            self.journal.group_transport(group_id, "sent")
        except BaseException:
            self.journal.group_transport(group_id, "uncertain")
            for request in requests:
                self.journal.uncertain(
                    request.client_order_id,
                    "Group transmission interrupted; staged members may exist. Reconcile all members; never resend.",
                )
            raise
        for request in requests:
            self._publish(request.client_order_id)
        return tuple(self.journal.get(r.client_order_id) for r in requests)

    async def observe(
        self, local_id: str, *, seconds: float = 15, until: str = "terminal"
    ) -> OrderRecord:
        """Wait for acknowledgment or terminal state; timeout does not cancel."""
        if until not in ("acknowledged", "terminal") or not 0 < seconds <= 300:
            raise ValueError(
                "until must be acknowledged/terminal; seconds must be in (0, 300]."
            )
        try:
            async with asyncio.timeout(seconds):
                while True:
                    self._changed.clear()
                    self._check()
                    row = self.journal.get(local_id)
                    if row.terminal or (until == "acknowledged" and row.acknowledged):
                        return row
                    await self._changed.wait()
        except TimeoutError:
            row = self.journal.get(local_id)
            if not row.acknowledged:
                self.journal.uncertain(
                    local_id,
                    "Observation timed out without acknowledgment; submission remains unresolved.",
                )
            else:
                self.journal.diagnostic(
                    local_id,
                    "Observation ended; order may still be working or cancellation pending.",
                )
            return self.journal.get(local_id)

    async def cancel(self, local_id: str) -> OrderRecord:
        """Request cancellation of one owned order; observe broker confirmation."""
        self._check()
        row = self.journal.get(local_id)
        if (row.account, row.broker_client_id) != (
            self.config.account,
            self.config.client_id,
        ):
            raise PaperAccountError(
                "Cannot cancel an order owned by another account/client."
            )
        if row.terminal:
            return row
        trade = self._trades.get(local_id)
        if (
            trade is None
            or trade.order.orderRef != local_id
            or trade.order.account != self.config.account
        ):
            raise PaperAccountError(
                "Order is not correlated in this session; reconcile before cancelling."
            )
        self.journal.cancel_requested(local_id)
        # Use the transport call: ib_async.cancelOrder can synthesize a local
        # Cancelled status for Inactive orders. Only broker confirmation counts.
        try:
            self._ib.client.cancelOrder(row.broker_order_id, "")
        except BaseException:
            self.journal.uncertain(
                local_id, "Cancellation transport failed; cancellation is unresolved."
            )
            raise
        self._publish(local_id)
        return self.journal.get(local_id)

    async def snapshot(self) -> PaperAccountSnapshot:
        """Read authoritative account balances and positions on this verified session."""
        from .ibkr import AccountValue, OpenOrder, PaperAccountSnapshot, _price

        self._check()
        started = datetime.now(timezone.utc)
        async with asyncio.timeout(self.config.timeout):
            summary = await self._ib.accountSummaryAsync(self.config.account)
            values = tuple(
                AccountValue(v.tag, v.value, v.currency)
                for v in summary
                if v.account == self.config.account
            )
            if not values:
                raise PaperAccountError("Broker returned no account summary.")
            positions = await self.positions()
            trades = await self._ib.reqAllOpenOrdersAsync()
            orders = tuple(
                OpenOrder(
                    _contract(t.contract),
                    t.order.orderId,
                    t.order.permId,
                    t.order.clientId,
                    t.order.action,
                    Decimal(str(t.order.totalQuantity)),
                    t.order.orderType,
                    t.order.tif,
                    _price(t.order.lmtPrice),
                    _price(t.order.auxPrice),
                    t.orderStatus.status,
                )
                for t in trades
                if t.order.account == self.config.account
            )
            self._check()
            return PaperAccountSnapshot(
                self.config.account,
                started,
                datetime.now(timezone.utc),
                values,
                positions,
                orders,
            )

    async def quote(self, request: StockQuoteRequest) -> Quote:
        """Read a stock snapshot; unavailable data preserves qualified identity.

        Market-data denial or timeout returns an incomplete diagnostic quote.
        Connection and contract-resolution failures still raise.
        """
        from ib_async import Stock, RequestError
        from .ibkr import _stock_quote, _MARKET_DATA_TYPES

        if not isinstance(request, StockQuoteRequest):
            raise TypeError("Stock quote requests are required.")
        self._check()
        receipts = {}
        resolved = None
        observed = None
        complete = False
        diagnostics = []

        def ticks(tickers):
            nonlocal observed
            for ticker in tickers:
                if resolved is not None and ticker.contract.conId == resolved.conId:
                    observed = ticker
                    for tick in ticker.ticks:
                        if tick.tickType in (1, 66):
                            receipts["bid"] = tick.time
                        elif tick.tickType in (2, 67):
                            receipts["ask"] = tick.time

        self._ib.pendingTickersEvent += ticks
        try:
            try:
                async with asyncio.timeout(self.config.timeout):
                    details = await self._ib.reqContractDetailsAsync(
                        Stock(
                            request.symbol,
                            request.exchange,
                            request.currency,
                            primaryExchange=request.primary_exchange,
                        )
                    )
                    if (
                        len(details) != 1
                        or details[0].contract.secType != "STK"
                        or details[0].contract.conId <= 0
                    ):
                        raise PaperAccountError(
                            "Stock must resolve uniquely before quoting."
                        )
                    resolved = details[0].contract
                    if (
                        resolved.symbol != request.symbol
                        or resolved.currency != request.currency
                    ):
                        raise PaperAccountError(
                            "Resolved stock identity differs from the quote request."
                        )
                    resolved.exchange = request.exchange
                    self._ib.reqMarketDataType(
                        _MARKET_DATA_TYPES[request.market_data_type]
                    )
                    values = await self._ib.reqTickersAsync(
                        resolved, regulatorySnapshot=False
                    )
                    if len(values) != 1:
                        raise PaperAccountError("Expected one quote snapshot.")
                    observed = values[0]
                    complete = True
            except (TimeoutError, RequestError) as error:
                if resolved is None:
                    raise PaperAccountError(
                        "Stock quote contract resolution failed; check connection and identity."
                    ) from error
                if isinstance(error, RequestError):
                    diagnostics.append(
                        f"IBKR {error.code}: {error.message.replace(self.config.account, '[ACCOUNT]')}"
                    )
                else:
                    diagnostics.append(
                        "Quote snapshot timed out; partial data is not a completed quote."
                    )
            self._check()
            assert resolved is not None
            return _stock_quote(
                resolved, request, observed, receipts, complete, diagnostics
            )
        finally:
            self._ib.pendingTickersEvent -= ticks

    async def positions(self) -> tuple[Position, ...]:
        """Current broker positions; does not replace strategy-attributed fills."""
        self._check()
        async with asyncio.timeout(self.config.timeout):
            values = await self._ib.reqPositionsAsync()
        return tuple(
            Position(
                _contract(v.contract), Decimal(str(v.position)), Decimal(str(v.avgCost))
            )
            for v in values
            if v.account == self.config.account
        )
