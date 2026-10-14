"""Small SQLite order journal. Decimal quantities/prices are stored as text."""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal

from .options import OptionIdentity, VerticalOrder


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StockOrder:
    client_order_id: str
    strategy_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    limit_price: Decimal | None
    session: Literal["regular", "extended", "overnight"] = "regular"
    currency: str = "USD"
    primary_exchange: str = ""
    time_in_force: Literal["DAY", "GTC"] = "DAY"
    order_type: Literal["limit", "stop", "stop_limit"] = "limit"
    stop_price: Decimal | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_:-]{1,48}", self.client_order_id):
            raise ValueError(
                "client_order_id must contain 1–48 letters, digits, underscores, colons or hyphens."
            )
        if not self.strategy_id.strip() or not self.symbol.strip():
            raise ValueError("strategy_id and symbol are required.")
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL.")
        if type(self.quantity) is not int or self.quantity <= 0:
            raise ValueError("quantity must be a positive whole number of shares.")
        if self.order_type not in ("limit", "stop", "stop_limit"):
            raise ValueError("Unsupported stock order type.")
        for name, required in (
            ("limit_price", self.order_type != "stop"),
            ("stop_price", self.order_type != "limit"),
        ):
            value = getattr(self, name)
            if required and (
                not isinstance(value, Decimal) or not value.is_finite() or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive Decimal.")
            if not required and value is not None:
                raise ValueError(f"{name} is not used for {self.order_type} orders.")
        if self.order_type != "limit" and self.session != "regular":
            raise ValueError("Native stops currently require regular sessions.")
        if self.session not in ("regular", "extended", "overnight"):
            raise ValueError("session must be regular, extended or overnight.")
        if self.currency != "USD":
            raise ValueError("This first stock-order implementation supports USD only.")
        if self.time_in_force not in ("DAY", "GTC"):
            raise ValueError("time_in_force must be DAY or GTC.")
        if self.session == "overnight" and self.time_in_force != "DAY":
            raise ValueError("Overnight stock orders require DAY time in force.")

    @property
    def exchange(self) -> str:
        return "OVERNIGHT" if self.session == "overnight" else "SMART"

    @property
    def risk_price(self) -> Decimal:
        """Reservation estimate; a triggered market stop has no execution cap."""
        return max(self.limit_price or Decimal(0), self.stop_price or Decimal(0))


@dataclass(frozen=True)
class Execution:
    execution_id: str
    client_order_id: str
    quantity: Decimal
    price: Decimal
    side: str
    occurred_at: str
    commission: Decimal | None = None
    commission_currency: str | None = None
    contract_id: int = 0


@dataclass(frozen=True)
class OrderRecord:
    request: StockOrder | VerticalOrder
    account: str
    broker_order_id: int
    broker_client_id: int
    contract_id: int
    permanent_id: int
    status: str
    acknowledged: bool
    uncertain: bool
    cancel_pending: bool
    filled_quantity: Decimal
    executions: tuple[Execution, ...]
    diagnostics: tuple[str, ...]
    created_at: datetime | None = None
    revision: int = 0
    replace_pending: bool = False
    pending_request: StockOrder | None = None
    group_id: str | None = None
    parent_order_id: str | None = None
    group_kind: str | None = None
    needs_terms: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in ("Filled", "Cancelled", "ApiCancelled", "Rejected")

    @property
    def executed_quantity(self) -> Decimal:
        if isinstance(self.request, VerticalOrder):
            return min(
                sum(
                    (
                        e.quantity
                        for e in self.executions
                        if e.contract_id == leg.contract_id
                    ),
                    Decimal(0),
                )
                for leg in (self.request.long_leg, self.request.short_leg)
            )
        return sum((e.quantity for e in self.executions), Decimal(0))

    @property
    def remaining_quantity(self) -> Decimal:
        return max(Decimal(0), Decimal(self.request.quantity) - self.filled_quantity)


class OrderJournal:
    """Own one local journal. Context manager closes it; each mutation commits.

    Reusing an ID never resends an order, including after uncertain submission.
    This is an execution journal, not an account cash ledger or full reconciliation.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS orders (
                local_id TEXT PRIMARY KEY, request TEXT NOT NULL, account TEXT NOT NULL,
                broker_id INTEGER NOT NULL, client_id INTEGER NOT NULL, contract_id INTEGER NOT NULL,
                perm_id INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'PendingSubmit',
                acknowledged INTEGER NOT NULL DEFAULT 0, uncertain INTEGER NOT NULL DEFAULT 1,
                cancel_pending INTEGER NOT NULL DEFAULT 0, reported_filled TEXT NOT NULL DEFAULT '0'
            );
            CREATE TABLE IF NOT EXISTS executions (
                account TEXT NOT NULL, exec_id TEXT NOT NULL, local_id TEXT NOT NULL REFERENCES orders(local_id),
                quantity TEXT NOT NULL, price TEXT NOT NULL, side TEXT NOT NULL, occurred_at TEXT NOT NULL,
                commission TEXT, commission_currency TEXT, PRIMARY KEY(account, exec_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY, local_id TEXT NOT NULL REFERENCES orders(local_id),
                recorded_at TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_intents (
                local_id TEXT PRIMARY KEY, account TEXT NOT NULL,
                public_id TEXT NOT NULL, command TEXT NOT NULL,
                UNIQUE(account, public_id)
            );
            CREATE TABLE IF NOT EXISTS order_revisions (
                local_id TEXT NOT NULL REFERENCES orders(local_id), revision INTEGER NOT NULL,
                command_id TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(local_id, revision), UNIQUE(local_id, command_id)
            );
            CREATE TABLE IF NOT EXISTS order_groups (
                group_id TEXT PRIMARY KEY, account TEXT NOT NULL, kind TEXT NOT NULL,
                intent TEXT NOT NULL, transport_state TEXT NOT NULL DEFAULT 'staged'
            );
            CREATE TABLE IF NOT EXISTS order_group_members (
                local_id TEXT PRIMARY KEY REFERENCES orders(local_id),
                group_id TEXT NOT NULL REFERENCES order_groups(group_id), parent_id TEXT
            );
        """)
        if "contract_id" not in {
            r[1] for r in self._db.execute("PRAGMA table_info(executions)")
        }:
            with self._db:
                self._db.execute(
                    "ALTER TABLE executions ADD COLUMN contract_id INTEGER NOT NULL DEFAULT 0"
                )
        if "needs_terms" not in {
            r[1] for r in self._db.execute("PRAGMA table_info(orders)")
        }:
            with self._db:
                self._db.execute(
                    "ALTER TABLE orders ADD COLUMN needs_terms INTEGER NOT NULL DEFAULT 0"
                )

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._db.close()

    def _event(self, local_id: str, kind: str, detail: str) -> None:
        self._db.execute(
            "INSERT INTO events(local_id,recorded_at,kind,detail) VALUES(?,?,?,?)",
            (local_id, _now(), kind, detail),
        )

    def intent(
        self,
        request: StockOrder | VerticalOrder,
        account: str,
        broker_id: int,
        client_id: int,
        contract_id: int,
    ) -> bool:
        data = asdict(request)
        if isinstance(request, VerticalOrder):
            data["kind"] = "vertical"
        payload = json.dumps(data, default=str, sort_keys=True)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._db.execute(
                "SELECT request,account FROM orders WHERE local_id=?",
                (request.client_order_id,),
            ).fetchone()
            if existing:
                # Decode defaults so journals written before TIF was explicit
                # remain retryable without rewriting their original evidence.
                if (
                    self.get(request.client_order_id).request != request
                    or existing["account"] != account
                ):
                    raise ValueError(
                        "client_order_id already belongs to a different intent/account."
                    )
                return False
            self._db.execute(
                "INSERT INTO orders(local_id,request,account,broker_id,client_id,contract_id) VALUES(?,?,?,?,?,?)",
                (
                    request.client_order_id,
                    payload,
                    account,
                    broker_id,
                    client_id,
                    contract_id,
                ),
            )
            self._event(request.client_order_id, "intent", payload)
        return True

    def runtime_intent(self, local_id: str) -> dict | None:
        """Return original strategy command metadata, absent on legacy orders."""
        row = self._db.execute(
            "SELECT * FROM runtime_intents WHERE local_id=?", (local_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def record_runtime_intent(
        self, local_id: str, account: str, public_id: str, command: str
    ) -> None:
        """Durably bind the complete original command before broker submission."""
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            old = self.runtime_intent(local_id)
            expected = dict(
                local_id=local_id, account=account, public_id=public_id, command=command
            )
            if old is not None:
                if old != expected:
                    raise ValueError(
                        "Durable order ID already belongs to a different intent/account."
                    )
                return
            self._db.execute(
                "INSERT INTO runtime_intents VALUES(?,?,?,?)",
                (local_id, account, public_id, command),
            )

    def broker_order_id(self, account: str, public_id: str) -> str:
        """Resolve a strategy-facing order ID; legacy broker IDs remain valid."""
        row = self._db.execute(
            "SELECT local_id FROM runtime_intents WHERE account=? AND public_id=?",
            (account, public_id),
        ).fetchone()
        return row[0] if row is not None else public_id

    def get(self, local_id: str) -> OrderRecord:
        row = self._db.execute(
            "SELECT * FROM orders WHERE local_id=?", (local_id,)
        ).fetchone()
        if row is None:
            raise KeyError(local_id)
        data = json.loads(row["request"])
        data["limit_price"] = (
            Decimal(data["limit_price"]) if data["limit_price"] is not None else None
        )
        if data.get("stop_price") is not None:
            data["stop_price"] = Decimal(data["stop_price"])
        if data.pop("kind", "stock") == "vertical":
            for name in ("long_leg", "short_leg"):
                leg = data[name]
                leg["expiry"] = date.fromisoformat(leg["expiry"])
                leg["strike"] = Decimal(leg["strike"])
                data[name] = OptionIdentity(**leg)
            request = VerticalOrder(**data)
        else:
            request = StockOrder(**data)
        fills = tuple(
            Execution(
                r["exec_id"],
                r["local_id"],
                Decimal(r["quantity"]),
                Decimal(r["price"]),
                r["side"],
                r["occurred_at"],
                Decimal(r["commission"]) if r["commission"] is not None else None,
                r["commission_currency"],
                r["contract_id"],
            )
            for r in self._db.execute(
                "SELECT * FROM executions WHERE local_id=? ORDER BY occurred_at,exec_id",
                (local_id,),
            )
        )
        executed = (
            min(
                (
                    sum(
                        (f.quantity for f in fills if f.contract_id == leg.contract_id),
                        Decimal(0),
                    )
                    for leg in (request.long_leg, request.short_leg)
                )
            )
            if isinstance(request, VerticalOrder)
            else sum((f.quantity for f in fills), Decimal(0))
        )
        filled = max(Decimal(row["reported_filled"]), executed)
        revision = self._db.execute(
            "SELECT * FROM order_revisions WHERE local_id=? ORDER BY revision DESC LIMIT 1",
            (local_id,),
        ).fetchone()
        pending = revision is not None and revision["status"] == "pending"
        accepted_revision = self._db.execute(
            "SELECT COUNT(*) FROM order_revisions WHERE local_id=? AND status='accepted'",
            (local_id,),
        ).fetchone()[0]
        desired = (
            json.loads(revision["request"])
            if pending and revision is not None
            else None
        )
        if desired is not None:
            for name in ("limit_price", "stop_price"):
                if desired.get(name) is not None:
                    desired[name] = Decimal(desired[name])
        group = self._db.execute(
            "SELECT m.group_id, m.parent_id, g.kind FROM order_group_members m JOIN order_groups g USING(group_id) WHERE local_id=?",
            (local_id,),
        ).fetchone()
        status = (
            "Filled" if filled >= data["quantity"] and not pending else row["status"]
        )
        if 0 < filled < data["quantity"] and status in ("Submitted", "PreSubmitted"):
            status = "PartiallyFilled"
        intent_time = self._db.execute(
            "SELECT recorded_at FROM events WHERE local_id=? AND kind='intent' ORDER BY sequence LIMIT 1",
            (local_id,),
        ).fetchone()
        return OrderRecord(
            request,
            row["account"],
            row["broker_id"],
            row["client_id"],
            row["contract_id"],
            row["perm_id"],
            status,
            bool(row["acknowledged"]),
            bool(row["uncertain"]),
            bool(row["cancel_pending"]) and status != "Filled",
            filled,
            fills,
            tuple(
                r[0]
                for r in self._db.execute(
                    "SELECT detail FROM events WHERE local_id=? AND kind='diagnostic' ORDER BY sequence",
                    (local_id,),
                )
            ),
            datetime.fromisoformat(intent_time[0]) if intent_time is not None else None,
            accepted_revision,
            pending,
            StockOrder(**desired) if desired else None,
            group["group_id"] if group else None,
            group["parent_id"] if group else None,
            group["kind"] if group else None,
            bool(row["needs_terms"]),
        )

    def orders(self) -> tuple[OrderRecord, ...]:
        return tuple(
            self.get(r[0])
            for r in self._db.execute("SELECT local_id FROM orders ORDER BY rowid")
        )

    def replace_requested(
        self, local_id: str, request: StockOrder, command_id: str
    ) -> bool:
        """Persist desired terms; accepted terms change only on matching broker evidence."""
        payload = json.dumps(asdict(request), default=str, sort_keys=True)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            prior = self._db.execute(
                "SELECT request FROM order_revisions WHERE local_id=? AND command_id=?",
                (local_id, command_id),
            ).fetchone()
            if prior is not None:
                if prior[0] != payload:
                    raise ValueError(
                        "Replacement command ID already belongs to different terms."
                    )
                return False
            row = self.get(local_id)
            if (
                row.terminal
                or row.cancel_pending
                or row.replace_pending
                or row.uncertain
            ):
                raise ValueError(
                    "Replacement requires a reconciled working order with no pending mutation."
                )
            if request.quantity <= row.filled_quantity:
                raise ValueError(
                    "Replacement total quantity must exceed cumulative fills; cancel the remainder instead."
                )
            revision = self._db.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM order_revisions WHERE local_id=?",
                (local_id,),
            ).fetchone()[0]
            if row.request == request:
                self._db.execute(
                    "INSERT INTO order_revisions VALUES(?,?,?,?,?)",
                    (local_id, revision, command_id, payload, "noop"),
                )
                self._event(local_id, "replace_unchanged", payload)
                return False
            self._db.execute(
                "INSERT INTO order_revisions VALUES(?,?,?,?,?)",
                (local_id, revision, command_id, payload, "pending"),
            )
            self._db.execute(
                "UPDATE orders SET uncertain=1,needs_terms=1 WHERE local_id=?",
                (local_id,),
            )
            self._event(local_id, "replace_requested", payload)
        return True

    def replacement_observed(self, local_id: str, request: StockOrder) -> bool:
        with self._db:
            row = self.get(local_id)
            if not row.replace_pending or row.pending_request != request:
                return False
            payload = json.dumps(asdict(request), default=str, sort_keys=True)
            self._db.execute(
                "UPDATE orders SET request=?, uncertain=0,needs_terms=0 WHERE local_id=?",
                (payload, local_id),
            )
            self._db.execute(
                "UPDATE order_revisions SET status='accepted' WHERE local_id=? AND status='pending'",
                (local_id,),
            )
            self._event(local_id, "replace_accepted", payload)
        return True

    def terms_confirmed(self, local_id: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE orders SET needs_terms=0 WHERE local_id=?", (local_id,)
            )

    def replacement_rejected(self, local_id: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE order_revisions SET status='rejected' WHERE local_id=? AND status='pending'",
                (local_id,),
            )
            # Rejecting a modification says nothing about the original order's
            # current working terms; an openOrder replay must settle uncertainty.
            self._db.execute(
                "UPDATE orders SET uncertain=1 WHERE local_id=?", (local_id,)
            )
            self._event(
                local_id,
                "replace_rejected",
                "Broker rejected modification; original order requires reconciliation.",
            )

    def latest_replacement(self, local_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM order_revisions WHERE local_id=? ORDER BY revision DESC LIMIT 1",
            (local_id,),
        ).fetchone()
        return dict(row) if row else None

    def replacement_command(self, local_id: str, command_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM order_revisions WHERE local_id=? AND command_id=?",
            (local_id, command_id),
        ).fetchone()
        return dict(row) if row else None

    def group(self, group_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM order_groups WHERE group_id=?", (group_id,)
        ).fetchone()
        return dict(row) if row else None

    def group_intent(
        self,
        group_id: str,
        kind: str,
        account: str,
        requests: tuple[StockOrder, ...],
        broker_ids: tuple[int, ...],
        client_id: int,
        contract_ids: tuple[int, ...],
    ) -> bool:
        payload = json.dumps(
            [asdict(request) for request in requests], default=str, sort_keys=True
        )
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self.group(group_id)
            if existing:
                if (existing["kind"], existing["account"], existing["intent"]) != (
                    kind,
                    account,
                    payload,
                ):
                    raise ValueError(
                        "Group ID already belongs to a different intent/account."
                    )
                return False
            self._db.execute(
                "INSERT INTO order_groups(group_id,account,kind,intent) VALUES(?,?,?,?)",
                (group_id, account, kind, payload),
            )
            for index, (request, broker_id, contract_id) in enumerate(
                zip(requests, broker_ids, contract_ids, strict=True)
            ):
                body = json.dumps(asdict(request), default=str, sort_keys=True)
                self._db.execute(
                    "INSERT INTO orders(local_id,request,account,broker_id,client_id,contract_id) VALUES(?,?,?,?,?,?)",
                    (
                        request.client_order_id,
                        body,
                        account,
                        broker_id,
                        client_id,
                        contract_id,
                    ),
                )
                parent = (
                    requests[0].client_order_id if kind == "bracket" and index else None
                )
                self._db.execute(
                    "INSERT INTO order_group_members VALUES(?,?,?)",
                    (request.client_order_id, group_id, parent),
                )
                self._event(request.client_order_id, "intent", body)
        return True

    def group_transport(self, group_id: str, state: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE order_groups SET transport_state=? WHERE group_id=?",
                (state, group_id),
            )
            if state != "sent":
                self._db.execute(
                    "UPDATE orders SET uncertain=1 WHERE local_id IN (SELECT local_id FROM order_group_members WHERE group_id=?)",
                    (group_id,),
                )

    def events(self, local_id: str) -> tuple[dict, ...]:
        return tuple(
            dict(r)
            for r in self._db.execute(
                "SELECT * FROM events WHERE local_id=? ORDER BY sequence", (local_id,)
            )
        )

    def status(
        self, local_id: str, status: str, filled: Decimal, perm_id: int = 0
    ) -> None:
        if not isinstance(filled, Decimal) or not filled.is_finite() or filled < 0:
            raise ValueError("Reported filled quantity must be finite and nonnegative.")
        with self._db:
            current = self.get(local_id)
            maximum = max(
                current.request.quantity,
                current.pending_request.quantity if current.pending_request else 0,
            )
            if filled > maximum:
                raise ValueError(
                    "Reported fills exceed the order quantity; reconcile broker state."
                )
            if (
                status == "Filled"
                and max(current.filled_quantity, filled) != current.request.quantity
            ):
                raise ValueError(
                    "Filled status must report the complete order quantity."
                )
            # PendingSubmit/PendingCancel may be local client echoes, not broker acknowledgments.
            ack = status in (
                "Submitted",
                "PreSubmitted",
                "Inactive",
                "Filled",
                "Cancelled",
                "ApiCancelled",
                "Rejected",
            )
            chosen = (
                current.status if current.terminal and status != "Filled" else status
            )
            if status == "Rejected" and current.status != "Filled":
                chosen = status
            self._db.execute(
                """UPDATE orders SET status=?, reported_filled=?, perm_id=?,
                acknowledged=MAX(acknowledged,?), uncertain=CASE WHEN ? THEN 0 ELSE uncertain END,
                cancel_pending=CASE WHEN ? THEN 0 ELSE cancel_pending END WHERE local_id=?""",
                (
                    chosen,
                    str(max(current.filled_quantity, filled)),
                    perm_id or current.permanent_id,
                    int(ack),
                    int(
                        ack and not current.replace_pending and not current.needs_terms
                    ),
                    int(chosen in ("Filled", "Cancelled", "ApiCancelled", "Rejected")),
                    local_id,
                ),
            )
            if chosen in ("Cancelled", "ApiCancelled", "Rejected"):
                self._db.execute(
                    "UPDATE order_revisions SET status='canceled' WHERE local_id=? AND status='pending'",
                    (local_id,),
                )
            self._event(
                local_id,
                "status",
                json.dumps(
                    dict(status=status, filled=str(filled), permanent_id=perm_id)
                ),
            )

    def uncertain(self, local_id: str, detail: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE orders SET uncertain=1 WHERE local_id=?", (local_id,)
            )
            self._event(local_id, "diagnostic", detail)

    def diagnostic(self, local_id: str, detail: str) -> None:
        with self._db:
            self._event(local_id, "diagnostic", detail)

    def cancel_requested(self, local_id: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE orders SET cancel_pending=1 WHERE local_id=?", (local_id,)
            )
            self._event(
                local_id,
                "cancel_requested",
                "Cancellation requested; fills may still arrive.",
            )

    def cancel_rejected(self, local_id: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE orders SET cancel_pending=0 WHERE local_id=?", (local_id,)
            )

    def execution(self, fill: Execution) -> bool:
        if (
            not fill.execution_id
            or not isinstance(fill.quantity, Decimal)
            or not fill.quantity.is_finite()
            or fill.quantity <= 0
            or not isinstance(fill.price, Decimal)
            or not fill.price.is_finite()
            or fill.price <= 0
            or fill.side not in ("BUY", "SELL")
        ):
            raise ValueError("Invalid execution identity, side, quantity or price.")
        occurred = datetime.fromisoformat(fill.occurred_at)
        if occurred.tzinfo is None or occurred.utcoffset() is None:
            raise ValueError("Execution timestamps must be timezone-aware.")
        if fill.commission is not None and (
            not isinstance(fill.commission, Decimal) or not fill.commission.is_finite()
        ):
            raise ValueError("Execution commission must be finite.")
        if (fill.commission is None) != (fill.commission_currency is None):
            raise ValueError(
                "Execution commission and currency must be supplied together."
            )
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            order = self.get(fill.client_order_id)
            expected_side = (
                order.request.leg_side(fill.contract_id)
                if isinstance(order.request, VerticalOrder)
                else order.request.side
            )
            if not isinstance(
                order.request, VerticalOrder
            ) and fill.contract_id not in (0, order.contract_id):
                raise ValueError(
                    "Execution contract does not match the accepted order."
                )
            if fill.side != expected_side:
                raise ValueError("Execution side does not match the accepted order.")
            old = self._db.execute(
                "SELECT * FROM executions WHERE account=? AND exec_id=?",
                (order.account, fill.execution_id),
            ).fetchone()
            if old:
                if (
                    old["local_id"],
                    Decimal(old["quantity"]),
                    Decimal(old["price"]),
                    old["side"],
                    old["contract_id"],
                ) != (
                    fill.client_order_id,
                    fill.quantity,
                    fill.price,
                    fill.side,
                    fill.contract_id,
                ):
                    raise ValueError(
                        "Conflicting replay of an execution ID; reconciliation required."
                    )
                if datetime.fromisoformat(old["occurred_at"]) != occurred:
                    self._db.execute(
                        "UPDATE executions SET occurred_at=? WHERE account=? AND exec_id=?",
                        (fill.occurred_at, order.account, fill.execution_id),
                    )
                    self._event(
                        fill.client_order_id,
                        "execution_time_reconciled",
                        json.dumps(
                            dict(
                                execution_id=fill.execution_id,
                                previous=old["occurred_at"],
                                broker_replay=fill.occurred_at,
                            )
                        ),
                    )
                return False
            maximum = max(
                order.request.quantity,
                order.pending_request.quantity if order.pending_request else 0,
            )
            if (
                sum(
                    (
                        f.quantity
                        for f in order.executions
                        if not isinstance(order.request, VerticalOrder)
                        or f.contract_id == fill.contract_id
                    ),
                    Decimal(0),
                )
                + fill.quantity
                > maximum
            ):
                raise ValueError(
                    "Executions exceed the accepted order quantity; reconciliation required."
                )
            self._db.execute(
                "INSERT INTO executions(account,exec_id,local_id,quantity,price,side,occurred_at,commission,commission_currency,contract_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    order.account,
                    fill.execution_id,
                    fill.client_order_id,
                    str(fill.quantity),
                    str(fill.price),
                    fill.side,
                    fill.occurred_at,
                    str(fill.commission) if fill.commission is not None else None,
                    fill.commission_currency,
                    fill.contract_id,
                ),
            )
            self._db.execute(
                "UPDATE orders SET acknowledged=1,uncertain=? WHERE local_id=?",
                (int(order.replace_pending or order.needs_terms), fill.client_order_id),
            )
            self._event(
                fill.client_order_id, "execution", json.dumps(asdict(fill), default=str)
            )
        return True

    def commission(
        self, account: str, execution_id: str, amount: Decimal, currency: str
    ) -> None:
        if not isinstance(amount, Decimal) or not amount.is_finite() or not currency:
            raise ValueError("Commission requires a finite amount and currency.")
        with self._db:
            self._db.execute(
                "UPDATE executions SET commission=?,commission_currency=? WHERE account=? AND exec_id=?",
                (str(amount), currency, account, execution_id),
            )

    def attributed_position(
        self, account: str, strategy_id: str, contract_id: int
    ) -> Decimal:
        total = Decimal(0)
        for order in self.orders():
            if (order.account, order.request.strategy_id) == (account, strategy_id):
                total += sum(
                    (
                        f.quantity if f.side == "BUY" else -f.quantity
                        for f in order.executions
                        if (f.contract_id or order.contract_id) == contract_id
                    ),
                    Decimal(0),
                )
        return total
