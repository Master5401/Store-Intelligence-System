"""
src/anomaly/pos_integration.py
────────────────────────────────
Correlates visual tracking with POS transaction log to detect:
  1. Scan Avoidance – item crosses scanner but no barcode in POS stream
  2. Statistical anomalies – excessive voids, discount abuse, per cashier
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class POSEvent:
    event_type: str   # scan | void | discount | payment | login | logout
    cashier_id: str
    item_id:    Optional[str]
    amount:     Optional[float]
    timestamp:  datetime = field(default_factory=datetime.utcnow)
    raw:        dict = field(default_factory=dict)


@dataclass
class CheckoutViolation:
    violation_type: str   # scan_avoidance | void_spike | discount_abuse
    cashier_id:     str
    timestamp:      datetime
    confidence:     float
    details:        dict = field(default_factory=dict)


class POSIntegration:
    """
    Correlates YOLO object tracking data with POS event streams.

    Usage
    -----
    pos = POSIntegration()
    pos.ingest_pos_event(POSEvent(...))
    pos.record_visual_crossing(cashier_id)
    violations = pos.check_pending_violations()
    """

    SCAN_WINDOW_SECONDS = 5.0
    VOID_THRESHOLD      = 0.25
    DISCOUNT_THRESHOLD  = 0.30

    def __init__(self) -> None:
        self._pos_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._pending_crossings: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._violations: list[CheckoutViolation] = []

    def ingest_pos_event(self, event: POSEvent) -> None:
        self._pos_history[event.cashier_id].append(event)
        if event.event_type == "scan":
            pending = self._pending_crossings[event.cashier_id]
            if pending:
                delta = (event.timestamp - pending[0]).total_seconds()
                if 0 <= delta <= self.SCAN_WINDOW_SECONDS:
                    pending.popleft()
                    logger.debug("POS: scan matched crossing (Δ=%.2fs)", delta)

    def record_visual_crossing(
        self,
        cashier_id: str,
        crossing_time: Optional[datetime] = None,
    ) -> None:
        t = crossing_time or datetime.utcnow()
        self._pending_crossings[cashier_id].append(t)

    def check_pending_violations(self) -> list[CheckoutViolation]:
        now = datetime.utcnow()
        new_violations: list[CheckoutViolation] = []

        for cashier_id, pending in self._pending_crossings.items():
            expired, remaining = [], []
            for t in pending:
                if (now - t).total_seconds() > self.SCAN_WINDOW_SECONDS * 2:
                    expired.append(t)
                else:
                    remaining.append(t)
            self._pending_crossings[cashier_id] = deque(remaining, maxlen=50)

            for t in expired:
                v = CheckoutViolation(
                    violation_type="scan_avoidance",
                    cashier_id=cashier_id,
                    timestamp=t,
                    confidence=0.82,
                    details={"visual_crossing_at": t.isoformat()},
                )
                new_violations.append(v)
                self._violations.append(v)
                logger.warning("POS VIOLATION: scan_avoidance cashier=%s", cashier_id)

        for cashier_id in list(self._pos_history.keys()):
            v = self._analyse_cashier_stats(cashier_id)
            if v:
                new_violations.append(v)
                self._violations.append(v)

        return new_violations

    def _analyse_cashier_stats(self, cashier_id: str) -> Optional[CheckoutViolation]:
        events = list(self._pos_history[cashier_id])
        if len(events) < 20:
            return None
        scans     = sum(1 for e in events if e.event_type == "scan")
        voids     = sum(1 for e in events if e.event_type == "void")
        discounts = sum(1 for e in events if e.event_type == "discount")
        total_ops = scans + voids + discounts
        if total_ops < 10:
            return None

        void_ratio     = voids / total_ops
        discount_ratio = discounts / total_ops

        if void_ratio > self.VOID_THRESHOLD:
            return CheckoutViolation(
                violation_type="void_spike",
                cashier_id=cashier_id,
                timestamp=datetime.utcnow(),
                confidence=min(0.95, 0.5 + void_ratio),
                details={"void_ratio": round(void_ratio, 3), "voids": voids, "scans": scans},
            )
        if discount_ratio > self.DISCOUNT_THRESHOLD:
            return CheckoutViolation(
                violation_type="discount_abuse",
                cashier_id=cashier_id,
                timestamp=datetime.utcnow(),
                confidence=min(0.90, 0.4 + discount_ratio),
                details={"discount_ratio": round(discount_ratio, 3), "discounts": discounts, "scans": scans},
            )
        return None

    def get_violations(self) -> list[CheckoutViolation]:
        return list(self._violations)

    def get_cashier_stats(self, cashier_id: str) -> dict:
        events = list(self._pos_history.get(cashier_id, []))
        if not events:
            return {"cashier_id": cashier_id, "status": "no_data"}
        scans     = sum(1 for e in events if e.event_type == "scan")
        voids     = sum(1 for e in events if e.event_type == "void")
        discounts = sum(1 for e in events if e.event_type == "discount")
        total     = len(events)
        return {
            "cashier_id":    cashier_id,
            "total_events":  total,
            "scans":         scans,
            "voids":         voids,
            "discounts":     discounts,
            "void_rate":     round(voids / total, 3) if total else 0,
            "discount_rate": round(discounts / total, 3) if total else 0,
        }
