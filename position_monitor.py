#!/usr/bin/env python3
"""
Position Monitor for Tastytrade Trading Bot

Background monitoring system that checks positions for management actions
and generates alerts for user review.

Author: Trading Bot
License: MIT
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from queue import Queue
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot import TastytradeBot

logger = logging.getLogger(__name__)


class AlertUrgency(Enum):
    """Urgency level for management alerts"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AlertAction(Enum):
    """Recommended action for an alert"""
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    CLOSE_OR_ROLL = "CLOSE_OR_ROLL"
    MONITOR = "MONITOR"


@dataclass
class ManagementAlert:
    """
    Alert generated when a position hits management criteria.

    Stored in a queue and displayed to user on next interaction.
    """
    # Position identification
    symbol: str
    underlying: str

    # Recommended action
    action: AlertAction
    reason: str
    urgency: AlertUrgency

    # Position metrics at time of alert
    pnl_percent: float
    dte: int
    is_tested: bool

    # Position details
    quantity: int = 0
    entry_credit: float = 0.0
    current_value: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    # Claude AI analysis (if available)
    claude_reasoning: Optional[str] = None
    claude_confidence: Optional[int] = None

    def urgency_marker(self) -> str:
        """Get urgency marker for display"""
        markers = {
            AlertUrgency.LOW: "",
            AlertUrgency.NORMAL: "[!]",
            AlertUrgency.HIGH: "[!!]",
            AlertUrgency.CRITICAL: "[!!!]"
        }
        return markers.get(self.urgency, "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'action': self.action.value,
            'reason': self.reason,
            'urgency': self.urgency.value,
            'pnl_percent': self.pnl_percent,
            'dte': self.dte,
            'is_tested': self.is_tested,
            'quantity': self.quantity,
            'entry_credit': self.entry_credit,
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'claude_reasoning': self.claude_reasoning,
            'claude_confidence': self.claude_confidence
        }


class PositionMonitor:
    """
    Background position monitor that runs on a configurable interval.

    Features:
    - Runs in background thread with its own asyncio loop
    - Checks positions against management criteria
    - Generates alerts for positions needing action
    - Thread-safe alert queue for chatbot integration
    """

    def __init__(
        self,
        bot: 'TastytradeBot',
        interval_minutes: int = 60,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the position monitor.

        Args:
            bot: Reference to the trading bot
            interval_minutes: Check interval in minutes
            config: Optional configuration dict
        """
        self.bot = bot
        self.interval_minutes = interval_minutes
        self.config = config or {}

        # Thread-safe alert queue
        self._alert_queue: Queue[ManagementAlert] = Queue()

        # Background thread management
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Alert history (for deduplication)
        self._recent_alerts: Dict[str, datetime] = {}
        self._alert_cooldown_minutes = 60  # Don't re-alert for same position within cooldown

        # Management thresholds from config
        management_rules = self.bot.config.get('management_rules', {})
        profit_taking = management_rules.get('profit_taking', {})
        loss_management = management_rules.get('loss_management', {})
        dte_management = management_rules.get('dte_management', {})

        self.profit_target = Decimal(str(profit_taking.get('profit_target_percent', 0.50)))
        self.stop_loss_multiplier = Decimal(str(loss_management.get('stop_loss_multiplier', 2.0)))
        self.management_dte = dte_management.get('management_dte', 21)

        logger.info(f"PositionMonitor initialized (interval={interval_minutes}min)")

    def start(self) -> None:
        """Start the background monitoring thread"""
        if self._running:
            logger.warning("Position monitor already running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Position monitor started")

    def stop(self) -> None:
        """Stop the background monitoring thread"""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("Position monitor stopped")

    def _run_loop(self) -> None:
        """Main monitoring loop running in background thread"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self._stop_event.is_set():
                try:
                    # Run the check
                    loop.run_until_complete(self._check_positions())
                except Exception as e:
                    logger.error(f"Position check failed: {e}")

                # Wait for next interval or stop signal
                self._stop_event.wait(timeout=self.interval_minutes * 60)
        finally:
            loop.close()

    async def _check_positions(self) -> None:
        """Check all positions and generate alerts as needed"""
        logger.info("Running position management check...")

        try:
            # Use the bot's existing manage_positions method
            recommendations = await self.bot.manage_positions()

            for rec in recommendations:
                if rec.get('action'):
                    self._process_recommendation(rec)

            if recommendations:
                logger.info(f"Position check complete: {len(recommendations)} action(s) found")
            else:
                logger.info("Position check complete: no actions needed")

        except Exception as e:
            logger.error(f"Position check error: {e}")

    def _process_recommendation(self, rec: Dict[str, Any]) -> None:
        """Process a management recommendation and generate alert if needed"""
        symbol = rec.get('symbol', 'Unknown')

        # Check cooldown to avoid duplicate alerts
        last_alert = self._recent_alerts.get(symbol)
        if last_alert:
            minutes_since = (datetime.now() - last_alert).total_seconds() / 60
            if minutes_since < self._alert_cooldown_minutes:
                logger.debug(f"Skipping alert for {symbol} (cooldown)")
                return

        # Determine urgency
        action_str = rec.get('action', 'MONITOR')
        pnl_percent = rec.get('pnl_percent', 0)
        is_tested = rec.get('is_tested', False)
        dte = rec.get('dte', 999)

        urgency = self._determine_urgency(action_str, pnl_percent, is_tested, dte)

        # Map action string to enum
        action_map = {
            'CLOSE': AlertAction.CLOSE,
            'ROLL': AlertAction.ROLL,
            'CLOSE_OR_ROLL': AlertAction.CLOSE_OR_ROLL,
            'MONITOR': AlertAction.MONITOR
        }
        action = action_map.get(action_str, AlertAction.MONITOR)

        # Create alert
        alert = ManagementAlert(
            symbol=symbol,
            underlying=rec.get('underlying', symbol),
            action=action,
            reason=rec.get('reason', ''),
            urgency=urgency,
            pnl_percent=pnl_percent,
            dte=dte,
            is_tested=is_tested,
            quantity=rec.get('quantity', 0),
            claude_reasoning=rec.get('claude_reasoning'),
            claude_confidence=rec.get('claude_confidence')
        )

        # Add to queue
        self._alert_queue.put(alert)
        self._recent_alerts[symbol] = datetime.now()

        logger.info(f"Alert generated: {alert.urgency_marker()} {symbol} - {action.value}")

    def _determine_urgency(
        self,
        action: str,
        pnl_percent: float,
        is_tested: bool,
        dte: int
    ) -> AlertUrgency:
        """Determine alert urgency based on position metrics"""
        # Stop loss = critical
        if pnl_percent <= -float(self.stop_loss_multiplier):
            return AlertUrgency.CRITICAL

        # Tested + low DTE = high
        if is_tested and dte <= self.management_dte:
            return AlertUrgency.HIGH

        # Tested = high
        if is_tested:
            return AlertUrgency.HIGH

        # Low DTE = normal
        if dte <= self.management_dte:
            return AlertUrgency.NORMAL

        # Profit target = normal
        if pnl_percent >= float(self.profit_target):
            return AlertUrgency.NORMAL

        return AlertUrgency.LOW

    def get_pending_alerts(self) -> List[ManagementAlert]:
        """
        Get all pending alerts from the queue.

        Called by chatbot before input() to display alerts.
        Non-blocking - returns empty list if no alerts.

        Returns:
            List of pending ManagementAlert objects
        """
        alerts = []
        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                alerts.append(alert)
            except Exception:
                break
        return alerts

    def has_pending_alerts(self) -> bool:
        """Check if there are pending alerts without consuming them"""
        return not self._alert_queue.empty()

    def clear_alerts(self) -> None:
        """Clear all pending alerts"""
        while not self._alert_queue.empty():
            try:
                self._alert_queue.get_nowait()
            except Exception:
                break

    def trigger_check_now(self) -> None:
        """
        Manually trigger a position check.

        Useful for testing or user-requested checks.
        """
        if self._running:
            # Interrupt the wait by setting and clearing a temporary event
            logger.info("Manual position check triggered")
            asyncio.run(self._check_positions())
        else:
            logger.warning("Monitor not running - starting check directly")
            asyncio.run(self._check_positions())

    @property
    def is_running(self) -> bool:
        """Check if monitor is currently running"""
        return self._running
