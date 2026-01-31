#!/usr/bin/env python3
"""
Response Formatter for Tastytrade Chatbot

Formats data into natural language responses suitable for CLI display.

Author: Trading Bot
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Dict, Any, Optional
from datetime import datetime


class ResponseFormatter:
    """
    Formats trading data into natural language responses.

    Optimized for CLI display with clear, concise formatting.
    """

    def format_portfolio_summary(self, portfolio: Dict[str, Any]) -> str:
        """Format portfolio summary for display"""
        nlv = portfolio.get('net_liquidating_value', 0)
        bp = portfolio.get('buying_power', 0)
        delta = portfolio.get('portfolio_delta', 0)
        theta = portfolio.get('portfolio_theta', 0)
        positions = portfolio.get('open_positions', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)

        lines = [
            "Portfolio Summary",
            "-" * 30,
            f"Net Liquidating Value: ${nlv:,.2f}",
            f"Buying Power:          ${bp:,.2f}",
            f"Open Positions:        {positions}",
            "",
            "Greeks (Beta-Weighted to SPY):",
            f"  Delta: {delta:+.1f}",
            f"  Theta: {theta:+.2f}",
            "",
            f"Today's P&L: ${daily_pnl:+,.2f}"
        ]

        return "\n".join(lines)

    def format_positions_list(self, positions: List[Dict[str, Any]]) -> str:
        """Format list of positions for display"""
        if not positions:
            return "No open positions."

        lines = ["Current Positions", "-" * 50]

        for pos in positions:
            symbol = pos.get('symbol', 'Unknown')
            underlying = pos.get('underlying', symbol)
            qty = pos.get('quantity', 0)
            pnl = pos.get('pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            dte = pos.get('dte', 'N/A')

            pnl_display = f"${pnl:+.2f}" if isinstance(pnl, (int, float, Decimal)) else pnl
            pnl_pct_display = f"({pnl_pct:+.1%})" if isinstance(pnl_pct, (int, float, Decimal)) else ""

            lines.append(f"  {underlying}: {qty} contracts | DTE: {dte} | P&L: {pnl_display} {pnl_pct_display}")

        return "\n".join(lines)

    def format_pending_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Format pending trades for display"""
        if not trades:
            return "No pending trades."

        lines = ["Pending Trades", "-" * 50]

        for trade in trades:
            trade_id = trade.get('id', 'Unknown')[:8]
            symbol = trade.get('underlying_symbol', 'Unknown')
            strategy = trade.get('strategy', 'Unknown')
            credit = trade.get('expected_credit', 0)
            max_loss = trade.get('max_loss', 0)
            iv_rank = trade.get('iv_rank', 0)
            dte = trade.get('dte', 0)

            lines.append(f"  [{trade_id}] {symbol} {strategy}")
            lines.append(f"       Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}")
            lines.append(f"       IV Rank: {iv_rank:.1f}% | DTE: {dte}")
            lines.append("")

        lines.append("Use 'approve <id>' or 'reject <id>' to manage trades.")
        return "\n".join(lines)

    def format_trade_proposal(self, trade: Dict[str, Any]) -> str:
        """Format a single trade proposal in detail"""
        trade_id = trade.get('id', 'Unknown')[:8]
        symbol = trade.get('underlying_symbol', 'Unknown')
        strategy = trade.get('strategy', 'Unknown')
        credit = trade.get('expected_credit', 0)
        limit_price = trade.get('limit_price') or credit  # Use limit_price if set, otherwise credit
        max_loss = trade.get('max_loss', 0)
        pop = trade.get('probability_of_profit', 0)
        iv_rank = trade.get('iv_rank', 0)
        dte = trade.get('dte', 0)
        delta = trade.get('delta', 0)
        theta = trade.get('theta', 0)
        legs = trade.get('legs', [])
        status = trade.get('status', 'pending')
        order_status = trade.get('order_status')

        lines = [
            f"Trade Proposal [{trade_id}]",
            "=" * 40,
            f"Symbol:   {symbol}",
            f"Strategy: {strategy}",
            f"Status:   {status}",
        ]

        if order_status:
            lines.append(f"Order:    {order_status.upper()}")

        lines.extend(["", "Legs:"])

        for leg in legs:
            action = leg.get('action', 'SELL')
            opt_type = leg.get('option_type', 'PUT')
            strike = leg.get('strike', 0)
            qty = leg.get('quantity', 1)
            exp = leg.get('expiration', 'N/A')
            lines.append(f"  {action} {qty}x {opt_type} @ ${strike} (exp: {exp})")

        lines.extend([
            "",
            "Pricing:",
            f"  Expected Credit: ${credit:.2f}",
            f"  LIMIT PRICE:     ${limit_price:.2f}  <-- Order will execute at this price",
        ])

        # Show if limit price differs from expected credit
        if trade.get('limit_price') and abs(float(limit_price) - float(credit)) > 0.001:
            lines.append(f"  (Modified from expected ${credit:.2f})")

        lines.extend([
            "",
            "Risk:",
            f"  Max Loss:    ${max_loss:.2f}",
            f"  IV Rank:     {iv_rank:.1f}%",
            f"  DTE:         {dte} days",
            f"  P.O.P:       {pop:.1%}",
            "",
            "Greeks:",
            f"  Delta: {delta:+.3f}",
            f"  Theta: {theta:+.3f}",
        ])

        return "\n".join(lines)

    def format_working_orders(self, orders: List[Dict[str, Any]]) -> str:
        """Format list of working (unfilled) orders"""
        if not orders:
            return "No working orders."

        lines = ["Working Orders (Unfilled)", "-" * 50]

        for order in orders:
            trade_id = order.get('id', 'Unknown')[:8]
            symbol = order.get('underlying_symbol', 'Unknown')
            strategy = order.get('strategy', 'Unknown')
            limit_price = order.get('limit_price') or order.get('expected_credit', 0)
            order_id = order.get('order_id', 'N/A')
            order_status = order.get('order_status', 'working')

            lines.append(f"  [{trade_id}] {symbol} {strategy}")
            lines.append(f"       Limit: ${limit_price:.2f} | Order ID: {order_id[:8] if order_id else 'N/A'}")
            lines.append(f"       Status: {order_status.upper()}")
            lines.append("")

        lines.append("Use 'modify price <id> <new_price>' to adjust limit price")
        lines.append("Use 'cancel order <id>' to cancel the order")
        return "\n".join(lines)

    def format_price_confirmation(self, trade: Dict[str, Any]) -> str:
        """Format limit price confirmation before execution"""
        trade_id = trade.get('id', 'Unknown')[:8]
        symbol = trade.get('underlying_symbol', 'Unknown')
        strategy = trade.get('strategy', 'Unknown')
        credit = trade.get('expected_credit', 0)
        limit_price = trade.get('limit_price') or credit

        lines = [
            "=" * 45,
            "        CONFIRM LIMIT ORDER PRICE",
            "=" * 45,
            "",
            f"Trade:     [{trade_id}] {symbol} {strategy}",
            "",
            f"LIMIT PRICE: ${limit_price:.2f}",
            "",
        ]

        if limit_price != credit:
            lines.append(f"(Expected credit was ${credit:.2f})")
            lines.append("")

        lines.extend([
            "This order will execute at or better than the limit price.",
            "",
            "Commands:",
            "  'execute' or 'confirm' - Submit the order",
            "  'set price <amount>'   - Change the limit price",
            "  'cancel'               - Cancel without executing",
        ])

        return "\n".join(lines)

    def format_iv_rank(self, symbol: str, iv_rank: float, threshold: float = 30.0) -> str:
        """Format IV rank information"""
        status = "ELEVATED" if iv_rank >= threshold else "LOW"
        recommendation = (
            "Above tastylive threshold - suitable for selling premium."
            if iv_rank >= threshold else
            "Below tastylive threshold - not ideal for selling premium."
        )

        return f"{symbol} IV Rank: {iv_rank:.1f}% ({status})\n{recommendation}"

    def format_research_result(self, research: Dict[str, Any]) -> str:
        """Format research/analysis results"""
        symbol = research.get('symbol', 'Unknown')
        strategy = research.get('strategy', 'Unknown')

        lines = [
            f"Research: {strategy} on {symbol}",
            "=" * 40
        ]

        if research.get('put_side'):
            put = research['put_side']
            lines.append(f"Put side:  Sell {put.get('short_strike')}P / Buy {put.get('long_strike')}P")

        if research.get('call_side'):
            call = research['call_side']
            lines.append(f"Call side: Sell {call.get('short_strike')}C / Buy {call.get('long_strike')}C")

        metrics = research.get('metrics', {})
        if metrics:
            lines.extend([
                "",
                f"Net credit: ${metrics.get('net_credit', 0):.2f}",
                f"Max loss:   ${metrics.get('max_loss', 0):.2f}",
                f"P.O.P:      {metrics.get('pop', 0):.1%}"
            ])

        if research.get('rationale'):
            lines.extend(["", "Analysis:", research['rationale']])

        return "\n".join(lines)

    def format_management_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Format position management recommendations"""
        if not actions:
            return "No positions require management action at this time."

        lines = ["Position Management Actions", "-" * 40]

        for action in actions:
            symbol = action.get('symbol', 'Unknown')
            rec_action = action.get('action', 'MONITOR')
            reason = action.get('reasoning', '')
            urgency = action.get('urgency', 'normal')

            urgency_marker = "[!]" if urgency in ('high', 'critical') else ""
            lines.append(f"  {urgency_marker}{symbol}: {rec_action}")
            if reason:
                lines.append(f"       {reason}")
            lines.append("")

        return "\n".join(lines)

    def format_risk_params(self, params: Dict[str, Any]) -> str:
        """Format risk parameters for display"""
        lines = [
            "Risk Parameters",
            "=" * 40,
            "",
            "Loss Limits:",
            f"  Max Daily Loss:    ${params.get('max_daily_loss', 0):,.2f}",
            f"  Max Weekly Loss:   ${params.get('max_weekly_loss', 0):,.2f}",
            f"  Max Position Loss: ${params.get('max_position_loss', 0):,.2f}",
            "",
            "Position Sizing:",
            f"  Max Position Size: {params.get('max_position_size_percent', 0):.1%} of portfolio",
            f"  Max Positions:     {params.get('max_total_positions', 0)}",
            "",
            "Entry Criteria:",
            f"  Min IV Rank:       {params.get('min_iv_rank', 0)}%",
            f"  Target DTE:        {params.get('target_dte', 45)} days",
            "",
            "Management:",
            f"  Profit Target:     {params.get('profit_target_percent', 0):.0%}",
            f"  Management DTE:    {params.get('management_dte', 21)} days"
        ]

        return "\n".join(lines)

    def format_trade_confirmation(
        self,
        trade_id: str,
        action: str,
        success: bool,
        sandbox: bool = False
    ) -> str:
        """Format trade action confirmation"""
        mode = "[SANDBOX] " if sandbox else ""
        status = "successfully" if success else "failed"

        if action == "approve":
            if success:
                return f"{mode}Trade {trade_id} approved. Ready for execution."
            return f"{mode}Failed to approve trade {trade_id}."

        elif action == "reject":
            if success:
                return f"{mode}Trade {trade_id} rejected and removed from pending."
            return f"{mode}Failed to reject trade {trade_id}."

        elif action == "execute":
            if success:
                return f"{mode}Trade {trade_id} executed {status}."
            return f"{mode}Trade {trade_id} execution failed."

        return f"{mode}Trade {trade_id}: {action} {status}."

    def format_scan_results(
        self,
        opportunities: List[Dict[str, Any]],
        symbols_scanned: List[str]
    ) -> str:
        """Format scan results"""
        data_source = "[MOCK DATA] " if True else ""  # Determined by context

        if not opportunities:
            return f"Scanned {', '.join(symbols_scanned)} - no opportunities found meeting criteria."

        lines = [
            f"Found {len(opportunities)} opportunity(ies):",
            ""
        ]

        for opp in opportunities:
            trade_id = opp.get('id', 'Unknown')[:8]
            symbol = opp.get('underlying_symbol', 'Unknown')
            strategy = opp.get('strategy', 'Unknown')
            iv_rank = opp.get('iv_rank', 0)
            delta = opp.get('delta', 0)
            dte = opp.get('dte', 0)
            credit = opp.get('expected_credit', 0)
            max_loss = opp.get('max_loss', 0)

            lines.append(f"  [{trade_id}] {symbol} {strategy}")
            lines.append(f"       IV: {iv_rank:.1f}% | Delta: {delta:.2f} | DTE: {dte}")
            lines.append(f"       Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}")
            lines.append("")

        lines.append("Use 'pending' to see details, 'approve <id>' to approve.")
        return "\n".join(lines)

    def format_help(self) -> str:
        """Format help message"""
        return """Trading Bot Commands
====================

Scanning & Research:
  "Find opportunities on SPY, QQQ"
  "What's the IV rank on NVDA?"
  "Research iron condors on IWM"

Trade Management:
  "Show pending trades"
  "Approve trade abc123"
  "Reject trade abc123"
  "Execute the approved trade"

Order Management:
  "Show working orders"     - View unfilled limit orders
  "Set price 2.50"          - Modify limit price before execution
  "Check order status"      - Check if order is filled
  "Cancel order abc123"     - Cancel an unfilled order

Portfolio:
  "What's my portfolio delta?"
  "Show my positions"
  "How much buying power do I have?"
  "What's my P&L today?"

Position Management:
  "Check if I should close anything"
  "Analyze my TSLA position"
  "Close SPY" - Close a position (with confirmation)
  "Roll SPY" - Roll a position to new expiration

Other:
  "Show risk parameters"
  "How's the market looking?"
  "help" - Show this message

The bot follows tastylive methodology:
- Sell premium when IV Rank > 30%
- Target 45 DTE entries
- 30 delta standard, 16 delta conservative
- Take profit at 50%
- Manage at 21 DTE

All orders execute as LIMIT orders. You can confirm
and modify the limit price before execution."""

    def format_error(self, message: str) -> str:
        """Format error message"""
        return f"Error: {message}"

    def format_not_found(self, item_type: str, identifier: str) -> str:
        """Format not found message"""
        return f"{item_type} '{identifier}' not found."

    def format_alert(self, alert: Any) -> str:
        """
        Format a management alert for display.

        Args:
            alert: ManagementAlert object

        Returns:
            Formatted alert string with urgency marker
        """
        # Get urgency marker
        urgency_marker = alert.urgency_marker() if hasattr(alert, 'urgency_marker') else ""

        # Get action string
        action = alert.action.value if hasattr(alert.action, 'value') else str(alert.action)

        lines = [
            f"{urgency_marker} {alert.underlying}: {action}",
            f"   Reason: {alert.reason}"
        ]

        # Add metrics
        metrics = []
        if alert.pnl_percent != 0:
            metrics.append(f"P&L: {alert.pnl_percent:.1%}")
        if alert.dte < 999:
            metrics.append(f"DTE: {alert.dte}")
        if alert.is_tested:
            metrics.append("TESTED")

        if metrics:
            lines.append(f"   {' | '.join(metrics)}")

        # Add Claude reasoning if available
        if alert.claude_reasoning:
            lines.append(f"   AI: {alert.claude_reasoning[:100]}...")

        return "\n".join(lines)

    def format_close_confirmation(self, position_data: Dict[str, Any]) -> str:
        """Format close position confirmation"""
        symbol = position_data.get('symbol', 'Unknown')
        positions = position_data.get('positions', [])

        lines = [
            f"Close Position: {symbol}",
            "-" * 40
        ]

        for pos in positions:
            opt_type = pos.get('option_type', 'OPT')
            strike = pos.get('strike', 0)
            qty = pos.get('quantity', 0)
            dte = pos.get('dte', 0)
            pnl_pct = pos.get('pnl_percent', 0)
            direction = "SHORT" if qty < 0 else "LONG"

            lines.append(f"  {direction} {abs(qty)}x {opt_type} @ ${strike:.0f}")
            lines.append(f"       DTE: {dte} | P&L: {pnl_pct:.1%}")

        return "\n".join(lines)

    def format_roll_confirmation(
        self,
        position_data: Dict[str, Any],
        target_dte: int = 45
    ) -> str:
        """Format roll position confirmation"""
        symbol = position_data.get('symbol', 'Unknown')
        positions = position_data.get('positions', [])

        lines = [
            f"Roll Position: {symbol} to ~{target_dte} DTE",
            "-" * 40
        ]

        for pos in positions:
            opt_type = pos.get('option_type', 'OPT')
            strike = pos.get('strike', 0)
            qty = pos.get('quantity', 0)
            dte = pos.get('dte', 0)
            direction = "SHORT" if qty < 0 else "LONG"

            lines.append(f"  Current: {direction} {abs(qty)}x {opt_type} @ ${strike:.0f} | DTE: {dte}")
            lines.append(f"  New:     {direction} {abs(qty)}x {opt_type} @ ${strike:.0f} | DTE: ~{target_dte}")

        lines.extend([
            "",
            "Note: Roll requires minimum $0.25 credit"
        ])

        return "\n".join(lines)
