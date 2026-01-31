#!/usr/bin/env python3
"""
Rich Display Layer for Tastytrade Bot

Centralized display formatting with Rich library support.
Can be disabled for debugging, logging, or testing.

Usage:
    display = RichDisplay(enabled=True)
    display.render_scan_results(opportunities, symbols)

Disable Rich:
    RICH_DISABLED=1 python3 chatbot.py

    Or at runtime:
    display.toggle(False)

Author: Trading Bot
License: MIT
"""

import os
from typing import List, Dict, Any, Optional
from decimal import Decimal

# Rich imports - gracefully handle if not installed
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class Styles:
    """Centralized style constants for consistent theming."""
    # P&L colors
    PROFIT = "green bold"
    LOSS = "red"
    NEUTRAL = "white"

    # Element styles
    SYMBOL = "cyan bold"
    TRADE_ID = "yellow"
    DIM = "dim"
    HEADER = "bold white"
    SUBHEADER = "bold"

    # IV Rank thresholds (tastylive methodology)
    IV_HIGH = "green"       # >= 40% (excellent)
    IV_MED = "yellow"       # 30-40% (acceptable)
    IV_LOW = "red"          # < 30% (avoid)

    # Probability of Profit thresholds
    POP_HIGH = "green"      # >= 70%
    POP_MED = "yellow"      # 60-70%
    POP_LOW = "red"         # < 60%

    # DTE thresholds (tastylive: 45 DTE target)
    DTE_GOOD = "green"      # 35-50
    DTE_OK = "yellow"       # outside sweet spot

    # Urgency markers
    CRITICAL = "red bold"
    HIGH = "yellow bold"
    NORMAL = "white"
    LOW = "dim"

    # Pass/Fail indicators
    PASS = "green"
    FAIL = "red"


class RichDisplay:
    """
    Centralized display layer with Rich formatting.

    Can be disabled for:
    - Debugging (see plain output)
    - Logging to files
    - Piping output
    - Testing

    Attributes:
        enabled: Whether Rich formatting is active
        default_layout: Default layout for scan results ("auto", "table", "cards", "compact")
    """

    def __init__(
        self,
        enabled: bool = None,
        default_layout: str = "auto"
    ):
        """
        Initialize RichDisplay.

        Args:
            enabled: Enable Rich formatting. If None, checks RICH_DISABLED env var.
            default_layout: Default layout for multi-item displays.
        """
        # Check environment variable if not explicitly set
        if enabled is None:
            env_disabled = os.environ.get("RICH_DISABLED", "0") == "1"
            enabled = RICH_AVAILABLE and not env_disabled

        # Force disable if Rich not installed
        if not RICH_AVAILABLE:
            enabled = False

        self._enabled = enabled
        self.default_layout = default_layout
        self._console = Console() if enabled else None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def console(self) -> Optional[Console]:
        return self._console

    def toggle(self, enabled: bool = None) -> str:
        """
        Toggle Rich on/off at runtime.

        Args:
            enabled: Explicit state, or None to toggle current state.

        Returns:
            Status message.
        """
        if not RICH_AVAILABLE and enabled:
            return "Rich library not installed. Run: pip install rich"

        if enabled is None:
            enabled = not self._enabled

        self._enabled = enabled
        self._console = Console() if enabled else None
        return f"Display mode: {'Rich' if enabled else 'Plain text'}"

    def set_layout(self, layout: str) -> str:
        """
        Change default layout.

        Args:
            layout: One of "auto", "table", "cards", "compact"

        Returns:
            Status message.
        """
        valid = ["auto", "table", "cards", "compact"]
        if layout not in valid:
            return f"Invalid layout. Choose from: {', '.join(valid)}"
        self.default_layout = layout
        return f"Default layout: {layout}"

    def print(self, content: str) -> None:
        """Print content using Rich console or plain print."""
        if self._enabled and self._console:
            self._console.print(content)
        else:
            print(content)

    # =========================================================================
    # STYLE HELPERS
    # =========================================================================

    def _iv_style(self, iv_rank: float) -> str:
        """Get style for IV rank value."""
        if iv_rank >= 40:
            return Styles.IV_HIGH
        elif iv_rank >= 30:
            return Styles.IV_MED
        return Styles.IV_LOW

    def _pop_style(self, pop: float) -> str:
        """Get style for probability of profit."""
        if pop >= 0.70:
            return Styles.POP_HIGH
        elif pop >= 0.60:
            return Styles.POP_MED
        return Styles.POP_LOW

    def _pnl_style(self, pnl: float) -> str:
        """Get style for P&L value."""
        if pnl > 0:
            return Styles.PROFIT
        elif pnl < 0:
            return Styles.LOSS
        return Styles.NEUTRAL

    def _dte_style(self, dte: int) -> str:
        """Get style for DTE value."""
        if 35 <= dte <= 50:
            return Styles.DTE_GOOD
        return Styles.DTE_OK

    def _format_strategy(self, strategy: str) -> str:
        """Format strategy name for display."""
        return strategy.replace('_', ' ').title()

    def _format_currency(self, value: float, show_sign: bool = False) -> str:
        """Format currency value."""
        if show_sign:
            return f"${value:+,.2f}"
        return f"${value:,.2f}"

    # =========================================================================
    # SCAN RESULTS
    # =========================================================================

    def render_scan_results(
        self,
        opportunities: List[Dict[str, Any]],
        symbols: List[str],
        layout: str = None,
        data_source: str = ""
    ) -> str:
        """
        Render scan results with specified layout.

        Args:
            opportunities: List of opportunity dicts from scan
            symbols: List of symbols that were scanned
            layout: "auto", "table", "cards", or "compact"
            data_source: Prefix like "[MOCK DATA] " or ""

        Returns:
            Formatted string (also prints if Rich enabled)
        """
        layout = layout or self.default_layout

        if not self._enabled:
            return self._plain_scan_results(opportunities, symbols, data_source)

        # Auto-select layout based on result count
        if layout == "auto":
            if len(opportunities) == 0:
                layout = "compact"
            elif len(opportunities) <= 3:
                layout = "cards"
            else:
                layout = "table"

        if layout == "table":
            return self._rich_scan_table(opportunities, symbols, data_source)
        elif layout == "cards":
            return self._rich_scan_cards(opportunities, symbols, data_source)
        else:
            return self._rich_scan_compact(opportunities, symbols, data_source)

    def _plain_scan_results(
        self,
        opportunities: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """Plain text fallback for scan results."""
        if not opportunities:
            return f"{data_source}Scanned {', '.join(symbols)} - no opportunities found meeting criteria."

        lines = [
            f"{data_source}Found {len(opportunities)} opportunity(ies):",
            ""
        ]

        for idx, opp in enumerate(opportunities, 1):
            trade_id = opp.get('id', 'Unknown')[:8]
            symbol = opp.get('underlying_symbol', 'Unknown')
            strategy = opp.get('strategy', 'Unknown')
            iv_rank = opp.get('iv_rank', 0)
            delta = opp.get('delta', 0)
            dte = opp.get('dte', 0)
            credit = opp.get('expected_credit', 0)
            max_loss = opp.get('max_loss', 0)
            pop = opp.get('probability_of_profit', 0)

            lines.append(f"  {idx}. [{trade_id}] {symbol} {strategy}")
            lines.append(f"       IV: {iv_rank:.1f}% | Delta: {delta:.2f} | DTE: {dte} | POP: {pop:.0%}")
            lines.append(f"       Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}")
            lines.append("")

        lines.append("Use 'select #' to create trade, 'approve <id>' to approve.")
        return "\n".join(lines)

    def _rich_scan_table(
        self,
        opportunities: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """Rich table layout for scan results."""
        if not opportunities:
            self._console.print(f"[dim]{data_source}Scanned {', '.join(symbols)} - no opportunities found[/dim]")
            return ""

        # Header
        self._console.print()
        self._console.print(f"[bold green]{data_source}Found {len(opportunities)} opportunities[/bold green]")
        self._console.print()

        # Create table
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
        )

        # Define columns
        table.add_column("#", style="bold white", width=3, justify="center")
        table.add_column("ID", style=Styles.TRADE_ID, width=8)
        table.add_column("Symbol", style=Styles.SYMBOL, width=6)
        table.add_column("Strategy", width=18)
        table.add_column("IV Rank", justify="right", width=8)
        table.add_column("Delta", justify="right", width=7)
        table.add_column("DTE", justify="right", width=5)
        table.add_column("Credit", justify="right", width=8)
        table.add_column("Max Loss", justify="right", width=9)
        table.add_column("P.O.P", justify="right", width=6)

        for idx, opp in enumerate(opportunities, 1):
            trade_id = opp.get('id', 'Unknown')[:8]
            symbol = opp.get('underlying_symbol', 'Unknown')
            strategy = self._format_strategy(opp.get('strategy', 'Unknown'))
            iv_rank = float(opp.get('iv_rank', 0))
            delta = float(opp.get('delta', 0))
            dte = int(opp.get('dte', 0))
            credit = float(opp.get('expected_credit', 0))
            max_loss = float(opp.get('max_loss', 0))
            pop = float(opp.get('probability_of_profit', 0))

            # Color coding
            iv_style = self._iv_style(iv_rank)
            pop_style = self._pop_style(pop)
            dte_style = self._dte_style(dte)
            max_loss_str = f"${max_loss:,.0f}" if max_loss > 0 else "[dim]undef[/dim]"

            table.add_row(
                str(idx),
                trade_id,
                symbol,
                strategy,
                f"[{iv_style}]{iv_rank:.1f}%[/{iv_style}]",
                f"{delta:+.2f}",
                f"[{dte_style}]{dte}[/{dte_style}]",
                f"[green]${credit:.2f}[/green]",
                max_loss_str,
                f"[{pop_style}]{pop:.0%}[/{pop_style}]",
            )

        self._console.print(table)
        self._console.print()
        self._console.print("[dim]Commands: 'select #' to create trade, 'approve <id>' after review[/dim]")

        return ""  # Already printed

    def _rich_scan_cards(
        self,
        opportunities: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """Rich cards layout for scan results."""
        if not opportunities:
            self._console.print(f"[dim]{data_source}Scanned {', '.join(symbols)} - no opportunities found[/dim]")
            return ""

        self._console.print()
        self._console.print(f"[bold green]{data_source}Found {len(opportunities)} opportunities[/bold green]")

        for idx, opp in enumerate(opportunities, 1):
            trade_id = opp.get('id', 'Unknown')[:8]
            symbol = opp.get('underlying_symbol', 'Unknown')
            strategy = self._format_strategy(opp.get('strategy', 'Unknown'))
            iv_rank = float(opp.get('iv_rank', 0))
            delta = float(opp.get('delta', 0))
            dte = int(opp.get('dte', 0))
            credit = float(opp.get('expected_credit', 0))
            max_loss = float(opp.get('max_loss', 0))
            pop = float(opp.get('probability_of_profit', 0))
            theta = float(opp.get('theta', 0))

            # Color coding
            iv_style = self._iv_style(iv_rank)
            pop_style = self._pop_style(pop)

            # Calculate return on risk
            ror = (credit * 100 / max_loss * 100) if max_loss > 0 else 0

            # Build strikes text based on strategy
            strikes_text = self._build_strikes_text(opp)

            # Build card content
            content = Text()
            content.append("Strategy: ", style="dim")
            content.append(f"{strategy}\n", style="bold")

            if strikes_text:
                content.append(f"{strikes_text}\n", style="dim")

            content.append("\n")

            # Metrics row 1
            content.append("IV Rank: ", style="dim")
            content.append(f"{iv_rank:.1f}%", style=iv_style)
            content.append("  |  ", style="dim")
            content.append("Delta: ", style="dim")
            content.append(f"{delta:+.2f}", style="white")
            content.append("  |  ", style="dim")
            content.append("DTE: ", style="dim")
            content.append(f"{dte}\n", style="white")

            # Metrics row 2
            content.append("Credit: ", style="dim")
            content.append(f"${credit:.2f}", style="green bold")
            content.append("  |  ", style="dim")
            content.append("Max Loss: ", style="dim")
            if max_loss > 0:
                content.append(f"${max_loss:,.0f}", style="red")
            else:
                content.append("undefined", style="dim italic")
            content.append("  |  ", style="dim")
            content.append("Theta: ", style="dim")
            content.append(f"+${theta:.3f}/day\n", style="cyan")

            # Metrics row 3
            content.append("\nP.O.P: ", style="dim")
            content.append(f"{pop:.0%}", style=pop_style + " bold")
            if max_loss > 0:
                content.append("  |  ", style="dim")
                content.append("Return on Risk: ", style="dim")
                content.append(f"{ror:.1f}%", style="cyan")

            # Create panel
            panel = Panel(
                content,
                title=f"[bold white]#{idx}[/bold white] [{Styles.TRADE_ID}]{trade_id}[/{Styles.TRADE_ID}] [{Styles.SYMBOL}]{symbol}[/{Styles.SYMBOL}]",
                title_align="left",
                border_style="blue",
                padding=(0, 1),
            )
            self._console.print(panel)

        self._console.print()
        self._console.print("[dim]Commands: 'select #' to create trade, 'approve <id>' after review[/dim]")

        return ""  # Already printed

    def _rich_scan_compact(
        self,
        opportunities: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """Rich compact layout for scan results."""
        if not opportunities:
            self._console.print(f"[dim]{data_source}Scanned {', '.join(symbols)} - no opportunities found[/dim]")
            return ""

        # Summary header
        header_text = Text()
        header_text.append("SCAN RESULTS", style="bold white")
        header_text.append("  |  ", style="dim")
        header_text.append(f"{len(opportunities)} opportunities", style="green")
        header_text.append("  |  ", style="dim")
        header_text.append(f"{data_source}Symbols: {', '.join(symbols)}", style="dim")

        self._console.print()
        self._console.print(Panel(header_text, box=box.HEAVY, style="dim"))

        for idx, opp in enumerate(opportunities, 1):
            trade_id = opp.get('id', 'Unknown')[:8]
            symbol = opp.get('underlying_symbol', 'Unknown')
            strategy = self._format_strategy(opp.get('strategy', 'Unknown'))
            iv_rank = float(opp.get('iv_rank', 0))
            credit = float(opp.get('expected_credit', 0))
            pop = float(opp.get('probability_of_profit', 0))
            dte = int(opp.get('dte', 0))

            # Color coding
            iv_style = self._iv_style(iv_rank)
            pop_style = self._pop_style(pop)

            # One-line summary
            line = Text()
            line.append(f" {idx}. ", style="bold white")
            line.append(f"[{trade_id}] ", style=Styles.TRADE_ID + " dim")
            line.append(f"{symbol:5}", style=Styles.SYMBOL)
            line.append(f" {strategy:20}", style="white")
            line.append("  IV:", style="dim")
            line.append(f"{iv_rank:5.1f}%", style=iv_style)
            line.append("  Credit:", style="dim")
            line.append(f"${credit:6.2f}", style="green bold")
            line.append("  POP:", style="dim")
            line.append(f"{pop:4.0%}", style=pop_style)
            line.append("  DTE:", style="dim")
            line.append(f"{dte:3}", style="white")

            self._console.print(line)

        self._console.print()
        self._console.print("[dim]" + "-" * 70 + "[/dim]")
        self._console.print("[dim]'select #' to create trade  |  'approve <id>' to approve[/dim]")

        return ""  # Already printed

    def _build_strikes_text(self, opp: Dict[str, Any]) -> str:
        """Build strikes description based on strategy type."""
        # Vertical spreads
        if 'short_strike' in opp and 'long_strike' in opp:
            return f"Strikes: {opp['short_strike']}/{opp['long_strike']}"

        # Iron condors
        if 'put_short_strike' in opp and 'call_short_strike' in opp:
            return f"Put: {opp['put_short_strike']}/{opp.get('put_long_strike', 'N/A')} | Call: {opp['call_short_strike']}/{opp.get('call_long_strike', 'N/A')}"

        # Strangles/Straddles (naked)
        if 'short_put_strike' in opp and 'short_call_strike' in opp:
            return f"Put: {opp['short_put_strike']} | Call: {opp['short_call_strike']}"

        return ""

    # =========================================================================
    # FILTERED CANDIDATES (risk criteria failed)
    # =========================================================================

    def render_filtered_candidates(
        self,
        candidates: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """
        Render filtered candidates that didn't pass risk criteria.

        Args:
            candidates: List of candidate dicts from scan
            symbols: List of symbols that were scanned
            data_source: Prefix like "[MOCK DATA] " or ""

        Returns:
            Formatted string (also prints if Rich enabled)
        """
        if not self._enabled:
            return self._plain_filtered_candidates(candidates, symbols, data_source)
        return self._rich_filtered_candidates(candidates, symbols, data_source)

    def _plain_filtered_candidates(
        self,
        candidates: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """Plain text fallback for filtered candidates."""
        if not candidates:
            return f"{data_source}Scanned {', '.join(symbols)} - no candidates found."

        lines = [
            f"{data_source}Found {len(candidates)} potential strateg{'y' if len(candidates) == 1 else 'ies'}:",
            ""
        ]

        for candidate in candidates:
            opt_num = candidate.get('option_number', 0)
            strategy = candidate.get('strategy', 'Unknown')
            symbol = candidate.get('underlying_symbol', 'Unknown')
            credit = float(candidate.get('expected_credit', 0))
            max_loss = float(candidate.get('max_loss', 0))
            pop = float(candidate.get('probability_of_profit', 0))
            kelly = float(candidate.get('kelly_score', 0))

            lines.append(
                f"{opt_num}. {strategy} on {symbol}\n"
                f"   Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}\n"
                f"   POP: {pop:.0%} | Kelly: {kelly:.3f}"
            )

        lines.append("")
        lines.append("These were filtered by risk criteria. To create one anyway, say 'option [number]'.")
        return "\n".join(lines)

    def _rich_filtered_candidates(
        self,
        candidates: List[Dict[str, Any]],
        symbols: List[str],
        data_source: str = ""
    ) -> str:
        """Rich display for filtered candidates."""
        if not candidates:
            self._console.print(f"[dim]{data_source}Scanned {', '.join(symbols)} - no candidates found[/dim]")
            return ""

        # Header with warning
        self._console.print()
        self._console.print(f"[bold yellow]{data_source}Found {len(candidates)} potential strateg{'y' if len(candidates) == 1 else 'ies'} (filtered by risk)[/bold yellow]")
        self._console.print()

        # Create table
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
            border_style="yellow dim",
            expand=False,
        )

        # Define columns
        table.add_column("#", style="bold white", width=3, justify="center")
        table.add_column("Symbol", style=Styles.SYMBOL, width=6)
        table.add_column("Strategy", width=18)
        table.add_column("Credit", justify="right", width=8)
        table.add_column("Max Loss", justify="right", width=9)
        table.add_column("P.O.P", justify="right", width=6)
        table.add_column("Kelly", justify="right", width=8)

        for candidate in candidates:
            opt_num = candidate.get('option_number', 0)
            symbol = candidate.get('underlying_symbol', 'Unknown')
            strategy = self._format_strategy(candidate.get('strategy', 'Unknown'))
            credit = float(candidate.get('expected_credit', 0))
            max_loss = float(candidate.get('max_loss', 0))
            pop = float(candidate.get('probability_of_profit', 0))
            kelly = float(candidate.get('kelly_score', 0))

            pop_style = self._pop_style(pop)
            kelly_style = "green" if kelly > 0 else "red" if kelly < -0.1 else "yellow"

            table.add_row(
                str(opt_num),
                symbol,
                strategy,
                f"[green]${credit:.2f}[/green]",
                f"${max_loss:,.0f}",
                f"[{pop_style}]{pop:.0%}[/{pop_style}]",
                f"[{kelly_style}]{kelly:.3f}[/{kelly_style}]",
            )

        self._console.print(table)
        self._console.print()
        self._console.print("[yellow]These were filtered by risk criteria.[/yellow]")
        self._console.print("[dim]To create one anyway, say 'option [number]'.[/dim]")

        return ""  # Already printed

    # =========================================================================
    # TRADE APPROVAL
    # =========================================================================

    def render_trade_approval(self, trade: Dict[str, Any]) -> str:
        """
        Render trade approval screen.

        Args:
            trade: Trade proposal dict

        Returns:
            Formatted string
        """
        if not self._enabled:
            return self._plain_trade_approval(trade)
        return self._rich_trade_approval(trade)

    def _plain_trade_approval(self, trade: Dict[str, Any]) -> str:
        """Plain text trade approval screen."""
        trade_id = trade.get('id', 'Unknown')[:8]
        symbol = trade.get('underlying_symbol', 'Unknown')
        strategy = trade.get('strategy', 'Unknown')
        iv_rank = float(trade.get('iv_rank', 0))
        delta = float(trade.get('delta', 0))
        dte = int(trade.get('dte', 0))
        credit = float(trade.get('expected_credit', 0))
        limit_price = float(trade.get('limit_price') or credit)
        max_loss = float(trade.get('max_loss', 0))
        pop = float(trade.get('probability_of_profit', 0))
        theta = float(trade.get('theta', 0))
        legs = trade.get('legs', [])
        status = trade.get('status', 'pending')
        order_status = trade.get('order_status')

        # Entry criteria checks
        iv_pass = "PASS" if iv_rank >= 30 else "FAIL"
        dte_pass = "PASS" if 30 <= dte <= 60 else "FAIL"
        pop_pass = "PASS" if pop >= 0.50 else "FAIL"

        lines = [
            f"Trade Proposal [{trade_id}]",
            "=" * 50,
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
            "Entry Criteria:",
            f"  [{iv_pass}] IV Rank >= 30%: {iv_rank:.1f}%",
            f"  [{dte_pass}] DTE 30-60: {dte} days",
            f"  [{pop_pass}] P.O.P >= 50%: {pop:.0%}",
            "",
            "Pricing:",
            f"  Expected Credit: ${credit:.2f}",
            f"  LIMIT PRICE:     ${limit_price:.2f}",
            "",
            "Risk:",
            f"  Max Loss: ${max_loss:.2f}",
            f"  Delta:    {delta:+.3f}",
            f"  Theta:    +${theta:.3f}/day",
            "",
            "Management Plan:",
            f"  Take profit at 50%: ${credit * 0.5:.2f}",
            f"  Roll/close at 21 DTE",
            f"  Stop loss at 2x: ${credit * 2:.2f}",
            "",
            "Commands: 'approve' | 'reject' | 'set price <amount>'"
        ])

        return "\n".join(lines)

    def _rich_trade_approval(self, trade: Dict[str, Any]) -> str:
        """Rich trade approval screen."""
        trade_id = trade.get('id', 'Unknown')[:8]
        symbol = trade.get('underlying_symbol', 'Unknown')
        strategy = self._format_strategy(trade.get('strategy', 'Unknown'))
        iv_rank = float(trade.get('iv_rank', 0))
        delta = float(trade.get('delta', 0))
        dte = int(trade.get('dte', 0))
        credit = float(trade.get('expected_credit', 0))
        limit_price = float(trade.get('limit_price') or credit)
        max_loss = float(trade.get('max_loss', 0))
        pop = float(trade.get('probability_of_profit', 0))
        theta = float(trade.get('theta', 0))

        # Entry criteria checks
        iv_pass = iv_rank >= 30
        dte_pass = 30 <= dte <= 60
        pop_pass = pop >= 0.50

        # Build checklist
        checklist = Text()
        checklist.append("Entry Criteria\n", style="bold underline")
        checklist.append("  ")
        checklist.append("+" if iv_pass else "x", style=Styles.PASS if iv_pass else Styles.FAIL)
        checklist.append(f" IV Rank >= 30%: {iv_rank:.1f}%\n")
        checklist.append("  ")
        checklist.append("+" if dte_pass else "x", style=Styles.PASS if dte_pass else Styles.FAIL)
        checklist.append(f" DTE 30-60: {dte} days\n")
        checklist.append("  ")
        checklist.append("+" if pop_pass else "x", style=Styles.PASS if pop_pass else Styles.FAIL)
        checklist.append(f" P.O.P >= 50%: {pop:.0%}\n")

        # Build pricing/metrics
        metrics = Text()
        metrics.append("Trade Details\n", style="bold underline")
        metrics.append(f"  Credit:      ", style="dim")
        metrics.append(f"${credit:.2f}\n", style="green")
        metrics.append(f"  Limit Price: ", style="dim")
        metrics.append(f"${limit_price:.2f}\n", style="green bold")
        metrics.append(f"  Max Loss:    ", style="dim")
        metrics.append(f"${max_loss:,.0f}\n", style="red" if max_loss > 0 else "dim")
        metrics.append(f"  Delta:       {delta:+.3f}\n")
        metrics.append(f"  Theta:       ", style="dim")
        metrics.append(f"+${theta:.3f}/day\n", style="cyan")

        # Management plan
        management = Text()
        management.append("Management Plan\n", style="bold underline")
        management.append(f"  Take profit 50%: ", style="dim")
        management.append(f"${credit * 0.5:.2f}\n", style="green")
        management.append(f"  Roll/close at:   ", style="dim")
        management.append(f"21 DTE\n", style="yellow")
        management.append(f"  Stop loss 2x:    ", style="dim")
        management.append(f"${credit * 2:.2f}\n", style="red")

        # Combine into columns
        columns = Columns([checklist, metrics, management], equal=True, expand=True)

        # Wrap in panel
        title = f"[bold white]TRADE APPROVAL[/bold white]  |  [{Styles.TRADE_ID}]{trade_id}[/{Styles.TRADE_ID}] [{Styles.SYMBOL}]{symbol}[/{Styles.SYMBOL}] {strategy}"
        panel = Panel(
            columns,
            title=title,
            title_align="left",
            border_style="yellow",
            padding=(1, 2),
        )

        self._console.print()
        self._console.print(panel)
        self._console.print()
        self._console.print("[bold yellow]Commands:[/bold yellow] 'approve' | 'reject' | 'set price <amount>'")

        return ""  # Already printed

    # =========================================================================
    # PORTFOLIO SUMMARY
    # =========================================================================

    def render_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """
        Render portfolio summary.

        Args:
            portfolio: Portfolio state dict

        Returns:
            Formatted string
        """
        if not self._enabled:
            return self._plain_portfolio(portfolio)
        return self._rich_portfolio(portfolio)

    def _plain_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """Plain text portfolio summary."""
        nlv = portfolio.get('net_liquidating_value', 0)
        bp = portfolio.get('buying_power', 0)
        delta = portfolio.get('portfolio_delta', 0)
        theta = portfolio.get('portfolio_theta', 0)
        vega = portfolio.get('portfolio_vega', 0)
        gamma = portfolio.get('portfolio_gamma', 0)
        positions = portfolio.get('open_positions', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)
        weekly_pnl = portfolio.get('weekly_pnl', 0)

        lines = [
            "Portfolio Summary",
            "-" * 40,
            f"Net Liquidating Value: ${nlv:,.2f}",
            f"Buying Power:          ${bp:,.2f}",
            f"Open Positions:        {positions}",
            "",
            "Greeks (Beta-Weighted to SPY):",
            f"  Delta: {delta:+.1f}",
            f"  Theta: {theta:+.2f}",
            f"  Vega:  {vega:+.2f}",
            f"  Gamma: {gamma:+.3f}",
            "",
            f"Today's P&L:  ${daily_pnl:+,.2f}",
            f"Week's P&L:   ${weekly_pnl:+,.2f}",
        ]

        return "\n".join(lines)

    def _rich_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """Rich portfolio summary."""
        nlv = float(portfolio.get('net_liquidating_value', 0))
        bp = float(portfolio.get('buying_power', 0))
        delta = float(portfolio.get('portfolio_delta', 0))
        theta = float(portfolio.get('portfolio_theta', 0))
        vega = float(portfolio.get('portfolio_vega', 0))
        gamma = float(portfolio.get('portfolio_gamma', 0))
        positions = int(portfolio.get('open_positions', 0))
        daily_pnl = float(portfolio.get('daily_pnl', 0))
        weekly_pnl = float(portfolio.get('weekly_pnl', 0))

        # Account section
        account = Text()
        account.append("Account\n", style="bold underline")
        account.append(f"  NLV:      ", style="dim")
        account.append(f"${nlv:,.2f}\n", style="bold white")
        account.append(f"  BP:       ", style="dim")
        account.append(f"${bp:,.2f}\n", style="cyan")
        account.append(f"  Positions: {positions}\n")

        # Greeks section
        greeks = Text()
        greeks.append("Greeks (Beta-Weighted)\n", style="bold underline")
        greeks.append(f"  Delta: {delta:+.1f}\n")
        greeks.append(f"  Theta: ", style="dim")
        greeks.append(f"{theta:+.2f}\n", style="cyan" if theta > 0 else "yellow")
        greeks.append(f"  Vega:  {vega:+.2f}\n")
        greeks.append(f"  Gamma: {gamma:+.3f}\n")

        # P&L section
        pnl = Text()
        pnl.append("P&L\n", style="bold underline")
        pnl.append(f"  Today: ", style="dim")
        pnl.append(f"${daily_pnl:+,.2f}\n", style=self._pnl_style(daily_pnl))
        pnl.append(f"  Week:  ", style="dim")
        pnl.append(f"${weekly_pnl:+,.2f}\n", style=self._pnl_style(weekly_pnl))

        columns = Columns([account, greeks, pnl], equal=True, expand=True)

        panel = Panel(
            columns,
            title="[bold white]PORTFOLIO SUMMARY[/bold white]",
            title_align="left",
            border_style="blue",
            padding=(1, 2),
        )

        self._console.print()
        self._console.print(panel)

        return ""  # Already printed

    # =========================================================================
    # POSITIONS LIST
    # =========================================================================

    def render_positions(self, positions: List[Dict[str, Any]]) -> str:
        """
        Render positions list.

        Args:
            positions: List of position dicts

        Returns:
            Formatted string
        """
        if not self._enabled:
            return self._plain_positions(positions)
        return self._rich_positions(positions)

    def _plain_positions(self, positions: List[Dict[str, Any]]) -> str:
        """Plain text positions list."""
        if not positions:
            return "No open positions."

        lines = ["Current Positions", "-" * 60]

        for pos in positions:
            symbol = pos.get('symbol', 'Unknown')
            underlying = pos.get('underlying', symbol)
            qty = pos.get('quantity', 0)
            pnl = pos.get('pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            dte = pos.get('dte', 'N/A')

            pnl_display = f"${pnl:+.2f}" if isinstance(pnl, (int, float, Decimal)) else str(pnl)
            pnl_pct_display = f"({pnl_pct:+.1%})" if isinstance(pnl_pct, (int, float, Decimal)) else ""

            lines.append(f"  {underlying}: {qty} contracts | DTE: {dte} | P&L: {pnl_display} {pnl_pct_display}")

        return "\n".join(lines)

    def _rich_positions(self, positions: List[Dict[str, Any]]) -> str:
        """Rich positions table."""
        if not positions:
            self._console.print("[dim]No open positions.[/dim]")
            return ""

        table = Table(
            title="[bold]Current Positions[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )

        table.add_column("Symbol", style=Styles.SYMBOL)
        table.add_column("Qty", justify="right")
        table.add_column("DTE", justify="right")
        table.add_column("P&L $", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Status", justify="center")

        for pos in positions:
            underlying = pos.get('underlying', pos.get('symbol', 'Unknown'))
            qty = pos.get('quantity', 0)
            pnl = float(pos.get('pnl', 0))
            pnl_pct = float(pos.get('pnl_pct', 0))
            dte = pos.get('dte', 'N/A')

            # Determine status
            status = ""
            status_style = "dim"
            if isinstance(dte, int):
                if dte <= 21:
                    status = "MANAGE"
                    status_style = "yellow"
            if pnl_pct >= 0.50:
                status = "PROFIT"
                status_style = "green"
            elif pnl_pct <= -1.0:
                status = "STOP"
                status_style = "red"

            pnl_style = self._pnl_style(pnl)

            table.add_row(
                underlying,
                str(qty),
                str(dte),
                f"[{pnl_style}]${pnl:+,.2f}[/{pnl_style}]",
                f"[{pnl_style}]{pnl_pct:+.1%}[/{pnl_style}]",
                f"[{status_style}]{status}[/{status_style}]" if status else "",
            )

        self._console.print()
        self._console.print(table)

        return ""  # Already printed

    # =========================================================================
    # PENDING TRADES
    # =========================================================================

    def render_pending_trades(self, trades: List[Dict[str, Any]]) -> str:
        """
        Render pending trades list.

        Args:
            trades: List of pending trade dicts

        Returns:
            Formatted string
        """
        if not self._enabled:
            return self._plain_pending_trades(trades)
        return self._rich_pending_trades(trades)

    def _plain_pending_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Plain text pending trades."""
        if not trades:
            return "No pending trades."

        lines = ["Pending Trades", "-" * 50]

        for trade in trades:
            trade_id = trade.get('id', 'Unknown')[:8]
            symbol = trade.get('underlying_symbol', 'Unknown')
            strategy = trade.get('strategy', 'Unknown')
            credit = trade.get('expected_credit', 0)
            max_loss = trade.get('max_loss', 0)
            status = trade.get('status', 'pending')

            lines.append(f"  [{trade_id}] {symbol} {strategy}")
            lines.append(f"       Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f} | Status: {status}")
            lines.append("")

        lines.append("Use 'approve <id>' or 'reject <id>' to manage trades.")
        return "\n".join(lines)

    def _rich_pending_trades(self, trades: List[Dict[str, Any]]) -> str:
        """Rich pending trades table."""
        if not trades:
            self._console.print("[dim]No pending trades.[/dim]")
            return ""

        table = Table(
            title="[bold]Pending Trades[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )

        table.add_column("ID", style=Styles.TRADE_ID)
        table.add_column("Symbol", style=Styles.SYMBOL)
        table.add_column("Strategy")
        table.add_column("Credit", justify="right")
        table.add_column("Max Loss", justify="right")
        table.add_column("Status", justify="center")

        for trade in trades:
            trade_id = trade.get('id', 'Unknown')[:8]
            symbol = trade.get('underlying_symbol', 'Unknown')
            strategy = self._format_strategy(trade.get('strategy', 'Unknown'))
            credit = float(trade.get('expected_credit', 0))
            max_loss = float(trade.get('max_loss', 0))
            status = trade.get('status', 'pending')

            status_style = "yellow" if status == "pending_approval" else "green" if status == "approved" else "white"

            table.add_row(
                trade_id,
                symbol,
                strategy,
                f"[green]${credit:.2f}[/green]",
                f"${max_loss:,.0f}",
                f"[{status_style}]{status}[/{status_style}]",
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        self._console.print("[dim]Commands: 'approve <id>' | 'reject <id>' | 'details <id>'[/dim]")

        return ""  # Already printed

    # =========================================================================
    # WORKING ORDERS
    # =========================================================================

    def render_working_orders(self, orders: List[Dict[str, Any]]) -> str:
        """
        Render working (unfilled) orders.

        Args:
            orders: List of working order dicts

        Returns:
            Formatted string
        """
        if not self._enabled:
            return self._plain_working_orders(orders)
        return self._rich_working_orders(orders)

    def _plain_working_orders(self, orders: List[Dict[str, Any]]) -> str:
        """Plain text working orders."""
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
            lines.append(f"       Limit: ${limit_price:.2f} | Order ID: {str(order_id)[:8] if order_id else 'N/A'}")
            lines.append(f"       Status: {order_status.upper()}")
            lines.append("")

        lines.append("Use 'modify price <id> <new_price>' to adjust limit price")
        lines.append("Use 'cancel order <id>' to cancel the order")
        return "\n".join(lines)

    def _rich_working_orders(self, orders: List[Dict[str, Any]]) -> str:
        """Rich working orders table."""
        if not orders:
            self._console.print("[dim]No working orders.[/dim]")
            return ""

        table = Table(
            title="[bold yellow]Working Orders (Unfilled)[/bold yellow]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="yellow",
        )

        table.add_column("ID", style=Styles.TRADE_ID)
        table.add_column("Symbol", style=Styles.SYMBOL)
        table.add_column("Strategy")
        table.add_column("Limit", justify="right")
        table.add_column("Order ID", style="dim")
        table.add_column("Status", justify="center")

        for order in orders:
            trade_id = order.get('id', 'Unknown')[:8]
            symbol = order.get('underlying_symbol', 'Unknown')
            strategy = self._format_strategy(order.get('strategy', 'Unknown'))
            limit_price = float(order.get('limit_price') or order.get('expected_credit', 0))
            order_id = order.get('order_id', 'N/A')
            order_status = order.get('order_status', 'working')

            table.add_row(
                trade_id,
                symbol,
                strategy,
                f"[green]${limit_price:.2f}[/green]",
                str(order_id)[:8] if order_id else "N/A",
                f"[yellow]{order_status.upper()}[/yellow]",
            )

        self._console.print()
        self._console.print(table)
        self._console.print()
        self._console.print("[dim]Commands: 'set price <id> <amount>' | 'cancel order <id>'[/dim]")

        return ""  # Already printed

    # =========================================================================
    # ALERTS
    # =========================================================================

    def render_alert(self, alert: Any) -> str:
        """
        Render a management alert.

        Args:
            alert: ManagementAlert object

        Returns:
            Formatted string
        """
        if not self._enabled:
            return self._plain_alert(alert)
        return self._rich_alert(alert)

    def _plain_alert(self, alert: Any) -> str:
        """Plain text alert."""
        urgency_marker = alert.urgency_marker() if hasattr(alert, 'urgency_marker') else ""
        action = alert.action.value if hasattr(alert.action, 'value') else str(alert.action)

        lines = [
            f"{urgency_marker} {alert.underlying}: {action}",
            f"   Reason: {alert.reason}"
        ]

        metrics = []
        if alert.pnl_percent != 0:
            metrics.append(f"P&L: {alert.pnl_percent:.1%}")
        if alert.dte < 999:
            metrics.append(f"DTE: {alert.dte}")
        if alert.is_tested:
            metrics.append("TESTED")

        if metrics:
            lines.append(f"   {' | '.join(metrics)}")

        return "\n".join(lines)

    def _rich_alert(self, alert: Any) -> str:
        """Rich alert panel."""
        action = alert.action.value if hasattr(alert.action, 'value') else str(alert.action)
        urgency = alert.urgency.value if hasattr(alert.urgency, 'value') else str(alert.urgency)

        # Determine style based on urgency
        if urgency == "critical":
            border_style = "red bold"
            icon = "[!]"
        elif urgency == "high":
            border_style = "yellow"
            icon = "[!]"
        else:
            border_style = "blue"
            icon = "[i]"

        content = Text()
        content.append(f"{icon} ", style=border_style)
        content.append(f"{alert.underlying}", style=Styles.SYMBOL)
        content.append(f": {action.upper()}\n", style="bold")
        content.append(f"   {alert.reason}", style="dim")

        # Metrics line
        metrics = []
        if alert.pnl_percent != 0:
            pnl_style = self._pnl_style(alert.pnl_percent)
            metrics.append(f"[{pnl_style}]P&L: {alert.pnl_percent:.1%}[/{pnl_style}]")
        if alert.dte < 999:
            metrics.append(f"DTE: {alert.dte}")
        if alert.is_tested:
            metrics.append("[red]TESTED[/red]")

        if metrics:
            content.append(f"\n   {' | '.join(metrics)}")

        panel = Panel(
            content,
            border_style=border_style,
            padding=(0, 1),
        )

        self._console.print(panel)

        return ""  # Already printed


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    # Demo the display modes
    display = RichDisplay(enabled=True)

    sample_opportunities = [
        {
            'id': 'abc12345-6789-0123-4567-890abcdef012',
            'underlying_symbol': 'SPY',
            'strategy': 'SHORT_PUT_SPREAD',
            'iv_rank': 42.5,
            'delta': -0.28,
            'dte': 45,
            'expected_credit': 1.25,
            'max_loss': 375.00,
            'probability_of_profit': 0.72,
            'theta': 0.045,
            'short_strike': 580,
            'long_strike': 575,
        },
        {
            'id': 'def98765-4321-fedc-ba98-765432109876',
            'underlying_symbol': 'QQQ',
            'strategy': 'IRON_CONDOR',
            'iv_rank': 38.2,
            'delta': 0.05,
            'dte': 42,
            'expected_credit': 2.10,
            'max_loss': 290.00,
            'probability_of_profit': 0.68,
            'theta': 0.082,
            'put_short_strike': 505,
            'put_long_strike': 500,
            'call_short_strike': 540,
            'call_long_strike': 545,
        },
    ]

    symbols = ['SPY', 'QQQ']

    print("=" * 60)
    print("RICH ENABLED - Table Layout")
    print("=" * 60)
    display.render_scan_results(sample_opportunities, symbols, layout="table")

    print("\n" + "=" * 60)
    print("RICH ENABLED - Cards Layout")
    print("=" * 60)
    display.render_scan_results(sample_opportunities, symbols, layout="cards")

    print("\n" + "=" * 60)
    print("RICH DISABLED - Plain Text")
    print("=" * 60)
    display.toggle(False)
    print(display.render_scan_results(sample_opportunities, symbols))
