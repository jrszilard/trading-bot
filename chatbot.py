#!/usr/bin/env python3
"""
Tastytrade Trading Chatbot

Natural language interface for the Tastytrade trading bot.
Interprets user intent and executes trading operations conversationally.

Author: Trading Bot
License: MIT
"""

import asyncio
import json
import logging
import os
import sys
from typing import Optional, Dict, Any, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from conversation import ConversationManager, ConversationState
from intent_router import IntentRouter, TradingIntent, ParsedIntent
from response_formatter import ResponseFormatter
from research import ResearchEngine
from position_monitor import PositionMonitor, ManagementAlert

# Import trading bot components
from trading_bot import TastytradeBot
from models import RiskParameters, TradeStatus, TradeProposal, PortfolioState

logger = logging.getLogger(__name__)


class ChatbotInterface:
    """
    Natural language chatbot interface for the trading bot.

    Provides conversational interaction for:
    - Scanning for opportunities
    - Managing trades (approve, reject, execute)
    - Portfolio queries
    - Research and analysis
    """

    def __init__(
        self,
        bot: TastytradeBot,
        use_streaming: bool = True
    ):
        self.bot = bot
        self.use_streaming = use_streaming

        # Initialize components
        self.conversation = ConversationManager()
        self.router = IntentRouter()
        self.formatter = ResponseFormatter()
        self.research = ResearchEngine(bot)

        # Position monitor (initialized in run())
        self.position_monitor: Optional[PositionMonitor] = None

        # Register intent handlers
        self._register_handlers()

        # Running state
        self.running = False

    def _register_handlers(self) -> None:
        """Register all intent handlers"""
        # Trading operations
        self.router.register_handler(TradingIntent.SCAN_OPPORTUNITIES, self._handle_scan)
        self.router.register_handler(TradingIntent.CREATE_TRADE, self._handle_create_trade)
        self.router.register_handler(TradingIntent.SELECT_OPTION, self._handle_select_option)
        self.router.register_handler(TradingIntent.SHOW_PENDING, self._handle_pending)
        self.router.register_handler(TradingIntent.APPROVE_TRADE, self._handle_approve)
        self.router.register_handler(TradingIntent.REJECT_TRADE, self._handle_reject)
        self.router.register_handler(TradingIntent.EXECUTE_TRADE, self._handle_execute)

        # Portfolio queries
        self.router.register_handler(TradingIntent.GET_PORTFOLIO, self._handle_portfolio)
        self.router.register_handler(TradingIntent.GET_POSITIONS, self._handle_positions)
        self.router.register_handler(TradingIntent.GET_BUYING_POWER, self._handle_buying_power)
        self.router.register_handler(TradingIntent.GET_PNL, self._handle_pnl)

        # Position management
        self.router.register_handler(TradingIntent.MANAGE_POSITIONS, self._handle_manage)
        self.router.register_handler(TradingIntent.POSITION_ANALYSIS, self._handle_position_analysis)
        self.router.register_handler(TradingIntent.CLOSE_POSITION, self._handle_close_position)
        self.router.register_handler(TradingIntent.ROLL_POSITION, self._handle_roll_position)

        # Research
        self.router.register_handler(TradingIntent.GET_IV_RANK, self._handle_iv_rank)
        self.router.register_handler(TradingIntent.RESEARCH_TRADE, self._handle_research)
        self.router.register_handler(TradingIntent.MARKET_ANALYSIS, self._handle_market)

        # Configuration
        self.router.register_handler(TradingIntent.GET_RISK_PARAMS, self._handle_risk_params)

        # Backtesting
        self.router.register_handler(TradingIntent.RUN_BACKTEST, self._handle_backtest)

        # Conversational
        self.router.register_handler(TradingIntent.HELP, self._handle_help)
        self.router.register_handler(TradingIntent.GENERAL_CHAT, self._handle_general_chat)
        self.router.register_handler(TradingIntent.CONFIRM_YES, self._handle_confirm_yes)
        self.router.register_handler(TradingIntent.CONFIRM_NO, self._handle_confirm_no)

    async def process_message(self, user_input: str) -> str:
        """
        Process a user message and return a response.

        Args:
            user_input: The user's natural language input

        Returns:
            The bot's response
        """
        user_input = user_input.strip()
        if not user_input:
            return ""

        # Add user message to conversation
        self.conversation.add_user_message(user_input)

        # Try quick pattern matching first
        parsed = self.router.parse_quick(user_input)

        # If low confidence, try Claude NLU if available
        if parsed.confidence < 0.7 and self.bot.claude_advisor:
            try:
                claude_parse = await self.bot.claude_advisor.parse_intent(
                    user_input,
                    conversation_history=self.conversation.get_llm_context(),
                    context_summary=self.conversation.get_context_summary()
                )
                if claude_parse and claude_parse.get('confidence', 0) > parsed.confidence:
                    # Update parsed intent with Claude's interpretation
                    parsed = ParsedIntent(
                        intent=TradingIntent(claude_parse['intent']),
                        confidence=claude_parse['confidence'],
                        raw_text=user_input,
                        symbols=claude_parse.get('symbols', []),
                        trade_id=claude_parse.get('trade_id'),
                        strategy=claude_parse.get('strategy'),
                        action=claude_parse.get('action')
                    )
            except Exception as e:
                logger.debug(f"Claude NLU failed, using pattern match: {e}")

        # Resolve references (e.g., "it", "that trade")
        self._resolve_references(parsed)

        # Build context for handler
        context = {
            'bot': self.bot,
            'conversation': self.conversation,
            'formatter': self.formatter,
            'research': self.research
        }

        # Route to handler
        response = await self.router.route(parsed, context)

        # Add assistant response to conversation
        self.conversation.add_assistant_message(
            response,
            intent=parsed.intent.value,
            entities={
                'symbols': parsed.symbols,
                'trade_id': parsed.trade_id
            }
        )

        return response

    def _resolve_references(self, parsed: ParsedIntent) -> None:
        """Resolve pronoun and reference resolution"""
        # Resolve symbol references
        if not parsed.symbols:
            resolved_symbol = self.conversation.resolve_symbol_reference(parsed.raw_text)
            if resolved_symbol:
                parsed.symbols = [resolved_symbol]

        # Resolve trade references
        if not parsed.trade_id:
            resolved_trade = self.conversation.resolve_trade_reference(parsed.raw_text)
            if resolved_trade:
                parsed.trade_id = resolved_trade

        # Update conversation entities
        if parsed.symbols:
            self.conversation.context.entities.symbols = parsed.symbols
        if parsed.trade_id:
            self.conversation.context.entities.trade_id = parsed.trade_id

    # =========================================================================
    # Intent Handlers
    # =========================================================================

    async def _handle_scan(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle scan for opportunities"""
        # Expanded default symbol universe for open-ended scans
        # Prioritize liquid, high IV rank opportunities
        if not parsed.symbols:
            symbols = [
                # Major indices (most liquid)
                'SPY', 'QQQ', 'IWM', 'DIA',
                # Sector ETFs (high liquidity)
                'XLF', 'XLE', 'XLK', 'XLV', 'XLI',
                # Popular high IV stocks
                'TSLA', 'AMD', 'NVDA'
            ]
        else:
            symbols = parsed.symbols

        data_source = "[MOCK DATA] " if self.bot.use_mock_data else ""
        response_parts = [f"{data_source}Scanning {', '.join(symbols)} for opportunities..."]

        try:
            # Unpack both approved opportunities and all candidates
            opportunities, all_candidates = await self.bot.scan_for_opportunities(symbols)

            # Store candidates in conversation context for option selection
            self.conversation.context.entities.last_scan_candidates = all_candidates
            if symbols:
                self.conversation.context.entities.last_scan_symbol = symbols[0]

            # If no approved opportunities but candidates exist, show filtered candidates
            if not opportunities and all_candidates:
                response_parts = [
                    f"{data_source}Found {len(all_candidates)} potential strateg{'y' if len(all_candidates) == 1 else 'ies'}:",
                    ""
                ]

                for candidate in all_candidates:
                    opt_num = candidate['option_number']
                    strategy = candidate['strategy']
                    symbol = candidate['symbol']
                    credit = candidate['credit']
                    max_loss = candidate['max_loss']
                    pop = candidate['pop']
                    kelly = candidate['kelly_score']

                    response_parts.append(
                        f"{opt_num}. {strategy} on {symbol}\n"
                        f"   Credit: ${credit:.2f} | Max Loss: ${max_loss:.2f}\n"
                        f"   POP: {pop:.0%} | Kelly: {kelly:.3f}"
                    )

                response_parts.append("")
                response_parts.append("These were filtered by risk criteria. To create one anyway, say 'option [number]'.")
                return "\n".join(response_parts)

            if not opportunities:
                return f"{data_source}No opportunities found on {', '.join(symbols)} meeting criteria."

            # Format approved results
            results = []
            for opp in opportunities:
                results.append({
                    'id': opp.id,
                    'underlying_symbol': opp.underlying_symbol,
                    'strategy': opp.strategy.value,
                    'iv_rank': float(opp.iv_rank),
                    'delta': float(opp.delta),
                    'dte': opp.dte,
                    'expected_credit': float(opp.expected_credit),
                    'max_loss': float(opp.max_loss)
                })

            return self.formatter.format_scan_results(results, symbols)

        except Exception as e:
            logger.error(f"Scan error: {e}")
            return f"Error scanning: {str(e)}"

    async def _handle_select_option(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle user selecting an option from scan results"""
        from models import StrategyType, TradeStatus
        from conversation import ConversationState
        from decimal import Decimal

        # Update portfolio state before creating trade (to get current NLV/BP for risk checks)
        await self.bot.update_portfolio_state()

        # Get option number from parsed intent
        option_num = parsed.parameters.get('option_number', 1)

        # Get stored candidates
        candidates = self.conversation.context.entities.last_scan_candidates

        if not candidates:
            return "No scan results to select from. Run a scan first."

        # Validate option number
        if option_num < 1 or option_num > len(candidates):
            return f"Invalid option. Please choose 1-{len(candidates)}."

        # Get selected candidate (1-indexed from user, but list is already 1-indexed)
        selected = next((c for c in candidates if c['option_number'] == option_num), None)

        if not selected:
            return f"Could not find option {option_num}."

        # Create trade proposal from selected candidate
        proposal = await self.bot.propose_trade(
            strategy=StrategyType(selected['strategy']),
            underlying=selected['symbol'],
            legs=selected['legs'],
            greeks=selected['greeks'],
            iv_rank=Decimal(str(selected['iv_rank'])),
            claude_analysis=None  # User override, skip Claude
        )

        if not proposal:
            return "Failed to create trade proposal (risk limits exceeded)."

        # Update conversation context
        self.conversation.context.entities.trade_id = proposal.id
        self.conversation.context.state = ConversationState.AWAITING_APPROVAL

        return (f"Created trade proposal from option {option_num}:\n"
                f"[{proposal.id}] {proposal.underlying_symbol} {proposal.strategy.value}\n"
                f"Credit: ${proposal.expected_credit:.2f} | Max Loss: ${proposal.max_loss:.2f}\n\n"
                f"Approve this trade? (yes/no)")

    async def _handle_pending(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle show pending trades"""
        pending = self.bot.get_pending_trades()

        if not pending:
            return "No pending trades."

        trades = []
        for t in pending:
            trades.append({
                'id': t.id,
                'underlying_symbol': t.underlying_symbol,
                'strategy': t.strategy.value,
                'expected_credit': float(t.expected_credit),
                'max_loss': float(t.max_loss),
                'iv_rank': float(t.iv_rank),
                'dte': t.dte
            })

            # Track most recent trade for reference resolution
            self.conversation.context.entities.trade_id = t.id

        return self.formatter.format_pending_trades(trades)

    async def _handle_approve(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle approve trade"""
        trade_id = parsed.trade_id

        if not trade_id:
            # Check if there's only one pending trade
            pending = self.bot.get_pending_trades()
            if len(pending) == 1:
                trade_id = pending[0].id
            elif len(pending) == 0:
                return "No pending trades to approve."
            else:
                return f"Multiple pending trades. Please specify: {', '.join(t.id[:8] for t in pending)}"

        # Find the trade
        trade = None
        for t in self.bot.pending_trades:
            if t.id.startswith(trade_id) or t.id == trade_id:
                trade = t
                break

        if not trade:
            return self.formatter.format_not_found("Trade", trade_id)

        # Approve it
        trade.status = TradeStatus.APPROVED
        trade.approved_by = "chatbot_user"

        # Update context for potential execution
        self.conversation.context.entities.trade_id = trade.id
        self.conversation.set_pending_action("execute", trade.id)
        self.conversation.context.set_state(ConversationState.AWAITING_EXECUTION)

        sandbox = "[SANDBOX] " if self.bot.sandbox_mode else ""
        return f"{sandbox}Trade {trade.id[:8]} approved. Execute now? (yes/no)"

    async def _handle_reject(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle reject trade"""
        trade_id = parsed.trade_id

        if not trade_id:
            pending = self.bot.get_pending_trades()
            if len(pending) == 1:
                trade_id = pending[0].id
            elif len(pending) == 0:
                return "No pending trades to reject."
            else:
                return f"Multiple pending trades. Please specify: {', '.join(t.id[:8] for t in pending)}"

        # Find and reject
        for t in self.bot.pending_trades[:]:
            if t.id.startswith(trade_id) or t.id == trade_id:
                t.status = TradeStatus.REJECTED
                t.rejection_reason = "User rejected via chatbot"
                self.bot.pending_trades.remove(t)
                self.bot.trade_history.append(t)

                sandbox = "[SANDBOX] " if self.bot.sandbox_mode else ""
                return f"{sandbox}Trade {t.id[:8]} rejected and removed."

        return self.formatter.format_not_found("Trade", trade_id)

    async def _handle_execute(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle execute trade"""
        trade_id = parsed.trade_id

        if not trade_id:
            # Look for approved trades
            approved = [t for t in self.bot.pending_trades if t.status == TradeStatus.APPROVED]
            if len(approved) == 1:
                trade_id = approved[0].id
            elif len(approved) == 0:
                return "No approved trades to execute. Approve a trade first."
            else:
                return f"Multiple approved trades. Please specify: {', '.join(t.id[:8] for t in approved)}"

        # Find the trade
        trade = None
        for t in self.bot.pending_trades:
            if t.id.startswith(trade_id) or t.id == trade_id:
                trade = t
                break

        if not trade:
            return self.formatter.format_not_found("Trade", trade_id)

        if trade.status != TradeStatus.APPROVED:
            return f"Trade {trade_id[:8]} is not approved (status: {trade.status.value}). Approve it first."

        # Execute
        success = await self.bot.execute_trade(trade)
        self.conversation.context.clear_state()

        sandbox = self.bot.sandbox_mode
        return self.formatter.format_trade_confirmation(
            trade.id[:8], "execute", success, sandbox
        )

    async def _handle_create_trade(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle direct trade creation requests (buy X shares of Y)"""
        from models import TradeProposal, StrategyType, TradeStatus
        from conversation import ConversationState
        from decimal import Decimal
        from datetime import datetime
        import uuid

        # Update portfolio state before creating trade (to get current NLV/BP for risk checks)
        await self.bot.update_portfolio_state()

        # Validate inputs
        if not parsed.symbols or len(parsed.symbols) == 0:
            return "Which symbol do you want to trade?"

        symbol = parsed.symbols[0]
        action = parsed.action  # "buy" or "sell"
        params = parsed.parameters or {}
        quantity = params.get('quantity', 1)
        instrument_type = params.get('instrument_type', 'stock')

        # For stock purchases, create a simple stock position proposal
        if instrument_type == 'stock':
            # Get current price (from API or mock data)
            if self.bot.use_mock_data and self.bot.mock_provider:
                quote = self.bot.mock_provider.get_quote(symbol, is_option=False)
                price = (quote.bid_price + quote.ask_price) / 2
            else:
                # Use real API to get current price
                # For now, we'll need to fetch the quote
                try:
                    await self.bot.update_portfolio_state()
                    # Try to get price from session
                    if self.bot.session:
                        from tastytrade import DXLinkStreamer
                        async with DXLinkStreamer(self.bot.session) as streamer:
                            await streamer.subscribe_quote([symbol])
                            quote = await streamer.get_event(symbol)
                            if quote:
                                price = (quote.bidPrice + quote.askPrice) / 2
                            else:
                                return f"Could not get current price for {symbol}"
                    else:
                        return "No active session to get price data"
                except Exception as e:
                    logger.error(f"Error getting price for {symbol}: {e}")
                    return f"Error getting price for {symbol}: {str(e)}"

            # Calculate cost
            total_cost = price * quantity

            # Create a simplified TradeProposal for stock
            leg = {
                'symbol': symbol,
                'instrument_type': 'stock',
                'quantity': quantity,
                'action': action,  # "buy" or "sell"
                'price': price,
                'total_cost': total_cost
            }

            proposal = TradeProposal(
                id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(),
                strategy=StrategyType.LONG_STOCK if action == "buy" else StrategyType.SHORT_STOCK,
                underlying_symbol=symbol,
                legs=[leg],
                expected_credit=Decimal("-" + str(total_cost)) if action == "buy" else Decimal(str(total_cost)),
                max_loss=Decimal(str(total_cost)) if action == "buy" else Decimal("999999"),  # unlimited for short
                probability_of_profit=Decimal("0.50"),  # 50/50 for directional stock
                iv_rank=Decimal("0"),
                dte=0,  # stocks don't expire
                delta=Decimal(str(quantity)) if action == "buy" else Decimal(str(-quantity)),
                theta=Decimal("0"),
                vega=Decimal("0"),
                rationale=f"{action.capitalize()} {quantity} share{'s' if quantity > 1 else ''} of {symbol} at ${price:.2f}"
            )

            # Check risk limits
            passed, reason = self.bot.check_risk_limits(proposal)
            if not passed:
                return f"Cannot create trade: {reason}"

            # Add to pending trades
            self.bot.pending_trades.append(proposal)

            # Update conversation context
            self.conversation.context.entities.trade_id = proposal.id
            self.conversation.context.state = ConversationState.AWAITING_APPROVAL

            data_source = "[MOCK DATA] " if self.bot.use_mock_data else ""
            return (f"{data_source}Created trade proposal [{proposal.id}]:\n"
                    f"{action.capitalize()} {quantity} share{'s' if quantity > 1 else ''} of {symbol}\n"
                    f"Estimated price: ${price:.2f}\n"
                    f"Total cost: ${total_cost:.2f}\n\n"
                    f"Approve this trade? (yes/no)")

        # For options, need more info (strike, expiration)
        elif instrument_type in ('call', 'put'):
            # Set research context and ask for strike/expiration
            self.conversation.context.entities.last_research_symbol = symbol
            self.conversation.context.entities.pending_action = f"{action}_{instrument_type}"
            return (f"To {action} {instrument_type}s on {symbol}, I need:\n"
                    f"• Strike price\n"
                    f"• Expiration date\n\n"
                    f"Or would you like me to scan {symbol} for optimal {instrument_type} strategies?")

        return "I didn't understand the trade request. Can you clarify?"

    async def _handle_portfolio(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle portfolio query"""
        await self.bot.update_portfolio_state()

        portfolio_data = {
            'net_liquidating_value': float(self.bot.portfolio_state.net_liquidating_value),
            'buying_power': float(self.bot.portfolio_state.buying_power),
            'portfolio_delta': float(self.bot.portfolio_state.portfolio_delta),
            'portfolio_theta': float(self.bot.portfolio_state.portfolio_theta),
            'open_positions': self.bot.portfolio_state.open_positions,
            'daily_pnl': float(self.bot.portfolio_state.daily_pnl)
        }

        return self.formatter.format_portfolio_summary(portfolio_data)

    async def _handle_positions(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle show positions"""
        # Update portfolio state
        await self.bot.update_portfolio_state()

        # Show detailed mock positions if available
        if self.bot.use_mock_data and self.bot.mock_positions:
            if not self.bot.mock_positions:
                return "No open positions."

            data_source = "[MOCK DATA] " if self.bot.use_mock_data else ""
            lines = [f"{data_source}Current Positions", "-" * 50]

            for pos in self.bot.mock_positions:
                symbol = pos['symbol']
                qty = pos.get('quantity', 0)
                strategy = pos.get('strategy', 'unknown')
                delta = pos.get('delta', 0)

                lines.append(f"\n{symbol} - {strategy}")
                lines.append(f"  Quantity: {qty}")
                lines.append(f"  Delta: {delta:.2f}")
                lines.append(f"  Entry: ${pos.get('entry_price', 0):.2f}")

            return "\n".join(lines)
        else:
            # Use standard positions from portfolio state
            positions = self.bot.portfolio_state.positions_by_underlying
            if not positions:
                return "No open positions."

            lines = ["Current Positions", "-" * 30]
            for symbol, count in positions.items():
                lines.append(f"  {symbol}: {count} position(s)")

            return "\n".join(lines)

    async def _handle_buying_power(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle buying power query"""
        await self.bot.update_portfolio_state()

        bp = self.bot.portfolio_state.buying_power
        nlv = self.bot.portfolio_state.net_liquidating_value
        used_pct = 1 - (float(bp) / float(nlv)) if nlv > 0 else 0

        return f"Buying Power: ${float(bp):,.2f}\n" \
               f"Net Liquidating Value: ${float(nlv):,.2f}\n" \
               f"BP Usage: {used_pct:.1%}"

    async def _handle_pnl(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle P&L query"""
        await self.bot.update_portfolio_state()

        daily = self.bot.portfolio_state.daily_pnl
        weekly = self.bot.portfolio_state.weekly_pnl

        return f"Today's P&L: ${float(daily):+,.2f}\n" \
               f"Week's P&L: ${float(weekly):+,.2f}"

    async def _handle_manage(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle position management check"""
        try:
            actions = await self.bot.manage_positions()

            if not actions:
                return "No positions require management action at this time."

            return self.formatter.format_management_actions(actions)
        except Exception as e:
            logger.error(f"Management check error: {e}")
            return f"Error checking positions: {str(e)}"

    async def _handle_position_analysis(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle specific position analysis"""
        symbol = parsed.primary_symbol()

        if not symbol:
            return "Which position would you like me to analyze? Specify the symbol."

        result = await self.research.analyze_position_action(symbol)

        action = result.get('action', 'MONITOR')
        reasoning = result.get('reasoning', 'No specific recommendation')
        urgency = result.get('urgency', 'normal')

        urgency_prefix = "[URGENT] " if urgency in ('high', 'critical') else ""

        return f"{urgency_prefix}{symbol} Recommendation: {action}\n{reasoning}"

    async def _handle_close_position(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle close position request"""
        symbol = parsed.primary_symbol()

        if not symbol:
            # Try to get from context
            symbol = self.conversation.context.entities.get_current_symbol()

        if not symbol:
            return "Which position would you like to close? Specify the symbol (e.g., 'close SPY')."

        # Get position details for confirmation
        position_details = await self.bot.get_position_details(symbol)

        if not position_details:
            return f"No open position found for {symbol}."

        # Store pending close data
        self.conversation.context.entities.set_pending_close(position_details)
        self.conversation.context.set_state(ConversationState.AWAITING_CLOSE_APPROVAL)

        # Format confirmation message
        sandbox = "[SANDBOX] " if self.bot.sandbox_mode else ""
        mock = "[MOCK] " if self.bot.use_mock_data else ""

        lines = [
            f"{sandbox}{mock}Close Position: {symbol}",
            "-" * 40
        ]

        for pos in position_details.get('positions', []):
            opt_type = pos.get('option_type', 'OPT')
            strike = pos.get('strike', 0)
            qty = pos.get('quantity', 0)
            dte = pos.get('dte', 0)
            pnl_pct = pos.get('pnl_percent', 0)
            direction = "short" if qty < 0 else "long"
            lines.append(f"  {direction.upper()} {abs(qty)}x {opt_type} @ ${strike:.0f} | DTE: {dte} | P&L: {pnl_pct:.1%}")

        total_pnl = position_details.get('total_pnl_percent', 0)
        lines.extend([
            "",
            f"Total P&L: {total_pnl:.1%}",
            "",
            "Are you sure you want to close this position? (yes/no)"
        ])

        return "\n".join(lines)

    async def _handle_roll_position(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle roll position request"""
        symbol = parsed.primary_symbol()

        if not symbol:
            # Try to get from context
            symbol = self.conversation.context.entities.get_current_symbol()

        if not symbol:
            return "Which position would you like to roll? Specify the symbol (e.g., 'roll SPY')."

        # Get position details for confirmation
        position_details = await self.bot.get_position_details(symbol)

        if not position_details:
            return f"No open position found for {symbol}."

        # Store pending roll data
        self.conversation.context.entities.set_pending_roll(position_details)
        self.conversation.context.set_state(ConversationState.AWAITING_ROLL_APPROVAL)

        # Format confirmation message
        sandbox = "[SANDBOX] " if self.bot.sandbox_mode else ""
        mock = "[MOCK] " if self.bot.use_mock_data else ""

        # Get target DTE from config
        target_dte = self.bot.risk_params.target_dte

        lines = [
            f"{sandbox}{mock}Roll Position: {symbol}",
            "-" * 40,
            f"Rolling to ~{target_dte} DTE",
            ""
        ]

        for pos in position_details.get('positions', []):
            opt_type = pos.get('option_type', 'OPT')
            strike = pos.get('strike', 0)
            qty = pos.get('quantity', 0)
            dte = pos.get('dte', 0)
            direction = "short" if qty < 0 else "long"
            lines.append(f"  Current: {direction.upper()} {abs(qty)}x {opt_type} @ ${strike:.0f} | DTE: {dte}")
            lines.append(f"  New:     {direction.upper()} {abs(qty)}x {opt_type} @ ${strike:.0f} | DTE: ~{target_dte}")

        lines.extend([
            "",
            "Note: Roll will be executed as credit if available (min $0.25)",
            "",
            "Are you sure you want to roll this position? (yes/no)"
        ])

        return "\n".join(lines)

    async def _handle_iv_rank(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle IV rank query"""
        symbol = parsed.primary_symbol()

        if not symbol:
            return "Which symbol would you like the IV rank for?"

        result = await self.research.research_symbol(symbol)

        # Set research context
        self.conversation.set_research_context(symbol)

        return self.formatter.format_iv_rank(
            symbol,
            result['iv_rank'],
            threshold=30.0
        )

    async def _handle_research(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle research/strategy analysis"""
        symbol = parsed.primary_symbol()
        strategy = parsed.strategy or 'short_put_spread'

        if not symbol:
            # Try to get from context
            symbol = self.conversation.context.entities.get_current_symbol()

        if not symbol:
            return "Which symbol would you like me to research?"

        result = await self.research.research_strategy(symbol, strategy)

        # Set research context
        self.conversation.set_research_context(symbol, strategy)

        response = self.formatter.format_research_result(result)

        # If it would meet criteria, offer to create trade
        if result.get('would_meet_criteria'):
            self.conversation.set_pending_action("create_trade", symbol)
            response += "\n\nWould you like me to create this as a pending trade?"

        return response

    async def _handle_market(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle market overview"""
        overview = await self.research.get_market_overview()

        lines = ["Market Overview", "=" * 30, ""]

        for idx in overview.get('market_indices', []):
            suitable = "suitable" if idx['suitable'] else "low"
            lines.append(f"  {idx['symbol']}: IV Rank {idx['iv_rank']:.1f}% ({suitable})")

        lines.extend([
            "",
            f"IV Environment: {overview.get('iv_environment', 'unknown').upper()}",
            f"Sentiment: {overview.get('overall_sentiment', 'neutral')}",
            "",
            overview.get('recommendation', '')
        ])

        return "\n".join(lines)

    async def _handle_risk_params(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle risk parameters query"""
        params = {
            'max_daily_loss': float(self.bot.risk_params.max_daily_loss),
            'max_weekly_loss': float(self.bot.risk_params.max_weekly_loss),
            'max_position_loss': float(self.bot.risk_params.max_position_loss),
            'max_position_size_percent': float(self.bot.risk_params.max_position_size_percent),
            'max_total_positions': self.bot.risk_params.max_total_positions,
            'min_iv_rank': float(self.bot.risk_params.min_iv_rank),
            'target_dte': self.bot.risk_params.target_dte,
            'profit_target_percent': float(self.bot.risk_params.profit_target_percent),
            'management_dte': self.bot.risk_params.management_dte
        }

        return self.formatter.format_risk_params(params)

    async def _handle_backtest(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle backtest requests"""
        from datetime import date, timedelta
        from decimal import Decimal
        from backtesting import BacktestEngine, BacktestAnalyzer

        params = parsed.parameters or {}

        # Determine time period
        if 'years' in params:
            days = params['years'] * 365
        elif 'months' in params:
            days = params['months'] * 30
        else:
            days = 180  # Default to 6 months

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Determine symbols
        symbols = parsed.symbols if parsed.symbols else [
            'SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA'
        ]

        # Create backtest engine
        backtest_engine = BacktestEngine(
            strategy_engine=self.bot.strategy_engine,
            initial_capital=Decimal("50000"),
            max_positions=10
        )

        try:
            # Run backtest
            result = backtest_engine.run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                scan_frequency_days=1
            )

            # Generate report
            analyzer = BacktestAnalyzer()
            report = analyzer.generate_report(result)

            # Also get analysis for recommendations
            analysis = analyzer.analyze_results(result)

            # Format for display
            response_lines = [
                "Backtest Complete!",
                "=" * 60,
                "",
                f"Period: {start_date} to {end_date} ({days} days)",
                f"Symbols: {', '.join(symbols)}",
                f"Initial Capital: ${result.initial_capital:,.2f}",
                f"Final Capital: ${result.final_capital:,.2f}",
                "",
                "PERFORMANCE SUMMARY",
                "-" * 60,
                f"Total Return: {result.total_return_percent:.2f}%",
                f"Total P&L: ${result.total_pnl:,.2f}",
                f"Total Trades: {result.total_trades}",
                f"Win Rate: {result.win_rate:.2f}%",
                f"Profit Factor: {result.profit_factor:.2f}",
                "",
                "RISK METRICS",
                "-" * 60,
                f"Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_percent:.2f}%)",
                f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
                f"Sortino Ratio: {result.sortino_ratio:.2f}",
                "",
                "STRATEGY BREAKDOWN",
                "-" * 60,
            ]

            # Add strategy details
            if result.trades_by_strategy:
                for strategy, count in sorted(result.trades_by_strategy.items()):
                    pnl = result.pnl_by_strategy[strategy]
                    win_rate = result.win_rate_by_strategy[strategy]
                    response_lines.append(
                        f"  {strategy:25s}  {count:3d} trades  "
                        f"${pnl:10,.2f}  {win_rate:6.2f}% wins"
                    )

            response_lines.extend(["", "KEY RECOMMENDATIONS", "-" * 60])

            # Add recommendations
            if analysis['recommendations']:
                for rec in analysis['recommendations'][:5]:  # Top 5
                    response_lines.append(f"• {rec}")
            else:
                response_lines.append("• No major issues identified. Continue current strategy.")

            response_lines.append("")
            response_lines.append("For the complete detailed report, check the logs.")

            return "\n".join(response_lines)

        except Exception as e:
            logger.error(f"Error running backtest: {e}", exc_info=True)
            return f"Failed to run backtest: {str(e)}"

    async def _handle_help(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle help request"""
        return self.formatter.format_help()

    async def _handle_general_chat(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle general trading chat/questions"""
        if self.bot.claude_advisor and self.bot.claude_advisor.is_available:
            try:
                response = await self.bot.claude_advisor.generate_conversational_response(
                    user_message=parsed.raw_text,
                    intent="general_chat",
                    context_data={
                        'portfolio_delta': float(self.bot.portfolio_state.portfolio_delta),
                        'open_positions': self.bot.portfolio_state.open_positions,
                        'pending_trades': len(self.bot.get_pending_trades())
                    },
                    conversation_history=self.conversation.get_llm_context()
                )
                return response
            except Exception as e:
                logger.error(f"General chat error: {e}")

        return "I'm not sure how to help with that. Type 'help' to see available commands."

    async def _handle_confirm_yes(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle yes confirmation"""
        state = self.conversation.context.state
        pending_action = self.conversation.context.entities.pending_action
        pending_target = self.conversation.context.entities.pending_target

        # Handle position close approval
        if state == ConversationState.AWAITING_CLOSE_APPROVAL:
            position_data = self.conversation.context.entities.pending_close_position
            if position_data:
                symbol = position_data.get('symbol')
                result = await self.bot.close_position(symbol)

                self.conversation.context.clear_state()

                if result.get('success'):
                    return result.get('message', f"Position {symbol} closed successfully.")
                else:
                    return f"Failed to close position: {result.get('error', 'Unknown error')}"

        # Handle position roll approval
        if state == ConversationState.AWAITING_ROLL_APPROVAL:
            position_data = self.conversation.context.entities.pending_roll_position
            if position_data:
                symbol = position_data.get('symbol')
                result = await self.bot.roll_position(symbol)

                self.conversation.context.clear_state()

                if result.get('success'):
                    net_credit = result.get('net_credit', 0)
                    message = result.get('message', f"Position {symbol} rolled successfully.")
                    if net_credit > 0:
                        message += f"\nNet credit: ${net_credit:.2f}"
                    return message
                else:
                    return f"Failed to roll position: {result.get('error', 'Unknown error')}"

        if pending_action == "execute" and pending_target:
            # Execute the pending trade
            parsed.trade_id = pending_target
            return await self._handle_execute(parsed, context)

        elif pending_action == "create_trade":
            # Would create the researched trade as pending
            symbol = self.conversation.context.entities.last_research_symbol
            strategy = self.conversation.context.entities.last_research_strategy

            if symbol:
                # Scan and create trade
                opportunities, _ = await self.bot.scan_for_opportunities([symbol])
                if opportunities:
                    trade = opportunities[0]
                    self.conversation.context.entities.trade_id = trade.id
                    return f"Created pending trade [{trade.id[:8]}]: {symbol} {trade.strategy.value}\n" \
                           f"Credit: ${float(trade.expected_credit):.2f}\n\n" \
                           f"Use 'approve' to approve this trade."
                else:
                    return f"Could not create trade for {symbol}. Criteria not met."

        # Handle trade approval (from CREATE_TRADE or SELECT_OPTION)
        if state == ConversationState.AWAITING_APPROVAL:
            trade_id = self.conversation.context.entities.trade_id
            if trade_id:
                # Approve the trade
                parsed.trade_id = trade_id
                return await self._handle_approve(parsed, context)

        self.conversation.context.clear_state()
        return "Confirmed. What would you like to do next?"

    async def _handle_confirm_no(
        self,
        parsed: ParsedIntent,
        context: Dict[str, Any]
    ) -> str:
        """Handle no confirmation"""
        self.conversation.context.clear_state()
        self.conversation.clear_pending_action()
        return "Okay, cancelled. What would you like to do?"

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    def _display_alerts(self, alerts: List[ManagementAlert]) -> None:
        """Display pending alerts to the user"""
        if not alerts:
            return

        print("\n" + "=" * 60)
        print("POSITION ALERTS")
        print("=" * 60)

        for alert in alerts:
            formatted = self.formatter.format_alert(alert)
            print(formatted)
            print("-" * 40)

        print("Use 'close <symbol>' or 'roll <symbol>' to take action")
        print("=" * 60 + "\n")

    async def run(self) -> None:
        """Main chatbot run loop"""
        self.running = True

        # Print banner
        print("\n" + "=" * 60)
        print("TASTYTRADE TRADING CHATBOT")
        print("Natural Language Interface | Tastylive Methodology")
        print("=" * 60)

        # Show mode
        mode_parts = []
        if self.bot.sandbox_mode:
            mode_parts.append("SANDBOX")
        else:
            mode_parts.append("PRODUCTION")
        if self.bot.use_mock_data:
            mode_parts.append("MOCK DATA")
        if self.bot.claude_advisor and self.bot.claude_advisor.is_available:
            mode_parts.append("CLAUDE AI")

        print(f"Mode: {' + '.join(mode_parts)}")
        print("\nType 'help' for commands, 'quit' to exit")
        print("=" * 60 + "\n")

        # Start position monitor
        interval_minutes = self.bot.config.get('bot_settings', {}).get(
            'management_check_interval_minutes', 60
        )
        self.position_monitor = PositionMonitor(
            self.bot,
            interval_minutes=interval_minutes,
            config=self.bot.config
        )
        self.position_monitor.start()
        print(f"Position monitor started (checking every {interval_minutes} minutes)\n")

        try:
            while self.running:
                try:
                    # Check for pending alerts before prompting
                    if self.position_monitor:
                        alerts = self.position_monitor.get_pending_alerts()
                        self._display_alerts(alerts)

                    # Get user input
                    user_input = input("You: ").strip()

                    if not user_input:
                        continue

                    # Check for exit
                    if user_input.lower() in ('quit', 'exit', 'bye', 'goodbye'):
                        print("\nBot: Goodbye! Happy trading!")
                        self.running = False
                        break

                    # Process and respond
                    response = await self.process_message(user_input)

                    if response:
                        print(f"\nBot: {response}\n")

                except KeyboardInterrupt:
                    print("\n\nBot: Interrupted. Goodbye!")
                    self.running = False
                    break
                except EOFError:
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}")
                    print(f"\nBot: Sorry, an error occurred: {str(e)}\n")
        finally:
            # Stop position monitor on exit
            if self.position_monitor:
                self.position_monitor.stop()
                print("Position monitor stopped.")


async def main():
    """Main entry point"""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Build risk parameters from config
    risk_config = config.get('risk_parameters', {})
    loss_limits = risk_config.get('loss_limits', {})
    position_sizing = risk_config.get('position_sizing', {})
    entry_criteria = config.get('entry_criteria', {})

    from decimal import Decimal
    risk_params = RiskParameters(
        max_daily_loss=Decimal(str(loss_limits.get('max_daily_loss', 500))),
        max_weekly_loss=Decimal(str(loss_limits.get('max_weekly_loss', 1500))),
        max_position_loss=Decimal(str(loss_limits.get('max_position_loss', 300))),
        max_position_size_percent=Decimal(str(position_sizing.get('max_position_size_percent', 0.05))),
        max_total_positions=position_sizing.get('max_total_positions', 15),
        min_iv_rank=Decimal(str(entry_criteria.get('iv_requirements', {}).get('min_iv_rank', 30))),
        target_dte=entry_criteria.get('dte_requirements', {}).get('target_dte', 45),
    )

    # Initialize bot
    sandbox_mode = config.get('bot_settings', {}).get('sandbox_mode', True)
    claude_config = config.get('claude_advisor', {})

    bot = TastytradeBot(
        risk_params=risk_params,
        sandbox_mode=sandbox_mode,
        claude_config=claude_config,
        config=config
    )

    # Try to connect
    if os.environ.get('TT_CLIENT_SECRET') and os.environ.get('TT_REFRESH_TOKEN'):
        connected = await bot.connect()
        if connected:
            print("Connected to Tastytrade API")
        else:
            print("Warning: Could not connect to Tastytrade API")
            print("Continuing with mock data only...")
    else:
        print("No Tastytrade credentials found. Using mock data mode.")

    # Create and run chatbot
    chatbot = ChatbotInterface(bot)
    await chatbot.run()


if __name__ == "__main__":
    asyncio.run(main())
