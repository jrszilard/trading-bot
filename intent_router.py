#!/usr/bin/env python3
"""
Intent Router for Tastytrade Chatbot

Handles intent classification and routing for natural language
trading commands.

Author: Trading Bot
License: MIT
"""

from dataclasses import dataclass, field
from difflib import get_close_matches
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class TradingIntent(Enum):
    """All supported trading intents"""

    # Trading Operations
    SCAN_OPPORTUNITIES = "scan_opportunities"
    CREATE_TRADE = "create_trade"
    SELECT_OPTION = "select_option"
    SHOW_PENDING = "show_pending"
    APPROVE_TRADE = "approve_trade"
    REJECT_TRADE = "reject_trade"
    EXECUTE_TRADE = "execute_trade"

    # Order Management
    WORKING_ORDERS = "working_orders"
    MODIFY_PRICE = "modify_price"
    CANCEL_ORDER = "cancel_order"
    CHECK_ORDER_STATUS = "check_order_status"

    # Portfolio Queries
    GET_PORTFOLIO = "get_portfolio"
    GET_POSITIONS = "get_positions"
    GET_BUYING_POWER = "get_buying_power"
    GET_PNL = "get_pnl"

    # Position Management
    MANAGE_POSITIONS = "manage_positions"
    POSITION_ANALYSIS = "position_analysis"
    CLOSE_POSITION = "close_position"
    ROLL_POSITION = "roll_position"

    # Research
    GET_IV_RANK = "get_iv_rank"
    RESEARCH_TRADE = "research_trade"
    MARKET_ANALYSIS = "market_analysis"

    # Risk & Configuration
    GET_RISK_PARAMS = "get_risk_params"

    # Backtesting
    RUN_BACKTEST = "run_backtest"
    SHOW_BACKTEST_RESULTS = "show_backtest_results"

    # Conversational
    CLARIFICATION = "clarification"
    GENERAL_CHAT = "general_chat"
    HELP = "help"

    # Confirmation responses
    CONFIRM_YES = "confirm_yes"
    CONFIRM_NO = "confirm_no"

    # Display settings
    SET_DISPLAY_MODE = "set_display_mode"
    SET_LAYOUT = "set_layout"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Result of intent parsing"""
    intent: TradingIntent
    confidence: float  # 0.0 to 1.0
    raw_text: str

    # Extracted entities
    symbols: List[str] = field(default_factory=list)
    trade_id: Optional[str] = None
    strategy: Optional[str] = None
    action: Optional[str] = None  # approve, reject, execute, etc.

    # Additional context
    parameters: Dict[str, Any] = field(default_factory=dict)

    def has_symbols(self) -> bool:
        return len(self.symbols) > 0

    def primary_symbol(self) -> Optional[str]:
        return self.symbols[0] if self.symbols else None


class IntentRouter:
    """
    Routes parsed intents to appropriate handler functions.

    Uses pattern matching for quick classification, with optional
    LLM-based parsing for complex cases.
    """

    # Common stock/ETF symbols for quick matching
    COMMON_SYMBOLS = {
        'SPY', 'QQQ', 'IWM', 'DIA',
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE',
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
        'VIX', 'TLT', 'GLD', 'SLV', 'USO', 'EEM', 'EFA'
    }

    # Strategy keywords
    STRATEGY_KEYWORDS = {
        # Vertical Spreads
        'put spread': 'short_put_spread',
        'call spread': 'short_call_spread',
        'credit spread': 'short_put_spread',
        'vertical spread': 'short_put_spread',

        # Multi-leg Premium Selling
        'iron condor': 'iron_condor',
        'strangle': 'short_strangle',
        'straddle': 'short_straddle',
        'jade lizard': 'jade_lizard',
        'lizard': 'jade_lizard',

        # Time Spreads
        'calendar': 'calendar_spread',
        'calendar spread': 'calendar_spread',
        'time spread': 'calendar_spread',
        'diagonal': 'diagonal_spread',
        'diagonal spread': 'diagonal_spread',
        'double calendar': 'double_calendar',

        # Naked Options
        'short put': 'short_put',
        'short call': 'short_call',
        'naked put': 'short_put',
        'naked call': 'short_call',

        # Volatility Expansion
        'long strangle': 'long_strangle',
        'long straddle': 'long_straddle',
        'buy strangle': 'long_strangle',
        'buy straddle': 'long_straddle',

        # Stock-based
        'covered call': 'covered_call',
        'pmcc': 'poor_mans_covered_call',
        "poor man's covered call": 'poor_mans_covered_call',
        'collar': 'collar',

        # Ratio & Advanced
        'ratio spread': 'put_ratio_spread',
        'broken wing': 'broken_wing_butterfly',
        'butterfly': 'broken_wing_butterfly'
    }

    # Command keywords for fuzzy matching (typo correction)
    COMMAND_KEYWORDS = [
        'scan', 'find', 'search', 'opportunities', 'opportunity',
        'approve', 'reject', 'execute', 'confirm',
        'pending', 'working', 'orders', 'order', 'trades', 'trade',
        'portfolio', 'positions', 'position', 'holdings',
        'close', 'roll', 'cancel', 'modify',
        'backtest', 'research', 'analyze', 'analysis',
        'help', 'status', 'filled', 'price', 'show', 'check'
    ]

    # Intent descriptions for user feedback
    INTENT_DESCRIPTIONS = {
        TradingIntent.SCAN_OPPORTUNITIES: "scan for trading opportunities",
        TradingIntent.SHOW_PENDING: "show pending trades",
        TradingIntent.APPROVE_TRADE: "approve a trade",
        TradingIntent.REJECT_TRADE: "reject a trade",
        TradingIntent.EXECUTE_TRADE: "execute a trade",
        TradingIntent.WORKING_ORDERS: "show working orders",
        TradingIntent.MODIFY_PRICE: "modify limit price",
        TradingIntent.CANCEL_ORDER: "cancel an order",
        TradingIntent.CHECK_ORDER_STATUS: "check order fill status",
        TradingIntent.GET_PORTFOLIO: "show portfolio summary",
        TradingIntent.GET_POSITIONS: "show current positions",
        TradingIntent.GET_BUYING_POWER: "show buying power",
        TradingIntent.GET_PNL: "show P&L",
        TradingIntent.MANAGE_POSITIONS: "check position management",
        TradingIntent.CLOSE_POSITION: "close a position",
        TradingIntent.ROLL_POSITION: "roll a position",
        TradingIntent.GET_IV_RANK: "get IV rank",
        TradingIntent.RESEARCH_TRADE: "research a trade",
        TradingIntent.RUN_BACKTEST: "run a backtest",
        TradingIntent.HELP: "show help",
        TradingIntent.GENERAL_CHAT: "general question",
    }

    def __init__(self):
        self.handlers: Dict[TradingIntent, Callable] = {}

    def register_handler(
        self,
        intent: TradingIntent,
        handler: Callable[..., Awaitable[str]]
    ) -> None:
        """Register a handler for an intent"""
        self.handlers[intent] = handler

    def get_intent_description(self, intent: TradingIntent) -> str:
        """Get human-readable description of an intent"""
        return self.INTENT_DESCRIPTIONS.get(intent, intent.value.replace('_', ' '))

    def suggest_typo_corrections(self, text: str) -> List[str]:
        """
        Suggest corrections for misspelled command keywords.

        Returns list of suggestions like "Did you mean 'approve' instead of 'appove'?"
        """
        words = text.lower().split()
        suggestions = []

        for word in words:
            # Skip very short words and common words
            if len(word) < 3:
                continue

            # Check for close matches
            matches = get_close_matches(word, self.COMMAND_KEYWORDS, n=1, cutoff=0.7)
            if matches and matches[0] != word:
                suggestions.append(f"'{word}' → '{matches[0]}'")

        return suggestions

    def get_contextual_suggestions(self, text: str) -> List[str]:
        """
        Get context-aware suggestions based on keywords in the input.

        Returns list of suggested commands the user might want.
        """
        text_lower = text.lower()
        suggestions = []

        # Order/trade related
        if any(word in text_lower for word in ['order', 'trade', 'fill', 'submit', 'sent']):
            suggestions.extend([
                "'show working orders' - View unfilled limit orders",
                "'check order status' - Check if an order filled",
                "'show pending trades' - View trades awaiting approval",
                "'cancel order [id]' - Cancel an unfilled order"
            ])

        # Position related
        if any(word in text_lower for word in ['position', 'holding', 'have', 'own', 'my']):
            suggestions.extend([
                "'show positions' - View current positions",
                "'manage positions' - Check for needed actions",
                "'close [symbol]' - Close a position",
                "'roll [symbol]' - Roll to new expiration"
            ])

        # Scanning/finding related
        if any(word in text_lower for word in ['find', 'look', 'search', 'trade', 'opportunity', 'good']):
            suggestions.extend([
                "'scan SPY, QQQ' - Find trading opportunities",
                "'research [symbol]' - Analyze a specific symbol",
                "'what's the IV rank on [symbol]?' - Check volatility"
            ])

        # Portfolio/account related
        if any(word in text_lower for word in ['account', 'money', 'balance', 'value', 'worth']):
            suggestions.extend([
                "'show portfolio' - View account summary",
                "'show buying power' - Check available capital",
                "'show P&L' - View profit/loss"
            ])

        # Price/modify related
        if any(word in text_lower for word in ['price', 'change', 'modify', 'adjust', 'different']):
            suggestions.extend([
                "'set price [amount]' - Change limit price",
                "'modify price [id] [amount]' - Update order price"
            ])

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)

        return unique_suggestions[:5]  # Limit to 5 suggestions

    def parse_quick(self, text: str) -> ParsedIntent:
        """
        Quick pattern-based intent parsing.
        Used for simple, common commands before falling back to LLM.
        """
        text_lower = text.lower().strip()

        # Extract symbols early for reuse
        symbols = self._extract_symbols(text)

        # Cancel order should be checked before confirmation patterns
        # (since "cancel" is in confirmation patterns)
        if self._matches_cancel_order(text_lower):
            return ParsedIntent(
                intent=TradingIntent.CANCEL_ORDER,
                confidence=0.90,
                raw_text=text,
                trade_id=self._extract_trade_id(text)
            )

        # Check for confirmation responses first (highest priority in context)
        if self._is_confirmation_yes(text_lower):
            return ParsedIntent(
                intent=TradingIntent.CONFIRM_YES,
                confidence=0.95,
                raw_text=text
            )

        if self._is_confirmation_no(text_lower):
            return ParsedIntent(
                intent=TradingIntent.CONFIRM_NO,
                confidence=0.95,
                raw_text=text
            )

        # Help
        if self._matches_help(text_lower):
            return ParsedIntent(
                intent=TradingIntent.HELP,
                confidence=0.95,
                raw_text=text
            )

        # Select option from scan results (option 2, #1, etc.)
        if self._matches_select_option(text_lower):
            return ParsedIntent(
                intent=TradingIntent.SELECT_OPTION,
                confidence=0.90,
                raw_text=text,
                parameters={'option_number': self._extract_option_number(text_lower)}
            )

        # Create trade directly (buy X shares of Y, sell Z calls, etc.)
        if self._matches_create_trade(text_lower):
            return ParsedIntent(
                intent=TradingIntent.CREATE_TRADE,
                confidence=0.85,
                raw_text=text,
                symbols=symbols,
                action=self._extract_trade_action(text_lower),
                parameters=self._extract_trade_parameters(text_lower)
            )

        # Research/strategy analysis (check before scan since both can match)
        if self._matches_research(text_lower):
            return ParsedIntent(
                intent=TradingIntent.RESEARCH_TRADE,
                confidence=0.85,
                raw_text=text,
                symbols=symbols,
                strategy=self._extract_strategy(text_lower)
            )

        # Scan/find opportunities
        if self._matches_scan(text_lower):
            return ParsedIntent(
                intent=TradingIntent.SCAN_OPPORTUNITIES,
                confidence=0.90,
                raw_text=text,
                symbols=symbols,
                strategy=self._extract_strategy(text_lower)
            )

        # Backtesting
        if self._matches_backtest(text_lower):
            return ParsedIntent(
                intent=TradingIntent.RUN_BACKTEST,
                confidence=0.90,
                raw_text=text,
                symbols=symbols,
                strategy=self._extract_strategy(text_lower),
                parameters=self._extract_backtest_params(text_lower)
            )

        # Show pending trades
        if self._matches_pending(text_lower):
            return ParsedIntent(
                intent=TradingIntent.SHOW_PENDING,
                confidence=0.90,
                raw_text=text
            )

        # Execute trade (check before approve since "execute" contains "e" patterns)
        if self._matches_execute(text_lower):
            return ParsedIntent(
                intent=TradingIntent.EXECUTE_TRADE,
                confidence=0.90,
                raw_text=text,
                trade_id=self._extract_trade_id(text),
                action="execute"
            )

        # Approve trade
        if self._matches_approve(text_lower):
            return ParsedIntent(
                intent=TradingIntent.APPROVE_TRADE,
                confidence=0.90,
                raw_text=text,
                trade_id=self._extract_trade_id(text),
                action="approve"
            )

        # Reject trade
        if self._matches_reject(text_lower):
            return ParsedIntent(
                intent=TradingIntent.REJECT_TRADE,
                confidence=0.90,
                raw_text=text,
                trade_id=self._extract_trade_id(text),
                action="reject"
            )

        # Check order status (more specific - check BEFORE working_orders)
        # Focuses on fill/execution status, not listing orders
        if self._matches_order_status(text_lower):
            return ParsedIntent(
                intent=TradingIntent.CHECK_ORDER_STATUS,
                confidence=0.90,
                raw_text=text,
                trade_id=self._extract_trade_id(text)
            )

        # Working orders (unfilled limit orders) - listing orders
        if self._matches_working_orders(text_lower):
            return ParsedIntent(
                intent=TradingIntent.WORKING_ORDERS,
                confidence=0.90,
                raw_text=text
            )

        # Modify limit price
        if self._matches_modify_price(text_lower):
            return ParsedIntent(
                intent=TradingIntent.MODIFY_PRICE,
                confidence=0.90,
                raw_text=text,
                trade_id=self._extract_trade_id(text),
                parameters=self._extract_price_params(text)
            )

        # Portfolio queries
        if self._matches_portfolio(text_lower):
            return ParsedIntent(
                intent=TradingIntent.GET_PORTFOLIO,
                confidence=0.85,
                raw_text=text
            )

        # Positions
        if self._matches_positions(text_lower):
            return ParsedIntent(
                intent=TradingIntent.GET_POSITIONS,
                confidence=0.85,
                raw_text=text
            )

        # Buying power
        if self._matches_buying_power(text_lower):
            return ParsedIntent(
                intent=TradingIntent.GET_BUYING_POWER,
                confidence=0.85,
                raw_text=text
            )

        # P&L
        if self._matches_pnl(text_lower):
            return ParsedIntent(
                intent=TradingIntent.GET_PNL,
                confidence=0.85,
                raw_text=text
            )

        # IV Rank
        if self._matches_iv_rank(text_lower):
            return ParsedIntent(
                intent=TradingIntent.GET_IV_RANK,
                confidence=0.85,
                raw_text=text,
                symbols=symbols
            )

        # Close position (explicit close command)
        if self._matches_close_position(text_lower):
            return ParsedIntent(
                intent=TradingIntent.CLOSE_POSITION,
                confidence=0.90,
                raw_text=text,
                symbols=symbols,
                action="close"
            )

        # Roll position (explicit roll command)
        if self._matches_roll_position(text_lower):
            return ParsedIntent(
                intent=TradingIntent.ROLL_POSITION,
                confidence=0.90,
                raw_text=text,
                symbols=symbols,
                action="roll"
            )

        # Position analysis (should I close X?) - check before general manage
        # This is more specific when a symbol is mentioned
        if self._matches_position_analysis(text_lower) and symbols:
            return ParsedIntent(
                intent=TradingIntent.POSITION_ANALYSIS,
                confidence=0.85,
                raw_text=text,
                symbols=symbols
            )

        # Position management (general)
        if self._matches_manage(text_lower):
            return ParsedIntent(
                intent=TradingIntent.MANAGE_POSITIONS,
                confidence=0.85,
                raw_text=text,
                symbols=symbols
            )

        # Market analysis
        if self._matches_market(text_lower):
            return ParsedIntent(
                intent=TradingIntent.MARKET_ANALYSIS,
                confidence=0.80,
                raw_text=text
            )

        # Risk parameters
        if self._matches_risk(text_lower):
            return ParsedIntent(
                intent=TradingIntent.GET_RISK_PARAMS,
                confidence=0.85,
                raw_text=text
            )

        # Display mode settings (/plain, /rich, etc.)
        if self._matches_display_mode(text_lower):
            mode = self._extract_display_mode(text_lower)
            return ParsedIntent(
                intent=TradingIntent.SET_DISPLAY_MODE,
                confidence=0.95,
                raw_text=text,
                parameters={'mode': mode}
            )

        # Layout settings (/layout table, etc.)
        if self._matches_layout(text_lower):
            layout = self._extract_layout(text_lower)
            return ParsedIntent(
                intent=TradingIntent.SET_LAYOUT,
                confidence=0.95,
                raw_text=text,
                parameters={'layout': layout}
            )

        # If we have a symbol and nothing else matched, might be a query about it
        if symbols:
            return ParsedIntent(
                intent=TradingIntent.GENERAL_CHAT,
                confidence=0.50,
                raw_text=text,
                symbols=symbols
            )

        # Default to general chat with low confidence
        return ParsedIntent(
            intent=TradingIntent.GENERAL_CHAT,
            confidence=0.40,
            raw_text=text
        )

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []

        # Common words to exclude (not stock symbols in this context)
        common_words = {
            'ON', 'IN', 'AT', 'TO', 'FOR', 'THE', 'IS', 'IT', 'MY', 'ME', 'DO', 'IF', 'OR', 'AN', 'AS', 'BE',
            'OF', 'A', 'WHAT', 'HOW', 'SHOW', 'GET', 'FIND', 'SCAN', 'IV', 'RANK', 'PUT', 'CALL', 'IRON',
            'SPREAD', 'CONDOR', 'STRANGLE', 'STRADDLE', 'CREDIT', 'DEBIT', 'BUY', 'SELL', 'OPEN', 'CLOSE',
            'APPROVE', 'REJECT', 'EXECUTE', 'PENDING', 'TRADE', 'TRADES', 'POSITION', 'POSITIONS',
            'PORTFOLIO', 'DELTA', 'THETA', 'VEGA', 'GAMMA', 'PNL', 'TODAY', 'WEEK', 'MARKET', 'RESEARCH',
            'ANALYZE', 'CHECK', 'HELP', 'ABOUT', 'CAN', 'YOU', 'SHOULD', 'WOULD', 'COULD', 'YES', 'NO',
            'LOOK', 'LOOKING', 'WITH', 'AND', 'LIKE', 'WANT', 'NEED', 'HAVE', 'HAS', 'HAD', 'WILL', 'ARE',
            'RUN', 'TEST', 'BACK', 'BACKTEST', 'SHARE', 'SHARES', 'STOCK', 'STOCKS', 'CONTRACT', 'CONTRACTS'
        }

        # Look for explicit symbols (uppercase 1-5 letters)
        words = text.upper().split()
        for word in words:
            # Clean punctuation
            clean = re.sub(r'[^\w]', '', word)

            # First check if it's a known symbol
            if clean in self.COMMON_SYMBOLS:
                symbols.append(clean)
            # Then check if it looks like a symbol and isn't a common word
            elif re.match(r'^[A-Z]{2,5}$', clean) and clean not in common_words:
                symbols.append(clean)

        return symbols

    def _extract_strategy(self, text: str) -> Optional[str]:
        """Extract strategy type from text"""
        text_lower = text.lower()
        for keyword, strategy in self.STRATEGY_KEYWORDS.items():
            if keyword in text_lower:
                return strategy
        return None

    def _extract_trade_id(self, text: str) -> Optional[str]:
        """Extract trade ID from text"""
        # Look for hex-like IDs (e.g., a7b2c3d4)
        match = re.search(r'\b([a-f0-9]{6,8})\b', text.lower())
        if match:
            return match.group(1)
        return None

    def _extract_backtest_params(self, text: str) -> Dict[str, Any]:
        """Extract backtest parameters from text"""
        params = {}

        # Extract time period
        if 'month' in text:
            months_match = re.search(r'(\d+)\s*month', text)
            if months_match:
                params['months'] = int(months_match.group(1))
            else:
                params['months'] = 6  # Default to 6 months

        if 'year' in text:
            years_match = re.search(r'(\d+)\s*year', text)
            if years_match:
                params['years'] = int(years_match.group(1))
            else:
                params['years'] = 1  # Default to 1 year

        # Extract mode preference
        if 'api' in text or 'real' in text or 'actual' in text:
            params['mode'] = 'api'
        elif 'synthetic' in text or 'simulated' in text:
            params['mode'] = 'synthetic'
        else:
            params['mode'] = 'auto'  # Auto-detect

        # Extract comparison flag
        if 'compare' in text or 'comparison' in text or 'all strategies' in text:
            params['compare_strategies'] = True

        return params

    # Pattern matching methods
    def _is_confirmation_yes(self, text: str) -> bool:
        # Don't match if it contains specific action words like "trade", "position"
        if any(word in text for word in ['trade', 'position', 'the ', 'that ', 'this ']):
            return False

        patterns = [
            r'^yes\b', r'^yeah\b', r'^yep\b', r'^sure\b',
            r'^ok\b', r'^okay\b', r'^do it\b', r'^go ahead\b',
            r'^confirm\b', r'^please do\b', r'^yes please\b', r"^let's do it\b"
        ]
        return any(re.search(p, text) for p in patterns)

    def _is_confirmation_no(self, text: str) -> bool:
        # Don't match if it contains specific action words
        if any(word in text for word in ['trade', 'position', 'the ', 'that ', 'this ']):
            return False

        patterns = [
            r'^no\b', r'^nope\b', r'^nah\b', r'^cancel\b',
            r'^stop\b', r'^nevermind\b', r'^never mind\b',
            r"^don't\b", r'^not now\b'
        ]
        return any(re.search(p, text) for p in patterns)

    def _matches_help(self, text: str) -> bool:
        patterns = [
            'help', 'what can you do', 'commands', 'how do i', 'how to',
            'what commands', 'show commands', 'list commands', 'usage',
            'what are my options', 'what can i say', 'how does this work'
        ]
        return any(p in text for p in patterns)

    def _matches_scan(self, text: str) -> bool:
        patterns = [
            'scan', 'find', 'look for', 'search', 'opportunities', 'trades on',
            'what looks good', 'any trades', 'find me', 'show me trades',
            'check for trades', 'trade ideas', 'what can i trade',
            'anything interesting', 'good opportunities', 'what should i trade',
            'looking for trades', 'hunt for', 'scout'
        ]
        return any(p in text for p in patterns)

    def _matches_pending(self, text: str) -> bool:
        patterns = [
            'pending', 'waiting', 'awaiting', 'queued', 'show trades',
            'trades i have', 'my trades', 'unapproved', 'not yet approved',
            'trades to approve', 'what needs approval'
        ]
        return any(p in text for p in patterns)

    def _matches_approve(self, text: str) -> bool:
        patterns = ['approve', 'accept', 'green light', 'thumbs up', 'looks good']
        # Must not be a question about approval
        if re.search(r'(should|can|do)\s+i\s+approve', text, re.IGNORECASE):
            return False
        return any(p in text for p in patterns)

    def _matches_reject(self, text: str) -> bool:
        patterns = ['reject', 'decline', 'pass on', 'skip this', 'thumbs down']
        return any(p in text for p in patterns)

    def _matches_execute(self, text: str) -> bool:
        # Check for execute intent, but NOT past tense questions about execution
        if re.search(r'(did|has|was|is).*(execute|filled)', text, re.IGNORECASE):
            return False  # This is asking about status, not requesting execution
        # Don't match if asking to view/show/list orders (those are queries, not execution)
        if re.search(r'(view|show|list|see)\s+.*(order|trade)', text, re.IGNORECASE):
            return False
        patterns = [
            'execute', 'place order', 'submit', 'send order',
            'pull the trigger', 'do it', 'make it happen', 'go ahead',
            'place the trade', 'submit the trade', 'send it'
        ]
        return any(p in text for p in patterns)

    def _matches_portfolio(self, text: str) -> bool:
        patterns = [
            'portfolio', 'account', 'holdings', 'my delta', 'my theta',
            'account summary', 'how am i looking', 'overview',
            'net liquidating', 'nlv', 'account value', 'my account'
        ]
        return any(p in text for p in patterns)

    def _matches_positions(self, text: str) -> bool:
        patterns = [
            'positions', 'what do i own', 'what am i holding', 'open positions',
            'what do i have', 'my holdings', 'what am i in', 'current trades',
            'open trades', 'show my trades', 'my open', 'what i own'
        ]
        return any(p in text for p in patterns)

    def _matches_buying_power(self, text: str) -> bool:
        patterns = [
            'buying power', 'bp', 'capital', 'available funds', 'how much can i',
            'available to trade', 'cash available', 'margin available',
            'what can i spend', 'trading power', 'funds available'
        ]
        return any(p in text for p in patterns)

    def _matches_pnl(self, text: str) -> bool:
        patterns = [
            'p&l', 'pnl', 'profit', 'loss', 'how am i doing', 'performance',
            'gains', 'losses', 'made money', 'lost money', 'up or down',
            'in the green', 'in the red', 'my returns', 'how much have i made'
        ]
        return any(p in text for p in patterns)

    def _matches_iv_rank(self, text: str) -> bool:
        patterns = [
            'iv rank', 'iv percentile', 'implied volatility', 'volatility rank',
            'iv on', 'volatility on', 'how volatile', 'is iv high',
            'is volatility high', 'ivr', 'ivp'
        ]
        return any(p in text for p in patterns)

    def _matches_research(self, text: str) -> bool:
        patterns = [
            'research', 'analyze', 'analysis', 'look at', 'what about',
            'tell me about', 'info on', 'information on', 'details on',
            'how does', 'what do you think about', 'evaluate'
        ]
        return any(p in text for p in patterns)

    def _matches_manage(self, text: str) -> bool:
        patterns = [
            'manage', 'check positions', 'should i close', 'roll', 'management',
            'check if i should close', 'anything to close', 'positions to manage',
            'need attention', 'needs action', 'check my positions',
            'any positions need', 'position check', 'review positions'
        ]
        return any(p in text for p in patterns)

    def _matches_backtest(self, text: str) -> bool:
        patterns = [
            'backtest', 'back test', 'historical test', 'test historically',
            'past performance', 'simulate', 'how would', 'performance test',
            'test strategy', 'historical data', 'run simulation'
        ]
        return any(p in text for p in patterns)

    def _matches_market(self, text: str) -> bool:
        patterns = [
            'market', 'how is the market', "how's the market", 'market overview',
            'market conditions', 'market today', 'market looking',
            'spy doing', 'qqq doing', 'indices', 'vix'
        ]
        return any(p in text for p in patterns)

    def _matches_risk(self, text: str) -> bool:
        patterns = [
            'risk', 'limits', 'parameters', 'constraints', 'rules',
            'risk settings', 'my limits', 'trading limits', 'max loss',
            'position limits', 'risk rules', 'risk params'
        ]
        return any(p in text for p in patterns)

    def _matches_position_analysis(self, text: str) -> bool:
        patterns = [
            'should i close', 'should i roll', 'what to do with', 'analyze position',
            'what should i do with', 'how is my position', 'position doing',
            'check on my', 'hows my', "how's my", 'status of my'
        ]
        return any(p in text for p in patterns)

    def _matches_close_position(self, text: str) -> bool:
        """Match explicit close position commands"""
        simple_patterns = [
            'close position', 'close my', 'close the', 'close out',
            'btc', 'buy to close', 'exit position', 'exit my',
            'get out of', 'sell my position', 'close it', 'exit trade',
            'take off', 'take it off', 'flatten'
        ]
        # Check simple patterns first
        if any(p in text for p in simple_patterns):
            return True

        # Check regex patterns
        regex_patterns = [
            r'^close\s+\w+',  # "close SPY" at start
            r'close\s+(the\s+)?[A-Z]{1,5}\b',  # "close SPY" or "close the SPY"
        ]
        for p in regex_patterns:
            if re.search(p, text, re.IGNORECASE):
                return True

        return False

    def _matches_roll_position(self, text: str) -> bool:
        """Match explicit roll position commands"""
        simple_patterns = [
            'roll position', 'roll my', 'roll the', 'roll out',
            'roll it', 'roll forward', 'extend expiration', 'push out',
            'move to next expiration', 'roll over'
        ]
        # Check simple patterns first
        if any(p in text for p in simple_patterns):
            return True

        # Check regex patterns
        regex_patterns = [
            r'^roll\s+\w+',  # "roll SPY" at start
            r'roll\s+(the\s+)?[A-Z]{1,5}\b',  # "roll SPY" or "roll the SPY"
        ]
        for p in regex_patterns:
            if re.search(p, text, re.IGNORECASE):
                return True

        return False

    def _matches_select_option(self, text: str) -> bool:
        """Match option selection patterns (option 2, #1, the second one)"""
        patterns = [
            r'option\s+\d+',
            r'#\d+',
            r'\b(first|second|third|1st|2nd|3rd)\s+(one|option|choice)',
            r'go with\s+(option\s+)?\d+',
            r'(choose|pick|select)\s+(option\s+)?\d+',
            r'let\'?s?\s+go\s+with\s+(option\s+)?\d+',
            r'i\'?ll\s+take\s+(option\s+)?\d+',
        ]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    def _matches_create_trade(self, text: str) -> bool:
        """Match buy/sell trade requests"""
        patterns = [
            r'\b(buy|purchase|long)\b.*\b(share|stock|call|put|contract)s?\b',
            r'\b(sell|short)\b.*\b(share|stock|call|put|contract)s?\b',
            r'\b\d+\s+(share|stock|call|put|contract)s?\s+of\b',
            r'\b(buy|sell|purchase|short|long)\b.*\b[A-Z]{1,5}\b',
        ]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    def _extract_option_number(self, text: str) -> int:
        """Extract option number from text (option 2 -> 2, #3 -> 3)"""
        # Try "option 2", "#2" pattern
        match = re.search(r'(?:option|#)\s*(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Try ordinal words
        ordinals = {
            'first': 1, '1st': 1,
            'second': 2, '2nd': 2,
            'third': 3, '3rd': 3,
            'fourth': 4, '4th': 4,
            'fifth': 5, '5th': 5
        }
        text_lower = text.lower()
        for word, num in ordinals.items():
            if word in text_lower:
                return num

        return 1  # default

    def _extract_trade_action(self, text: str) -> str:
        """Extract buy/sell action from text"""
        if re.search(r'\b(buy|purchase|long)\b', text, re.IGNORECASE):
            return "buy"
        elif re.search(r'\b(sell|short)\b', text, re.IGNORECASE):
            return "sell"
        return "buy"  # default

    def _extract_trade_parameters(self, text: str) -> Dict[str, Any]:
        """Extract quantity and instrument type from text"""
        params = {
            'quantity': 1,
            'instrument_type': 'stock'
        }

        # Extract quantity (1 share, 5 shares, etc.)
        qty_match = re.search(r'(\d+)\s+(share|stock|call|put|contract)', text, re.IGNORECASE)
        if qty_match:
            params['quantity'] = int(qty_match.group(1))

        # Extract instrument type
        if re.search(r'\bcall', text, re.IGNORECASE):
            params['instrument_type'] = 'call'
        elif re.search(r'\bput', text, re.IGNORECASE):
            params['instrument_type'] = 'put'
        elif re.search(r'\b(share|stock)', text, re.IGNORECASE):
            params['instrument_type'] = 'stock'

        return params

    # Order management pattern matchers
    def _matches_working_orders(self, text: str) -> bool:
        """
        Match working/unfilled order queries with flexible natural language.
        Focus on LISTING orders, not checking fill status.

        Examples that should match:
        - "show working orders"
        - "any unfilled orders?"
        - "what orders are still open"
        - "list my orders"
        - "orders that haven't filled"
        - "check on my orders"
        """
        # Simple keyword patterns - focus on working/unfilled/open orders
        simple_patterns = [
            'working order', 'unfilled order', 'open order',
            'active order', 'submitted order',
            'show order', 'list order', 'my order',
            'orders out', 'orders in'
        ]
        if any(p in text for p in simple_patterns):
            return True

        # Regex patterns for more natural language - listing orders
        regex_patterns = [
            r'order.*(not|hasn\'?t|haven\'?t).*(fill|complete|execute)',  # orders not filled
            r'(any|what|which|show|list).*(order).*(waiting|open|active|out)',
            r'(waiting|working).*(order)',
            r'orders?\s+(i|we)\s+(have|placed|submitted)',
            r'check\s+on\s+(my\s+)?order',
            r'(see|view)\s+(my\s+)?order',
        ]
        for pattern in regex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _matches_modify_price(self, text: str) -> bool:
        """
        Match limit price modification requests with flexible natural language.

        Examples that should match:
        - "set price to 2.50"
        - "change the limit price"
        - "I want a better price"
        - "adjust the price to 1.75"
        - "make it 2.00"
        - "use 1.50 as the price"
        - "can we get 2.25?"
        - "try for 1.80"
        """
        # Simple keyword patterns
        simple_patterns = [
            'modify price', 'change price', 'update price',
            'set price', 'new price', 'adjust price',
            'limit price', 'change limit', 'update limit',
            'different price', 'better price'
        ]
        if any(p in text for p in simple_patterns):
            return True

        # Regex patterns for more natural language
        regex_patterns = [
            r'(set|make|use|try)\s+(it\s+)?(to\s+|for\s+|at\s+)?\$?\d+\.?\d*',  # "set it to 2.50"
            r'(price|limit)\s+(of|at|to)\s+\$?\d+\.?\d*',  # "price of 2.50"
            r'\$?\d+\.?\d*\s+(as\s+)?(the\s+)?(price|limit)',  # "2.50 as the price"
            r'(can\s+we|let\'?s?|try\s+(for|to))\s+(get\s+)?\$?\d+\.?\d*',  # "can we get 2.50"
            r'(want|need|prefer)\s+(a\s+)?(price|limit)\s+(of|at)?\s*\$?\d+\.?\d*',
            r'(raise|lower|increase|decrease)\s+(the\s+)?(price|limit)',
            r'(improve|better)\s+(the\s+)?price',
        ]
        for pattern in regex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _matches_cancel_order(self, text: str) -> bool:
        """
        Match order cancellation requests with flexible natural language.

        Examples that should match:
        - "cancel the order"
        - "kill my order"
        - "don't execute the trade"
        - "withdraw the order"
        - "pull the order"
        - "nevermind on that order"
        - "forget about the trade"
        """
        # Simple keyword patterns
        simple_patterns = [
            'cancel order', 'cancel the order', 'cancel my order',
            'cancel that order', 'cancel this order',
            'kill order', 'kill the order', 'kill my order',
            'delete order', 'remove order', 'withdraw order',
            'withdraw the order', 'withdraw my order',
            'pull the order', 'pull my order',
            'nevermind order', 'nevermind the order',
            'never mind order', 'never mind the order',
        ]
        if any(p in text for p in simple_patterns):
            return True

        # Regex patterns for more natural language
        regex_patterns = [
            r'(don\'?t|do\s+not)\s+(execute|submit|place|send)\s+(the\s+|my\s+)?(order|trade)',
            r'(stop|abort|halt)\s+(the\s+|my\s+)?(order|trade)',
            r'(nevermind|never\s*mind)\s+(about\s+)?(the\s+|that\s+|my\s+|on\s+)?(order|trade|that)',
            r'(scratch|nix|scrap)\s+(the\s+|that\s+|my\s+)?(order|trade)',
            r'(take\s+back|pull\s+back)\s+(the\s+|my\s+)?(order|trade)',
            r'forget\s+(about\s+)?(the\s+|that\s+|my\s+)?(order|trade)',
            r'(cancel|kill|withdraw)\s+it\b',  # "cancel it"
        ]
        for pattern in regex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _matches_order_status(self, text: str) -> bool:
        """
        Match order status check requests with flexible natural language.
        Focus on FILL/EXECUTION STATUS, not listing orders.

        Examples that should match:
        - "did my order fill?"
        - "is it filled?"
        - "what's the fill status"
        - "order go through?"
        - "did we get filled?"
        - "confirm the execution"
        """
        # Simple keyword patterns - focus on fill/execution status
        simple_patterns = [
            'fill status', 'execution status',
            'filled yet', 'executed yet',
            'is it filled', 'was it filled',
            'did it fill', 'has it filled',
            'get filled', 'got filled',
            'what happened to my order', "what's happening with",
            'status of my order', 'where is my order',
            'did the order go through', 'order status',
            'any fills', 'any executions'
        ]
        if any(p in text for p in simple_patterns):
            return True

        # Regex patterns focused on checking fill/execution
        regex_patterns = [
            r'(did|has|have)\s+(my\s+|the\s+|that\s+|it\s+)?(order|trade)?\s*(been\s+)?(fill|execute)',
            r'(is|was)\s+(it\s+|the\s+order\s+|my\s+order\s+)?(fill|execute)',
            r'(order|trade)\s+(go|went|gone)\s+through',
            r'(we|i|it)\s+(get\s+)?fill',
            r'(confirm|verify)\s+(the\s+)?(fill|execution)',
            r'(check|what).*(fill|execution)\s*(status)?$',
            r'\bfill(ed)?\?',  # "filled?" at end
            r'(order|trade)\s+(been\s+)?(execute|fill)',  # "order been executed"
            r'what\s+happened\s+(to|with)',  # "what happened to/with..."
        ]
        for pattern in regex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _extract_price_params(self, text: str) -> Dict[str, Any]:
        """
        Extract new price from text with flexible parsing.

        Handles formats like:
        - "set price 2.50"
        - "change price to $1.75"
        - "make it 2.00"
        - "try for 1.80"
        - "use $2.25"
        - "2.50 please"
        """
        params = {}

        # Try to find price patterns with context words
        price_patterns = [
            r'(?:price|limit)\s*(?:of|at|to|=)?\s*\$?(\d+\.?\d*)',  # price to 2.50
            r'(?:set|make|use|try)\s+(?:it\s+)?(?:to\s+|for\s+|at\s+)?\$?(\d+\.?\d*)',  # set it to 2.50
            r'\$(\d+\.?\d*)',  # $2.50
            r'(\d+\.\d{2})\b',  # 2.50 (must have decimal for standalone)
            r'(?:get|want|need)\s+\$?(\d+\.?\d*)',  # get 2.50
        ]

        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    if 0 < price < 1000:  # Sanity check for option prices
                        params['new_price'] = price
                        break
                except ValueError:
                    continue

        return params

    def _matches_display_mode(self, text: str) -> bool:
        """Match display mode commands (/plain, /rich, etc.)"""
        patterns = [
            r'^/plain\b', r'^/rich\b', r'^/display\b',
            r'^plain\s*mode', r'^rich\s*mode', r'^text\s*mode',
            r'(switch|change)\s+(to\s+)?(plain|rich|text)\s*(mode|display)?',
            r'(turn|switch)\s+(off|on)\s+(rich|formatting)',
            r'(disable|enable)\s+(rich|formatting)',
            r'(no|without)\s+(formatting|colors)',
            r'plain\s*text', r'debug\s*mode'
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_display_mode(self, text: str) -> str:
        """Extract display mode from text (plain/rich)"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['plain', 'text', 'debug', 'off', 'disable', 'no ', 'without']):
            return 'plain'
        elif any(word in text_lower for word in ['rich', 'on', 'enable', 'color']):
            return 'rich'
        return 'toggle'  # Default to toggle

    def _matches_layout(self, text: str) -> bool:
        """Match layout commands (/layout table, etc.)"""
        patterns = [
            r'^/layout\b',
            r'(set|change|switch)\s+(to\s+)?(layout|display)\s+(to\s+)?(table|cards?|compact|auto)',
            r'(use|show)\s+(table|cards?|compact|auto)\s*(layout|view|mode)?',
            r'(table|cards?|compact|auto)\s+(layout|view|mode|display)'
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_layout(self, text: str) -> str:
        """Extract layout type from text"""
        text_lower = text.lower()
        if 'table' in text_lower:
            return 'table'
        elif 'card' in text_lower:
            return 'cards'
        elif 'compact' in text_lower:
            return 'compact'
        return 'auto'

    async def route(
        self,
        parsed: ParsedIntent,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Route a parsed intent to its handler.

        Args:
            parsed: The parsed intent
            context: Optional context dict with bot, conversation state, etc.

        Returns:
            Response string from the handler
        """
        handler = self.handlers.get(parsed.intent)

        if handler is None:
            # Check for fallback handler
            handler = self.handlers.get(TradingIntent.GENERAL_CHAT)

        if handler is None:
            return "I'm not sure how to handle that request."

        try:
            return await handler(parsed, context)
        except Exception as e:
            logger.error(f"Handler error for {parsed.intent}: {e}")
            return f"Sorry, there was an error processing your request: {str(e)}"
