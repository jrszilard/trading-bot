#!/usr/bin/env python3
"""
Intent Router for Tastytrade Chatbot

Handles intent classification and routing for natural language
trading commands.

Author: Trading Bot
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable
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

    def __init__(self):
        self.handlers: Dict[TradingIntent, Callable] = {}

    def register_handler(
        self,
        intent: TradingIntent,
        handler: Callable[..., Awaitable[str]]
    ) -> None:
        """Register a handler for an intent"""
        self.handlers[intent] = handler

    def parse_quick(self, text: str) -> ParsedIntent:
        """
        Quick pattern-based intent parsing.
        Used for simple, common commands before falling back to LLM.
        """
        text_lower = text.lower().strip()

        # Extract symbols early for reuse
        symbols = self._extract_symbols(text)

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
        patterns = ['help', 'what can you do', 'commands', 'how do i', 'how to']
        return any(p in text for p in patterns)

    def _matches_scan(self, text: str) -> bool:
        patterns = ['scan', 'find', 'look for', 'search', 'opportunities', 'trades on']
        return any(p in text for p in patterns)

    def _matches_pending(self, text: str) -> bool:
        patterns = ['pending', 'waiting', 'awaiting', 'queued', 'show trades']
        return any(p in text for p in patterns)

    def _matches_approve(self, text: str) -> bool:
        return 'approve' in text

    def _matches_reject(self, text: str) -> bool:
        return 'reject' in text

    def _matches_execute(self, text: str) -> bool:
        patterns = ['execute', 'place order', 'submit', 'send order']
        return any(p in text for p in patterns)

    def _matches_portfolio(self, text: str) -> bool:
        patterns = ['portfolio', 'account', 'holdings', 'my delta', 'my theta']
        return any(p in text for p in patterns)

    def _matches_positions(self, text: str) -> bool:
        patterns = ['positions', 'what do i own', 'what am i holding', 'open positions']
        return any(p in text for p in patterns)

    def _matches_buying_power(self, text: str) -> bool:
        patterns = ['buying power', 'bp', 'capital', 'available funds', 'how much can i']
        return any(p in text for p in patterns)

    def _matches_pnl(self, text: str) -> bool:
        patterns = ['p&l', 'pnl', 'profit', 'loss', 'how am i doing', 'performance']
        return any(p in text for p in patterns)

    def _matches_iv_rank(self, text: str) -> bool:
        patterns = ['iv rank', 'iv percentile', 'implied volatility', 'volatility rank']
        return any(p in text for p in patterns)

    def _matches_research(self, text: str) -> bool:
        patterns = ['research', 'analyze', 'analysis', 'look at', 'what about']
        return any(p in text for p in patterns)

    def _matches_manage(self, text: str) -> bool:
        patterns = ['manage', 'check positions', 'should i close', 'roll', 'management',
                    'check if i should close', 'anything to close', 'positions to manage']
        return any(p in text for p in patterns)

    def _matches_backtest(self, text: str) -> bool:
        patterns = ['backtest', 'back test', 'historical test', 'test historically',
                    'past performance', 'simulate', 'how would', 'performance test']
        return any(p in text for p in patterns)

    def _matches_market(self, text: str) -> bool:
        patterns = ['market', 'how is the market', "how's the market", 'market overview', 'market conditions']
        return any(p in text for p in patterns)

    def _matches_risk(self, text: str) -> bool:
        patterns = ['risk', 'limits', 'parameters', 'constraints', 'rules']
        return any(p in text for p in patterns)

    def _matches_position_analysis(self, text: str) -> bool:
        patterns = ['should i close', 'should i roll', 'what to do with', 'analyze position']
        return any(p in text for p in patterns)

    def _matches_close_position(self, text: str) -> bool:
        """Match explicit close position commands"""
        patterns = [
            'close position', 'close my', 'close the', 'close out',
            'btc', 'buy to close',
            r'^close\s+\w+',  # "close SPY" at start
        ]
        # Check simple patterns
        for p in patterns:
            if p.startswith('^'):
                import re
                if re.search(p, text):
                    return True
            elif p in text:
                return True
        return False

    def _matches_roll_position(self, text: str) -> bool:
        """Match explicit roll position commands"""
        patterns = [
            'roll position', 'roll my', 'roll the', 'roll out',
            r'^roll\s+\w+',  # "roll SPY" at start
        ]
        # Check simple patterns
        for p in patterns:
            if p.startswith('^'):
                import re
                if re.search(p, text):
                    return True
            elif p in text:
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
