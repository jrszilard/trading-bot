#!/usr/bin/env python3
"""
Tastytrade Trading Bot with Tastylive Best Practices

This bot implements the tastylive trading methodology:
- Sell options when IV is high (IV Rank > 30)
- Target 45 DTE expiration
- Sell options at ~30 delta (or 16 delta for more conservative)
- Take profit at 50% of max premium
- Manage/roll at 21 DTE
- Trade small and often (position sizing)
- Beta-weight portfolio to SPY

Risk Management Features:
- Maximum loss limits (daily, weekly, position-level)
- Trade approval workflow before execution
- Portfolio beta weighting
- Position size limits based on account value

Author: Trading Bot
License: MIT
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Optional, List, Callable, Any, Dict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Import models from separate module
from models import (
    TradeStatus,
    StrategyType,
    RiskParameters,
    TradeProposal,
    PortfolioState,
    TradeCandidate
)

# Import risk management functions
from risk_manager import check_risk_limits, generate_trade_rationale

# Import strategy engine
from strategy_engine import StrategyEngine

# Import Claude Trade Advisor
try:
    from claude_advisor import ClaudeTradeAdvisor, ClaudeAnalysis, ManagementAnalysis
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ClaudeTradeAdvisor = None

# Import Mock Data Provider
try:
    from mock_data import MockDataProvider, create_mock_provider
    MOCK_DATA_AVAILABLE = True
except ImportError:
    MOCK_DATA_AVAILABLE = False
    MockDataProvider = None
    create_mock_provider = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TastytradeBot:
    """
    Main trading bot implementing tastylive methodology
    
    Key Features:
    1. Scans for opportunities based on IV Rank and probability
    2. Proposes trades following tastylive mechanics
    3. Requires approval before execution
    4. Enforces strict risk management
    5. Manages positions at 21 DTE or 50% profit
    """
    
    def __init__(
        self,
        session=None,
        risk_params: Optional[RiskParameters] = None,
        approval_callback: Optional[Callable[[TradeProposal], bool]] = None,
        sandbox_mode: bool = True,
        claude_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trading bot

        Args:
            session: Tastytrade session object
            risk_params: Risk management parameters
            approval_callback: Function to call for trade approval
            sandbox_mode: If True, use sandbox environment
            claude_config: Configuration for Claude AI advisor
            config: Full configuration dict (for mock data settings, etc.)
        """
        self.session = session
        self.risk_params = risk_params or RiskParameters()
        self.approval_callback = approval_callback
        self.sandbox_mode = sandbox_mode
        self.config = config or {}

        # State tracking
        self.portfolio_state = PortfolioState()
        self.pending_trades: List[TradeProposal] = []
        self.trade_history: List[TradeProposal] = []

        # Watchlist for scanning
        self.watchlist: List[str] = []

        # Initialize Mock Data Provider (for sandbox testing without real market data)
        self.mock_provider = None
        self.use_mock_data = False
        mock_config = self.config.get('mock_data', {})
        if MOCK_DATA_AVAILABLE and mock_config.get('enabled', False):
            if sandbox_mode:
                self.mock_provider = MockDataProvider(self.config)
                self.use_mock_data = True
                logger.info("Mock data provider initialized (sandbox mode with mock market data)")
            else:
                logger.warning("Mock data is only available in sandbox mode")
        elif not MOCK_DATA_AVAILABLE and mock_config.get('enabled', False):
            logger.warning("Mock data requested but mock_data module not available")

        # Initialize Claude AI Trade Advisor
        self.claude_advisor = None
        self.claude_config = claude_config or {}
        if CLAUDE_AVAILABLE and self.claude_config.get('enabled', True):
            self.claude_advisor = ClaudeTradeAdvisor(
                model=self.claude_config.get('model', 'claude-sonnet-4-20250514'),
                max_tokens=self.claude_config.get('max_tokens', 1024),
                temperature=self.claude_config.get('temperature', 0.3)
            )
            if self.claude_advisor.is_available:
                logger.info("Claude AI Trade Advisor initialized")
            else:
                logger.warning("Claude AI Trade Advisor not available (missing API key)")
                self.claude_advisor = None
        elif not CLAUDE_AVAILABLE:
            logger.info("Claude advisor module not available")

        # Initialize Strategy Engine
        self.strategy_engine = StrategyEngine(self.risk_params)

        mock_status = ", mock_data=enabled" if self.use_mock_data else ""
        logger.info(f"TastytradeBot initialized (sandbox_mode={sandbox_mode}{mock_status})")
    
    async def connect(
        self,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None
    ) -> bool:
        """
        Connect to Tastytrade API using OAuth2

        OAuth Setup (one-time):
        1. Go to https://developer.tastytrade.com/ and create an OAuth application
        2. Check required scopes, add http://localhost:8000 as callback
        3. Save your client_secret
        4. Go to OAuth Applications > Manage > Create Grant to get a refresh_token

        Args:
            client_secret: OAuth client secret (or set TT_CLIENT_SECRET env var)
            refresh_token: OAuth refresh token (or set TT_REFRESH_TOKEN env var)

        Environment variables (recommended for security):
            TT_CLIENT_SECRET: Your OAuth client secret
            TT_REFRESH_TOKEN: Your OAuth refresh token

        Note: Refresh tokens last forever. Session tokens expire every 15 minutes
        but are automatically refreshed when needed.
        """
        try:
            from tastytrade import Session

            # Get credentials from args or environment
            secret = client_secret or os.environ.get('TT_CLIENT_SECRET')
            token = refresh_token or os.environ.get('TT_REFRESH_TOKEN')

            if not secret or not token:
                logger.error(
                    "Missing OAuth credentials. Provide client_secret and refresh_token "
                    "as arguments or set TT_CLIENT_SECRET and TT_REFRESH_TOKEN environment variables."
                )
                return False

            # Create session with sandbox mode if enabled
            self.session = Session(secret, token, is_test=self.sandbox_mode)
            self._session_secret = secret
            self._session_token = token

            env_type = "SANDBOX" if self.sandbox_mode else "PRODUCTION"
            logger.info(f"Connected to Tastytrade API ({env_type})")
            return True

        except ImportError:
            logger.error("tastytrade package not installed. Run: pip install tastytrade")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def ensure_session_valid(self) -> bool:
        """
        Ensure the session token is valid, refreshing if needed.

        OAuth session tokens expire every 15 minutes. This method checks
        expiration and refreshes automatically.

        Returns:
            True if session is valid, False if refresh failed
        """
        if not self.session:
            logger.error("No active session")
            return False

        try:
            from tastytrade.utils import now_in_new_york

            # Check if session token is expired or about to expire (1 min buffer)
            if now_in_new_york() >= self.session.session_expiration - timedelta(minutes=1):
                logger.info("Session token expired, refreshing...")
                self.session.refresh()
                logger.info("Session token refreshed successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to refresh session: {e}")
            return False
    
    async def update_portfolio_state(self) -> PortfolioState:
        """
        Fetch current portfolio state from Tastytrade

        Updates:
        - Account balances and buying power
        - Current P&L
        - Beta-weighted portfolio Greeks
        - Position counts
        """
        if not self.session:
            logger.warning("No active session - using mock portfolio state")
            return self.portfolio_state

        # Ensure session token is valid (refreshes if expired)
        if not await self.ensure_session_valid():
            logger.error("Session validation failed")
            return self.portfolio_state
        
        try:
            from tastytrade import Account
            
            # Get first account (could be extended to support multiple)
            accounts = Account.get(self.session)
            if not accounts:
                logger.error("No accounts found")
                return self.portfolio_state
            
            account = accounts[0]
            
            # Get balances (with fallback for uninitialized sandbox accounts)
            try:
                balances = account.get_balances(self.session)
                self.portfolio_state.account_value = Decimal(str(balances.margin_equity or 0))
                self.portfolio_state.buying_power = Decimal(str(balances.derivative_buying_power or 0))
                self.portfolio_state.net_liquidating_value = Decimal(str(balances.net_liquidating_value or 0))
                self.portfolio_state.cash_balance = Decimal(str(balances.cash_balance or 0))
            except Exception as balance_error:
                if "record_not_found" in str(balance_error) or "404" in str(balance_error):
                    logger.warning("Sandbox account has no balances - using mock values. "
                                   "Initialize your account at https://cert.tastytrade.com")
                    # Use mock balances for sandbox testing
                    self.portfolio_state.account_value = Decimal("100000.00")
                    self.portfolio_state.buying_power = Decimal("50000.00")
                    self.portfolio_state.net_liquidating_value = Decimal("100000.00")
                    self.portfolio_state.cash_balance = Decimal("100000.00")
                else:
                    raise
            
            # Get positions and calculate Greeks
            positions = account.get_positions(self.session)
            self.portfolio_state.open_positions = len(positions)
            
            # Count positions by underlying
            self.portfolio_state.positions_by_underlying = {}
            for pos in positions:
                symbol = pos.underlying_symbol
                self.portfolio_state.positions_by_underlying[symbol] = \
                    self.portfolio_state.positions_by_underlying.get(symbol, 0) + 1

            # Calculate beta-weighted Greeks using streaming data
            await self._calculate_beta_weighted_greeks(positions)
            
            logger.info(f"Portfolio updated: NLV=${self.portfolio_state.net_liquidating_value}, "
                       f"BP=${self.portfolio_state.buying_power}, "
                       f"Positions={self.portfolio_state.open_positions}")
            
            return self.portfolio_state

        except Exception as e:
            logger.error(f"Failed to update portfolio state: {e}")
            return self.portfolio_state

    async def _calculate_beta_weighted_greeks(self, positions: List[Any]) -> None:
        """
        Calculate beta-weighted portfolio Greeks relative to SPY

        Beta-weighting normalizes all positions to SPY-equivalent deltas,
        allowing comparison across different underlyings.
        """
        try:
            from tastytrade import DXLinkStreamer
            from tastytrade.instruments import Option, Equity
            from tastytrade.dxfeed import Greeks, Quote
            from tastytrade.metrics import get_market_metrics

            # Reset portfolio Greeks
            self.portfolio_state.portfolio_delta = Decimal("0")
            self.portfolio_state.portfolio_theta = Decimal("0")
            self.portfolio_state.portfolio_vega = Decimal("0")
            self.portfolio_state.portfolio_gamma = Decimal("0")

            if not positions:
                return

            # Get SPY price as the reference
            spy_quote = None
            async with DXLinkStreamer(self.session) as streamer:
                await streamer.subscribe(Quote, ['SPY'])
                try:
                    spy_quote = await asyncio.wait_for(
                        streamer.get_event(Quote), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Could not get SPY quote for beta-weighting")
                    return

            spy_price = (Decimal(str(spy_quote.bid_price or 0)) +
                        Decimal(str(spy_quote.ask_price or 0))) / 2

            if spy_price == 0:
                logger.warning("SPY price is zero, cannot beta-weight")
                return

            # Get beta values for all underlyings
            underlyings = list(set(pos.underlying_symbol for pos in positions))
            metrics = get_market_metrics(self.session, underlyings)
            betas = {m.symbol: Decimal(str(m.beta or 1)) for m in metrics}

            # Get underlying prices
            underlying_prices = {}
            async with DXLinkStreamer(self.session) as streamer:
                await streamer.subscribe(Quote, underlyings)
                timeout_count = 0
                while len(underlying_prices) < len(underlyings) and timeout_count < 30:
                    try:
                        quote = await asyncio.wait_for(
                            streamer.get_event(Quote), timeout=0.5
                        )
                        mid = (Decimal(str(quote.bid_price or 0)) +
                              Decimal(str(quote.ask_price or 0))) / 2
                        underlying_prices[quote.event_symbol] = mid
                    except asyncio.TimeoutError:
                        timeout_count += 1

            # Collect option positions for Greeks streaming
            option_positions = [p for p in positions if p.instrument_type == 'Equity Option']
            equity_positions = [p for p in positions if p.instrument_type == 'Equity']

            # Add equity position deltas (delta = 1 per share)
            for pos in equity_positions:
                symbol = pos.underlying_symbol
                beta = betas.get(symbol, Decimal("1"))
                underlying_price = underlying_prices.get(symbol, Decimal("0"))

                if underlying_price > 0:
                    # Beta-weighted delta = shares * beta * (underlying_price / SPY_price)
                    bw_delta = Decimal(str(pos.quantity)) * beta * (underlying_price / spy_price)
                    self.portfolio_state.portfolio_delta += bw_delta

            # Stream Greeks for option positions
            if option_positions:
                streamer_symbols = []
                option_map = {}
                for pos in option_positions:
                    try:
                        opt = Option.get_option(self.session, pos.symbol)
                        streamer_symbols.append(opt.streamer_symbol)
                        option_map[opt.streamer_symbol] = {
                            'position': pos,
                            'option': opt
                        }
                    except Exception:
                        continue

                if streamer_symbols:
                    async with DXLinkStreamer(self.session) as streamer:
                        await streamer.subscribe(Greeks, streamer_symbols)

                        greeks_data = {}
                        timeout_count = 0
                        while len(greeks_data) < len(streamer_symbols) and timeout_count < 50:
                            try:
                                g = await asyncio.wait_for(
                                    streamer.get_event(Greeks), timeout=0.5
                                )
                                greeks_data[g.event_symbol] = g
                            except asyncio.TimeoutError:
                                timeout_count += 1

                        # Calculate beta-weighted Greeks for each option
                        for streamer_sym, data in option_map.items():
                            pos = data['position']
                            opt = data['option']
                            g = greeks_data.get(streamer_sym)

                            if g is None:
                                continue

                            symbol = pos.underlying_symbol
                            beta = betas.get(symbol, Decimal("1"))
                            underlying_price = underlying_prices.get(symbol, spy_price)

                            # Contract multiplier (usually 100 for options)
                            multiplier = Decimal("100")
                            quantity = Decimal(str(pos.quantity))

                            # Raw Greeks
                            delta = Decimal(str(g.delta or 0))
                            theta = Decimal(str(g.theta or 0))
                            vega = Decimal(str(g.vega or 0))
                            gamma = Decimal(str(g.gamma or 0))

                            # Beta-weighted delta = delta * quantity * multiplier * beta * (underlying/SPY)
                            bw_multiplier = beta * (underlying_price / spy_price) if spy_price > 0 else Decimal("1")

                            self.portfolio_state.portfolio_delta += delta * quantity * multiplier * bw_multiplier
                            self.portfolio_state.portfolio_theta += theta * quantity * multiplier
                            self.portfolio_state.portfolio_vega += vega * quantity * multiplier
                            self.portfolio_state.portfolio_gamma += gamma * quantity * multiplier

            logger.info(f"Beta-weighted Greeks: Δ={self.portfolio_state.portfolio_delta:.1f}, "
                       f"Θ={self.portfolio_state.portfolio_theta:.2f}, "
                       f"V={self.portfolio_state.portfolio_vega:.2f}")

        except ImportError as e:
            logger.warning(f"Cannot calculate beta-weighted Greeks: {e}")
        except Exception as e:
            logger.error(f"Beta-weighted Greeks calculation failed: {e}")

    def _evaluate_strategies_for_symbol(
        self,
        symbol: str,
        options: List,
        greeks_by_strike: dict,
        quotes_by_strike: dict,
        iv_rank: Decimal,
        target_exp: date,
        max_candidates: int = 3
    ) -> List[TradeCandidate]:
        """
        Evaluate all strategies for a symbol using the strategy engine.
        Delegates to StrategyEngine for actual evaluation.
        """
        return self.strategy_engine.evaluate_strategies_for_symbol(
            symbol=symbol,
            options=options,
            greeks_by_strike=greeks_by_strike,
            quotes_by_strike=quotes_by_strike,
            iv_rank=iv_rank,
            target_exp=target_exp,
            max_candidates=max_candidates
        )

    def check_risk_limits(self, proposal: TradeProposal) -> tuple[bool, str]:
        """
        Check if a proposed trade passes all risk limits.
        Delegates to risk_manager module.
        """
        return check_risk_limits(
            proposal=proposal,
            risk_params=self.risk_params,
            portfolio_state=self.portfolio_state,
            sandbox_mode=self.sandbox_mode
        )

    def generate_trade_rationale(self, proposal: TradeProposal) -> str:
        """
        Generate human-readable rationale for the trade.
        Delegates to risk_manager module.
        """
        return generate_trade_rationale(proposal, self.risk_params)

    async def propose_trade(
        self,
        strategy: StrategyType,
        underlying: str,
        legs: List[dict],
        greeks: dict,
        iv_rank: Decimal,
        claude_analysis: Optional[Any] = None
    ) -> Optional[TradeProposal]:
        """
        Create a trade proposal for approval

        Args:
            strategy: Type of options strategy
            underlying: Underlying symbol (e.g., 'SPY')
            legs: List of option legs with strike, expiration, etc.
            greeks: Position Greeks (delta, theta, vega, gamma)
            iv_rank: Current IV Rank percentage
            claude_analysis: Optional Claude AI analysis of the opportunity

        Returns:
            TradeProposal if created, None if rejected by risk checks
        """
        import uuid

        # Calculate expected credit and max loss from legs
        # For spreads: net credit = credits received - debits paid
        total_credits = sum(Decimal(str(leg.get('credit', 0))) for leg in legs)
        total_debits = sum(Decimal(str(leg.get('debit', 0))) for leg in legs)
        expected_credit = total_credits - total_debits

        # For spreads: max loss = spread width - net credit (calculated during scanning)
        # Check if max_loss is explicitly provided in any leg
        explicit_max_loss = sum(Decimal(str(leg.get('max_loss', 0))) for leg in legs)
        if explicit_max_loss > 0:
            max_loss = explicit_max_loss
        elif strategy in (StrategyType.SHORT_PUT_SPREAD, StrategyType.SHORT_CALL_SPREAD):
            # Calculate max loss for vertical spreads: width * 100 - net_credit * 100
            strikes = [Decimal(str(leg.get('strike', 0))) for leg in legs]
            if len(strikes) >= 2:
                spread_width = abs(max(strikes) - min(strikes))
                max_loss = (spread_width * 100) - (expected_credit * 100)
            else:
                max_loss = Decimal("0")
        elif strategy == StrategyType.IRON_CONDOR:
            # Calculate max loss for iron condor: max(put_spread_width, call_spread_width) * 100 - net_credit * 100
            # Separate legs by option type
            put_strikes = [Decimal(str(leg.get('strike', 0))) for leg in legs if leg.get('option_type') == 'PUT']
            call_strikes = [Decimal(str(leg.get('strike', 0))) for leg in legs if leg.get('option_type') == 'CALL']

            put_width = abs(max(put_strikes) - min(put_strikes)) if len(put_strikes) >= 2 else Decimal("0")
            call_width = abs(max(call_strikes) - min(call_strikes)) if len(call_strikes) >= 2 else Decimal("0")

            # Max loss is the wider spread width minus total credit received
            wider_spread = max(put_width, call_width)
            max_loss = (wider_spread * 100) - (expected_credit * 100)
        else:
            max_loss = Decimal("0")

        # Get DTE from first leg
        dte = legs[0].get('dte', self.risk_params.target_dte) if legs else self.risk_params.target_dte

        # Create proposal
        proposal = TradeProposal(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            strategy=strategy,
            underlying_symbol=underlying,
            legs=legs,
            expected_credit=expected_credit,
            max_loss=max_loss,
            probability_of_profit=Decimal(str(greeks.get('pop', 0.70))),
            iv_rank=iv_rank,
            dte=dte,
            delta=Decimal(str(greeks.get('delta', 0))),
            theta=Decimal(str(greeks.get('theta', 0))),
            vega=Decimal(str(greeks.get('vega', 0))),
            rationale=""
        )

        # Store Claude analysis if available
        if claude_analysis:
            proposal.claude_analysis = {
                'recommendation': claude_analysis.recommendation,
                'suggested_strategy': claude_analysis.suggested_strategy,
                'suggested_delta': claude_analysis.suggested_delta,
                'key_risks': claude_analysis.key_risks,
                'rationale': claude_analysis.rationale
            }
            proposal.claude_confidence = claude_analysis.confidence
        
        # Generate rationale
        proposal.rationale = self.generate_trade_rationale(proposal)
        
        # Check risk limits
        passed, reason = self.check_risk_limits(proposal)
        if not passed:
            logger.warning(f"Trade proposal rejected: {reason}")
            proposal.status = TradeStatus.REJECTED
            proposal.rejection_reason = reason
            self.trade_history.append(proposal)
            return None
        
        # Add to pending for approval
        self.pending_trades.append(proposal)
        logger.info(f"Trade proposal created: {proposal.id} - {proposal.underlying_symbol} {proposal.strategy.value}")
        
        return proposal
    
    async def request_approval(self, proposal: TradeProposal) -> bool:
        """
        Request approval for a trade proposal
        
        This is the critical human-in-the-loop step.
        The bot will NEVER execute without explicit approval.
        """
        if proposal.status != TradeStatus.PENDING_APPROVAL:
            logger.warning(f"Trade {proposal.id} not in pending status")
            return False

        # Get Claude's summary if available
        ai_summary = None
        if self.claude_advisor:
            try:
                portfolio_context = {
                    'portfolio_delta': float(self.portfolio_state.portfolio_delta),
                    'open_positions': self.portfolio_state.open_positions,
                    'buying_power': float(self.portfolio_state.buying_power),
                    'net_liquidating_value': float(self.portfolio_state.net_liquidating_value),
                    'daily_pnl': float(self.portfolio_state.daily_pnl)
                }
                ai_summary = await self.claude_advisor.generate_approval_summary(
                    proposal=proposal,
                    portfolio_context=portfolio_context
                )
            except Exception as e:
                logger.warning(f"Failed to generate Claude summary: {e}")

        # Display proposal details
        print("\n" + "="*60)
        print("TRADE APPROVAL REQUEST")
        print("="*60)

        # Show Claude's AI analysis first if available
        if ai_summary:
            print("\n[AI ANALYSIS]")
            print("-" * 40)
            print(ai_summary)
            print("-" * 40)
        elif proposal.claude_analysis:
            # Show stored analysis if live summary failed
            print("\n[AI ANALYSIS]")
            print("-" * 40)
            print(f"Recommendation: {proposal.claude_analysis.get('recommendation', 'N/A')}")
            print(f"Confidence: {proposal.claude_confidence}/10")
            print(f"Rationale: {proposal.claude_analysis.get('rationale', 'N/A')}")
            if proposal.claude_analysis.get('key_risks'):
                print("Key Risks:")
                for risk in proposal.claude_analysis['key_risks']:
                    print(f"  - {risk}")
            print("-" * 40)

        print(f"\nProposal ID: {proposal.id}")
        print(f"Timestamp: {proposal.timestamp}")
        print(f"\n{proposal.rationale}")
        print("\nLegs:")
        for i, leg in enumerate(proposal.legs, 1):
            print(f"  {i}. {leg}")
        print("="*60)
        
        # Use callback if provided, otherwise wait for manual input
        if self.approval_callback:
            approved = self.approval_callback(proposal)
        else:
            # Console-based approval
            response = input("\nApprove this trade? (yes/no): ").strip().lower()
            approved = response in ('yes', 'y', 'approve')
        
        if approved:
            proposal.status = TradeStatus.APPROVED
            proposal.approved_by = "user"
            proposal.approval_timestamp = datetime.now()
            logger.info(f"Trade {proposal.id} APPROVED")
            return True
        else:
            proposal.status = TradeStatus.REJECTED
            proposal.rejection_reason = "User rejected"
            proposal.approval_timestamp = datetime.now()
            logger.info(f"Trade {proposal.id} REJECTED by user")
            # Move to history
            self.pending_trades.remove(proposal)
            self.trade_history.append(proposal)
            return False
    
    async def execute_trade(self, proposal: TradeProposal) -> bool:
        """
        Execute an approved trade
        
        IMPORTANT: Only executes if:
        1. Trade status is APPROVED
        2. Risk checks still pass (re-validated)
        3. Session is active
        """
        if proposal.status != TradeStatus.APPROVED:
            logger.error(f"Cannot execute trade {proposal.id} - not approved (status: {proposal.status})")
            return False
        
        # Re-check risk limits before execution
        await self.update_portfolio_state()
        passed, reason = self.check_risk_limits(proposal)
        if not passed:
            logger.error(f"Trade {proposal.id} failed re-validation: {reason}")
            proposal.status = TradeStatus.FAILED
            proposal.execution_result = {"error": reason}
            return False
        
        if not self.session:
            logger.error("No active session - cannot execute trade")
            proposal.status = TradeStatus.FAILED
            proposal.execution_result = {"error": "No session"}
            return False

        # Ensure session token is valid before execution
        if not await self.ensure_session_valid():
            logger.error("Session validation failed - cannot execute trade")
            proposal.status = TradeStatus.FAILED
            proposal.execution_result = {"error": "Session refresh failed"}
            return False

        # Handle mock data execution (sandbox with synthetic options)
        if self.use_mock_data and self.sandbox_mode:
            logger.info(f"[MOCK EXECUTION] Simulating trade {proposal.id} - {proposal.underlying_symbol} {proposal.strategy.value}")
            logger.info(f"  Credit: ${proposal.expected_credit:.2f} | Max Loss: ${proposal.max_loss:.2f}")
            for leg in proposal.legs:
                logger.info(f"  Leg: {leg.get('action')} {leg.get('quantity', 1)}x {leg.get('symbol')}")

            proposal.status = TradeStatus.EXECUTED
            proposal.execution_result = {
                "mock_execution": True,
                "sandbox": True,
                "message": "Trade simulated with mock data (market closed or using synthetic options)"
            }

            # Move to history
            if proposal in self.pending_trades:
                self.pending_trades.remove(proposal)
            self.trade_history.append(proposal)

            return True

        try:
            from tastytrade import Account
            from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType

            account = Account.get(self.session)[0]

            # Build order legs
            order_legs = []
            for leg in proposal.legs:
                from tastytrade.instruments import Option
                from tastytrade.order import Leg

                # Fetch the actual option instrument
                option = Option.get_option(self.session, leg['symbol'])

                # Map action string to OrderAction enum
                action_map = {
                    'SELL_TO_OPEN': OrderAction.SELL_TO_OPEN,
                    'BUY_TO_OPEN': OrderAction.BUY_TO_OPEN,
                    'SELL_TO_CLOSE': OrderAction.SELL_TO_CLOSE,
                    'BUY_TO_CLOSE': OrderAction.BUY_TO_CLOSE,
                }
                action = action_map.get(leg.get('action', 'SELL_TO_OPEN'), OrderAction.SELL_TO_OPEN)

                order_leg = Leg(
                    instrument_type=option.instrument_type,
                    symbol=option.symbol,
                    action=action,
                    quantity=leg.get('quantity', 1)
                )
                order_legs.append(order_leg)

            if not order_legs:
                raise ValueError("No valid order legs could be constructed")
            
            # Create the order
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.LIMIT,
                legs=order_legs,
                price=-proposal.expected_credit  # Negative = credit
            )
            
            # Dry run first (always)
            dry_run_result = account.place_order(self.session, order, dry_run=True)
            logger.info(f"Dry run result: {dry_run_result}")
            
            if self.sandbox_mode:
                logger.info(f"SANDBOX MODE: Would execute trade {proposal.id}")
                proposal.status = TradeStatus.EXECUTED
                proposal.execution_result = {"sandbox": True, "dry_run": str(dry_run_result)}
            else:
                # Actually execute
                result = account.place_order(self.session, order, dry_run=False)
                proposal.status = TradeStatus.EXECUTED
                proposal.execution_result = {"order_id": str(result.order.id) if result.order else None}
                logger.info(f"Trade {proposal.id} EXECUTED: {result}")
            
            # Move to history
            if proposal in self.pending_trades:
                self.pending_trades.remove(proposal)
            self.trade_history.append(proposal)
            
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            proposal.status = TradeStatus.FAILED
            proposal.execution_result = {"error": str(e)}
            return False

    async def close_position(self, position_symbol: str) -> Dict[str, Any]:
        """
        Close an existing position by buying/selling to close.

        Args:
            position_symbol: The underlying symbol of the position to close

        Returns:
            Dict with 'success' boolean and 'message' or 'error' string
        """
        result = {'success': False, 'message': '', 'error': None}

        # Use mock data if enabled
        if self.use_mock_data:
            logger.info(f"[MOCK] Closing position: {position_symbol}")
            result['success'] = True
            result['message'] = f"[SANDBOX] Position {position_symbol} closed successfully (mock)"
            result['order_id'] = 'mock-close-' + position_symbol.lower()
            return result

        if not self.session:
            result['error'] = "No active session"
            return result

        if not await self.ensure_session_valid():
            result['error'] = "Session validation failed"
            return result

        try:
            from tastytrade import Account
            from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType, Leg
            from tastytrade.instruments import Option
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Quote

            account = Account.get(self.session)[0]
            positions = account.get_positions(self.session)

            # Find positions matching the symbol
            matching_positions = [
                p for p in positions
                if p.underlying_symbol.upper() == position_symbol.upper()
                and p.instrument_type == 'Equity Option'
            ]

            if not matching_positions:
                result['error'] = f"No open option positions found for {position_symbol}"
                return result

            # Build close order legs
            order_legs = []
            for pos in matching_positions:
                try:
                    option = Option.get_option(self.session, pos.symbol)

                    # Determine close action based on position direction
                    if pos.quantity > 0:
                        # Long position - sell to close
                        action = OrderAction.SELL_TO_CLOSE
                    else:
                        # Short position - buy to close
                        action = OrderAction.BUY_TO_CLOSE

                    leg = Leg(
                        instrument_type=option.instrument_type,
                        symbol=option.symbol,
                        action=action,
                        quantity=abs(pos.quantity)
                    )
                    order_legs.append(leg)
                except Exception as e:
                    logger.warning(f"Could not build leg for {pos.symbol}: {e}")

            if not order_legs:
                result['error'] = "Could not build order legs for position"
                return result

            # Get current mid price for limit order
            streamer_symbols = [Option.get_option(self.session, pos.symbol).streamer_symbol
                               for pos in matching_positions]

            total_price = Decimal("0")
            async with DXLinkStreamer(self.session) as streamer:
                await streamer.subscribe(Quote, streamer_symbols)
                quotes = {}
                timeout_count = 0
                while len(quotes) < len(streamer_symbols) and timeout_count < 20:
                    try:
                        quote = await asyncio.wait_for(
                            streamer.get_event(Quote), timeout=0.5
                        )
                        quotes[quote.event_symbol] = quote
                    except asyncio.TimeoutError:
                        timeout_count += 1

                for sym, quote in quotes.items():
                    mid = (Decimal(str(quote.bid_price or 0)) +
                           Decimal(str(quote.ask_price or 0))) / 2
                    total_price += mid

            # Create close order (negative price for credit, positive for debit)
            # For BTC orders, we're buying so it's a debit (positive price)
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.LIMIT,
                legs=order_legs,
                price=total_price  # Positive = debit for BTC
            )

            # Dry run first
            dry_run_result = account.place_order(self.session, order, dry_run=True)
            logger.info(f"Close order dry run: {dry_run_result}")

            if self.sandbox_mode:
                result['success'] = True
                result['message'] = f"[SANDBOX] Close order submitted for {position_symbol}"
                result['dry_run'] = str(dry_run_result)
            else:
                order_result = account.place_order(self.session, order, dry_run=False)
                result['success'] = True
                result['message'] = f"Close order executed for {position_symbol}"
                result['order_id'] = str(order_result.order.id) if order_result.order else None

            logger.info(f"Position {position_symbol} close: {result['message']}")

        except Exception as e:
            logger.error(f"Close position failed: {e}")
            result['error'] = str(e)

        return result

    async def roll_position(
        self,
        position_symbol: str,
        target_dte: int = 45
    ) -> Dict[str, Any]:
        """
        Roll an existing position to a new expiration.

        Closes the current position and opens a new one at the same strike
        but with a later expiration (typically next monthly).

        Args:
            position_symbol: The underlying symbol of the position to roll
            target_dte: Target DTE for the new position (default 45)

        Returns:
            Dict with 'success' boolean, 'message', 'net_credit' and/or 'error'
        """
        result = {'success': False, 'message': '', 'error': None, 'net_credit': 0}

        # Use mock data if enabled
        if self.use_mock_data:
            logger.info(f"[MOCK] Rolling position: {position_symbol} to ~{target_dte} DTE")
            result['success'] = True
            result['message'] = f"[SANDBOX] Position {position_symbol} rolled to ~{target_dte} DTE (mock)"
            result['net_credit'] = 0.35  # Mock credit
            result['order_id'] = 'mock-roll-' + position_symbol.lower()
            return result

        if not self.session:
            result['error'] = "No active session"
            return result

        if not await self.ensure_session_valid():
            result['error'] = "Session validation failed"
            return result

        try:
            from tastytrade import Account
            from tastytrade.order import NewOrder, OrderAction, OrderTimeInForce, OrderType, Leg
            from tastytrade.instruments import Option, get_option_chain
            from tastytrade.utils import get_tasty_monthly
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Quote

            account = Account.get(self.session)[0]
            positions = account.get_positions(self.session)

            # Find positions matching the symbol
            matching_positions = [
                p for p in positions
                if p.underlying_symbol.upper() == position_symbol.upper()
                and p.instrument_type == 'Equity Option'
            ]

            if not matching_positions:
                result['error'] = f"No open option positions found for {position_symbol}"
                return result

            # Get target expiration (next monthly)
            target_exp = get_tasty_monthly()

            # Get option chain for new positions
            chain = get_option_chain(self.session, position_symbol)
            if target_exp not in chain:
                result['error'] = f"No options available at target expiration for {position_symbol}"
                return result

            new_options = chain[target_exp]

            order_legs = []
            total_close_price = Decimal("0")
            total_open_price = Decimal("0")

            for pos in matching_positions:
                try:
                    old_option = Option.get_option(self.session, pos.symbol)

                    # Find matching strike in new expiration
                    new_option = None
                    for opt in new_options:
                        if (opt.strike_price == old_option.strike_price and
                            opt.option_type == old_option.option_type):
                            new_option = opt
                            break

                    if new_option is None:
                        logger.warning(f"Could not find matching strike for roll: {pos.symbol}")
                        continue

                    # Determine actions based on position direction
                    if pos.quantity > 0:
                        # Long position - sell to close, buy to open
                        close_action = OrderAction.SELL_TO_CLOSE
                        open_action = OrderAction.BUY_TO_OPEN
                    else:
                        # Short position - buy to close, sell to open
                        close_action = OrderAction.BUY_TO_CLOSE
                        open_action = OrderAction.SELL_TO_OPEN

                    # Close leg
                    close_leg = Leg(
                        instrument_type=old_option.instrument_type,
                        symbol=old_option.symbol,
                        action=close_action,
                        quantity=abs(pos.quantity)
                    )
                    order_legs.append(close_leg)

                    # Open leg
                    open_leg = Leg(
                        instrument_type=new_option.instrument_type,
                        symbol=new_option.symbol,
                        action=open_action,
                        quantity=abs(pos.quantity)
                    )
                    order_legs.append(open_leg)

                except Exception as e:
                    logger.warning(f"Could not build roll legs for {pos.symbol}: {e}")

            if not order_legs:
                result['error'] = "Could not build order legs for roll"
                return result

            # Get current prices for all options
            old_symbols = [Option.get_option(self.session, pos.symbol).streamer_symbol
                          for pos in matching_positions]
            new_symbols = []
            for pos in matching_positions:
                old_opt = Option.get_option(self.session, pos.symbol)
                for opt in new_options:
                    if (opt.strike_price == old_opt.strike_price and
                        opt.option_type == old_opt.option_type):
                        new_symbols.append(opt.streamer_symbol)
                        break

            all_symbols = old_symbols + new_symbols

            async with DXLinkStreamer(self.session) as streamer:
                await streamer.subscribe(Quote, all_symbols)
                quotes = {}
                timeout_count = 0
                while len(quotes) < len(all_symbols) and timeout_count < 30:
                    try:
                        quote = await asyncio.wait_for(
                            streamer.get_event(Quote), timeout=0.5
                        )
                        quotes[quote.event_symbol] = quote
                    except asyncio.TimeoutError:
                        timeout_count += 1

                # Calculate net credit
                for sym in old_symbols:
                    if sym in quotes:
                        mid = (Decimal(str(quotes[sym].bid_price or 0)) +
                               Decimal(str(quotes[sym].ask_price or 0))) / 2
                        total_close_price += mid

                for sym in new_symbols:
                    if sym in quotes:
                        mid = (Decimal(str(quotes[sym].bid_price or 0)) +
                               Decimal(str(quotes[sym].ask_price or 0))) / 2
                        total_open_price += mid

            # For short positions rolling: close debit, open credit
            # Net credit = open credit - close debit (for short positions)
            net_credit = total_open_price - total_close_price

            # Check minimum credit requirement
            min_credit = Decimal(str(
                self.config.get('management_rules', {})
                .get('rolling_rules', {})
                .get('min_credit_for_roll', 0.25)
            ))

            if net_credit < min_credit:
                result['error'] = f"Net credit ${net_credit:.2f} below minimum ${min_credit:.2f}"
                return result

            # Create combo order
            order = NewOrder(
                time_in_force=OrderTimeInForce.DAY,
                order_type=OrderType.LIMIT,
                legs=order_legs,
                price=-net_credit  # Negative for credit
            )

            # Dry run first
            dry_run_result = account.place_order(self.session, order, dry_run=True)
            logger.info(f"Roll order dry run: {dry_run_result}")

            if self.sandbox_mode:
                result['success'] = True
                result['message'] = f"[SANDBOX] Roll order submitted for {position_symbol}"
                result['net_credit'] = float(net_credit)
                result['dry_run'] = str(dry_run_result)
            else:
                order_result = account.place_order(self.session, order, dry_run=False)
                result['success'] = True
                result['message'] = f"Roll order executed for {position_symbol}"
                result['net_credit'] = float(net_credit)
                result['order_id'] = str(order_result.order.id) if order_result.order else None

            logger.info(f"Position {position_symbol} roll: {result['message']} (credit: ${net_credit:.2f})")

        except Exception as e:
            logger.error(f"Roll position failed: {e}")
            result['error'] = str(e)

        return result

    async def get_position_details(self, position_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get details about a position for display/confirmation.

        Args:
            position_symbol: The underlying symbol

        Returns:
            Dict with position details or None if not found
        """
        # Use mock data if enabled
        if self.use_mock_data:
            # Return mock position data
            return {
                'symbol': position_symbol,
                'underlying': position_symbol,
                'positions': [
                    {
                        'option_symbol': f'{position_symbol}  250221P00450000',
                        'option_type': 'PUT',
                        'strike': 450.0,
                        'expiration': '2025-02-21',
                        'quantity': -1,
                        'entry_price': 2.50,
                        'current_price': 1.25,
                        'pnl_percent': 0.50,
                        'dte': 28
                    }
                ],
                'total_pnl_percent': 0.50,
                'total_dte': 28,
                'is_tested': False
            }

        if not self.session:
            return None

        try:
            from tastytrade import Account
            from tastytrade.instruments import Option
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Quote

            account = Account.get(self.session)[0]
            positions = account.get_positions(self.session)

            matching = [
                p for p in positions
                if p.underlying_symbol.upper() == position_symbol.upper()
                and p.instrument_type == 'Equity Option'
            ]

            if not matching:
                return None

            details = {
                'symbol': position_symbol,
                'underlying': position_symbol,
                'positions': [],
                'total_pnl_percent': 0,
                'total_dte': 999,
                'is_tested': False
            }

            for pos in matching:
                try:
                    opt = Option.get_option(self.session, pos.symbol)
                    dte = (opt.expiration_date - date.today()).days

                    pos_detail = {
                        'option_symbol': pos.symbol,
                        'option_type': opt.option_type,
                        'strike': float(opt.strike_price),
                        'expiration': str(opt.expiration_date),
                        'quantity': pos.quantity,
                        'entry_price': float(pos.average_open_price or 0),
                        'current_price': 0,
                        'pnl_percent': 0,
                        'dte': dte
                    }

                    details['positions'].append(pos_detail)
                    if dte < details['total_dte']:
                        details['total_dte'] = dte

                except Exception as e:
                    logger.warning(f"Could not get details for {pos.symbol}: {e}")

            return details

        except Exception as e:
            logger.error(f"Get position details failed: {e}")
            return None

    async def scan_for_opportunities(self, symbols: List[str]) -> List[TradeProposal]:
        """
        Scan symbols for trading opportunities based on tastylive criteria

        Looks for:
        1. Elevated IV Rank (>30)
        2. Options at ~45 DTE
        3. Strikes at ~30 delta (or 16 delta for conservative)
        4. Sufficient liquidity

        When use_mock_data is enabled (sandbox mode), uses mock market data
        instead of real streaming data.
        """
        opportunities = []

        # Use mock data path if enabled
        if self.use_mock_data and self.mock_provider:
            return await self._scan_with_mock_data(symbols)

        if not self.session:
            logger.warning("No session - cannot scan for opportunities")
            return opportunities

        # Ensure session token is valid
        if not await self.ensure_session_valid():
            logger.error("Session validation failed")
            return opportunities

        try:
            from tastytrade.instruments import get_option_chain
            from tastytrade.utils import get_tasty_monthly
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Greeks, Quote
            from tastytrade.metrics import get_market_metrics

            # Get 45 DTE expiration (tastylive standard)
            target_exp = get_tasty_monthly()

            # Fetch IV Rank for all symbols at once
            metrics = get_market_metrics(self.session, symbols)
            iv_ranks = {m.symbol: Decimal(str(m.implied_volatility_index_rank or 0))
                       for m in metrics}

            for symbol in symbols:
                try:
                    # Get IV Rank from market metrics
                    iv_rank = iv_ranks.get(symbol, Decimal("0"))

                    if iv_rank < self.risk_params.min_iv_rank:
                        logger.debug(f"{symbol} IV Rank {iv_rank} too low")
                        continue

                    # Get option chain
                    chain = get_option_chain(self.session, symbol)

                    if target_exp not in chain:
                        logger.debug(f"No options at target expiration for {symbol}")
                        continue

                    options = chain[target_exp]

                    # Build maps for all options (puts and calls)
                    opt_by_streamer = {opt.streamer_symbol: opt for opt in options}

                    # Stream Greeks for all options
                    async with DXLinkStreamer(self.session) as streamer:
                        await streamer.subscribe(Greeks, [opt.streamer_symbol for opt in options])

                        # Collect Greeks
                        option_greeks = {}
                        timeout_count = 0
                        while len(option_greeks) < len(options) and timeout_count < 100:
                            try:
                                greeks_data = await asyncio.wait_for(
                                    streamer.get_event(Greeks), timeout=0.5
                                )
                                option_greeks[greeks_data.event_symbol] = greeks_data
                            except asyncio.TimeoutError:
                                timeout_count += 1
                                continue

                        # Subscribe to quotes for all options
                        await streamer.subscribe(Quote, [opt.streamer_symbol for opt in options])

                        # Collect quotes
                        option_quotes = {}
                        timeout_count = 0
                        while len(option_quotes) < len(options) and timeout_count < 100:
                            try:
                                quote = await asyncio.wait_for(
                                    streamer.get_event(Quote), timeout=0.5
                                )
                                option_quotes[quote.event_symbol] = quote
                            except asyncio.TimeoutError:
                                timeout_count += 1
                                continue

                        # Build greeks and quotes by strike maps
                        greeks_by_strike = {}
                        quotes_by_strike = {}

                        for opt in options:
                            strike = opt.strike_price
                            if opt.streamer_symbol in option_greeks:
                                greeks_by_strike[strike] = option_greeks[opt.streamer_symbol]
                            if opt.streamer_symbol in option_quotes:
                                q = option_quotes[opt.streamer_symbol]
                                quotes_by_strike[strike] = (
                                    Decimal(str(q.bid_price or 0)),
                                    Decimal(str(q.ask_price or 0))
                                )

                        # Evaluate all strategies and get top 3 candidates
                        candidates = self._evaluate_strategies_for_symbol(
                            symbol=symbol,
                            options=options,
                            greeks_by_strike=greeks_by_strike,
                            quotes_by_strike=quotes_by_strike,
                            iv_rank=iv_rank,
                            target_exp=target_exp,
                            max_candidates=3
                        )

                        if not candidates:
                            logger.debug(f"No valid candidates found for {symbol}")
                            continue

                        # Log all candidates for comparison
                        logger.info(f"{symbol} - Top {len(candidates)} strategies:")
                        for i, cand in enumerate(candidates, 1):
                            ror = cand.return_on_risk()
                            if cand.strategy == StrategyType.IRON_CONDOR:
                                logger.info(f"  {i}. {cand.strategy.value}: "
                                           f"{cand.put_short_strike}P/{cand.put_long_strike}P + "
                                           f"{cand.call_short_strike}C/{cand.call_long_strike}C | "
                                           f"Credit=${cand.net_credit:.2f} | MaxLoss=${cand.max_loss:.2f} | "
                                           f"POP={cand.probability_of_profit:.0%} | RoR={ror:.1f}% | "
                                           f"Kelly={cand.score:.3f}")
                            else:
                                logger.info(f"  {i}. {cand.strategy.value}: "
                                           f"{cand.short_strike}/{cand.long_strike} | "
                                           f"Credit=${cand.net_credit:.2f} | MaxLoss=${cand.max_loss:.2f} | "
                                           f"POP={cand.probability_of_profit:.0%} | RoR={ror:.1f}% | "
                                           f"Kelly={cand.score:.3f}")

                        # Select the best candidate (highest Kelly score)
                        best = candidates[0]
                        dte = best.dte

                        # Get Claude's analysis if available
                        claude_analysis = None
                        if self.claude_advisor:
                            try:
                                # Build strategy-specific summary
                                if best.strategy == StrategyType.IRON_CONDOR:
                                    chain_summary = {
                                        'target_expiration': str(target_exp),
                                        'dte': dte,
                                        'strategy': 'iron_condor',
                                        'put_short_strike': str(best.put_short_strike),
                                        'put_long_strike': str(best.put_long_strike),
                                        'call_short_strike': str(best.call_short_strike),
                                        'call_long_strike': str(best.call_long_strike),
                                        'net_credit': str(best.net_credit),
                                        'max_loss': str(best.max_loss),
                                        'prob_otm': float(best.probability_of_profit),
                                        'kelly_score': float(best.score),
                                        'alternatives_evaluated': len(candidates)
                                    }
                                else:
                                    chain_summary = {
                                        'target_expiration': str(target_exp),
                                        'dte': dte,
                                        'strategy': best.strategy.value,
                                        'short_strike': str(best.short_strike),
                                        'long_strike': str(best.long_strike),
                                        'spread_width': str(best.spread_width),
                                        'net_credit': str(best.net_credit),
                                        'max_loss': str(best.max_loss),
                                        'prob_otm': float(best.probability_of_profit),
                                        'kelly_score': float(best.score),
                                        'alternatives_evaluated': len(candidates)
                                    }

                                portfolio_context = {
                                    'portfolio_delta': float(self.portfolio_state.portfolio_delta),
                                    'open_positions': self.portfolio_state.open_positions,
                                    'buying_power': float(self.portfolio_state.buying_power),
                                    'buying_power_used_pct': float(
                                        (1 - self.portfolio_state.buying_power /
                                         max(self.portfolio_state.net_liquidating_value, Decimal('1'))) * 100
                                    ) if self.portfolio_state.net_liquidating_value > 0 else 0,
                                    'daily_pnl': float(self.portfolio_state.daily_pnl)
                                }

                                claude_analysis = await self.claude_advisor.analyze_opportunity(
                                    symbol=symbol,
                                    iv_rank=float(iv_rank),
                                    current_price=float(best.short_strike or best.put_short_strike or 0),
                                    option_chain_summary=chain_summary,
                                    portfolio_state=portfolio_context
                                )

                                if claude_analysis:
                                    confidence_threshold = self.claude_config.get(
                                        'confidence_thresholds', {}
                                    ).get('opportunity_threshold', 7)

                                    if claude_analysis.recommendation == 'SKIP':
                                        logger.info(f"Claude recommends SKIP for {symbol}: {claude_analysis.rationale}")
                                        continue
                                    elif claude_analysis.recommendation == 'WAIT':
                                        logger.info(f"Claude recommends WAIT for {symbol}: {claude_analysis.rationale}")
                                        continue
                                    elif claude_analysis.confidence < confidence_threshold:
                                        logger.info(f"Claude confidence {claude_analysis.confidence} below threshold for {symbol}")
                                        continue

                                    logger.info(f"Claude recommends TRADE for {symbol} (confidence: {claude_analysis.confidence}/10)")

                            except Exception as e:
                                logger.warning(f"Claude analysis failed for {symbol}: {e}")

                        # Create proposal from best candidate
                        proposal = await self.propose_trade(
                            strategy=best.strategy,
                            underlying=symbol,
                            legs=best.legs,
                            greeks=best.greeks,
                            iv_rank=iv_rank,
                            claude_analysis=claude_analysis
                        )

                        if proposal:
                            logger.info(f"Selected {best.strategy.value} for {symbol} "
                                       f"(Kelly={best.score:.3f}, Credit=${best.net_credit:.2f})")
                            opportunities.append(proposal)

                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
        except Exception as e:
            logger.error(f"Scan failed: {e}")

        return opportunities

    async def _scan_with_mock_data(self, symbols: List[str]) -> List[TradeProposal]:
        """
        Scan for opportunities using mock market data with multi-strategy evaluation.

        Evaluates put spreads, call spreads, and iron condors for each symbol,
        ranking them by Kelly Criterion score and returning the best candidates.
        """
        opportunities = []

        if not self.mock_provider:
            logger.error("Mock provider not available")
            return opportunities

        # Update portfolio state for risk checks
        await self.update_portfolio_state()

        logger.info(f"Scanning with MOCK DATA (multi-strategy): {symbols}")

        # Get mock metrics for all symbols
        metrics = self.mock_provider.get_market_metrics(symbols)
        iv_ranks = {m.symbol: Decimal(str(m.implied_volatility_index_rank))
                   for m in metrics}

        for symbol in symbols:
            try:
                # Get IV Rank from mock metrics
                iv_rank = iv_ranks.get(symbol, Decimal("0"))

                if iv_rank < self.risk_params.min_iv_rank:
                    logger.debug(f"{symbol} IV Rank {iv_rank} too low (mock data)")
                    continue

                # Get mock option chain
                chain = self.mock_provider.get_option_chain(
                    symbol,
                    target_dte=self.risk_params.target_dte
                )

                if not chain:
                    logger.debug(f"No mock options for {symbol}")
                    continue

                # Get first (and likely only) expiration date
                target_exp = list(chain.keys())[0]
                options = chain[target_exp]

                # Build greeks and quotes maps for all options
                # Key by (strike, option_type) to keep puts and calls separate
                greeks_by_strike = {}
                quotes_by_strike = {}

                for opt in options:
                    strike = opt.strike_price
                    opt_type = opt.option_type
                    key = (strike, opt_type)
                    greeks_by_strike[key] = self.mock_provider.get_greeks(opt)
                    quote = self.mock_provider.get_quote(
                        symbol=opt.symbol,
                        is_option=True,
                        option=opt
                    )
                    quotes_by_strike[key] = (
                        Decimal(str(quote.bid_price)),
                        Decimal(str(quote.ask_price))
                    )

                # Evaluate all strategies and get top 3 candidates
                candidates = self._evaluate_strategies_for_symbol(
                    symbol=symbol,
                    options=options,
                    greeks_by_strike=greeks_by_strike,
                    quotes_by_strike=quotes_by_strike,
                    iv_rank=iv_rank,
                    target_exp=target_exp,
                    max_candidates=3
                )

                if not candidates:
                    logger.debug(f"No valid candidates found for {symbol}")
                    continue

                # Log all candidates for comparison
                logger.info(f"[MOCK] {symbol} - Top {len(candidates)} strategies:")
                for i, cand in enumerate(candidates, 1):
                    ror = cand.return_on_risk()
                    ev = cand.expected_value()
                    if cand.strategy == StrategyType.IRON_CONDOR:
                        logger.info(f"  {i}. {cand.strategy.value}: "
                                   f"{cand.put_short_strike}P/{cand.put_long_strike}P + "
                                   f"{cand.call_short_strike}C/{cand.call_long_strike}C | "
                                   f"Credit=${cand.net_credit:.2f} | MaxLoss=${cand.max_loss:.2f} | "
                                   f"POP={cand.probability_of_profit:.0%} | RoR={ror:.1f}% | "
                                   f"Kelly={cand.score:.3f}")
                    else:
                        logger.info(f"  {i}. {cand.strategy.value}: "
                                   f"{cand.short_strike}/{cand.long_strike} | "
                                   f"Credit=${cand.net_credit:.2f} | MaxLoss=${cand.max_loss:.2f} | "
                                   f"POP={cand.probability_of_profit:.0%} | RoR={ror:.1f}% | "
                                   f"Kelly={cand.score:.3f}")

                # Select the best candidate (highest Kelly score)
                best = candidates[0]
                dte = best.dte

                # Get Claude's analysis if available
                claude_analysis = None
                if self.claude_advisor:
                    try:
                        underlying_price = self.mock_provider.get_underlying_price(symbol)

                        # Build strategy-specific summary
                        if best.strategy == StrategyType.IRON_CONDOR:
                            chain_summary = {
                                'target_expiration': str(target_exp),
                                'dte': dte,
                                'strategy': 'iron_condor',
                                'put_short_strike': str(best.put_short_strike),
                                'put_long_strike': str(best.put_long_strike),
                                'call_short_strike': str(best.call_short_strike),
                                'call_long_strike': str(best.call_long_strike),
                                'net_credit': str(best.net_credit),
                                'max_loss': str(best.max_loss),
                                'prob_otm': float(best.probability_of_profit),
                                'kelly_score': float(best.score),
                                'alternatives_evaluated': len(candidates)
                            }
                        else:
                            chain_summary = {
                                'target_expiration': str(target_exp),
                                'dte': dte,
                                'strategy': best.strategy.value,
                                'short_strike': str(best.short_strike),
                                'long_strike': str(best.long_strike),
                                'spread_width': str(best.spread_width),
                                'net_credit': str(best.net_credit),
                                'max_loss': str(best.max_loss),
                                'prob_otm': float(best.probability_of_profit),
                                'kelly_score': float(best.score),
                                'alternatives_evaluated': len(candidates)
                            }

                        portfolio_context = {
                            'portfolio_delta': float(self.portfolio_state.portfolio_delta),
                            'open_positions': self.portfolio_state.open_positions,
                            'buying_power': float(self.portfolio_state.buying_power),
                            'buying_power_used_pct': float(
                                (1 - self.portfolio_state.buying_power /
                                 max(self.portfolio_state.net_liquidating_value, Decimal('1'))) * 100
                            ) if self.portfolio_state.net_liquidating_value > 0 else 0,
                            'daily_pnl': float(self.portfolio_state.daily_pnl)
                        }

                        claude_analysis = await self.claude_advisor.analyze_opportunity(
                            symbol=symbol,
                            iv_rank=float(iv_rank),
                            current_price=underlying_price,
                            option_chain_summary=chain_summary,
                            portfolio_state=portfolio_context
                        )

                        if claude_analysis:
                            confidence_threshold = self.claude_config.get(
                                'confidence_thresholds', {}
                            ).get('opportunity_threshold', 7)

                            if claude_analysis.recommendation == 'SKIP':
                                logger.info(f"Claude recommends SKIP for {symbol}: {claude_analysis.rationale}")
                                continue
                            elif claude_analysis.recommendation == 'WAIT':
                                logger.info(f"Claude recommends WAIT for {symbol}: {claude_analysis.rationale}")
                                continue
                            elif claude_analysis.confidence < confidence_threshold:
                                logger.info(f"Claude confidence {claude_analysis.confidence} below threshold for {symbol}")
                                continue

                            logger.info(f"Claude recommends TRADE for {symbol} (confidence: {claude_analysis.confidence}/10)")

                    except Exception as e:
                        logger.warning(f"Claude analysis failed for {symbol}: {e}")

                # Create proposal from best candidate
                proposal = await self.propose_trade(
                    strategy=best.strategy,
                    underlying=symbol,
                    legs=best.legs,
                    greeks=best.greeks,
                    iv_rank=iv_rank,
                    claude_analysis=claude_analysis
                )

                if proposal:
                    logger.info(f"[MOCK] Selected {best.strategy.value} for {symbol} "
                               f"(Kelly={best.score:.3f}, Credit=${best.net_credit:.2f})")
                    opportunities.append(proposal)

            except Exception as e:
                logger.error(f"Error scanning {symbol} with mock data: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info(f"Mock scan complete: found {len(opportunities)} opportunities")
        return opportunities
    
    async def manage_positions(self) -> List[dict]:
        """
        Check existing positions for management actions

        Tastylive management rules:
        1. Take profit at 50% of max profit
        2. Roll or close at 21 DTE
        3. Consider rolling tested positions

        Returns:
            List of management recommendations (does NOT auto-execute)
        """
        recommendations = []

        if not self.session:
            logger.warning("No session - cannot manage positions")
            return recommendations

        # Ensure session token is valid
        if not await self.ensure_session_valid():
            logger.error("Session validation failed")
            return recommendations

        try:
            from tastytrade import Account, DXLinkStreamer
            from tastytrade.instruments import Option
            from tastytrade.dxfeed import Quote

            account = Account.get(self.session)[0]
            positions = account.get_positions(self.session)

            # Filter to option positions only
            option_positions = [p for p in positions if p.instrument_type == 'Equity Option']

            if not option_positions:
                logger.info("No option positions to manage")
                return recommendations

            # Get current quotes for all positions
            streamer_symbols = []
            position_map = {}
            for pos in option_positions:
                try:
                    opt = Option.get_option(self.session, pos.symbol)
                    streamer_symbols.append(opt.streamer_symbol)
                    position_map[opt.streamer_symbol] = {
                        'position': pos,
                        'option': opt
                    }
                except Exception as e:
                    logger.warning(f"Could not fetch option {pos.symbol}: {e}")

            if not streamer_symbols:
                return recommendations

            # Stream quotes to get current prices
            async with DXLinkStreamer(self.session) as streamer:
                await streamer.subscribe(Quote, streamer_symbols)

                quotes = {}
                timeout_count = 0
                while len(quotes) < len(streamer_symbols) and timeout_count < 50:
                    try:
                        quote = await asyncio.wait_for(
                            streamer.get_event(Quote), timeout=0.5
                        )
                        quotes[quote.event_symbol] = quote
                    except asyncio.TimeoutError:
                        timeout_count += 1

            # Analyze each position
            for streamer_sym, data in position_map.items():
                pos = data['position']
                opt = data['option']
                quote = quotes.get(streamer_sym)

                if quote is None:
                    continue

                # Calculate current P&L percentage
                avg_open_price = Decimal(str(pos.average_open_price or 0))
                current_mid = (Decimal(str(quote.bid_price or 0)) +
                              Decimal(str(quote.ask_price or 0))) / 2

                if avg_open_price != 0:
                    # For short positions, profit when price decreases
                    if pos.quantity < 0:  # Short position
                        pnl_percent = (avg_open_price - current_mid) / avg_open_price
                    else:  # Long position
                        pnl_percent = (current_mid - avg_open_price) / avg_open_price
                else:
                    pnl_percent = Decimal("0")

                # Calculate DTE
                dte = (opt.expiration_date - date.today()).days

                # Check if position is being tested (strike breached)
                # Would need underlying price - fetch from quote
                underlying_price = Decimal(str(pos.mark_price or opt.strike_price))
                is_tested = False
                if opt.option_type == 'P' and underlying_price < opt.strike_price:
                    is_tested = True
                elif opt.option_type == 'C' and underlying_price > opt.strike_price:
                    is_tested = True

                recommendation = {
                    'symbol': pos.symbol,
                    'underlying': pos.underlying_symbol,
                    'quantity': pos.quantity,
                    'dte': dte,
                    'pnl_percent': float(pnl_percent),
                    'is_tested': is_tested,
                    'action': None,
                    'reason': None
                }

                # Check management triggers (priority order)
                if pnl_percent >= self.risk_params.profit_target_percent:
                    recommendation['action'] = 'CLOSE'
                    recommendation['reason'] = f"Profit target reached ({pnl_percent:.1%} >= {self.risk_params.profit_target_percent:.0%})"
                    logger.info(f"[MANAGE] {pos.symbol}: {recommendation['reason']}")

                elif pnl_percent <= -self.risk_params.stop_loss_multiplier:
                    recommendation['action'] = 'CLOSE'
                    recommendation['reason'] = f"Stop loss triggered ({pnl_percent:.1%} loss >= {self.risk_params.stop_loss_multiplier}x)"
                    logger.warning(f"[MANAGE] {pos.symbol}: {recommendation['reason']}")

                elif dte <= self.risk_params.management_dte:
                    if is_tested:
                        recommendation['action'] = 'CLOSE_OR_ROLL'
                        recommendation['reason'] = f"At {dte} DTE and position is tested - evaluate closing or rolling"
                    else:
                        recommendation['action'] = 'ROLL'
                        recommendation['reason'] = f"At {dte} DTE - consider rolling to next cycle"
                    logger.info(f"[MANAGE] {pos.symbol}: {recommendation['reason']}")

                elif is_tested:
                    recommendation['action'] = 'MONITOR'
                    recommendation['reason'] = "Position is being tested - monitor closely"
                    logger.warning(f"[MANAGE] {pos.symbol}: {recommendation['reason']}")

                # Get Claude's analysis for positions needing action
                if recommendation['action'] and self.claude_advisor:
                    try:
                        position_info = {
                            'symbol': pos.symbol,
                            'underlying': pos.underlying_symbol,
                            'option_type': opt.option_type,
                            'strike': float(opt.strike_price),
                            'quantity': pos.quantity,
                            'entry_credit': float(avg_open_price)
                        }

                        portfolio_context = {
                            'open_positions': self.portfolio_state.open_positions,
                            'portfolio_delta': float(self.portfolio_state.portfolio_delta),
                            'daily_pnl': float(self.portfolio_state.daily_pnl)
                        }

                        claude_mgmt = await self.claude_advisor.evaluate_management_action(
                            position=position_info,
                            current_pnl_pct=float(pnl_percent),
                            dte_remaining=dte,
                            is_tested=is_tested,
                            portfolio_context=portfolio_context
                        )

                        if claude_mgmt:
                            recommendation['claude_action'] = claude_mgmt.action
                            recommendation['claude_reasoning'] = claude_mgmt.reasoning
                            recommendation['claude_confidence'] = claude_mgmt.confidence
                            recommendation['claude_urgency'] = claude_mgmt.urgency
                            if claude_mgmt.roll_suggestion:
                                recommendation['claude_roll_suggestion'] = claude_mgmt.roll_suggestion

                            logger.info(f"[CLAUDE] {pos.symbol}: {claude_mgmt.action} "
                                       f"(confidence: {claude_mgmt.confidence}/10, urgency: {claude_mgmt.urgency})")

                    except Exception as e:
                        logger.warning(f"Claude management analysis failed for {pos.symbol}: {e}")

                if recommendation['action']:
                    recommendations.append(recommendation)

            if recommendations:
                logger.info(f"Position management found {len(recommendations)} action(s) needed")
            else:
                logger.info("All positions within normal parameters")

        except ImportError as e:
            logger.error(f"Missing dependency for position management: {e}")
        except Exception as e:
            logger.error(f"Position management failed: {e}")

        return recommendations

    async def get_portfolio_analysis(self) -> str:
        """
        Get Claude's analysis of overall portfolio health

        Returns:
            Formatted portfolio analysis string
        """
        if not self.claude_advisor:
            return "Portfolio analysis unavailable (Claude not configured)"

        await self.update_portfolio_state()

        # Get positions for analysis
        positions = []
        if self.session:
            try:
                from tastytrade import Account
                from tastytrade.instruments import Option

                account = Account.get(self.session)[0]
                raw_positions = account.get_positions(self.session)

                for pos in raw_positions:
                    if pos.instrument_type == 'Equity Option':
                        try:
                            opt = Option.get_option(self.session, pos.symbol)
                            dte = (opt.expiration_date - date.today()).days
                            positions.append({
                                'symbol': pos.symbol,
                                'underlying': pos.underlying_symbol,
                                'quantity': pos.quantity,
                                'pnl_pct': float(pos.multiplier) if hasattr(pos, 'multiplier') else 0,
                                'dte': dte
                            })
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Could not fetch positions for analysis: {e}")

        # Get recent trades
        recent_trades = [
            {
                'symbol': t.underlying_symbol,
                'strategy': t.strategy.value,
                'pnl': float(t.expected_credit) if t.status.value == 'executed' else 0
            }
            for t in self.trade_history[-10:]
        ]

        portfolio_state = {
            'net_liquidating_value': float(self.portfolio_state.net_liquidating_value),
            'buying_power': float(self.portfolio_state.buying_power),
            'daily_pnl': float(self.portfolio_state.daily_pnl),
            'weekly_pnl': float(self.portfolio_state.weekly_pnl),
            'portfolio_delta': float(self.portfolio_state.portfolio_delta),
            'portfolio_theta': float(self.portfolio_state.portfolio_theta),
            'open_positions': self.portfolio_state.open_positions
        }

        try:
            analysis = await self.claude_advisor.analyze_portfolio_health(
                portfolio_state=portfolio_state,
                positions=positions,
                recent_trades=recent_trades
            )
            return analysis
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return f"Portfolio analysis failed: {e}"

    def get_pending_trades(self) -> List[TradeProposal]:
        """Get all trades awaiting approval"""
        return [t for t in self.pending_trades if t.status == TradeStatus.PENDING_APPROVAL]
    
    def get_trade_history(self, limit: int = 50) -> List[TradeProposal]:
        """Get recent trade history"""
        return self.trade_history[-limit:]
    
    def export_state(self, filepath: str) -> bool:
        """Export bot state to JSON for persistence

        Args:
            filepath: Path to export file

        Returns:
            True if export succeeded, False otherwise
        """
        state = {
            'timestamp': datetime.now().isoformat(),
            'risk_parameters': {
                'max_daily_loss': str(self.risk_params.max_daily_loss),
                'max_weekly_loss': str(self.risk_params.max_weekly_loss),
                'max_position_loss': str(self.risk_params.max_position_loss),
                'min_iv_rank': str(self.risk_params.min_iv_rank),
                'target_dte': self.risk_params.target_dte,
            },
            'portfolio_state': {
                'net_liquidating_value': str(self.portfolio_state.net_liquidating_value),
                'buying_power': str(self.portfolio_state.buying_power),
                'open_positions': self.portfolio_state.open_positions,
            },
            'pending_trades': [
                {
                    'id': t.id,
                    'strategy': t.strategy.value,
                    'underlying': t.underlying_symbol,
                    'status': t.status.value,
                }
                for t in self.pending_trades
            ],
            'trade_history_count': len(self.trade_history)
        }

        try:
            # Validate and resolve path
            export_path = Path(filepath).resolve()

            # Ensure parent directory exists
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"State exported to {export_path}")
            return True

        except (OSError, IOError) as e:
            logger.error(f"Failed to export state to {filepath}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting state: {e}")
            return False

    # =========================================================================
    # CHATBOT HELPER METHODS
    # =========================================================================

    async def get_position_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a specific symbol.

        Args:
            symbol: The underlying symbol to look up

        Returns:
            Position details dict or None if not found
        """
        if not self.session:
            # Return mock position data if in mock mode
            if self.use_mock_data:
                return None  # No mock positions by default
            return None

        try:
            from tastytrade import Account

            if not await self.ensure_session_valid():
                return None

            account = Account.get(self.session)[0]
            positions = account.get_positions(self.session)

            for pos in positions:
                if pos.underlying_symbol == symbol:
                    return {
                        'symbol': pos.symbol,
                        'underlying': pos.underlying_symbol,
                        'quantity': pos.quantity,
                        'instrument_type': pos.instrument_type,
                        'average_open_price': float(pos.average_open_price or 0),
                        'close_price': float(pos.close_price or 0),
                        'pnl': float((pos.close_price or 0) - (pos.average_open_price or 0)) * pos.quantity * 100,
                        'pnl_pct': 0,  # Would need more data to calculate
                        'dte': 0,  # Would need option expiration
                        'is_tested': False  # Would need current price vs strike
                    }

            return None

        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return None

    async def get_iv_rank(self, symbol: str) -> Optional[Decimal]:
        """
        Get IV Rank for a specific symbol.

        Args:
            symbol: The symbol to get IV rank for

        Returns:
            IV Rank as Decimal or None if unavailable
        """
        # Use mock data if enabled
        if self.use_mock_data and self.mock_provider:
            mock_data = self.mock_provider.get_symbol_data(symbol)
            if mock_data:
                return Decimal(str(mock_data.get('iv_rank', 0)))
            return None

        if not self.session:
            return None

        try:
            from tastytrade.metrics import get_market_metrics

            if not await self.ensure_session_valid():
                return None

            metrics = get_market_metrics(self.session, [symbol])
            if metrics:
                iv_rank = metrics[0].implied_volatility_index_rank
                if iv_rank is not None:
                    return Decimal(str(iv_rank))

            return None

        except Exception as e:
            logger.error(f"Error getting IV rank for {symbol}: {e}")
            return None

    async def research_strategy(
        self,
        symbol: str,
        strategy: str = "short_put_spread"
    ) -> Optional[Dict[str, Any]]:
        """
        Research a strategy on a symbol without creating a trade.

        Args:
            symbol: The underlying symbol
            strategy: Strategy type to research

        Returns:
            Research results dict or None if unavailable
        """
        result = {
            "symbol": symbol,
            "strategy": strategy,
            "iv_rank": 0.0,
            "put_side": None,
            "call_side": None,
            "metrics": {},
            "rationale": "",
            "would_meet_criteria": False
        }

        # Get IV rank first
        iv_rank = await self.get_iv_rank(symbol)
        if iv_rank is None:
            result["rationale"] = f"Could not get IV rank for {symbol}"
            return result

        result["iv_rank"] = float(iv_rank)

        if iv_rank < self.risk_params.min_iv_rank:
            result["rationale"] = f"IV Rank ({iv_rank:.1f}%) below {self.risk_params.min_iv_rank}% threshold"
            return result

        # Use mock data for structure if available
        if self.use_mock_data and self.mock_provider:
            mock_data = self.mock_provider.get_symbol_data(symbol)
            if mock_data:
                price = float(mock_data.get('price', 100))
                strike_interval = float(mock_data.get('option_strike_interval', 1.0))

                # Build approximate structure
                if strategy in ('short_put', 'short_put_spread', 'iron_condor'):
                    short_put = round(price * 0.95 / strike_interval) * strike_interval
                    long_put = short_put - (strike_interval * 5)
                    result["put_side"] = {
                        "short_strike": short_put,
                        "long_strike": long_put
                    }

                if strategy in ('short_call', 'short_call_spread', 'iron_condor'):
                    short_call = round(price * 1.05 / strike_interval) * strike_interval
                    long_call = short_call + (strike_interval * 5)
                    result["call_side"] = {
                        "short_strike": short_call,
                        "long_strike": long_call
                    }

                # Estimate metrics
                spread_width = strike_interval * 5
                estimated_credit = spread_width * 0.33
                max_loss = (spread_width * 100) - (estimated_credit * 100)

                result["metrics"] = {
                    "net_credit": estimated_credit,
                    "max_loss": max_loss,
                    "pop": 0.70,
                    "dte": 45
                }

                result["would_meet_criteria"] = True
                result["rationale"] = (
                    f"IV Rank at {iv_rank:.1f}% meets criteria. "
                    f"Estimated credit: ${estimated_credit:.2f}, Max loss: ${max_loss:.2f}"
                )

        return result


class TradingBotCLI:
    """
    Command-line interface for the trading bot
    
    Provides interactive control with approval workflow
    """
    
    def __init__(self, bot: TastytradeBot):
        self.bot = bot
        self.running = False
    
    async def run(self):
        """Main CLI loop"""
        self.running = True
        print("\n" + "="*60)
        print("TASTYTRADE TRADING BOT")
        print("Following Tastylive Best Practices")
        print("="*60)

        # Show mode status
        mode_parts = []
        if self.bot.sandbox_mode:
            mode_parts.append("SANDBOX")
        else:
            mode_parts.append("PRODUCTION")
        if self.bot.use_mock_data:
            mode_parts.append("MOCK DATA")
        print(f"Mode: {' + '.join(mode_parts)}")

        if self.bot.use_mock_data:
            print("\nMock Data Info:")
            print("  - Market data (IV rank, Greeks, quotes) uses simulated values")
            print("  - Order execution uses real sandbox API")
            print("  - Configure mock prices in config.json -> mock_data -> symbols")

        print("\nCommands:")
        print("  scan <symbols>  - Scan for opportunities (e.g., scan SPY,QQQ,IWM)")
        print("  pending         - Show pending trades")
        print("  approve <id>    - Approve a trade")
        print("  reject <id>     - Reject a trade")
        print("  execute <id>    - Execute approved trade")
        print("  manage          - Check positions for management actions")
        print("  portfolio       - Show portfolio state")
        print("  risk            - Show risk parameters")
        print("  history         - Show trade history")
        print("  export          - Export state to file")
        print("  quit            - Exit")
        print("="*60)
        
        while self.running:
            try:
                cmd = input("\n> ").strip()
                if not cmd:
                    continue
                
                parts = cmd.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if command == 'quit' or command == 'exit':
                    self.running = False
                    print("Goodbye!")
                    
                elif command == 'scan':
                    symbols = args[0].split(',') if args else ['SPY', 'QQQ', 'IWM']
                    data_source = "[MOCK DATA]" if self.bot.use_mock_data else "[LIVE DATA]"
                    print(f"Scanning {data_source}: {symbols}")
                    opportunities = await self.bot.scan_for_opportunities(symbols)
                    print(f"\nFound {len(opportunities)} opportunities")
                    if opportunities:
                        for opp in opportunities:
                            print(f"  [{opp.id}] {opp.underlying_symbol} {opp.strategy.value}")
                            print(f"       IV Rank: {opp.iv_rank}% | Delta: {opp.delta:.2f} | DTE: {opp.dte}")
                            print(f"       Credit: ${opp.expected_credit:.2f} | Max Loss: ${opp.max_loss:.2f}")
                        print("\nUse 'pending' to see full details, 'approve <id>' to approve")
                    
                elif command == 'pending':
                    pending = self.bot.get_pending_trades()
                    if not pending:
                        print("No pending trades")
                    else:
                        for t in pending:
                            print(f"  [{t.id}] {t.underlying_symbol} {t.strategy.value} - ${t.expected_credit}")
                    
                elif command == 'approve' and args:
                    trade_id = args[0]
                    for t in self.bot.pending_trades:
                        if t.id == trade_id:
                            await self.bot.request_approval(t)
                            break
                    else:
                        print(f"Trade {trade_id} not found")
                    
                elif command == 'reject' and args:
                    trade_id = args[0]
                    for t in self.bot.pending_trades[:]:  # Iterate over copy
                        if t.id == trade_id:
                            t.status = TradeStatus.REJECTED
                            t.rejection_reason = "User rejected via CLI"
                            t.approval_timestamp = datetime.now()
                            # Move to history
                            self.bot.pending_trades.remove(t)
                            self.bot.trade_history.append(t)
                            print(f"Trade {trade_id} rejected")
                            break
                    else:
                        print(f"Trade {trade_id} not found")
                    
                elif command == 'execute' and args:
                    trade_id = args[0]
                    for t in self.bot.pending_trades:
                        if t.id == trade_id:
                            if t.status == TradeStatus.APPROVED:
                                await self.bot.execute_trade(t)
                            else:
                                print(f"Trade must be approved first (status: {t.status.value})")
                            break
                    else:
                        print(f"Trade {trade_id} not found")
                    
                elif command == 'manage':
                    print("Checking positions for management actions...")
                    recommendations = await self.bot.manage_positions()
                    if not recommendations:
                        print("No positions require management action")
                    else:
                        print(f"\n{'='*60}")
                        print("POSITION MANAGEMENT RECOMMENDATIONS")
                        print("(These are recommendations only - no auto-execution)")
                        print(f"{'='*60}")
                        for rec in recommendations:
                            status = "⚠️ TESTED" if rec['is_tested'] else ""
                            print(f"\n  {rec['underlying']} | {rec['symbol']}")
                            print(f"    DTE: {rec['dte']} | P&L: {rec['pnl_percent']:.1%} {status}")
                            print(f"    Action: {rec['action']}")
                            print(f"    Reason: {rec['reason']}")
                        print(f"\n{'='*60}")
                        print("Use 'approve' workflow to act on these recommendations")

                elif command == 'portfolio':
                    await self.bot.update_portfolio_state()
                    state = self.bot.portfolio_state
                    print(f"\nPortfolio State:")
                    print(f"  Net Liquidating Value: ${state.net_liquidating_value}")
                    print(f"  Buying Power: ${state.buying_power}")
                    print(f"  Cash Balance: ${state.cash_balance}")
                    print(f"  Open Positions: {state.open_positions}")
                    print(f"  Daily P&L: ${state.daily_pnl}")
                    print(f"  Portfolio Delta (β-weighted): {state.portfolio_delta}")
                    
                elif command == 'risk':
                    params = self.bot.risk_params
                    print(f"\nRisk Parameters:")
                    print(f"  Max Daily Loss: ${params.max_daily_loss}")
                    print(f"  Max Weekly Loss: ${params.max_weekly_loss}")
                    print(f"  Max Position Loss: ${params.max_position_loss}")
                    print(f"  Max Position Size: {params.max_position_size_percent:.1%}")
                    print(f"  Min IV Rank: {params.min_iv_rank}%")
                    print(f"  Target DTE: {params.target_dte}")
                    print(f"  Profit Target: {params.profit_target_percent:.0%}")
                    
                elif command == 'history':
                    history = self.bot.get_trade_history(10)
                    if not history:
                        print("No trade history")
                    else:
                        print("\nRecent Trades:")
                        for t in history:
                            print(f"  [{t.id}] {t.timestamp.strftime('%Y-%m-%d')} "
                                  f"{t.underlying_symbol} {t.strategy.value} - {t.status.value}")
                    
                elif command == 'export':
                    filepath = args[0] if args else 'bot_state.json'
                    self.bot.export_state(filepath)
                    
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")


# Example usage and main entry point
async def main():
    """
    Main entry point demonstrating bot usage

    OAuth Setup:
    1. Create OAuth app at https://developer.tastytrade.com/
    2. Set environment variables:
       export TT_CLIENT_SECRET='your_client_secret'
       export TT_REFRESH_TOKEN='your_refresh_token'
    3. Run this script

    Or pass credentials directly to bot.connect()
    """
    # Load configuration from config.json
    config = {}
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")

    # Extract settings from config
    bot_settings = config.get('bot_settings', {})
    risk_config = config.get('risk_parameters', {})
    claude_config = config.get('claude_advisor', {})

    # Build risk parameters from config
    loss_limits = risk_config.get('loss_limits', {})
    position_sizing = risk_config.get('position_sizing', {})
    entry_criteria = config.get('entry_criteria', {})
    dte_requirements = entry_criteria.get('dte_requirements', {})
    iv_requirements = entry_criteria.get('iv_requirements', {})

    risk_params = RiskParameters(
        max_daily_loss=Decimal(str(loss_limits.get('max_daily_loss', 500))),
        max_weekly_loss=Decimal(str(loss_limits.get('max_weekly_loss', 1500))),
        max_position_loss=Decimal(str(loss_limits.get('max_position_loss', 300))),
        max_position_size_percent=Decimal(str(position_sizing.get('max_position_size_percent', 0.05))),
        max_total_positions=position_sizing.get('max_total_positions', 15),
        min_iv_rank=Decimal(str(iv_requirements.get('min_iv_rank', 30))),
        target_dte=dte_requirements.get('target_dte', 45),
        min_dte=dte_requirements.get('min_dte', 30),
        max_dte=dte_requirements.get('max_dte', 60),
    )

    # Create bot with config
    sandbox_mode = bot_settings.get('sandbox_mode', True)
    bot = TastytradeBot(
        risk_params=risk_params,
        sandbox_mode=sandbox_mode,
        claude_config=claude_config,
        config=config  # Pass full config for mock data settings
    )

    # Connect using environment variables (recommended)
    # Set TT_CLIENT_SECRET and TT_REFRESH_TOKEN env vars
    if os.environ.get('TT_CLIENT_SECRET') and os.environ.get('TT_REFRESH_TOKEN'):
        connected = await bot.connect()
        if connected:
            mode_str = "sandbox" if sandbox_mode else "production"
            mock_str = " with mock data" if bot.use_mock_data else ""
            print(f"Connected to Tastytrade API ({mode_str} mode{mock_str})")
        else:
            print("Failed to connect - running in offline mode")
            if bot.use_mock_data:
                print("Mock data is enabled - you can still scan for opportunities")
    else:
        print("No OAuth credentials found. Set TT_CLIENT_SECRET and TT_REFRESH_TOKEN")
        if bot.use_mock_data:
            print("Mock data is enabled - you can scan for opportunities without API connection")
        else:
            print("Running in offline/demo mode - API features will not work")

    # Run CLI
    cli = TradingBotCLI(bot)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
