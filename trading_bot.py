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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Callable, Any, Dict
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Status of a proposed trade"""
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StrategyType(Enum):
    """Supported option strategies following tastylive methodology"""
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    SHORT_STRANGLE = "short_strangle"
    SHORT_STRADDLE = "short_straddle"
    IRON_CONDOR = "iron_condor"
    SHORT_PUT_SPREAD = "short_put_spread"
    SHORT_CALL_SPREAD = "short_call_spread"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"


@dataclass
class RiskParameters:
    """
    Risk management parameters following tastylive guidelines
    
    Tastylive recommends:
    - Trade small: 1-5% of portfolio per position
    - Keep portfolio delta neutral to slightly directional
    - Maintain buying power reserve
    """
    # Maximum loss limits
    max_daily_loss: Decimal = Decimal("1000.00")  # Maximum daily loss in dollars
    max_weekly_loss: Decimal = Decimal("3000.00")  # Maximum weekly loss in dollars
    max_position_loss: Decimal = Decimal("500.00")  # Maximum loss per position
    max_portfolio_loss_percent: Decimal = Decimal("0.05")  # 5% max portfolio drawdown
    
    # Position sizing (tastylive: trade small and often)
    max_position_size_percent: Decimal = Decimal("0.05")  # 5% of portfolio max per position
    max_total_positions: int = 20  # Maximum number of open positions
    max_positions_per_underlying: int = 2  # Max positions per stock/ETF
    
    # Buying power management
    min_buying_power_reserve: Decimal = Decimal("0.30")  # Keep 30% BP in reserve
    max_buying_power_usage: Decimal = Decimal("0.50")  # Use max 50% of buying power
    
    # Portfolio Greeks limits (beta-weighted to SPY)
    max_portfolio_delta: Decimal = Decimal("500")  # Max beta-weighted delta
    max_portfolio_theta: Decimal = Decimal("-100")  # Max daily theta decay
    
    # IV and probability filters
    min_iv_rank: Decimal = Decimal("30")  # Minimum IV Rank to enter (tastylive standard)
    min_probability_otm: Decimal = Decimal("0.65")  # Minimum probability of profit
    
    # DTE management (tastylive: 45 DTE entry, 21 DTE management)
    target_dte: int = 45
    min_dte: int = 30
    max_dte: int = 60
    management_dte: int = 21  # Roll or close at this DTE
    
    # Profit/Loss targets (tastylive: 50% profit target)
    profit_target_percent: Decimal = Decimal("0.50")  # Take profit at 50%
    stop_loss_multiplier: Decimal = Decimal("2.0")  # Stop at 2x credit received


@dataclass
class TradeProposal:
    """A proposed trade awaiting approval"""
    id: str
    timestamp: datetime
    strategy: StrategyType
    underlying_symbol: str
    legs: List[dict]  # List of option legs
    expected_credit: Decimal
    max_loss: Decimal
    probability_of_profit: Decimal
    iv_rank: Decimal
    dte: int
    delta: Decimal
    theta: Decimal
    vega: Decimal
    rationale: str
    status: TradeStatus = TradeStatus.PENDING_APPROVAL
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    execution_result: Optional[dict] = None


@dataclass 
class PortfolioState:
    """Current portfolio state for risk calculations"""
    account_value: Decimal = Decimal("0")
    buying_power: Decimal = Decimal("0")
    net_liquidating_value: Decimal = Decimal("0")
    cash_balance: Decimal = Decimal("0")
    
    # Current P&L
    daily_pnl: Decimal = Decimal("0")
    weekly_pnl: Decimal = Decimal("0")
    
    # Beta-weighted portfolio Greeks (to SPY)
    portfolio_delta: Decimal = Decimal("0")
    portfolio_theta: Decimal = Decimal("0")
    portfolio_vega: Decimal = Decimal("0")
    portfolio_gamma: Decimal = Decimal("0")
    
    # Position counts
    open_positions: int = 0
    positions_by_underlying: dict = field(default_factory=dict)


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
        sandbox_mode: bool = True
    ):
        """
        Initialize the trading bot
        
        Args:
            session: Tastytrade session object
            risk_params: Risk management parameters
            approval_callback: Function to call for trade approval
            sandbox_mode: If True, use sandbox environment
        """
        self.session = session
        self.risk_params = risk_params or RiskParameters()
        self.approval_callback = approval_callback
        self.sandbox_mode = sandbox_mode
        
        # State tracking
        self.portfolio_state = PortfolioState()
        self.pending_trades: List[TradeProposal] = []
        self.trade_history: List[TradeProposal] = []
        
        # Watchlist for scanning
        self.watchlist: List[str] = []
        
        logger.info(f"TastytradeBot initialized (sandbox_mode={sandbox_mode})")
    
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
            
            # Get balances
            balances = account.get_balances(self.session)
            self.portfolio_state.account_value = Decimal(str(balances.equity_value or 0))
            self.portfolio_state.buying_power = Decimal(str(balances.derivative_buying_power or 0))
            self.portfolio_state.net_liquidating_value = Decimal(str(balances.net_liquidating_value or 0))
            self.portfolio_state.cash_balance = Decimal(str(balances.cash_balance or 0))
            
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

    def check_risk_limits(self, proposal: TradeProposal) -> tuple[bool, str]:
        """
        Check if a proposed trade passes all risk limits
        
        Returns:
            (passed, reason) - Boolean and explanation
        """
        params = self.risk_params
        state = self.portfolio_state
        
        # Check daily loss limit
        if state.daily_pnl <= -params.max_daily_loss:
            return False, f"Daily loss limit reached (${params.max_daily_loss})"
        
        # Check weekly loss limit
        if state.weekly_pnl <= -params.max_weekly_loss:
            return False, f"Weekly loss limit reached (${params.max_weekly_loss})"
        
        # Check position max loss
        if proposal.max_loss > params.max_position_loss:
            return False, f"Position max loss ${proposal.max_loss} exceeds limit ${params.max_position_loss}"
        
        # Check position size as percent of portfolio
        if state.net_liquidating_value > 0:
            position_percent = proposal.max_loss / state.net_liquidating_value
            if position_percent > params.max_position_size_percent:
                return False, f"Position size {position_percent:.1%} exceeds max {params.max_position_size_percent:.1%}"
        
        # Check total positions limit
        if state.open_positions >= params.max_total_positions:
            return False, f"Maximum positions ({params.max_total_positions}) reached"
        
        # Check positions per underlying
        underlying_count = state.positions_by_underlying.get(proposal.underlying_symbol, 0)
        if underlying_count >= params.max_positions_per_underlying:
            return False, f"Max positions for {proposal.underlying_symbol} reached ({params.max_positions_per_underlying})"
        
        # Check buying power usage
        bp_usage = (state.buying_power - proposal.max_loss) / state.buying_power if state.buying_power > 0 else 0
        if bp_usage < params.min_buying_power_reserve:
            return False, f"Insufficient buying power reserve (need {params.min_buying_power_reserve:.0%})"
        
        # Check IV Rank (tastylive: enter when IV is elevated)
        if proposal.iv_rank < params.min_iv_rank:
            return False, f"IV Rank {proposal.iv_rank} below minimum {params.min_iv_rank}"
        
        # Check probability of profit
        if proposal.probability_of_profit < params.min_probability_otm:
            return False, f"Probability {proposal.probability_of_profit:.0%} below minimum {params.min_probability_otm:.0%}"
        
        # Check DTE range (tastylive: ~45 DTE)
        if not (params.min_dte <= proposal.dte <= params.max_dte):
            return False, f"DTE {proposal.dte} outside range {params.min_dte}-{params.max_dte}"
        
        # Check portfolio delta impact
        new_delta = state.portfolio_delta + proposal.delta
        if abs(new_delta) > params.max_portfolio_delta:
            return False, f"Would exceed portfolio delta limit ({params.max_portfolio_delta})"
        
        return True, "All risk checks passed"
    
    def generate_trade_rationale(self, proposal: TradeProposal) -> str:
        """
        Generate human-readable rationale for the trade
        following tastylive methodology
        """
        rationale_parts = [
            f"Strategy: {proposal.strategy.value}",
            f"Underlying: {proposal.underlying_symbol}",
            f"",
            "Tastylive Criteria Met:",
            f"  ✓ IV Rank: {proposal.iv_rank}% (target: >{self.risk_params.min_iv_rank}%)",
            f"  ✓ DTE: {proposal.dte} days (target: ~{self.risk_params.target_dte})",
            f"  ✓ Probability OTM: {proposal.probability_of_profit:.1%} (target: >{self.risk_params.min_probability_otm:.0%})",
            f"",
            "Trade Details:",
            f"  Expected Credit: ${proposal.expected_credit}",
            f"  Max Loss: ${proposal.max_loss}",
            f"  Delta: {proposal.delta}",
            f"  Theta: {proposal.theta} (daily decay)",
            f"",
            "Management Plan:",
            f"  • Take profit at 50% (${proposal.expected_credit * Decimal('0.5'):.2f})",
            f"  • Roll or close at {self.risk_params.management_dte} DTE",
            f"  • Stop loss at 2x credit (${proposal.expected_credit * Decimal('2'):.2f} debit)",
        ]
        
        return "\n".join(rationale_parts)
    
    async def propose_trade(
        self,
        strategy: StrategyType,
        underlying: str,
        legs: List[dict],
        greeks: dict,
        iv_rank: Decimal
    ) -> Optional[TradeProposal]:
        """
        Create a trade proposal for approval
        
        Args:
            strategy: Type of options strategy
            underlying: Underlying symbol (e.g., 'SPY')
            legs: List of option legs with strike, expiration, etc.
            greeks: Position Greeks (delta, theta, vega, gamma)
            iv_rank: Current IV Rank percentage
        
        Returns:
            TradeProposal if created, None if rejected by risk checks
        """
        import uuid
        
        # Calculate expected credit and max loss from legs
        expected_credit = sum(Decimal(str(leg.get('credit', 0))) for leg in legs)
        max_loss = sum(Decimal(str(leg.get('max_loss', 0))) for leg in legs)
        
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
        
        # Display proposal details
        print("\n" + "="*60)
        print("TRADE APPROVAL REQUEST")
        print("="*60)
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
    
    async def scan_for_opportunities(self, symbols: List[str]) -> List[TradeProposal]:
        """
        Scan symbols for trading opportunities based on tastylive criteria

        Looks for:
        1. Elevated IV Rank (>30)
        2. Options at ~45 DTE
        3. Strikes at ~30 delta (or 16 delta for conservative)
        4. Sufficient liquidity
        """
        opportunities = []

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

                    # Get puts only for short put strategy
                    puts = [opt for opt in options if opt.option_type == 'P']
                    if not puts:
                        continue

                    # Stream Greeks to find ~30 delta put
                    async with DXLinkStreamer(self.session) as streamer:
                        await streamer.subscribe(Greeks, [opt.streamer_symbol for opt in puts])

                        # Collect Greeks for all puts
                        option_greeks = {}
                        timeout_count = 0
                        while len(option_greeks) < len(puts) and timeout_count < 50:
                            try:
                                greeks_data = await asyncio.wait_for(
                                    streamer.get_event(Greeks), timeout=0.5
                                )
                                option_greeks[greeks_data.event_symbol] = greeks_data
                            except asyncio.TimeoutError:
                                timeout_count += 1
                                continue

                        # Find the put closest to 30 delta
                        target_delta = Decimal("-0.30")
                        best_option = None
                        best_delta_diff = Decimal("1.0")
                        best_greeks = None

                        for opt in puts:
                            if opt.streamer_symbol in option_greeks:
                                g = option_greeks[opt.streamer_symbol]
                                if g.delta is not None:
                                    delta = Decimal(str(g.delta))
                                    delta_diff = abs(delta - target_delta)
                                    if delta_diff < best_delta_diff:
                                        best_delta_diff = delta_diff
                                        best_option = opt
                                        best_greeks = g

                        if best_option is None or best_greeks is None:
                            logger.debug(f"No suitable delta found for {symbol}")
                            continue

                        # Get quote for credit pricing
                        await streamer.subscribe(Quote, [best_option.streamer_symbol])
                        try:
                            quote = await asyncio.wait_for(
                                streamer.get_event(Quote), timeout=2.0
                            )
                            # Use mid price for credit estimate
                            bid = Decimal(str(quote.bid_price or 0))
                            ask = Decimal(str(quote.ask_price or 0))
                            credit = (bid + ask) / 2
                        except asyncio.TimeoutError:
                            logger.warning(f"Quote timeout for {best_option.symbol}")
                            credit = Decimal("0")

                        # Calculate max loss for cash-secured put
                        max_loss = (best_option.strike_price * 100) - (credit * 100)

                        dte = (target_exp - date.today()).days

                        # Calculate probability OTM from delta
                        prob_otm = Decimal("1") + Decimal(str(best_greeks.delta or -0.30))

                        legs = [{
                            'symbol': best_option.symbol,
                            'streamer_symbol': best_option.streamer_symbol,
                            'option_type': 'PUT',
                            'strike': str(best_option.strike_price),
                            'expiration': str(target_exp),
                            'action': 'SELL_TO_OPEN',
                            'quantity': 1,
                            'credit': str(credit),
                            'max_loss': str(max_loss),
                            'dte': dte
                        }]

                        greeks = {
                            'delta': float(best_greeks.delta or 0),
                            'theta': float(best_greeks.theta or 0),
                            'vega': float(best_greeks.vega or 0),
                            'gamma': float(best_greeks.gamma or 0),
                            'pop': float(prob_otm)
                        }

                        proposal = await self.propose_trade(
                            strategy=StrategyType.SHORT_PUT,
                            underlying=symbol,
                            legs=legs,
                            greeks=greeks,
                            iv_rank=iv_rank
                        )

                        if proposal:
                            opportunities.append(proposal)

                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
        except Exception as e:
            logger.error(f"Scan failed: {e}")

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
                    print(f"Scanning: {symbols}")
                    opportunities = await self.bot.scan_for_opportunities(symbols)
                    print(f"Found {len(opportunities)} opportunities")
                    
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
    # Create bot with default risk parameters
    bot = TastytradeBot(
        risk_params=RiskParameters(
            max_daily_loss=Decimal("500"),
            max_weekly_loss=Decimal("1500"),
            max_position_loss=Decimal("300"),
            min_iv_rank=Decimal("30"),
            target_dte=45,
        ),
        sandbox_mode=True  # Always start in sandbox!
    )

    # Connect using environment variables (recommended)
    # Set TT_CLIENT_SECRET and TT_REFRESH_TOKEN env vars
    if os.environ.get('TT_CLIENT_SECRET') and os.environ.get('TT_REFRESH_TOKEN'):
        connected = await bot.connect()
        if connected:
            print("Connected to Tastytrade API (sandbox mode)")
        else:
            print("Failed to connect - running in offline mode")
    else:
        print("No OAuth credentials found. Set TT_CLIENT_SECRET and TT_REFRESH_TOKEN")
        print("Running in offline/demo mode - API features will not work")

    # Run CLI
    cli = TradingBotCLI(bot)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
