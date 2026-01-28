"""
Data models for the Tastytrade Trading Bot

This module contains all dataclasses and enums used throughout the trading bot.
These are the core data structures that represent trades, strategies, risk parameters,
and portfolio state.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any


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
    # Premium Selling Strategies
    SHORT_PUT = "short_put"
    SHORT_CALL = "short_call"
    SHORT_STRANGLE = "short_strangle"
    SHORT_STRADDLE = "short_straddle"
    IRON_CONDOR = "iron_condor"
    SHORT_PUT_SPREAD = "short_put_spread"
    SHORT_CALL_SPREAD = "short_call_spread"
    JADE_LIZARD = "jade_lizard"

    # Time Decay Strategies
    CALENDAR_SPREAD = "calendar_spread"
    DOUBLE_CALENDAR = "double_calendar"
    DIAGONAL_SPREAD = "diagonal_spread"

    # Leverage Strategies
    POOR_MANS_COVERED_CALL = "poor_mans_covered_call"
    SYNTHETIC_LONG = "synthetic_long"
    SYNTHETIC_SHORT = "synthetic_short"

    # Ratio Strategies
    PUT_RATIO_SPREAD = "put_ratio_spread"
    CALL_RATIO_SPREAD = "call_ratio_spread"

    # Advanced Strategies
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"

    # Stock-based Strategies
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    COLLAR = "collar"


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
    # Claude AI analysis
    claude_analysis: Optional[Dict[str, Any]] = None
    claude_confidence: Optional[int] = None


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


@dataclass
class TradeCandidate:
    """
    A candidate trade for multi-strategy evaluation.
    Used internally during scanning to compare different strategies.
    """
    strategy: StrategyType
    underlying: str
    legs: List[dict]
    net_credit: Decimal
    max_loss: Decimal
    probability_of_profit: Decimal
    iv_rank: Decimal
    dte: int
    greeks: dict
    score: Decimal = Decimal("0")  # Kelly-inspired score

    # Strategy-specific details
    short_strike: Optional[Decimal] = None
    long_strike: Optional[Decimal] = None
    spread_width: Optional[Decimal] = None
    # For iron condors
    put_short_strike: Optional[Decimal] = None
    put_long_strike: Optional[Decimal] = None
    call_short_strike: Optional[Decimal] = None
    call_long_strike: Optional[Decimal] = None

    def calculate_kelly_score(self) -> Decimal:
        """
        Calculate Kelly Criterion inspired score.

        Kelly formula: f* = p - q/b
        Where:
        - p = probability of profit
        - q = probability of loss (1 - p)
        - b = reward/risk ratio (total_credit/max_loss)

        Note: net_credit is per-share, max_loss is total dollars.
        We multiply credit by 100 (contract multiplier) for comparison.

        Higher score = better risk-adjusted opportunity
        """
        if self.max_loss <= 0 or self.net_credit <= 0:
            return Decimal("-999")

        p = self.probability_of_profit
        q = Decimal("1") - p

        # Convert per-share credit to total dollars (1 contract = 100 shares)
        total_credit = self.net_credit * 100
        b = total_credit / self.max_loss  # Reward/risk ratio

        # Kelly score: p - (q / b)
        # Avoid division by zero
        if b <= 0:
            return Decimal("-999")

        self.score = p - (q / b)
        return self.score

    def return_on_risk(self) -> Decimal:
        """Calculate return on risk percentage (total credit / max loss)."""
        if self.max_loss <= 0:
            return Decimal("0")
        # Convert per-share credit to total dollars
        total_credit = self.net_credit * 100
        return (total_credit / self.max_loss) * 100

    def expected_value(self) -> Decimal:
        """Calculate expected value of the trade."""
        p = self.probability_of_profit
        # Convert per-share credit to total dollars (1 contract = 100 shares)
        return (p * self.net_credit * 100) - ((1 - p) * self.max_loss)


class MarketBias(Enum):
    """Market directional bias"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"  # Expecting big move in either direction
    RANGE_BOUND = "range_bound"  # Tight range expectation


class IVEnvironment(Enum):
    """IV environment classification"""
    VERY_LOW = "very_low"  # IV Rank < 20
    LOW = "low"  # IV Rank 20-40
    MODERATE = "moderate"  # IV Rank 40-60
    HIGH = "high"  # IV Rank 60-80
    VERY_HIGH = "very_high"  # IV Rank > 80


@dataclass
class MarketCondition:
    """
    Market condition analysis for strategy selection.

    This class analyzes current market state to determine which
    strategies are most appropriate.
    """
    symbol: str
    iv_rank: Decimal
    iv_environment: IVEnvironment

    # Price action
    current_price: Optional[Decimal] = None
    price_trend: Optional[MarketBias] = None

    # Volatility indicators
    expected_move: Optional[Decimal] = None  # Expected move by expiration
    vol_trend: Optional[str] = None  # "expanding" or "contracting"

    # Technical levels
    support_level: Optional[Decimal] = None
    resistance_level: Optional[Decimal] = None

    # Event risk
    days_to_earnings: Optional[int] = None
    is_pre_earnings: bool = False
    is_post_earnings: bool = False

    # Additional context
    liquidity_score: Optional[Decimal] = None  # 0-100, higher is better

    def get_iv_environment(self) -> IVEnvironment:
        """Classify IV environment from IV rank"""
        if self.iv_rank < 20:
            return IVEnvironment.VERY_LOW
        elif self.iv_rank < 40:
            return IVEnvironment.LOW
        elif self.iv_rank < 60:
            return IVEnvironment.MODERATE
        elif self.iv_rank < 80:
            return IVEnvironment.HIGH
        else:
            return IVEnvironment.VERY_HIGH

    def recommend_strategy_types(self) -> List[str]:
        """
        Recommend strategy types based on market conditions.

        Returns list of strategy type names in priority order.
        """
        recommendations = []

        iv_env = self.get_iv_environment()

        # High IV environments favor premium selling
        if iv_env in [IVEnvironment.HIGH, IVEnvironment.VERY_HIGH]:
            recommendations.extend([
                "short_strangle",
                "short_straddle",
                "iron_condor",
                "jade_lizard",
                "short_put_spread",
                "short_call_spread"
            ])

        # Low IV environments favor debit strategies and time spreads
        elif iv_env == IVEnvironment.VERY_LOW:
            recommendations.extend([
                "calendar_spread",
                "diagonal_spread",
                "long_strangle",  # If expecting volatility expansion
                "poor_mans_covered_call"
            ])

        # Moderate IV is mixed
        else:
            recommendations.extend([
                "short_put_spread",
                "short_call_spread",
                "iron_condor",
                "calendar_spread"
            ])

        # Pre-earnings: capitalize on IV crush
        if self.is_pre_earnings and self.days_to_earnings and self.days_to_earnings <= 7:
            if iv_env in [IVEnvironment.HIGH, IVEnvironment.VERY_HIGH]:
                recommendations.insert(0, "short_strangle")
                recommendations.insert(1, "iron_condor")

        # Post-earnings: IV low, might expand
        if self.is_post_earnings and iv_env == IVEnvironment.VERY_LOW:
            recommendations.insert(0, "calendar_spread")

        return recommendations
