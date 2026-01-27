"""
Data models for the Tastytrade Trading Bot

This module contains all dataclasses and enums used throughout the trading bot.
These are the core data structures that represent trades, strategies, risk parameters,
and portfolio state.
"""

from dataclasses import dataclass, field
from datetime import datetime
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
        return (p * self.net_credit) - ((1 - p) * self.max_loss)
