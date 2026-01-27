"""
Risk Management Functions for the Tastytrade Trading Bot

This module contains pure functions for risk calculations and trade validation.
These functions are stateless and take all required parameters explicitly,
making them easy to test in isolation.
"""

import logging
from decimal import Decimal
from typing import Tuple

from models import RiskParameters, TradeProposal, PortfolioState

logger = logging.getLogger(__name__)


def check_risk_limits(
    proposal: TradeProposal,
    risk_params: RiskParameters,
    portfolio_state: PortfolioState,
    sandbox_mode: bool = False
) -> Tuple[bool, str]:
    """
    Check if a proposed trade passes all risk limits.

    Args:
        proposal: The trade proposal to validate
        risk_params: Risk management parameters
        portfolio_state: Current portfolio state
        sandbox_mode: If True, relaxes some checks for sandbox testing

    Returns:
        (passed, reason) - Boolean indicating if trade passes and explanation
    """
    params = risk_params
    state = portfolio_state

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
    if state.buying_power > 0:
        bp_usage = (state.buying_power - proposal.max_loss) / state.buying_power
        if bp_usage < params.min_buying_power_reserve:
            return False, f"Insufficient buying power reserve (need {params.min_buying_power_reserve:.0%})"
    elif state.buying_power <= 0:
        if sandbox_mode and state.net_liquidating_value > 0:
            # Sandbox with 0 BP but positive NLV - skip check (uninitialized sandbox account)
            logger.debug("Sandbox mode: skipping BP reserve check (BP=0, NLV>0)")
        else:
            # Production mode or no NLV - reject trade
            return False, "Insufficient buying power (BP=0)"

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


def generate_trade_rationale(
    proposal: TradeProposal,
    risk_params: RiskParameters
) -> str:
    """
    Generate human-readable rationale for the trade following tastylive methodology.

    Args:
        proposal: The trade proposal
        risk_params: Risk management parameters

    Returns:
        Multi-line string with trade rationale
    """
    rationale_parts = [
        f"Strategy: {proposal.strategy.value}",
        f"Underlying: {proposal.underlying_symbol}",
        f"",
        "Tastylive Criteria Met:",
        f"  - IV Rank: {proposal.iv_rank}% (target: >{risk_params.min_iv_rank}%)",
        f"  - DTE: {proposal.dte} days (target: ~{risk_params.target_dte})",
        f"  - Probability OTM: {proposal.probability_of_profit:.1%} (target: >{risk_params.min_probability_otm:.0%})",
        f"",
        "Trade Details:",
        f"  Expected Credit: ${proposal.expected_credit}",
        f"  Max Loss: ${proposal.max_loss}",
        f"  Delta: {proposal.delta}",
        f"  Theta: {proposal.theta} (daily decay)",
        f"",
        "Management Plan:",
        f"  - Take profit at 50% (${proposal.expected_credit * Decimal('0.5'):.2f})",
        f"  - Roll or close at {risk_params.management_dte} DTE",
        f"  - Stop loss at 2x credit (${proposal.expected_credit * Decimal('2'):.2f} debit)",
    ]

    return "\n".join(rationale_parts)
