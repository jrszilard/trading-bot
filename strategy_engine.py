"""
Strategy Engine for the Tastytrade Trading Bot

This module contains the StrategyEngine class that evaluates and builds
option strategy candidates (put spreads, call spreads, iron condors).
It implements Kelly Criterion scoring for risk-adjusted strategy selection.
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional, List

from models import (
    StrategyType, TradeCandidate, RiskParameters,
    MarketCondition, IVEnvironment, MarketBias, PortfolioState
)

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Evaluates and builds option strategy candidates.

    Implements tastylive methodology:
    - 30 delta for directional spreads
    - 16 delta for iron condor wings
    - Kelly Criterion scoring for risk-adjusted selection
    """

    def __init__(self, risk_params: RiskParameters):
        """
        Initialize the strategy engine.

        Args:
            risk_params: Risk management parameters
        """
        self.risk_params = risk_params

    def analyze_market_conditions(
        self,
        symbol: str,
        iv_rank: Decimal,
        current_price: Optional[Decimal] = None,
        underlying_data: Optional[dict] = None
    ) -> MarketCondition:
        """
        Analyze market conditions for a symbol to guide strategy selection.

        Args:
            symbol: Underlying symbol
            iv_rank: Current IV rank
            current_price: Current price of underlying
            underlying_data: Additional data (earnings, technicals, etc.)

        Returns:
            MarketCondition object with analysis
        """
        # Determine IV environment
        if iv_rank < 20:
            iv_env = IVEnvironment.VERY_LOW
        elif iv_rank < 40:
            iv_env = IVEnvironment.LOW
        elif iv_rank < 60:
            iv_env = IVEnvironment.MODERATE
        elif iv_rank < 80:
            iv_env = IVEnvironment.HIGH
        else:
            iv_env = IVEnvironment.VERY_HIGH

        condition = MarketCondition(
            symbol=symbol,
            iv_rank=iv_rank,
            iv_environment=iv_env,
            current_price=current_price
        )

        # Add additional context if provided
        if underlying_data:
            condition.days_to_earnings = underlying_data.get('days_to_earnings')
            condition.support_level = underlying_data.get('support_level')
            condition.resistance_level = underlying_data.get('resistance_level')
            condition.liquidity_score = underlying_data.get('liquidity_score')

            # Check if pre/post earnings
            if condition.days_to_earnings:
                condition.is_pre_earnings = 0 < condition.days_to_earnings <= 7
                condition.is_post_earnings = -7 <= condition.days_to_earnings < 0

        return condition

    def build_put_spread_candidate(
        self,
        symbol: str,
        puts: List,
        greeks_by_strike: dict,
        quotes_by_strike: dict,
        iv_rank: Decimal,
        target_exp: date,
        target_delta: Decimal = Decimal("-0.30")
    ) -> Optional[TradeCandidate]:
        """
        Build a put credit spread candidate for evaluation.

        Args:
            symbol: Underlying symbol
            puts: List of put options sorted by strike descending
            greeks_by_strike: Map of strike -> greeks
            quotes_by_strike: Map of strike -> (bid, ask) tuple
            iv_rank: Current IV rank
            target_exp: Target expiration date
            target_delta: Target delta for short put (default -0.30)

        Returns:
            TradeCandidate if valid spread found, None otherwise
        """
        # Find the short put at target delta
        short_put = None
        short_greeks = None
        best_delta_diff = Decimal("1.0")

        for opt in puts:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in greeks_by_strike:
                continue
            g = greeks_by_strike[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                short_put = opt
                short_greeks = g

        if short_put is None:
            return None

        short_strike = short_put.strike_price if hasattr(short_put, 'strike_price') else Decimal(str(short_put.get('strike_price', 0)))

        if short_strike not in quotes_by_strike:
            return None

        short_bid, short_ask = quotes_by_strike[short_strike]
        short_credit = (short_bid + short_ask) / 2

        if short_credit <= 0:
            return None

        # Calculate optimal spread width
        estimated_net_credit = short_credit * Decimal("0.40")
        max_width = (self.risk_params.max_position_loss + estimated_net_credit * 100) / 100

        # Get strike interval
        sorted_strikes = sorted(greeks_by_strike.keys(), reverse=True)
        if len(sorted_strikes) >= 2:
            strike_interval = abs(sorted_strikes[0] - sorted_strikes[1])
        else:
            strike_interval = Decimal("1.0")

        # Find optimal long put
        available_strikes = [s for s in sorted_strikes if s < short_strike]
        if not available_strikes:
            return None

        long_strike = None
        long_greeks = None
        net_credit = Decimal("0")
        max_loss = Decimal("0")
        spread_width = Decimal("0")

        for strike in sorted(available_strikes, reverse=True):
            if strike not in quotes_by_strike:
                continue

            long_bid, long_ask = quotes_by_strike[strike]
            long_debit = (long_bid + long_ask) / 2

            width = short_strike - strike
            test_net_credit = short_credit - long_debit
            test_max_loss = (width * 100) - (test_net_credit * 100)

            if test_max_loss <= self.risk_params.max_position_loss and test_net_credit > Decimal("0.10"):
                long_strike = strike
                long_greeks = greeks_by_strike[strike]
                spread_width = width
                net_credit = test_net_credit
                max_loss = test_max_loss
                break

        if long_strike is None:
            return None

        dte = (target_exp - date.today()).days
        short_delta = Decimal(str(short_greeks.delta if hasattr(short_greeks, 'delta') else short_greeks.get('delta', 0)))
        prob_otm = Decimal("1") + short_delta

        # Build legs
        legs = self._create_spread_legs(
            short_put, long_strike, short_credit, net_credit,
            target_exp, dte, 'PUT', greeks_by_strike, quotes_by_strike,
            options=puts
        )

        # Net greeks
        long_delta = Decimal(str(long_greeks.delta if hasattr(long_greeks, 'delta') else long_greeks.get('delta', 0)))
        long_theta = Decimal(str(long_greeks.theta if hasattr(long_greeks, 'theta') else long_greeks.get('theta', 0)))
        long_vega = Decimal(str(long_greeks.vega if hasattr(long_greeks, 'vega') else long_greeks.get('vega', 0)))
        long_gamma = Decimal(str(long_greeks.gamma if hasattr(long_greeks, 'gamma') else long_greeks.get('gamma', 0)))
        short_theta = Decimal(str(short_greeks.theta if hasattr(short_greeks, 'theta') else short_greeks.get('theta', 0)))
        short_vega = Decimal(str(short_greeks.vega if hasattr(short_greeks, 'vega') else short_greeks.get('vega', 0)))
        short_gamma = Decimal(str(short_greeks.gamma if hasattr(short_greeks, 'gamma') else short_greeks.get('gamma', 0)))

        greeks = {
            'delta': float(short_delta + long_delta),
            'theta': float(short_theta + long_theta),
            'vega': float(short_vega + long_vega),
            'gamma': float(short_gamma + long_gamma),
            'pop': float(prob_otm)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.SHORT_PUT_SPREAD,
            underlying=symbol,
            legs=legs,
            net_credit=net_credit,
            max_loss=max_loss,
            probability_of_profit=prob_otm,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=spread_width
        )
        candidate.calculate_kelly_score()

        return candidate

    def build_call_spread_candidate(
        self,
        symbol: str,
        calls: List,
        greeks_by_strike: dict,
        quotes_by_strike: dict,
        iv_rank: Decimal,
        target_exp: date,
        target_delta: Decimal = Decimal("0.30")
    ) -> Optional[TradeCandidate]:
        """
        Build a call credit spread candidate for evaluation.

        Args:
            symbol: Underlying symbol
            calls: List of call options sorted by strike ascending
            greeks_by_strike: Map of strike -> greeks
            quotes_by_strike: Map of strike -> (bid, ask) tuple
            iv_rank: Current IV rank
            target_exp: Target expiration date
            target_delta: Target delta for short call (default 0.30)

        Returns:
            TradeCandidate if valid spread found, None otherwise
        """
        # Find the short call at target delta
        short_call = None
        short_greeks = None
        best_delta_diff = Decimal("1.0")

        for opt in calls:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in greeks_by_strike:
                continue
            g = greeks_by_strike[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                short_call = opt
                short_greeks = g

        if short_call is None:
            return None

        short_strike = short_call.strike_price if hasattr(short_call, 'strike_price') else Decimal(str(short_call.get('strike_price', 0)))

        if short_strike not in quotes_by_strike:
            return None

        short_bid, short_ask = quotes_by_strike[short_strike]
        short_credit = (short_bid + short_ask) / 2

        if short_credit <= 0:
            return None

        # Calculate optimal spread width
        estimated_net_credit = short_credit * Decimal("0.40")
        max_width = (self.risk_params.max_position_loss + estimated_net_credit * 100) / 100

        # Get strike interval
        sorted_strikes = sorted(greeks_by_strike.keys())
        if len(sorted_strikes) >= 2:
            strike_interval = abs(sorted_strikes[1] - sorted_strikes[0])
        else:
            strike_interval = Decimal("1.0")

        # Find optimal long call (higher strike for call spread)
        available_strikes = [s for s in sorted_strikes if s > short_strike]
        if not available_strikes:
            return None

        long_strike = None
        long_greeks = None
        net_credit = Decimal("0")
        max_loss = Decimal("0")
        spread_width = Decimal("0")

        for strike in sorted(available_strikes):  # Ascending for calls
            if strike not in quotes_by_strike:
                continue

            long_bid, long_ask = quotes_by_strike[strike]
            long_debit = (long_bid + long_ask) / 2

            width = strike - short_strike
            test_net_credit = short_credit - long_debit
            test_max_loss = (width * 100) - (test_net_credit * 100)

            if test_max_loss <= self.risk_params.max_position_loss and test_net_credit > Decimal("0.10"):
                long_strike = strike
                long_greeks = greeks_by_strike[strike]
                spread_width = width
                net_credit = test_net_credit
                max_loss = test_max_loss
                break

        if long_strike is None:
            return None

        dte = (target_exp - date.today()).days
        short_delta = Decimal(str(short_greeks.delta if hasattr(short_greeks, 'delta') else short_greeks.get('delta', 0)))
        # For calls, prob OTM = 1 - delta
        prob_otm = Decimal("1") - short_delta

        # Build legs
        legs = self._create_spread_legs(
            short_call, long_strike, short_credit, net_credit,
            target_exp, dte, 'CALL', greeks_by_strike, quotes_by_strike,
            options=calls
        )

        # Net greeks
        long_delta = Decimal(str(long_greeks.delta if hasattr(long_greeks, 'delta') else long_greeks.get('delta', 0)))
        long_theta = Decimal(str(long_greeks.theta if hasattr(long_greeks, 'theta') else long_greeks.get('theta', 0)))
        long_vega = Decimal(str(long_greeks.vega if hasattr(long_greeks, 'vega') else long_greeks.get('vega', 0)))
        long_gamma = Decimal(str(long_greeks.gamma if hasattr(long_greeks, 'gamma') else long_greeks.get('gamma', 0)))
        short_theta = Decimal(str(short_greeks.theta if hasattr(short_greeks, 'theta') else short_greeks.get('theta', 0)))
        short_vega = Decimal(str(short_greeks.vega if hasattr(short_greeks, 'vega') else short_greeks.get('vega', 0)))
        short_gamma = Decimal(str(short_greeks.gamma if hasattr(short_greeks, 'gamma') else short_greeks.get('gamma', 0)))

        greeks = {
            'delta': float(short_delta + long_delta),
            'theta': float(short_theta + long_theta),
            'vega': float(short_vega + long_vega),
            'gamma': float(short_gamma + long_gamma),
            'pop': float(prob_otm)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.SHORT_CALL_SPREAD,
            underlying=symbol,
            legs=legs,
            net_credit=net_credit,
            max_loss=max_loss,
            probability_of_profit=prob_otm,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            short_strike=short_strike,
            long_strike=long_strike,
            spread_width=spread_width
        )
        candidate.calculate_kelly_score()

        return candidate

    def build_iron_condor_candidate(
        self,
        symbol: str,
        puts: List,
        calls: List,
        put_greeks: dict,
        call_greeks: dict,
        put_quotes: dict,
        call_quotes: dict,
        iv_rank: Decimal,
        target_exp: date,
        put_delta: Decimal = Decimal("-0.16"),
        call_delta: Decimal = Decimal("0.16")
    ) -> Optional[TradeCandidate]:
        """
        Build an iron condor candidate for evaluation.

        Iron condor = put spread + call spread (neutral strategy)
        Uses 16 delta for wider wings (tastylive standard for IC)

        Returns:
            TradeCandidate if valid iron condor found, None otherwise
        """
        # Build put spread side
        put_candidate = self.build_put_spread_candidate(
            symbol, puts, put_greeks, put_quotes, iv_rank, target_exp,
            target_delta=put_delta
        )

        # Build call spread side
        call_candidate = self.build_call_spread_candidate(
            symbol, calls, call_greeks, call_quotes, iv_rank, target_exp,
            target_delta=call_delta
        )

        if put_candidate is None or call_candidate is None:
            return None

        # Combine into iron condor
        # Check combined max loss fits within limit
        # For iron condor, max loss is the wider spread width (since you can only lose on one side)
        combined_max_loss = max(put_candidate.max_loss, call_candidate.max_loss)
        combined_credit = put_candidate.net_credit + call_candidate.net_credit

        if combined_max_loss > self.risk_params.max_position_loss:
            return None

        # Combined probability: product of both sides being OTM
        combined_prob = put_candidate.probability_of_profit * call_candidate.probability_of_profit

        dte = put_candidate.dte

        # Combine legs
        legs = put_candidate.legs + call_candidate.legs

        # Combined greeks (mostly cancel out for neutral strategy)
        greeks = {
            'delta': put_candidate.greeks['delta'] + call_candidate.greeks['delta'],
            'theta': put_candidate.greeks['theta'] + call_candidate.greeks['theta'],
            'vega': put_candidate.greeks['vega'] + call_candidate.greeks['vega'],
            'gamma': put_candidate.greeks['gamma'] + call_candidate.greeks['gamma'],
            'pop': float(combined_prob)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.IRON_CONDOR,
            underlying=symbol,
            legs=legs,
            net_credit=combined_credit,
            max_loss=combined_max_loss,
            probability_of_profit=combined_prob,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            put_short_strike=put_candidate.short_strike,
            put_long_strike=put_candidate.long_strike,
            call_short_strike=call_candidate.short_strike,
            call_long_strike=call_candidate.long_strike
        )
        candidate.calculate_kelly_score()

        return candidate

    def build_short_strangle_candidate(
        self,
        symbol: str,
        puts: List,
        calls: List,
        put_greeks: dict,
        call_greeks: dict,
        put_quotes: dict,
        call_quotes: dict,
        iv_rank: Decimal,
        target_exp: date,
        put_delta: Decimal = Decimal("-0.16"),
        call_delta: Decimal = Decimal("0.16")
    ) -> Optional[TradeCandidate]:
        """
        Build a short strangle candidate (undefined risk).

        Short strangle = sell OTM put + sell OTM call (no protection)
        Higher premium than iron condor but undefined risk on both sides.

        Best for: High IV rank (>50), expecting range-bound movement
        """
        # Find short put at target delta
        short_put = None
        short_put_greeks = None
        best_delta_diff = Decimal("1.0")

        for opt in puts:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in put_greeks:
                continue
            g = put_greeks[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - put_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                short_put = opt
                short_put_greeks = g

        if short_put is None:
            return None

        # Find short call at target delta
        short_call = None
        short_call_greeks = None
        best_delta_diff = Decimal("1.0")

        for opt in calls:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in call_greeks:
                continue
            g = call_greeks[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - call_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                short_call = opt
                short_call_greeks = g

        if short_call is None:
            return None

        # Get strikes and quotes
        put_strike = short_put.strike_price if hasattr(short_put, 'strike_price') else Decimal(str(short_put.get('strike_price', 0)))
        call_strike = short_call.strike_price if hasattr(short_call, 'strike_price') else Decimal(str(short_call.get('strike_price', 0)))

        if put_strike not in put_quotes or call_strike not in call_quotes:
            return None

        put_bid, put_ask = put_quotes[put_strike]
        call_bid, call_ask = call_quotes[call_strike]

        put_credit = (put_bid + put_ask) / 2
        call_credit = (call_bid + call_ask) / 2
        total_credit = put_credit + call_credit

        if total_credit <= Decimal("0.20"):
            return None

        # For undefined risk, estimate max loss based on wing width expectation
        # Use expected move or default to reasonable estimate
        estimated_max_loss = total_credit * 100 * 3  # Conservative 3:1 risk/reward assumption

        dte = (target_exp - date.today()).days

        # Combined probability (both sides OTM)
        put_delta_val = Decimal(str(short_put_greeks.delta if hasattr(short_put_greeks, 'delta') else short_put_greeks.get('delta', 0)))
        call_delta_val = Decimal(str(short_call_greeks.delta if hasattr(short_call_greeks, 'delta') else short_call_greeks.get('delta', 0)))

        put_prob_otm = Decimal("1") + put_delta_val
        call_prob_otm = Decimal("1") - call_delta_val
        combined_prob = put_prob_otm * call_prob_otm

        # Build legs
        put_symbol = short_put.symbol if hasattr(short_put, 'symbol') else short_put.get('symbol', '')
        put_streamer = short_put.streamer_symbol if hasattr(short_put, 'streamer_symbol') else short_put.get('streamer_symbol', '')
        call_symbol = short_call.symbol if hasattr(short_call, 'symbol') else short_call.get('symbol', '')
        call_streamer = short_call.streamer_symbol if hasattr(short_call, 'streamer_symbol') else short_call.get('streamer_symbol', '')

        legs = [
            {
                'symbol': put_symbol,
                'streamer_symbol': put_streamer,
                'option_type': 'PUT',
                'strike': str(put_strike),
                'expiration': str(target_exp),
                'action': 'SELL_TO_OPEN',
                'quantity': 1,
                'credit': str(put_credit),
                'dte': dte
            },
            {
                'symbol': call_symbol,
                'streamer_symbol': call_streamer,
                'option_type': 'CALL',
                'strike': str(call_strike),
                'expiration': str(target_exp),
                'action': 'SELL_TO_OPEN',
                'quantity': 1,
                'credit': str(call_credit),
                'dte': dte
            }
        ]

        # Combined greeks
        call_theta = Decimal(str(short_call_greeks.theta if hasattr(short_call_greeks, 'theta') else short_call_greeks.get('theta', 0)))
        call_vega = Decimal(str(short_call_greeks.vega if hasattr(short_call_greeks, 'vega') else short_call_greeks.get('vega', 0)))
        call_gamma = Decimal(str(short_call_greeks.gamma if hasattr(short_call_greeks, 'gamma') else short_call_greeks.get('gamma', 0)))
        put_theta = Decimal(str(short_put_greeks.theta if hasattr(short_put_greeks, 'theta') else short_put_greeks.get('theta', 0)))
        put_vega = Decimal(str(short_put_greeks.vega if hasattr(short_put_greeks, 'vega') else short_put_greeks.get('vega', 0)))
        put_gamma = Decimal(str(short_put_greeks.gamma if hasattr(short_put_greeks, 'gamma') else short_put_greeks.get('gamma', 0)))

        greeks = {
            'delta': float(put_delta_val + call_delta_val),
            'theta': float(put_theta + call_theta),
            'vega': float(put_vega + call_vega),
            'gamma': float(put_gamma + call_gamma),
            'pop': float(combined_prob)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.SHORT_STRANGLE,
            underlying=symbol,
            legs=legs,
            net_credit=total_credit,
            max_loss=estimated_max_loss,
            probability_of_profit=combined_prob,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            put_short_strike=put_strike,
            call_short_strike=call_strike
        )
        candidate.calculate_kelly_score()

        return candidate

    def build_short_straddle_candidate(
        self,
        symbol: str,
        puts: List,
        calls: List,
        put_greeks: dict,
        call_greeks: dict,
        put_quotes: dict,
        call_quotes: dict,
        iv_rank: Decimal,
        target_exp: date,
        current_price: Optional[Decimal] = None
    ) -> Optional[TradeCandidate]:
        """
        Build a short straddle candidate (undefined risk).

        Short straddle = sell ATM put + sell ATM call
        Maximum premium collection but highest risk.

        Best for: Very high IV rank (>70), tight range expectation
        """
        if current_price is None:
            # Try to infer from available strikes
            put_strikes = sorted(put_greeks.keys())
            if put_strikes:
                current_price = put_strikes[len(put_strikes) // 2]
            else:
                return None

        # Find ATM strike (closest to current price)
        atm_strike = None
        min_distance = Decimal("999999")

        for strike in put_greeks.keys():
            if strike in call_greeks and strike in put_quotes and strike in call_quotes:
                distance = abs(strike - current_price)
                if distance < min_distance:
                    min_distance = distance
                    atm_strike = strike

        if atm_strike is None:
            return None

        # Find put and call options at ATM strike
        short_put = None
        short_call = None

        for opt in puts:
            opt_strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if opt_strike == atm_strike:
                short_put = opt
                break

        for opt in calls:
            opt_strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if opt_strike == atm_strike:
                short_call = opt
                break

        if short_put is None or short_call is None:
            return None

        # Get quotes and greeks
        put_bid, put_ask = put_quotes[atm_strike]
        call_bid, call_ask = call_quotes[atm_strike]

        put_credit = (put_bid + put_ask) / 2
        call_credit = (call_bid + call_ask) / 2
        total_credit = put_credit + call_credit

        if total_credit <= Decimal("0.50"):
            return None

        put_greeks_obj = put_greeks[atm_strike]
        call_greeks_obj = call_greeks[atm_strike]

        # Estimate max loss (undefined, use conservative estimate)
        estimated_max_loss = total_credit * 100 * 3

        dte = (target_exp - date.today()).days

        # ATM straddle probability is lower (needs to stay within credit collected)
        # Approximate: probability that price stays within credit range
        # For simplicity, use ~60% as typical ATM straddle POP
        combined_prob = Decimal("0.60")

        # Build legs
        put_symbol = short_put.symbol if hasattr(short_put, 'symbol') else short_put.get('symbol', '')
        put_streamer = short_put.streamer_symbol if hasattr(short_put, 'streamer_symbol') else short_put.get('streamer_symbol', '')
        call_symbol = short_call.symbol if hasattr(short_call, 'symbol') else short_call.get('symbol', '')
        call_streamer = short_call.streamer_symbol if hasattr(short_call, 'streamer_symbol') else short_call.get('streamer_symbol', '')

        legs = [
            {
                'symbol': put_symbol,
                'streamer_symbol': put_streamer,
                'option_type': 'PUT',
                'strike': str(atm_strike),
                'expiration': str(target_exp),
                'action': 'SELL_TO_OPEN',
                'quantity': 1,
                'credit': str(put_credit),
                'dte': dte
            },
            {
                'symbol': call_symbol,
                'streamer_symbol': call_streamer,
                'option_type': 'CALL',
                'strike': str(atm_strike),
                'expiration': str(target_exp),
                'action': 'SELL_TO_OPEN',
                'quantity': 1,
                'credit': str(call_credit),
                'dte': dte
            }
        ]

        # Combined greeks
        put_delta = Decimal(str(put_greeks_obj.delta if hasattr(put_greeks_obj, 'delta') else put_greeks_obj.get('delta', 0)))
        call_delta = Decimal(str(call_greeks_obj.delta if hasattr(call_greeks_obj, 'delta') else call_greeks_obj.get('delta', 0)))
        put_theta = Decimal(str(put_greeks_obj.theta if hasattr(put_greeks_obj, 'theta') else put_greeks_obj.get('theta', 0)))
        call_theta = Decimal(str(call_greeks_obj.theta if hasattr(call_greeks_obj, 'theta') else call_greeks_obj.get('theta', 0)))
        put_vega = Decimal(str(put_greeks_obj.vega if hasattr(put_greeks_obj, 'vega') else put_greeks_obj.get('vega', 0)))
        call_vega = Decimal(str(call_greeks_obj.vega if hasattr(call_greeks_obj, 'vega') else call_greeks_obj.get('vega', 0)))
        put_gamma = Decimal(str(put_greeks_obj.gamma if hasattr(put_greeks_obj, 'gamma') else put_greeks_obj.get('gamma', 0)))
        call_gamma = Decimal(str(call_greeks_obj.gamma if hasattr(call_greeks_obj, 'gamma') else call_greeks_obj.get('gamma', 0)))

        greeks = {
            'delta': float(put_delta + call_delta),
            'theta': float(put_theta + call_theta),
            'vega': float(put_vega + call_vega),
            'gamma': float(put_gamma + call_gamma),
            'pop': float(combined_prob)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.SHORT_STRADDLE,
            underlying=symbol,
            legs=legs,
            net_credit=total_credit,
            max_loss=estimated_max_loss,
            probability_of_profit=combined_prob,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            short_strike=atm_strike
        )
        candidate.calculate_kelly_score()

        return candidate

    def build_calendar_spread_candidate(
        self,
        symbol: str,
        options: List,  # All options including multiple expirations
        greeks_by_strike: dict,
        quotes_by_strike: dict,
        iv_rank: Decimal,
        front_exp: date,
        back_exp: date,
        option_type: str = 'PUT',
        target_delta: Decimal = Decimal("-0.30")
    ) -> Optional[TradeCandidate]:
        """
        Build a calendar spread candidate (time spread).

        Calendar spread = Long back-month + short front-month (same strike)
        Profits from time decay differential and vega expansion.

        Best for: Low IV rank (<40), expecting IV expansion
        """
        if (back_exp - front_exp).days < 20:
            return None  # Need sufficient time spread

        # Filter options by type and expiration
        front_options = [
            opt for opt in options
            if (opt.option_type if hasattr(opt, 'option_type') else opt.get('option_type')) == option_type[0]
            and (opt.expiration_date if hasattr(opt, 'expiration_date') else date.fromisoformat(opt.get('expiration_date'))) == front_exp
        ]

        back_options = [
            opt for opt in options
            if (opt.option_type if hasattr(opt, 'option_type') else opt.get('option_type')) == option_type[0]
            and (opt.expiration_date if hasattr(opt, 'expiration_date') else date.fromisoformat(opt.get('expiration_date'))) == back_exp
        ]

        if not front_options or not back_options:
            return None

        # Find strike at target delta in front month
        target_strike = None
        best_delta_diff = Decimal("1.0")

        for opt in front_options:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            key = (strike, option_type[0]) if isinstance(list(greeks_by_strike.keys())[0], tuple) else strike

            if key not in greeks_by_strike:
                continue

            g = greeks_by_strike[key]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                target_strike = strike

        if target_strike is None:
            return None

        # Find front and back month options at target strike
        front_opt = None
        back_opt = None

        for opt in front_options:
            opt_strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if opt_strike == target_strike:
                front_opt = opt
                break

        for opt in back_options:
            opt_strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if opt_strike == target_strike:
                back_opt = opt
                break

        if front_opt is None or back_opt is None:
            return None

        # Get quotes (need to handle tuple keys for quotes too)
        front_key = (target_strike, option_type[0]) if isinstance(list(quotes_by_strike.keys())[0], tuple) else target_strike
        back_key = front_key  # Same structure

        if front_key not in quotes_by_strike or back_key not in quotes_by_strike:
            return None

        front_bid, front_ask = quotes_by_strike[front_key]
        back_bid, back_ask = quotes_by_strike[back_key]

        front_credit = (front_bid + front_ask) / 2
        back_debit = (back_bid + back_ask) / 2

        # Calendar is a debit spread
        net_debit = back_debit - front_credit

        if net_debit <= 0 or net_debit > Decimal("2.00"):  # Unrealistic calendar
            return None

        # Max loss is net debit paid
        max_loss = net_debit * 100

        if max_loss > self.risk_params.max_position_loss:
            return None

        front_dte = (front_exp - date.today()).days
        back_dte = (back_exp - date.today()).days

        # Calendar probability estimation (typically 60-70%)
        prob_profit = Decimal("0.65")

        # Build legs
        front_symbol = front_opt.symbol if hasattr(front_opt, 'symbol') else front_opt.get('symbol', '')
        front_streamer = front_opt.streamer_symbol if hasattr(front_opt, 'streamer_symbol') else front_opt.get('streamer_symbol', '')
        back_symbol = back_opt.symbol if hasattr(back_opt, 'symbol') else back_opt.get('symbol', '')
        back_streamer = back_opt.streamer_symbol if hasattr(back_opt, 'streamer_symbol') else back_opt.get('streamer_symbol', '')

        legs = [
            {
                'symbol': front_symbol,
                'streamer_symbol': front_streamer,
                'option_type': option_type,
                'strike': str(target_strike),
                'expiration': str(front_exp),
                'action': 'SELL_TO_OPEN',
                'quantity': 1,
                'credit': str(front_credit),
                'dte': front_dte
            },
            {
                'symbol': back_symbol,
                'streamer_symbol': back_streamer,
                'option_type': option_type,
                'strike': str(target_strike),
                'expiration': str(back_exp),
                'action': 'BUY_TO_OPEN',
                'quantity': 1,
                'debit': str(back_debit),
                'dte': back_dte
            }
        ]

        # Greeks (positive vega from long back month)
        greeks = {
            'delta': 0.0,  # Roughly delta neutral
            'theta': 5.0,  # Positive theta from short front month
            'vega': 10.0,  # Positive vega from long back month
            'gamma': 0.0,
            'pop': float(prob_profit)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.CALENDAR_SPREAD,
            underlying=symbol,
            legs=legs,
            net_credit=Decimal("0"),  # It's a debit, but we track differently
            max_loss=max_loss,
            probability_of_profit=prob_profit,
            iv_rank=iv_rank,
            dte=front_dte,
            greeks=greeks,
            short_strike=target_strike,
            long_strike=target_strike
        )

        # Adjust Kelly score for debit spreads
        candidate.score = prob_profit - ((Decimal("1") - prob_profit) / (net_debit / max_loss)) if max_loss > 0 else Decimal("-999")

        return candidate

    def build_jade_lizard_candidate(
        self,
        symbol: str,
        puts: List,
        calls: List,
        put_greeks: dict,
        call_greeks: dict,
        put_quotes: dict,
        call_quotes: dict,
        iv_rank: Decimal,
        target_exp: date
    ) -> Optional[TradeCandidate]:
        """
        Build a jade lizard candidate.

        Jade Lizard = Short call spread + short put
        - No upside risk (credit > call spread width)
        - Undefined downside risk
        - Bullish bias

        Best for: High IV, bullish outlook
        """
        # Build call spread (use 30 delta for short call)
        call_spread = self.build_call_spread_candidate(
            symbol, calls, call_greeks, call_quotes, iv_rank, target_exp,
            target_delta=Decimal("0.30")
        )

        if call_spread is None:
            return None

        # Find short put (16 delta for put side)
        short_put = None
        short_put_greeks = None
        best_delta_diff = Decimal("1.0")
        target_delta = Decimal("-0.16")

        for opt in puts:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in put_greeks:
                continue
            g = put_greeks[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                short_put = opt
                short_put_greeks = g

        if short_put is None:
            return None

        put_strike = short_put.strike_price if hasattr(short_put, 'strike_price') else Decimal(str(short_put.get('strike_price', 0)))

        if put_strike not in put_quotes:
            return None

        put_bid, put_ask = put_quotes[put_strike]
        put_credit = (put_bid + put_ask) / 2

        if put_credit <= 0:
            return None

        # Total credit
        total_credit = call_spread.net_credit + put_credit

        # Check that credit > call spread width (jade lizard requirement)
        call_spread_width = call_spread.spread_width if call_spread.spread_width else Decimal("0")
        if total_credit <= call_spread_width:
            return None  # Not a jade lizard without no-risk on upside

        # Max loss is on downside (undefined, estimate conservatively)
        estimated_max_loss = total_credit * 100 * 3

        dte = call_spread.dte

        # Combined probability
        put_delta_val = Decimal(str(short_put_greeks.delta if hasattr(short_put_greeks, 'delta') else short_put_greeks.get('delta', 0)))
        put_prob_otm = Decimal("1") + put_delta_val
        combined_prob = call_spread.probability_of_profit * put_prob_otm

        # Build put leg
        put_symbol = short_put.symbol if hasattr(short_put, 'symbol') else short_put.get('symbol', '')
        put_streamer = short_put.streamer_symbol if hasattr(short_put, 'streamer_symbol') else short_put.get('streamer_symbol', '')

        put_leg = {
            'symbol': put_symbol,
            'streamer_symbol': put_streamer,
            'option_type': 'PUT',
            'strike': str(put_strike),
            'expiration': str(target_exp),
            'action': 'SELL_TO_OPEN',
            'quantity': 1,
            'credit': str(put_credit),
            'dte': dte
        }

        # Combine legs
        legs = call_spread.legs + [put_leg]

        # Combined greeks
        put_theta = Decimal(str(short_put_greeks.theta if hasattr(short_put_greeks, 'theta') else short_put_greeks.get('theta', 0)))
        put_vega = Decimal(str(short_put_greeks.vega if hasattr(short_put_greeks, 'vega') else short_put_greeks.get('vega', 0)))
        put_gamma = Decimal(str(short_put_greeks.gamma if hasattr(short_put_greeks, 'gamma') else short_put_greeks.get('gamma', 0)))

        greeks = {
            'delta': call_spread.greeks['delta'] + float(put_delta_val),
            'theta': call_spread.greeks['theta'] + float(put_theta),
            'vega': call_spread.greeks['vega'] + float(put_vega),
            'gamma': call_spread.greeks['gamma'] + float(put_gamma),
            'pop': float(combined_prob)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.JADE_LIZARD,
            underlying=symbol,
            legs=legs,
            net_credit=total_credit,
            max_loss=estimated_max_loss,
            probability_of_profit=combined_prob,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            put_short_strike=put_strike,
            call_short_strike=call_spread.short_strike,
            call_long_strike=call_spread.long_strike
        )
        candidate.calculate_kelly_score()

        return candidate

    def build_long_strangle_candidate(
        self,
        symbol: str,
        puts: List,
        calls: List,
        put_greeks: dict,
        call_greeks: dict,
        put_quotes: dict,
        call_quotes: dict,
        iv_rank: Decimal,
        target_exp: date
    ) -> Optional[TradeCandidate]:
        """
        Build a long strangle candidate (volatility expansion play).

        Long strangle = Buy OTM put + Buy OTM call
        Profits from large price move in either direction.

        Best for: Very low IV rank (<20), expecting volatility expansion
        """
        # Find OTM put (20 delta)
        long_put = None
        long_put_greeks = None
        best_delta_diff = Decimal("1.0")
        target_delta = Decimal("-0.20")

        for opt in puts:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in put_greeks:
                continue
            g = put_greeks[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                long_put = opt
                long_put_greeks = g

        if long_put is None:
            return None

        # Find OTM call (20 delta)
        long_call = None
        long_call_greeks = None
        best_delta_diff = Decimal("1.0")
        target_delta = Decimal("0.20")

        for opt in calls:
            strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
            if strike not in call_greeks:
                continue
            g = call_greeks[strike]
            delta = Decimal(str(g.delta if hasattr(g, 'delta') else g.get('delta', 0)))
            delta_diff = abs(delta - target_delta)

            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                long_call = opt
                long_call_greeks = g

        if long_call is None:
            return None

        # Get strikes and quotes
        put_strike = long_put.strike_price if hasattr(long_put, 'strike_price') else Decimal(str(long_put.get('strike_price', 0)))
        call_strike = long_call.strike_price if hasattr(long_call, 'strike_price') else Decimal(str(long_call.get('strike_price', 0)))

        if put_strike not in put_quotes or call_strike not in call_quotes:
            return None

        put_bid, put_ask = put_quotes[put_strike]
        call_bid, call_ask = call_quotes[call_strike]

        put_debit = (put_bid + put_ask) / 2
        call_debit = (call_bid + call_ask) / 2
        total_debit = put_debit + call_debit

        if total_debit <= Decimal("0.20"):
            return None

        # Max loss is debit paid
        max_loss = total_debit * 100

        if max_loss > self.risk_params.max_position_loss:
            return None

        dte = (target_exp - date.today()).days

        # Long strangle probability (typically 40-50%, needs big move)
        prob_profit = Decimal("0.45")

        # Build legs
        put_symbol = long_put.symbol if hasattr(long_put, 'symbol') else long_put.get('symbol', '')
        put_streamer = long_put.streamer_symbol if hasattr(long_put, 'streamer_symbol') else long_put.get('streamer_symbol', '')
        call_symbol = long_call.symbol if hasattr(long_call, 'symbol') else long_call.get('symbol', '')
        call_streamer = long_call.streamer_symbol if hasattr(long_call, 'streamer_symbol') else long_call.get('streamer_symbol', '')

        legs = [
            {
                'symbol': put_symbol,
                'streamer_symbol': put_streamer,
                'option_type': 'PUT',
                'strike': str(put_strike),
                'expiration': str(target_exp),
                'action': 'BUY_TO_OPEN',
                'quantity': 1,
                'debit': str(put_debit),
                'dte': dte
            },
            {
                'symbol': call_symbol,
                'streamer_symbol': call_streamer,
                'option_type': 'CALL',
                'strike': str(call_strike),
                'expiration': str(target_exp),
                'action': 'BUY_TO_OPEN',
                'quantity': 1,
                'debit': str(call_debit),
                'dte': dte
            }
        ]

        # Combined greeks (long options have positive vega, negative theta)
        put_delta = Decimal(str(long_put_greeks.delta if hasattr(long_put_greeks, 'delta') else long_put_greeks.get('delta', 0)))
        call_delta = Decimal(str(long_call_greeks.delta if hasattr(long_call_greeks, 'delta') else long_call_greeks.get('delta', 0)))
        put_theta = Decimal(str(long_put_greeks.theta if hasattr(long_put_greeks, 'theta') else long_put_greeks.get('theta', 0)))
        call_theta = Decimal(str(long_call_greeks.theta if hasattr(long_call_greeks, 'theta') else long_call_greeks.get('theta', 0)))
        put_vega = Decimal(str(long_put_greeks.vega if hasattr(long_put_greeks, 'vega') else long_put_greeks.get('vega', 0)))
        call_vega = Decimal(str(long_call_greeks.vega if hasattr(long_call_greeks, 'vega') else long_call_greeks.get('vega', 0)))
        put_gamma = Decimal(str(long_put_greeks.gamma if hasattr(long_put_greeks, 'gamma') else long_put_greeks.get('gamma', 0)))
        call_gamma = Decimal(str(long_call_greeks.gamma if hasattr(long_call_greeks, 'gamma') else long_call_greeks.get('gamma', 0)))

        greeks = {
            'delta': float(put_delta + call_delta),
            'theta': float(put_theta + call_theta),
            'vega': float(put_vega + call_vega),
            'gamma': float(put_gamma + call_gamma),
            'pop': float(prob_profit)
        }

        candidate = TradeCandidate(
            strategy=StrategyType.LONG_STRANGLE,
            underlying=symbol,
            legs=legs,
            net_credit=Decimal("0"),  # It's a debit
            max_loss=max_loss,
            probability_of_profit=prob_profit,
            iv_rank=iv_rank,
            dte=dte,
            greeks=greeks,
            put_short_strike=put_strike,  # Using these fields for put/call strikes
            call_short_strike=call_strike
        )

        # Adjust score for debit spread
        candidate.score = prob_profit - ((Decimal("1") - prob_profit) / (total_debit / max_loss)) if max_loss > 0 else Decimal("-999")

        return candidate

    def _create_spread_legs(
        self,
        short_option,
        long_strike: Decimal,
        short_credit: Decimal,
        net_credit: Decimal,
        target_exp: date,
        dte: int,
        option_type: str,
        greeks_by_strike: dict,
        quotes_by_strike: dict,
        options: List = None
    ) -> List[dict]:
        """Helper to create legs for a vertical spread."""
        short_strike = short_option.strike_price if hasattr(short_option, 'strike_price') else Decimal(str(short_option.get('strike_price', 0)))

        # Get option symbols
        short_symbol = short_option.symbol if hasattr(short_option, 'symbol') else short_option.get('symbol', '')
        short_streamer = short_option.streamer_symbol if hasattr(short_option, 'streamer_symbol') else short_option.get('streamer_symbol', '')

        # Find the long option to get its symbol
        long_symbol = ''
        long_streamer = ''
        if options:
            for opt in options:
                opt_strike = opt.strike_price if hasattr(opt, 'strike_price') else Decimal(str(opt.get('strike_price', 0)))
                opt_type = opt.option_type if hasattr(opt, 'option_type') else opt.get('option_type', '')
                if opt_strike == long_strike and opt_type == option_type[0]:  # 'P' or 'C'
                    long_symbol = opt.symbol if hasattr(opt, 'symbol') else opt.get('symbol', '')
                    long_streamer = opt.streamer_symbol if hasattr(opt, 'streamer_symbol') else opt.get('streamer_symbol', '')
                    break

        # Calculate long debit
        long_debit = short_credit - net_credit

        legs = [
            {
                'symbol': short_symbol,
                'streamer_symbol': short_streamer,
                'option_type': option_type,
                'strike': str(short_strike),
                'expiration': str(target_exp),
                'action': 'SELL_TO_OPEN',
                'quantity': 1,
                'credit': str(short_credit),
                'dte': dte
            },
            {
                'symbol': long_symbol,
                'streamer_symbol': long_streamer,
                'option_type': option_type,
                'strike': str(long_strike),
                'expiration': str(target_exp),
                'action': 'BUY_TO_OPEN',
                'quantity': 1,
                'debit': str(long_debit),
                'dte': dte
            }
        ]

        return legs

    def select_strategies_to_evaluate(
        self,
        market_condition: MarketCondition,
        portfolio_state: Optional[PortfolioState] = None
    ) -> List[str]:
        """
        Intelligently select which strategies to evaluate based on
        market conditions and portfolio structure.

        Args:
            market_condition: Current market conditions for the symbol
            portfolio_state: Current portfolio state for diversification

        Returns:
            List of strategy names to evaluate in priority order
        """
        strategies = []
        iv_env = market_condition.iv_environment

        # === HIGH IV ENVIRONMENT (60-100 IV Rank) ===
        # Favor premium selling strategies
        if iv_env in [IVEnvironment.HIGH, IVEnvironment.VERY_HIGH]:
            # Pre-earnings: Capitalize on IV crush
            if market_condition.is_pre_earnings:
                strategies.extend([
                    'short_strangle',  # Max premium, undefined risk
                    'short_straddle',  # ATM for max IV crush
                    'iron_condor',  # Defined risk alternative
                    'jade_lizard'  # Bullish bias with no upside risk
                ])
            else:
                # General high IV premium selling
                strategies.extend([
                    'iron_condor',  # Defined risk, neutral
                    'short_strangle',  # Higher premium, undefined risk
                    'short_put_spread',  # Bullish bias
                    'short_call_spread',  # Bearish bias
                    'jade_lizard'  # Bullish with downside risk only
                ])

        # === LOW IV ENVIRONMENT (0-40 IV Rank) ===
        # Favor debit strategies and time spreads
        elif iv_env in [IVEnvironment.VERY_LOW, IVEnvironment.LOW]:
            # Post-earnings or after volatility crush
            if market_condition.is_post_earnings or iv_env == IVEnvironment.VERY_LOW:
                strategies.extend([
                    'calendar_spread',  # Profit from IV expansion
                    'long_strangle',  # Expecting big move
                    'diagonal_spread',  # Directional with time decay
                ])
            else:
                # Moderate low IV
                strategies.extend([
                    'calendar_spread',
                    'short_put_spread',  # Still viable with lower premium
                    'short_call_spread'
                ])

        # === MODERATE IV ENVIRONMENT (40-60 IV Rank) ===
        # Mixed approach
        else:
            strategies.extend([
                'iron_condor',  # Balanced defined-risk
                'short_put_spread',  # Directional flexibility
                'short_call_spread',
                'calendar_spread',  # Time decay benefit
                'jade_lizard'  # Bullish opportunities
            ])

        # === PORTFOLIO DIVERSIFICATION ADJUSTMENTS ===
        if portfolio_state:
            # Check portfolio delta
            portfolio_delta = float(portfolio_state.portfolio_delta)

            # If portfolio is too bullish (high positive delta), favor bearish strategies
            if portfolio_delta > 200:
                # Move bearish strategies to front
                if 'short_call_spread' in strategies:
                    strategies.remove('short_call_spread')
                    strategies.insert(0, 'short_call_spread')

            # If portfolio is too bearish (high negative delta), favor bullish strategies
            elif portfolio_delta < -200:
                # Move bullish strategies to front
                if 'short_put_spread' in strategies:
                    strategies.remove('short_put_spread')
                    strategies.insert(0, 'short_put_spread')
                if 'jade_lizard' in strategies:
                    strategies.remove('jade_lizard')
                    strategies.insert(0, 'jade_lizard')

            # Check if we're approaching position limits
            if portfolio_state.open_positions >= self.risk_params.max_total_positions * 0.8:
                # Prioritize higher probability, defined-risk strategies
                priority_strategies = ['iron_condor', 'short_put_spread', 'short_call_spread']
                strategies = [s for s in priority_strategies if s in strategies] + \
                           [s for s in strategies if s not in priority_strategies]

            # Check symbol-specific concentration
            if portfolio_state.positions_by_underlying:
                symbol_count = portfolio_state.positions_by_underlying.get(market_condition.symbol, 0)
                if symbol_count >= self.risk_params.max_positions_per_underlying:
                    # Skip this symbol entirely
                    return []

        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for s in strategies:
            if s not in seen:
                seen.add(s)
                unique_strategies.append(s)

        return unique_strategies

    def evaluate_strategies_for_symbol(
        self,
        symbol: str,
        options: List,
        greeks_by_strike: dict,
        quotes_by_strike: dict,
        iv_rank: Decimal,
        target_exp: date,
        max_candidates: int = 5,
        current_price: Optional[Decimal] = None,
        portfolio_state: Optional[PortfolioState] = None,
        underlying_data: Optional[dict] = None
    ) -> List[TradeCandidate]:
        """
        Intelligently evaluate strategies for a symbol based on market conditions
        and portfolio structure.

        Args:
            symbol: Underlying symbol
            options: All options for the symbol (all expirations if evaluating calendars)
            greeks_by_strike: Combined greeks map (puts and calls)
            quotes_by_strike: Combined quotes map (puts and calls)
            iv_rank: Current IV rank
            target_exp: Target expiration
            max_candidates: Maximum candidates to return (default 5)
            current_price: Current price of underlying
            portfolio_state: Current portfolio state for diversification
            underlying_data: Additional market data (earnings, technicals, etc.)

        Returns:
            List of top TradeCandidate objects sorted by Kelly score
        """
        # Step 1: Analyze market conditions
        market_condition = self.analyze_market_conditions(
            symbol, iv_rank, current_price, underlying_data
        )

        # Step 2: Select strategies to evaluate based on conditions
        strategies_to_eval = self.select_strategies_to_evaluate(
            market_condition, portfolio_state
        )

        if not strategies_to_eval:
            logger.info(f"{symbol}: No strategies selected (may have position limit)")
            return []

        logger.info(f"{symbol} IV Rank={iv_rank:.1f} ({market_condition.iv_environment.value}): "
                   f"Evaluating {len(strategies_to_eval)} strategies")

        # Step 3: Prepare option data
        candidates = []

        # Separate puts and calls
        puts = sorted(
            [opt for opt in options if (opt.option_type if hasattr(opt, 'option_type') else opt.get('option_type')) == 'P'],
            key=lambda x: x.strike_price if hasattr(x, 'strike_price') else Decimal(str(x.get('strike_price', 0))),
            reverse=True
        )
        calls = sorted(
            [opt for opt in options if (opt.option_type if hasattr(opt, 'option_type') else opt.get('option_type')) == 'C'],
            key=lambda x: x.strike_price if hasattr(x, 'strike_price') else Decimal(str(x.get('strike_price', 0)))
        )

        # Separate greeks and quotes by type
        put_greeks = {}
        call_greeks = {}
        put_quotes = {}
        call_quotes = {}

        for key, val in greeks_by_strike.items():
            if isinstance(key, tuple):
                strike, opt_type = key
                if opt_type == 'P':
                    put_greeks[strike] = val
                else:
                    call_greeks[strike] = val
            else:
                put_greeks[key] = val
                call_greeks[key] = val

        for key, val in quotes_by_strike.items():
            if isinstance(key, tuple):
                strike, opt_type = key
                if opt_type == 'P':
                    put_quotes[strike] = val
                else:
                    call_quotes[strike] = val
            else:
                put_quotes[key] = val
                call_quotes[key] = val

        # Step 4: Build candidates for each selected strategy
        for strategy_name in strategies_to_eval:
            candidate = None

            try:
                if strategy_name == 'short_put_spread':
                    candidate = self.build_put_spread_candidate(
                        symbol, puts, put_greeks, put_quotes, iv_rank, target_exp,
                        target_delta=Decimal("-0.30")
                    )

                elif strategy_name == 'short_call_spread':
                    candidate = self.build_call_spread_candidate(
                        symbol, calls, call_greeks, call_quotes, iv_rank, target_exp,
                        target_delta=Decimal("0.30")
                    )

                elif strategy_name == 'iron_condor':
                    candidate = self.build_iron_condor_candidate(
                        symbol, puts, calls, put_greeks, call_greeks,
                        put_quotes, call_quotes, iv_rank, target_exp
                    )

                elif strategy_name == 'short_strangle':
                    candidate = self.build_short_strangle_candidate(
                        symbol, puts, calls, put_greeks, call_greeks,
                        put_quotes, call_quotes, iv_rank, target_exp
                    )

                elif strategy_name == 'short_straddle':
                    candidate = self.build_short_straddle_candidate(
                        symbol, puts, calls, put_greeks, call_greeks,
                        put_quotes, call_quotes, iv_rank, target_exp,
                        current_price=current_price
                    )

                elif strategy_name == 'jade_lizard':
                    candidate = self.build_jade_lizard_candidate(
                        symbol, puts, calls, put_greeks, call_greeks,
                        put_quotes, call_quotes, iv_rank, target_exp
                    )

                elif strategy_name == 'long_strangle':
                    candidate = self.build_long_strangle_candidate(
                        symbol, puts, calls, put_greeks, call_greeks,
                        put_quotes, call_quotes, iv_rank, target_exp
                    )

                elif strategy_name == 'calendar_spread':
                    # Calendar needs multiple expirations
                    # For now, skip if we don't have back month data
                    # TODO: Implement when scanner provides multiple expirations
                    logger.debug(f"{symbol}: Calendar spread requires multi-expiration data")
                    continue

                elif strategy_name == 'diagonal_spread':
                    # Diagonal needs multiple expirations
                    logger.debug(f"{symbol}: Diagonal spread requires multi-expiration data")
                    continue

                if candidate:
                    candidates.append(candidate)
                    logger.debug(f"{symbol} {strategy_name.upper()}: "
                               f"Score={candidate.score:.3f}, "
                               f"Credit=${candidate.net_credit:.2f}, "
                               f"MaxLoss=${candidate.max_loss:.2f}, "
                               f"POP={candidate.probability_of_profit:.2%}")

            except Exception as e:
                logger.warning(f"{symbol} {strategy_name} failed: {e}")
                continue

        # Step 5: Sort by Kelly score and return top candidates
        candidates.sort(key=lambda x: x.score, reverse=True)

        if candidates:
            logger.info(f"{symbol}: Evaluated {len(candidates)} strategies, "
                       f"best={candidates[0].strategy.value} "
                       f"(score={candidates[0].score:.3f}, "
                       f"credit=${candidates[0].net_credit:.2f})")
        else:
            logger.info(f"{symbol}: No valid candidates found")

        return candidates[:max_candidates]
