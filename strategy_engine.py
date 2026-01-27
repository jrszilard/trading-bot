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

from models import StrategyType, TradeCandidate, RiskParameters

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

    def evaluate_strategies_for_symbol(
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
        Evaluate all defined-risk strategies for a symbol and return top candidates.

        Args:
            symbol: Underlying symbol
            options: All options for the symbol at target expiration
            greeks_by_strike: Combined greeks map (puts and calls)
            quotes_by_strike: Combined quotes map (puts and calls)
            iv_rank: Current IV rank
            target_exp: Target expiration
            max_candidates: Maximum candidates to return (default 3)

        Returns:
            List of top TradeCandidate objects sorted by Kelly score
        """
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
        # Handle both old format (strike only) and new format ((strike, opt_type) tuple)
        put_greeks = {}
        call_greeks = {}
        put_quotes = {}
        call_quotes = {}

        for key, val in greeks_by_strike.items():
            if isinstance(key, tuple):
                # New format: (strike, option_type)
                strike, opt_type = key
                if opt_type == 'P':
                    put_greeks[strike] = val
                else:
                    call_greeks[strike] = val
            else:
                # Old format: just strike - add to both (legacy compatibility)
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

        # Strategy 1: Put Credit Spread (30 delta - bullish)
        put_spread = self.build_put_spread_candidate(
            symbol, puts, put_greeks, put_quotes, iv_rank, target_exp,
            target_delta=Decimal("-0.30")
        )
        if put_spread:
            candidates.append(put_spread)
            logger.debug(f"{symbol} PUT SPREAD: {put_spread.short_strike}P/{put_spread.long_strike}P "
                        f"Score={put_spread.score:.3f}, Credit=${put_spread.net_credit:.2f}")

        # Strategy 2: Call Credit Spread (30 delta - bearish)
        call_spread = self.build_call_spread_candidate(
            symbol, calls, call_greeks, call_quotes, iv_rank, target_exp,
            target_delta=Decimal("0.30")
        )
        if call_spread:
            candidates.append(call_spread)
            logger.debug(f"{symbol} CALL SPREAD: {call_spread.short_strike}C/{call_spread.long_strike}C "
                        f"Score={call_spread.score:.3f}, Credit=${call_spread.net_credit:.2f}")

        # Strategy 3: Iron Condor (16 delta wings - neutral)
        iron_condor = self.build_iron_condor_candidate(
            symbol, puts, calls, put_greeks, call_greeks, put_quotes, call_quotes,
            iv_rank, target_exp
        )
        if iron_condor:
            candidates.append(iron_condor)
            logger.debug(f"{symbol} IRON CONDOR: {iron_condor.put_short_strike}P/{iron_condor.put_long_strike}P "
                        f"+ {iron_condor.call_short_strike}C/{iron_condor.call_long_strike}C "
                        f"Score={iron_condor.score:.3f}, Credit=${iron_condor.net_credit:.2f}")

        # Sort by Kelly score (descending) and return top N
        candidates.sort(key=lambda x: x.score, reverse=True)

        if candidates:
            logger.info(f"{symbol}: Evaluated {len(candidates)} strategies, "
                       f"best={candidates[0].strategy.value} (score={candidates[0].score:.3f})")

        return candidates[:max_candidates]
