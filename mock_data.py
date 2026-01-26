#!/usr/bin/env python3
"""
Mock Data Provider for Tastytrade Trading Bot

Provides realistic mock market data for testing when real market data
APIs are unavailable (e.g., sandbox mode without DXLink access).

This allows testing the full bot workflow:
- Market data (IV rank, option chains, Greeks, quotes) uses mock data
- Order execution uses real Tastytrade sandbox API
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class MockMetric:
    """Mock market metrics for a symbol"""
    symbol: str
    implied_volatility_index_rank: float  # IV Rank 0-100
    beta: float  # Beta relative to SPY


@dataclass
class MockOption:
    """Mock option contract"""
    symbol: str  # e.g., "SPY   250131P00480000"
    streamer_symbol: str  # e.g., ".SPY250131P480"
    strike_price: Decimal
    expiration_date: date
    option_type: str  # 'P' or 'C'
    underlying_symbol: str


@dataclass
class MockGreeks:
    """Mock Greeks for an option"""
    event_symbol: str
    delta: float  # -1 to 1
    theta: float  # Daily decay (negative)
    vega: float   # Sensitivity to IV
    gamma: float  # Rate of delta change


@dataclass
class MockQuote:
    """Mock quote for a symbol"""
    event_symbol: str
    bid_price: float
    ask_price: float


class MockDataProvider:
    """
    Provides realistic mock market data for testing.

    All data is configurable via the config parameter.
    Default values are provided for common symbols.
    """

    # Default mock data for common symbols
    DEFAULT_SYMBOLS = {
        'SPY': {'price': 480.0, 'iv_rank': 35, 'beta': 1.0, 'option_strike_interval': 1.0},
        'QQQ': {'price': 410.0, 'iv_rank': 42, 'beta': 1.2, 'option_strike_interval': 1.0},
        'IWM': {'price': 200.0, 'iv_rank': 38, 'beta': 1.3, 'option_strike_interval': 1.0},
        'DIA': {'price': 380.0, 'iv_rank': 32, 'beta': 0.9, 'option_strike_interval': 1.0},
        'AAPL': {'price': 185.0, 'iv_rank': 28, 'beta': 1.1, 'option_strike_interval': 2.5},
        'MSFT': {'price': 420.0, 'iv_rank': 30, 'beta': 1.0, 'option_strike_interval': 2.5},
        'GOOGL': {'price': 165.0, 'iv_rank': 33, 'beta': 1.1, 'option_strike_interval': 2.5},
        'AMZN': {'price': 195.0, 'iv_rank': 36, 'beta': 1.2, 'option_strike_interval': 2.5},
        'META': {'price': 550.0, 'iv_rank': 45, 'beta': 1.3, 'option_strike_interval': 5.0},
        'NVDA': {'price': 140.0, 'iv_rank': 52, 'beta': 1.6, 'option_strike_interval': 2.5},
        'TSLA': {'price': 250.0, 'iv_rank': 55, 'beta': 1.8, 'option_strike_interval': 5.0},
        'AMD': {'price': 120.0, 'iv_rank': 48, 'beta': 1.5, 'option_strike_interval': 1.0},
        'XLF': {'price': 45.0, 'iv_rank': 28, 'beta': 1.1, 'option_strike_interval': 0.5},
        'XLE': {'price': 90.0, 'iv_rank': 35, 'beta': 1.2, 'option_strike_interval': 1.0},
        'XLK': {'price': 210.0, 'iv_rank': 32, 'beta': 1.1, 'option_strike_interval': 1.0},
        'XLV': {'price': 140.0, 'iv_rank': 25, 'beta': 0.8, 'option_strike_interval': 1.0},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock data provider.

        Args:
            config: Configuration dict with 'mock_data' section from config.json
        """
        self.config = config or {}
        mock_config = self.config.get('mock_data', {})

        # Merge default symbols with configured symbols
        self.symbols = dict(self.DEFAULT_SYMBOLS)
        configured_symbols = mock_config.get('symbols', {})
        self.symbols.update(configured_symbols)

        # Greeks model parameters
        greeks_config = mock_config.get('greeks_model', {})
        self.risk_free_rate = greeks_config.get('risk_free_rate', 0.05)
        self.default_iv = greeks_config.get('default_iv', 0.25)

        # Quote parameters
        self.default_bid_ask_spread_pct = mock_config.get('default_bid_ask_spread_pct', 0.03)
        self.default_iv_rank = mock_config.get('default_iv_rank', 35)

        logger.info(f"MockDataProvider initialized with {len(self.symbols)} symbols")

    def get_market_metrics(self, symbols: List[str]) -> List[MockMetric]:
        """
        Get mock market metrics for symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            List of MockMetric objects with IV rank and beta
        """
        metrics = []
        for symbol in symbols:
            data = self.symbols.get(symbol, {})
            metric = MockMetric(
                symbol=symbol,
                implied_volatility_index_rank=data.get('iv_rank', self.default_iv_rank),
                beta=data.get('beta', 1.0)
            )
            metrics.append(metric)
            logger.debug(f"Mock metric for {symbol}: IV Rank={metric.implied_volatility_index_rank}, Beta={metric.beta}")

        return metrics

    def get_option_chain(
        self,
        symbol: str,
        target_dte: int = 45
    ) -> Dict[date, List[MockOption]]:
        """
        Generate mock option chain for a symbol.

        Args:
            symbol: Underlying ticker symbol
            target_dte: Target days to expiration (finds nearest monthly)

        Returns:
            Dict mapping expiration dates to lists of MockOption objects
        """
        data = self.symbols.get(symbol, {})
        price = data.get('price', 100.0)
        strike_interval = data.get('option_strike_interval', 1.0)

        # Find the next monthly expiration near target DTE
        today = date.today()
        target_date = today + timedelta(days=target_dte)

        # Find the third Friday of the target month (monthly options)
        expiration = self._get_third_friday(target_date.year, target_date.month)

        # If expiration is in the past, use next month
        if expiration <= today:
            if target_date.month == 12:
                expiration = self._get_third_friday(target_date.year + 1, 1)
            else:
                expiration = self._get_third_friday(target_date.year, target_date.month + 1)

        # Generate strikes: -10% to +10% of current price
        min_strike = price * 0.90
        max_strike = price * 1.10

        options = []
        strike = Decimal(str(min_strike)).quantize(Decimal(str(strike_interval)))

        while float(strike) <= max_strike:
            # Generate both put and call at each strike
            for opt_type in ['P', 'C']:
                option = self._create_mock_option(
                    symbol=symbol,
                    strike=strike,
                    expiration=expiration,
                    option_type=opt_type
                )
                options.append(option)

            strike += Decimal(str(strike_interval))

        logger.debug(f"Generated {len(options)} mock options for {symbol} at {expiration}")

        return {expiration: options}

    def get_greeks(
        self,
        option: MockOption,
        underlying_price: Optional[float] = None
    ) -> MockGreeks:
        """
        Calculate mock Greeks using simplified Black-Scholes approximations.

        Args:
            option: The option to calculate Greeks for
            underlying_price: Current price of underlying (uses default if not provided)

        Returns:
            MockGreeks object with delta, theta, vega, gamma
        """
        symbol_data = self.symbols.get(option.underlying_symbol, {})
        price = underlying_price or symbol_data.get('price', 100.0)
        iv_rank = symbol_data.get('iv_rank', self.default_iv_rank)

        # Convert IV rank to actual IV (rough approximation)
        # Higher IV rank = higher actual IV
        iv = self.default_iv * (1 + (iv_rank - 50) / 100)
        iv = max(0.10, min(1.0, iv))  # Clamp between 10% and 100%

        strike = float(option.strike_price)
        dte = (option.expiration_date - date.today()).days
        t = max(dte / 365.0, 0.001)  # Time in years

        # Simplified Black-Scholes approximations
        d1 = self._calculate_d1(price, strike, t, self.risk_free_rate, iv)
        d2 = d1 - iv * math.sqrt(t)

        # Delta
        if option.option_type == 'C':
            delta = self._norm_cdf(d1)
        else:  # Put
            delta = self._norm_cdf(d1) - 1

        # Gamma (same for puts and calls)
        gamma = self._norm_pdf(d1) / (price * iv * math.sqrt(t))

        # Theta (daily decay, negative for long options)
        # Simplified: more decay as expiration approaches
        theta_annual = -(price * self._norm_pdf(d1) * iv) / (2 * math.sqrt(t))
        theta = theta_annual / 365.0

        # Vega (sensitivity to 1% change in IV)
        vega = price * math.sqrt(t) * self._norm_pdf(d1) / 100.0

        return MockGreeks(
            event_symbol=option.streamer_symbol,
            delta=round(delta, 4),
            theta=round(theta, 4),
            vega=round(vega, 4),
            gamma=round(gamma, 6)
        )

    def get_quote(
        self,
        symbol: str,
        is_option: bool = False,
        option: Optional[MockOption] = None
    ) -> MockQuote:
        """
        Generate mock quote with bid/ask spread.

        Args:
            symbol: The symbol to get quote for
            is_option: Whether this is an option quote
            option: MockOption object (required if is_option=True)

        Returns:
            MockQuote with bid and ask prices
        """
        if is_option and option:
            # Calculate theoretical option price
            greeks = self.get_greeks(option)
            symbol_data = self.symbols.get(option.underlying_symbol, {})
            price = symbol_data.get('price', 100.0)
            strike = float(option.strike_price)

            # Very rough option price approximation using delta
            # For ATM options, price is roughly 0.4 * price * IV * sqrt(t)
            dte = (option.expiration_date - date.today()).days
            t = max(dte / 365.0, 0.001)
            iv_rank = symbol_data.get('iv_rank', self.default_iv_rank)
            iv = self.default_iv * (1 + (iv_rank - 50) / 100)

            # Intrinsic value
            if option.option_type == 'C':
                intrinsic = max(0, price - strike)
            else:
                intrinsic = max(0, strike - price)

            # Time value approximation
            atm_time_value = price * iv * math.sqrt(t) * 0.4
            delta_factor = 1 - abs(abs(greeks.delta) - 0.5) * 2  # Peaks at 0.5 delta
            time_value = atm_time_value * delta_factor

            mid_price = intrinsic + time_value
            mid_price = max(0.05, mid_price)  # Minimum price

            # Wider spread for options (3-5% of mid price)
            spread_pct = self.default_bid_ask_spread_pct
            half_spread = mid_price * spread_pct / 2

            return MockQuote(
                event_symbol=option.streamer_symbol,
                bid_price=round(mid_price - half_spread, 2),
                ask_price=round(mid_price + half_spread, 2)
            )
        else:
            # Equity quote
            data = self.symbols.get(symbol, {})
            price = data.get('price', 100.0)

            # Tight spread for equities (0.01-0.02%)
            spread = max(0.01, price * 0.0002)

            return MockQuote(
                event_symbol=symbol,
                bid_price=round(price - spread / 2, 2),
                ask_price=round(price + spread / 2, 2)
            )

    def get_underlying_price(self, symbol: str) -> float:
        """Get the mock price for an underlying symbol."""
        data = self.symbols.get(symbol, {})
        return data.get('price', 100.0)

    def _create_mock_option(
        self,
        symbol: str,
        strike: Decimal,
        expiration: date,
        option_type: str
    ) -> MockOption:
        """Create a mock option with proper symbol format."""
        # Format: SPY   250131P00480000 (OCC format)
        exp_str = expiration.strftime('%y%m%d')
        strike_str = f"{int(strike * 1000):08d}"
        occ_symbol = f"{symbol:<6}{exp_str}{option_type}{strike_str}"

        # Streamer symbol format: .SPY250131P480
        streamer_symbol = f".{symbol}{exp_str}{option_type}{int(strike)}"

        return MockOption(
            symbol=occ_symbol,
            streamer_symbol=streamer_symbol,
            strike_price=strike,
            expiration_date=expiration,
            option_type=option_type,
            underlying_symbol=symbol
        )

    def _get_third_friday(self, year: int, month: int) -> date:
        """Get the third Friday of a given month (monthly options expiration)."""
        # Find the first day of the month
        first_day = date(year, month, 1)

        # Find the first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)

        return third_friday

    def _calculate_d1(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        sigma: float
    ) -> float:
        """Calculate d1 for Black-Scholes formula."""
        if sigma <= 0 or t <= 0:
            return 0
        return (math.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))

    def _norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def _norm_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return math.exp(-x ** 2 / 2.0) / math.sqrt(2.0 * math.pi)


# Convenience function for integration
def create_mock_provider(config: Dict[str, Any]) -> Optional[MockDataProvider]:
    """
    Create a MockDataProvider if mock data is enabled in config.

    Args:
        config: Full configuration dict

    Returns:
        MockDataProvider instance if enabled, None otherwise
    """
    mock_config = config.get('mock_data', {})
    if mock_config.get('enabled', False):
        return MockDataProvider(config)
    return None
