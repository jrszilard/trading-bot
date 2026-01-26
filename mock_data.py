#!/usr/bin/env python3
"""
Mock Data Provider for Tastytrade Trading Bot

Provides realistic mock market data for testing when real market data
APIs are unavailable (e.g., sandbox mode without DXLink access).

This allows testing the full bot workflow:
- Market data (IV rank, option chains, Greeks, quotes) uses mock data
- Order execution uses real Tastytrade sandbox API

Supports Brave Search API for dynamic real-time data fetching.
"""

import math
import os
import re
import json
import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging

# HTTP client for Brave Search API
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CachedData:
    """Cache entry for market data"""
    data: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minute default TTL

    def is_expired(self) -> bool:
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds


class BraveSearchClient:
    """
    Client for Brave Search API to fetch real-time market data.

    Uses Brave Search to get current stock prices, IV rank estimates,
    and other market information for more realistic mock data.
    """

    BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 300):
        """
        Initialize Brave Search client.

        Args:
            api_key: Brave Search API key (or set BRAVE_API_KEY env var)
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self.api_key = api_key or os.environ.get('BRAVE_API_KEY')
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, CachedData] = {}
        self._client: Optional['httpx.AsyncClient'] = None

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available - Brave Search disabled. Install with: pip install httpx")
        elif not self.api_key:
            logger.info("No BRAVE_API_KEY found - using static mock data only")

    @property
    def is_available(self) -> bool:
        """Check if Brave Search is available"""
        return HTTPX_AVAILABLE and bool(self.api_key)

    async def _get_client(self) -> 'httpx.AsyncClient':
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Perform a Brave Search query.

        Args:
            query: The search query

        Returns:
            Search results dict or None if failed
        """
        if not self.is_available:
            return None

        # Check cache
        cache_key = query.lower()
        if cache_key in self._cache and not self._cache[cache_key].is_expired():
            logger.debug(f"Cache hit for query: {query}")
            return self._cache[cache_key].data

        try:
            client = await self._get_client()
            response = await client.get(
                self.BRAVE_API_URL,
                params={
                    "q": query,
                    "count": 5,
                    "safesearch": "off",
                    "freshness": "pd"  # Past day for fresh data
                },
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key
                }
            )

            if response.status_code == 200:
                data = response.json()
                self._cache[cache_key] = CachedData(
                    data=data,
                    timestamp=datetime.now(),
                    ttl_seconds=self.cache_ttl
                )
                return data
            else:
                logger.warning(f"Brave Search API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Brave Search request failed: {e}")
            return None

    async def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get current stock price via Brave Search.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None if not found
        """
        query = f"{symbol} stock price today"
        results = await self.search(query)

        if not results:
            return None

        # Try to extract price from search results
        price = self._extract_price_from_results(results, symbol)
        if price:
            logger.info(f"Brave Search: {symbol} price = ${price}")
        return price

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market data for a symbol via Brave Search.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with price, iv_rank estimate, and other data
        """
        data = {
            "symbol": symbol,
            "price": None,
            "iv_rank": None,
            "change_pct": None,
            "source": "brave_search"
        }

        # Get price
        price = await self.get_stock_price(symbol)
        if price:
            data["price"] = price

        # Get IV rank (search for implied volatility info)
        iv_data = await self._get_iv_data(symbol)
        if iv_data:
            data.update(iv_data)

        return data

    async def _get_iv_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get implied volatility data from search"""
        query = f"{symbol} implied volatility IV rank options"
        results = await self.search(query)

        if not results:
            return None

        iv_rank = self._extract_iv_from_results(results, symbol)
        if iv_rank is not None:
            logger.info(f"Brave Search: {symbol} IV rank estimate = {iv_rank}")
            return {"iv_rank": iv_rank}

        return None

    def _extract_price_from_results(
        self,
        results: Dict[str, Any],
        symbol: str
    ) -> Optional[float]:
        """Extract stock price from Brave Search results"""
        # Check for infobox (often contains stock data)
        if "infobox" in results:
            infobox = results["infobox"]
            # Look for price in infobox data
            if "results" in infobox:
                for item in infobox["results"]:
                    if "stock" in str(item).lower() or "price" in str(item).lower():
                        price = self._parse_price(str(item))
                        if price:
                            return price

        # Search through web results
        web_results = results.get("web", {}).get("results", [])
        for result in web_results:
            title = result.get("title", "")
            description = result.get("description", "")
            text = f"{title} {description}"

            # Look for price patterns
            price = self._parse_price(text)
            if price and 1 < price < 10000:  # Sanity check
                return price

        return None

    def _extract_iv_from_results(
        self,
        results: Dict[str, Any],
        symbol: str
    ) -> Optional[float]:
        """Extract IV rank from Brave Search results"""
        web_results = results.get("web", {}).get("results", [])

        for result in web_results:
            title = result.get("title", "")
            description = result.get("description", "")
            text = f"{title} {description}".lower()

            # Look for IV rank patterns
            iv_patterns = [
                r'iv\s*rank[:\s]+(\d+(?:\.\d+)?)\s*%?',
                r'iv\s*percentile[:\s]+(\d+(?:\.\d+)?)\s*%?',
                r'implied\s*volatility\s*rank[:\s]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*%?\s*iv\s*rank',
            ]

            for pattern in iv_patterns:
                match = re.search(pattern, text)
                if match:
                    iv_rank = float(match.group(1))
                    if 0 <= iv_rank <= 100:
                        return iv_rank

        return None

    def _parse_price(self, text: str) -> Optional[float]:
        """Parse a price from text"""
        # Common price patterns
        patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $123.45 or $1,234.56
            r'(?:price|trading at|at)\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*(?:usd|dollars)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue

        return None


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

        # Initialize Brave Search client for dynamic data
        brave_config = mock_config.get('brave_search', {})
        self.use_brave_search = brave_config.get('enabled', True)
        cache_ttl = brave_config.get('cache_ttl_seconds', 300)

        self.brave_client = BraveSearchClient(
            api_key=brave_config.get('api_key'),
            cache_ttl=cache_ttl
        )

        # Cache for dynamically fetched data
        self._dynamic_cache: Dict[str, CachedData] = {}

        mode = "Brave Search enabled" if self.brave_client.is_available else "static data only"
        logger.info(f"MockDataProvider initialized with {len(self.symbols)} symbols ({mode})")

    def get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol data, using cached dynamic data if available.

        This is a synchronous method that returns cached data.
        Use refresh_symbol_data() to fetch fresh data from Brave Search.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with price, iv_rank, beta, and option_strike_interval
        """
        # Check dynamic cache first
        if symbol in self._dynamic_cache and not self._dynamic_cache[symbol].is_expired():
            return self._dynamic_cache[symbol].data

        # Fall back to static data
        static_data = self.symbols.get(symbol, {})
        return {
            'symbol': symbol,
            'price': static_data.get('price', 100.0),
            'iv_rank': static_data.get('iv_rank', self.default_iv_rank),
            'beta': static_data.get('beta', 1.0),
            'option_strike_interval': static_data.get('option_strike_interval', 1.0),
            'source': 'static'
        }

    async def refresh_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fresh market data for a symbol using Brave Search.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with price, iv_rank, and other market data
        """
        # Get static defaults
        static_data = self.symbols.get(symbol, {})
        result = {
            'symbol': symbol,
            'price': static_data.get('price', 100.0),
            'iv_rank': static_data.get('iv_rank', self.default_iv_rank),
            'beta': static_data.get('beta', 1.0),
            'option_strike_interval': static_data.get('option_strike_interval', 1.0),
            'source': 'static'
        }

        # Try Brave Search if available
        if self.use_brave_search and self.brave_client.is_available:
            try:
                brave_data = await self.brave_client.get_market_data(symbol)

                if brave_data.get('price'):
                    result['price'] = brave_data['price']
                    result['source'] = 'brave_search'

                    # Update strike interval based on price
                    result['option_strike_interval'] = self._calculate_strike_interval(brave_data['price'])

                if brave_data.get('iv_rank') is not None:
                    result['iv_rank'] = brave_data['iv_rank']

                logger.info(f"Refreshed {symbol} data from Brave Search: price=${result['price']}, IV={result['iv_rank']}")

            except Exception as e:
                logger.warning(f"Brave Search failed for {symbol}, using static data: {e}")

        # Cache the result
        self._dynamic_cache[symbol] = CachedData(
            data=result,
            timestamp=datetime.now(),
            ttl_seconds=300
        )

        return result

    async def refresh_all_symbols(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Refresh data for multiple symbols concurrently.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbols to their market data
        """
        tasks = [self.refresh_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to refresh {symbol}: {result}")
                data[symbol] = self.get_symbol_data(symbol)
            else:
                data[symbol] = result

        return data

    def _calculate_strike_interval(self, price: float) -> float:
        """Calculate appropriate option strike interval based on price"""
        if price < 25:
            return 0.5
        elif price < 50:
            return 1.0
        elif price < 200:
            return 2.5
        elif price < 500:
            return 5.0
        else:
            return 10.0

    async def close(self):
        """Close any open connections"""
        await self.brave_client.close()

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
            # Use get_symbol_data which checks dynamic cache first
            data = self.get_symbol_data(symbol)
            metric = MockMetric(
                symbol=symbol,
                implied_volatility_index_rank=data.get('iv_rank', self.default_iv_rank),
                beta=data.get('beta', 1.0)
            )
            metrics.append(metric)
            source = data.get('source', 'static')
            logger.debug(f"Mock metric for {symbol}: IV Rank={metric.implied_volatility_index_rank}, Beta={metric.beta} [{source}]")

        return metrics

    async def get_market_metrics_async(self, symbols: List[str]) -> List[MockMetric]:
        """
        Get market metrics for symbols, refreshing from Brave Search if available.

        Args:
            symbols: List of ticker symbols

        Returns:
            List of MockMetric objects with IV rank and beta
        """
        # Refresh data from Brave Search
        if self.brave_client.is_available:
            await self.refresh_all_symbols(symbols)

        # Return the metrics (now with fresh data in cache)
        return self.get_market_metrics(symbols)

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
        data = self.get_symbol_data(symbol)
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
        data = self.get_symbol_data(symbol)
        return data.get('price', 100.0)

    async def get_underlying_price_async(self, symbol: str) -> float:
        """Get the price for an underlying symbol, refreshing from Brave Search."""
        if self.brave_client.is_available:
            await self.refresh_symbol_data(symbol)
        return self.get_underlying_price(symbol)

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
