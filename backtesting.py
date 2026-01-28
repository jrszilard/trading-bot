"""
Backtesting System for Tastytrade Trading Bot

This module implements a comprehensive backtesting framework to:
1. Simulate historical trading performance
2. Generate synthetic historical data for sandbox testing
3. Calculate performance metrics (win rate, Sharpe ratio, max drawdown, etc.)
4. Analyze strategy performance across different market conditions
5. Identify optimization opportunities while respecting risk constraints

Author: Trading Bot
License: MIT
"""

import logging
import random
import asyncio
import aiohttp
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, Any

from models import (
    StrategyType, TradeProposal, TradeStatus, RiskParameters,
    PortfolioState, MarketCondition, IVEnvironment, TradeCandidate
)
from strategy_engine import StrategyEngine

logger = logging.getLogger(__name__)


class TastytradeBacktestClient:
    """
    Client for Tastytrade's backtesting API.

    Interfaces with https://backtester.vast.tastyworks.com to run
    backtests using real historical data when available.
    """

    def __init__(self, session_token: str):
        """
        Initialize the backtest client.

        Args:
            session_token: Authentication token for Tastytrade API
        """
        self.base_url = "https://backtester.vast.tastyworks.com"
        self.session_token = session_token
        self.headers = {
            "Authorization": session_token,
            "Content-Type": "application/json"
        }

    async def get_available_dates(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get available historical date ranges for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            Dict with available date ranges or None if unavailable
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/available-dates"
                params = {"symbol": symbol}

                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.warning(
                            f"Failed to get available dates for {symbol}: "
                            f"{response.status}"
                        )
                        return None
            except Exception as e:
                logger.error(f"Error fetching available dates for {symbol}: {e}")
                return None

    async def create_backtest(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        legs: List[Dict[str, Any]],
        entry_conditions: Optional[Dict[str, Any]] = None,
        exit_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new backtest.

        Args:
            symbol: Underlying symbol
            start_date: Backtest start date
            end_date: Backtest end date
            legs: List of leg definitions
            entry_conditions: Entry trigger conditions
            exit_conditions: Exit trigger conditions

        Returns:
            Backtest ID if successful, None otherwise
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/backtests"

                payload = {
                    "symbol": symbol,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "legs": legs,
                    "entry_conditions": entry_conditions or {},
                    "exit_conditions": exit_conditions or {}
                }

                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status == 201:
                        data = await response.json()
                        backtest_id = data.get("id")
                        logger.info(f"Created backtest {backtest_id} for {symbol}")
                        return backtest_id
                    else:
                        logger.error(
                            f"Failed to create backtest: {response.status}"
                        )
                        return None
            except Exception as e:
                logger.error(f"Error creating backtest: {e}")
                return None

    async def get_backtest_status(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status and results of a backtest.

        Args:
            backtest_id: Backtest ID

        Returns:
            Dict with backtest status and results
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/backtests/{backtest_id}"

                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(
                            f"Failed to get backtest status: {response.status}"
                        )
                        return None
            except Exception as e:
                logger.error(f"Error fetching backtest status: {e}")
                return None

    async def wait_for_backtest_completion(
        self,
        backtest_id: str,
        timeout_seconds: int = 300,
        poll_interval: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a backtest to complete.

        Args:
            backtest_id: Backtest ID
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks

        Returns:
            Completed backtest results or None if timeout/error
        """
        start_time = datetime.now()

        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout_seconds:
                logger.error(f"Backtest {backtest_id} timed out after {timeout_seconds}s")
                return None

            status_data = await self.get_backtest_status(backtest_id)
            if not status_data:
                return None

            status = status_data.get("status")

            if status == "completed":
                logger.info(f"Backtest {backtest_id} completed successfully")
                return status_data
            elif status == "failed":
                logger.error(f"Backtest {backtest_id} failed")
                return None
            elif status in ["pending", "running"]:
                logger.debug(f"Backtest {backtest_id} status: {status}")
                await asyncio.sleep(poll_interval)
            else:
                logger.warning(f"Unknown backtest status: {status}")
                await asyncio.sleep(poll_interval)

    async def cancel_backtest(self, backtest_id: str) -> bool:
        """
        Cancel a running backtest.

        Args:
            backtest_id: Backtest ID

        Returns:
            True if cancelled successfully
        """
        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.base_url}/backtests/{backtest_id}/cancel"

                async with session.post(url, headers=self.headers) as response:
                    if response.status == 200:
                        logger.info(f"Cancelled backtest {backtest_id}")
                        return True
                    else:
                        logger.error(f"Failed to cancel backtest: {response.status}")
                        return False
            except Exception as e:
                logger.error(f"Error cancelling backtest: {e}")
                return False

    def convert_strategy_to_legs(
        self,
        strategy_type: StrategyType,
        dte: int = 45,
        delta_short: float = 0.30,
        delta_long: float = 0.16
    ) -> List[Dict[str, Any]]:
        """
        Convert a StrategyType to Tastytrade API leg definitions.

        Args:
            strategy_type: Strategy type to convert
            dte: Days to expiration
            delta_short: Delta for short strikes
            delta_long: Delta for long strikes (protection)

        Returns:
            List of leg definitions for the API
        """
        legs = []

        if strategy_type == StrategyType.SHORT_PUT_SPREAD:
            legs = [
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": delta_short},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "long",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": delta_long},
                    "days_until_expiration": dte
                }
            ]

        elif strategy_type == StrategyType.SHORT_CALL_SPREAD:
            legs = [
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": delta_short},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "long",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": delta_long},
                    "days_until_expiration": dte
                }
            ]

        elif strategy_type == StrategyType.IRON_CONDOR:
            legs = [
                # Put spread
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.16},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "long",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.10},
                    "days_until_expiration": dte
                },
                # Call spread
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.16},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "long",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.10},
                    "days_until_expiration": dte
                }
            ]

        elif strategy_type == StrategyType.SHORT_STRANGLE:
            legs = [
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.16},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.16},
                    "days_until_expiration": dte
                }
            ]

        elif strategy_type == StrategyType.SHORT_STRADDLE:
            legs = [
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.50},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.50},
                    "days_until_expiration": dte
                }
            ]

        elif strategy_type == StrategyType.JADE_LIZARD:
            legs = [
                # Short put
                {
                    "type": "equity-option",
                    "option_type": "put",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.30},
                    "days_until_expiration": dte
                },
                # Short call spread
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "short",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.30},
                    "days_until_expiration": dte
                },
                {
                    "type": "equity-option",
                    "option_type": "call",
                    "direction": "long",
                    "quantity": 1,
                    "strike_selection": {"method": "delta", "value": 0.16},
                    "days_until_expiration": dte
                }
            ]

        else:
            logger.warning(f"Unsupported strategy type for API: {strategy_type}")

        return legs


@dataclass
class BacktestTrade:
    """A simulated trade during backtesting"""
    trade_id: str
    entry_date: date
    exit_date: Optional[date]
    strategy: StrategyType
    symbol: str
    entry_price: Decimal  # Credit received
    exit_price: Decimal  # Debit paid to close
    max_loss: Decimal
    dte_at_entry: int
    dte_at_exit: Optional[int]
    iv_rank_at_entry: Decimal
    iv_rank_at_exit: Optional[Decimal]
    pnl: Decimal  # Profit/Loss (positive = profit)
    pnl_percent: Decimal  # P&L as % of max risk
    is_winner: bool
    exit_reason: str  # "profit_target", "stop_loss", "dte_management", "expiration"

    # Greeks at entry
    delta_at_entry: Decimal
    theta_at_entry: Decimal

    # Trade metadata
    kelly_score: Decimal
    probability_of_profit: Decimal


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    start_date: date
    end_date: date
    initial_capital: Decimal
    final_capital: Decimal

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # P&L metrics
    total_pnl: Decimal
    total_return_percent: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    profit_factor: Decimal  # Total wins / Total losses

    # Risk metrics
    max_drawdown: Decimal
    max_drawdown_percent: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal

    # Strategy breakdown
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)
    pnl_by_strategy: Dict[str, Decimal] = field(default_factory=dict)
    win_rate_by_strategy: Dict[str, Decimal] = field(default_factory=dict)

    # Market condition analysis
    trades_by_iv_environment: Dict[str, int] = field(default_factory=dict)
    pnl_by_iv_environment: Dict[str, Decimal] = field(default_factory=dict)
    win_rate_by_iv_environment: Dict[str, Decimal] = field(default_factory=dict)

    # All trades for detailed analysis
    all_trades: List[BacktestTrade] = field(default_factory=list)


class HistoricalDataGenerator:
    """
    Generates synthetic historical option data for backtesting.

    This is necessary because the sandbox environment has limited historical data.
    The generator creates realistic option chains with proper Greeks and pricing.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the historical data generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Base volatility levels for different symbols
        self.symbol_volatilities = {
            'SPY': 0.15,
            'QQQ': 0.20,
            'IWM': 0.22,
            'DIA': 0.14,
            'AAPL': 0.25,
            'TSLA': 0.50,
            'AMD': 0.40,
            'NVDA': 0.45,
            'XLF': 0.18,
            'XLE': 0.25,
            'XLK': 0.20,
            'XLV': 0.16,
        }

    def generate_iv_rank_series(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Dict[date, Decimal]:
        """
        Generate a time series of IV ranks.

        Simulates realistic IV rank patterns:
        - Mean reverting behavior
        - Occasional spikes (earnings, market events)
        - Autocorrelation

        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date

        Returns:
            Dict mapping dates to IV ranks
        """
        iv_ranks = {}
        current_date = start_date

        # Starting IV rank (random between 20-60)
        current_iv = Decimal(str(random.uniform(20, 60)))

        while current_date <= end_date:
            # Mean reversion force (pulls toward 40)
            mean_reversion = (Decimal("40") - current_iv) * Decimal("0.05")

            # Random walk component
            random_change = Decimal(str(random.gauss(0, 3)))

            # Occasional IV spikes (5% chance per day)
            if random.random() < 0.05:
                random_change += Decimal(str(random.uniform(10, 30)))

            # Update IV rank
            current_iv = current_iv + mean_reversion + random_change

            # Clamp to [0, 100]
            current_iv = max(Decimal("0"), min(Decimal("100"), current_iv))

            iv_ranks[current_date] = current_iv
            current_date += timedelta(days=1)

        return iv_ranks

    def generate_price_series(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        initial_price: Decimal
    ) -> Dict[date, Decimal]:
        """
        Generate a time series of prices using geometric Brownian motion.

        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date
            initial_price: Starting price

        Returns:
            Dict mapping dates to prices
        """
        prices = {}
        current_date = start_date
        current_price = initial_price

        # Get volatility for this symbol
        annual_vol = self.symbol_volatilities.get(symbol, 0.20)
        daily_vol = annual_vol / (252 ** 0.5)  # Convert to daily

        # Slight upward drift (market tends up)
        daily_drift = 0.0003  # ~7.5% annual

        while current_date <= end_date:
            # Geometric Brownian motion
            random_return = random.gauss(daily_drift, daily_vol)
            current_price = current_price * Decimal(str(1 + random_return))

            prices[current_date] = current_price
            current_date += timedelta(days=1)

        return prices

    def generate_option_chain(
        self,
        symbol: str,
        current_price: Decimal,
        iv_rank: Decimal,
        expiration_date: date,
        current_date: date
    ) -> Dict[str, Any]:
        """
        Generate a synthetic option chain for a given date.

        Args:
            symbol: Underlying symbol
            current_price: Current stock price
            iv_rank: Current IV rank
            expiration_date: Option expiration date
            current_date: Current date

        Returns:
            Dict containing puts and calls with Greeks and quotes
        """
        dte = (expiration_date - current_date).days

        # Calculate implied volatility based on IV rank
        # IV rank represents percentile of IV over past year
        base_vol = self.symbol_volatilities.get(symbol, 0.20)

        # IV ranges from 0.8*base to 2.0*base depending on IV rank
        iv = base_vol * (0.8 + (float(iv_rank) / 100) * 1.2)

        # Generate strikes around current price (every $5 for most underlyings)
        strike_interval = Decimal("5")
        if current_price > 1000:
            strike_interval = Decimal("10")
        elif current_price < 50:
            strike_interval = Decimal("1")

        strikes = []
        for i in range(-10, 11):  # 21 strikes total
            strike = (current_price + i * strike_interval).quantize(Decimal("0.01"))
            strikes.append(strike)

        puts = []
        calls = []

        for strike in strikes:
            # Calculate moneyness
            moneyness = float(strike / current_price)

            # Simplified Black-Scholes for delta calculation
            # This is a rough approximation for backtesting
            if moneyness > 1:  # OTM put, ITM call
                put_delta = self._calculate_delta(moneyness, iv, dte, is_put=True)
                call_delta = self._calculate_delta(moneyness, iv, dte, is_put=False)
            else:  # ITM put, OTM call
                put_delta = self._calculate_delta(moneyness, iv, dte, is_put=True)
                call_delta = self._calculate_delta(moneyness, iv, dte, is_put=False)

            # Calculate option prices (very simplified)
            put_price = self._calculate_option_price(
                current_price, strike, dte, iv, is_put=True
            )
            call_price = self._calculate_option_price(
                current_price, strike, dte, iv, is_put=False
            )

            # Create put option
            puts.append({
                'strike_price': strike,
                'strike': strike,  # Keep for backward compatibility
                'expiration_date': expiration_date.isoformat(),
                'option_type': 'P',
                'delta': put_delta,
                'theta': Decimal(str(-0.01 * float(put_price))),  # Rough estimate
                'vega': Decimal(str(0.10)),
                'gamma': Decimal(str(0.05)),
                'bid': put_price * Decimal("0.98"),
                'ask': put_price * Decimal("1.02"),
                'mid': put_price,
            })

            # Create call option
            calls.append({
                'strike_price': strike,
                'strike': strike,  # Keep for backward compatibility
                'expiration_date': expiration_date.isoformat(),
                'option_type': 'C',
                'delta': call_delta,
                'theta': Decimal(str(-0.01 * float(call_price))),
                'vega': Decimal(str(0.10)),
                'gamma': Decimal(str(0.05)),
                'bid': call_price * Decimal("0.98"),
                'ask': call_price * Decimal("1.02"),
                'mid': call_price,
            })

        return {
            'puts': puts,
            'calls': calls,
            'underlying_price': current_price,
            'iv_rank': iv_rank,
            'expiration_date': expiration_date,
        }

    def _calculate_delta(
        self,
        moneyness: float,
        iv: float,
        dte: int,
        is_put: bool
    ) -> Decimal:
        """Calculate approximate option delta using Black-Scholes"""
        from math import log, erf, sqrt

        if dte <= 0 or iv <= 0:
            # At expiration or zero IV, delta is binary
            if is_put:
                return Decimal("-1.00") if moneyness > 1 else Decimal("0.00")
            else:
                return Decimal("1.00") if moneyness < 1 else Decimal("0.00")

        # Time to expiration in years
        T = dte / 365
        sqrt_T = sqrt(T)

        # Calculate d1 using Black-Scholes formula
        # moneyness = strike / spot, so we need ln(spot/strike) = -ln(moneyness)
        try:
            d1 = -log(moneyness) / (iv * sqrt_T)
        except (ValueError, ZeroDivisionError):
            d1 = 0

        # Cumulative normal distribution
        N_d1 = 0.5 * (1 + erf(d1 / sqrt(2)))

        if is_put:
            delta = N_d1 - 1  # Put delta is N(d1) - 1
        else:
            delta = N_d1  # Call delta is N(d1)

        # Clamp to reasonable range
        delta = max(-1.0, min(1.0, delta))

        return Decimal(str(delta)).quantize(Decimal("0.01"))

    def _calculate_option_price(
        self,
        spot: Decimal,
        strike: Decimal,
        dte: int,
        iv: float,
        is_put: bool
    ) -> Decimal:
        """Calculate approximate option price using simplified Black-Scholes"""
        from math import log, sqrt, exp

        # Intrinsic value
        intrinsic = max(Decimal("0"), (strike - spot) if is_put else (spot - strike))

        # Calculate time value based on moneyness and IV
        # Using a simplified approximation based on Black-Scholes principles

        if dte <= 0:
            return intrinsic

        T = dte / 365.0  # Time to expiration in years
        sqrt_T = sqrt(T)

        # Calculate moneyness (spot/strike for puts, strike/spot normalized)
        moneyness = float(strike / spot) if is_put else float(spot / strike)

        # For OTM options: use extrinsic value approximation
        # Price scales with IV, time, and distance from strike
        if intrinsic == 0:
            # OTM option - time value only
            # Approximate using simplified BSM: spot * IV * sqrt(T) * adjustment
            distance_factor = exp(-abs(1 - moneyness) * 2)  # Decay as we move OTM
            time_value = float(spot) * iv * sqrt_T * distance_factor * 0.4
        else:
            # ITM option - has both intrinsic and time value
            # Time value is smaller for ITM options
            distance_factor = exp(-abs(1 - moneyness))
            time_value = float(spot) * iv * sqrt_T * distance_factor * 0.2

        total_price = intrinsic + Decimal(str(time_value))

        return max(Decimal("0.01"), total_price.quantize(Decimal("0.01")))


class BacktestEngine:
    """
    Core backtesting engine that simulates trading over historical periods.

    Respects all risk management rules and simulates realistic trade execution,
    profit targets, stop losses, and position management.
    """

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        initial_capital: Decimal = Decimal("50000"),
        max_positions: int = 10,
        api_client: Optional[TastytradeBacktestClient] = None
    ):
        """
        Initialize the backtest engine.

        Args:
            strategy_engine: Strategy evaluation engine
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
            api_client: Optional Tastytrade API client for real historical data
        """
        self.strategy_engine = strategy_engine
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.api_client = api_client

        self.data_generator = HistoricalDataGenerator(seed=42)

    async def run_api_backtest(
        self,
        symbol: str,
        strategy_type: StrategyType,
        start_date: date,
        end_date: date,
        min_iv_rank: Optional[Decimal] = None
    ) -> Optional[BacktestResult]:
        """
        Run a backtest using Tastytrade's API with real historical data.

        This method interfaces with Tastytrade's backtesting API to run
        backtests using actual market data (when connected to live API).

        Args:
            symbol: Symbol to backtest
            strategy_type: Strategy type to test
            start_date: Start date
            end_date: End date
            min_iv_rank: Minimum IV rank filter (optional)

        Returns:
            BacktestResult or None if API unavailable
        """
        if not self.api_client:
            logger.warning("No API client configured. Use synthetic backtest instead.")
            return None

        logger.info(f"Running API backtest for {strategy_type.value} on {symbol}")

        # Check if historical data is available
        available_dates = await self.api_client.get_available_dates(symbol)
        if not available_dates:
            logger.warning(f"No historical data available for {symbol} via API")
            return None

        # Convert strategy to API leg definitions
        legs = self.api_client.convert_strategy_to_legs(strategy_type)
        if not legs:
            logger.error(f"Could not convert {strategy_type} to API legs")
            return None

        # Set up entry conditions
        entry_conditions = {
            "frequency": "daily"  # Scan daily for opportunities
        }

        if min_iv_rank:
            # Note: Tastytrade API may not support IV rank filtering directly
            # This would need to be confirmed in their API docs
            entry_conditions["min_iv_rank"] = float(min_iv_rank)

        # Set up exit conditions based on tastylive methodology
        exit_conditions = {
            "profit_target_percent": 50,  # Exit at 50% profit
            "stop_loss_percent": 200,  # Stop at 200% loss (2x credit)
            "dte_threshold": 21  # Close at 21 DTE
        }

        # Create the backtest
        backtest_id = await self.api_client.create_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            legs=legs,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions
        )

        if not backtest_id:
            logger.error("Failed to create backtest via API")
            return None

        # Wait for completion
        logger.info(f"Waiting for backtest {backtest_id} to complete...")
        results = await self.api_client.wait_for_backtest_completion(backtest_id)

        if not results:
            logger.error("Backtest failed or timed out")
            return None

        # Parse API results into BacktestResult format
        return self._parse_api_results(
            results, symbol, strategy_type, start_date, end_date
        )

    def _parse_api_results(
        self,
        api_results: Dict[str, Any],
        symbol: str,
        strategy_type: StrategyType,
        start_date: date,
        end_date: date
    ) -> BacktestResult:
        """
        Parse Tastytrade API backtest results into BacktestResult format.

        Args:
            api_results: Raw API response
            symbol: Symbol tested
            strategy_type: Strategy type tested
            start_date: Start date
            end_date: End date

        Returns:
            BacktestResult object
        """
        # Extract trials (individual trades) from API results
        trials = api_results.get("trials", [])
        snapshots = api_results.get("snapshots", [])
        statistics = api_results.get("statistics", {})

        # Convert trials to BacktestTrade objects
        all_trades = []
        for trial in trials:
            trade = BacktestTrade(
                trade_id=str(trial.get("id", "")),
                entry_date=date.fromisoformat(trial.get("entry_date")),
                exit_date=date.fromisoformat(trial.get("exit_date")) if trial.get("exit_date") else None,
                strategy=strategy_type,
                symbol=symbol,
                entry_price=Decimal(str(trial.get("entry_price", 0))),
                exit_price=Decimal(str(trial.get("exit_price", 0))),
                max_loss=Decimal(str(trial.get("max_loss", 0))),
                dte_at_entry=trial.get("dte_at_entry", 0),
                dte_at_exit=trial.get("dte_at_exit"),
                iv_rank_at_entry=Decimal(str(trial.get("iv_rank_at_entry", 0))),
                iv_rank_at_exit=Decimal(str(trial.get("iv_rank_at_exit", 0))) if trial.get("iv_rank_at_exit") else None,
                pnl=Decimal(str(trial.get("pnl", 0))),
                pnl_percent=Decimal(str(trial.get("pnl_percent", 0))),
                is_winner=trial.get("pnl", 0) > 0,
                exit_reason=trial.get("exit_reason", ""),
                delta_at_entry=Decimal(str(trial.get("delta_at_entry", 0))),
                theta_at_entry=Decimal(str(trial.get("theta_at_entry", 0))),
                kelly_score=Decimal("0"),  # Not provided by API
                probability_of_profit=Decimal(str(trial.get("probability_of_profit", 0))),
            )
            all_trades.append(trade)

        # Extract statistics
        total_trades = len(all_trades)
        winning_trades = len([t for t in all_trades if t.is_winner])
        losing_trades = total_trades - winning_trades

        # Build equity curve from snapshots
        equity_curve = []
        for snapshot in snapshots:
            snapshot_date = date.fromisoformat(snapshot.get("date"))
            equity = Decimal(str(snapshot.get("equity", 0)))
            equity_curve.append((snapshot_date, equity))

        # Calculate metrics
        if total_trades > 0:
            win_rate = Decimal(winning_trades) / Decimal(total_trades) * 100
        else:
            win_rate = Decimal("0")

        total_pnl = sum(t.pnl for t in all_trades)
        total_return_percent = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else Decimal("0")

        # Extract other metrics from API statistics
        avg_win = Decimal(str(statistics.get("avg_win", 0)))
        avg_loss = Decimal(str(statistics.get("avg_loss", 0)))
        largest_win = Decimal(str(statistics.get("largest_win", 0)))
        largest_loss = Decimal(str(statistics.get("largest_loss", 0)))
        profit_factor = Decimal(str(statistics.get("profit_factor", 0)))
        max_drawdown = Decimal(str(statistics.get("max_drawdown", 0)))
        max_drawdown_percent = Decimal(str(statistics.get("max_drawdown_percent", 0)))
        sharpe_ratio = Decimal(str(statistics.get("sharpe_ratio", 0)))
        sortino_ratio = Decimal(str(statistics.get("sortino_ratio", 0)))

        final_capital = self.initial_capital + total_pnl

        # Build strategy breakdown
        trades_by_strategy = {strategy_type.value: total_trades}
        pnl_by_strategy = {strategy_type.value: total_pnl}
        win_rate_by_strategy = {strategy_type.value: win_rate}

        # Build IV environment breakdown
        trades_by_iv = defaultdict(int)
        pnl_by_iv = defaultdict(lambda: Decimal("0"))
        wins_by_iv = defaultdict(int)

        for trade in all_trades:
            iv_rank = trade.iv_rank_at_entry
            if iv_rank < 20:
                iv_env = "very_low"
            elif iv_rank < 40:
                iv_env = "low"
            elif iv_rank < 60:
                iv_env = "moderate"
            elif iv_rank < 80:
                iv_env = "high"
            else:
                iv_env = "very_high"

            trades_by_iv[iv_env] += 1
            pnl_by_iv[iv_env] += trade.pnl
            if trade.is_winner:
                wins_by_iv[iv_env] += 1

        win_rate_by_iv = {
            iv_env: (Decimal(wins_by_iv[iv_env]) / Decimal(trades_by_iv[iv_env]) * 100)
            for iv_env in trades_by_iv
        }

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_percent=total_return_percent,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            trades_by_strategy=trades_by_strategy,
            pnl_by_strategy=pnl_by_strategy,
            win_rate_by_strategy=win_rate_by_strategy,
            trades_by_iv_environment=dict(trades_by_iv),
            pnl_by_iv_environment=dict(pnl_by_iv),
            win_rate_by_iv_environment=dict(win_rate_by_iv),
            all_trades=all_trades,
        )

    def run_backtest(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        scan_frequency_days: int = 1
    ) -> BacktestResult:
        """
        Run a backtest over a date range.

        Args:
            symbols: List of symbols to trade
            start_date: Start date
            end_date: End date
            scan_frequency_days: How often to scan for new opportunities

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial capital: ${self.initial_capital}")

        # Generate historical data for all symbols
        logger.info("Generating historical data...")
        historical_data = self._generate_historical_data(symbols, start_date, end_date)

        # Initialize portfolio
        portfolio = PortfolioState(
            account_value=self.initial_capital,
            buying_power=self.initial_capital,
            net_liquidating_value=self.initial_capital,
            cash_balance=self.initial_capital
        )

        # Track open positions and all trades
        open_positions: List[BacktestTrade] = []
        closed_trades: List[BacktestTrade] = []

        # Track equity curve for drawdown calculation
        equity_curve: List[Tuple[date, Decimal]] = [(start_date, self.initial_capital)]

        # Main backtest loop - iterate day by day
        current_date = start_date
        days_since_scan = 0

        while current_date <= end_date:
            # Manage existing positions
            for position in open_positions[:]:  # Copy list to allow removal
                exit_info = self._check_position_exit(
                    position, current_date, historical_data
                )

                if exit_info:
                    # Close the position
                    position.exit_date = current_date
                    position.exit_price = exit_info['exit_price']
                    position.dte_at_exit = exit_info['dte_at_exit']
                    position.iv_rank_at_exit = exit_info.get('iv_rank')
                    position.exit_reason = exit_info['reason']

                    # Calculate P&L
                    # For credit spreads: profit = credit received - debit paid to close
                    position.pnl = (position.entry_price - position.exit_price) * 100
                    position.pnl_percent = (position.pnl / position.max_loss) * 100
                    position.is_winner = position.pnl > 0

                    # Update portfolio
                    portfolio.cash_balance += position.pnl
                    portfolio.buying_power += position.max_loss  # Release buying power

                    # Move to closed trades
                    open_positions.remove(position)
                    closed_trades.append(position)

                    logger.debug(
                        f"Closed {position.strategy.value} on {position.symbol}: "
                        f"P&L ${position.pnl:.2f} ({position.exit_reason})"
                    )

            # Update portfolio value
            portfolio.account_value = portfolio.cash_balance + sum(
                pos.max_loss for pos in open_positions
            )
            portfolio.net_liquidating_value = portfolio.account_value
            portfolio.open_positions = len(open_positions)

            equity_curve.append((current_date, portfolio.account_value))

            # Scan for new opportunities on schedule
            if days_since_scan >= scan_frequency_days:
                if len(open_positions) < self.max_positions:
                    new_trades = self._scan_for_opportunities(
                        symbols,
                        current_date,
                        historical_data,
                        portfolio,
                        self.max_positions - len(open_positions)
                    )

                    for trade in new_trades:
                        # Check if we have buying power
                        if portfolio.buying_power >= trade.max_loss:
                            # Open the position
                            open_positions.append(trade)
                            portfolio.cash_balance += trade.entry_price * 100  # Credit received
                            portfolio.buying_power -= trade.max_loss

                            logger.debug(
                                f"Opened {trade.strategy.value} on {trade.symbol}: "
                                f"Credit ${trade.entry_price:.2f}, Max Loss ${trade.max_loss:.2f}"
                            )

                days_since_scan = 0

            # Move to next day
            current_date += timedelta(days=1)
            days_since_scan += 1

        # Close any remaining positions at end of backtest
        for position in open_positions:
            position.exit_date = end_date
            position.exit_price = position.entry_price * Decimal("0.5")  # Assume 50% profit
            position.dte_at_exit = 0
            position.exit_reason = "backtest_end"
            position.pnl = (position.entry_price - position.exit_price) * 100
            position.pnl_percent = (position.pnl / position.max_loss) * 100
            position.is_winner = position.pnl > 0
            closed_trades.append(position)
            portfolio.cash_balance += position.pnl

        # Calculate final metrics
        return self._calculate_results(
            closed_trades,
            equity_curve,
            start_date,
            end_date,
            self.initial_capital,
            portfolio.account_value
        )

    def _generate_historical_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[str, Dict[str, Any]]:
        """Generate historical data for all symbols"""
        data = {}

        for symbol in symbols:
            # Starting prices
            initial_prices = {
                'SPY': Decimal("400"),
                'QQQ': Decimal("350"),
                'IWM': Decimal("180"),
                'DIA': Decimal("340"),
                'AAPL': Decimal("170"),
                'TSLA': Decimal("200"),
                'AMD': Decimal("100"),
                'NVDA': Decimal("450"),
                'XLF': Decimal("40"),
                'XLE': Decimal("80"),
                'XLK': Decimal("160"),
                'XLV': Decimal("130"),
            }

            initial_price = initial_prices.get(symbol, Decimal("100"))

            # Generate price and IV rank series
            prices = self.data_generator.generate_price_series(
                symbol, start_date, end_date, initial_price
            )
            iv_ranks = self.data_generator.generate_iv_rank_series(
                symbol, start_date, end_date
            )

            data[symbol] = {
                'prices': prices,
                'iv_ranks': iv_ranks,
            }

        return data

    def _scan_for_opportunities(
        self,
        symbols: List[str],
        current_date: date,
        historical_data: Dict[str, Dict[str, Any]],
        portfolio: PortfolioState,
        max_new_trades: int
    ) -> List[BacktestTrade]:
        """Scan for trading opportunities on a given date"""
        candidates = []

        for symbol in symbols:
            # Get current market data
            symbol_data = historical_data[symbol]
            current_price = symbol_data['prices'].get(current_date)
            iv_rank = symbol_data['iv_ranks'].get(current_date)

            if not current_price or not iv_rank:
                continue

            # Check if IV rank meets minimum threshold
            if iv_rank < self.strategy_engine.risk_params.min_iv_rank:
                continue

            # Analyze market conditions
            market_condition = self.strategy_engine.analyze_market_conditions(
                symbol, iv_rank, current_price
            )

            # Select appropriate strategies for current conditions
            strategy_types = self.strategy_engine.select_strategies_to_evaluate(
                market_condition, portfolio
            )

            # Generate option chain for 45 DTE
            target_exp = current_date + timedelta(days=45)
            option_chain = self.data_generator.generate_option_chain(
                symbol, current_price, iv_rank, target_exp, current_date
            )

            # Evaluate strategies
            for strategy_name in strategy_types[:3]:  # Top 3 strategies
                try:
                    strategy_type = StrategyType(strategy_name)
                    trade_candidate = self._evaluate_strategy(
                        strategy_type,
                        symbol,
                        option_chain,
                        iv_rank,
                        current_date
                    )

                    if trade_candidate:
                        candidates.append(trade_candidate)

                except (ValueError, Exception) as e:
                    logger.debug(f"Could not evaluate {strategy_name}: {e}")
                    continue

        # Sort by Kelly score and take top candidates
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Convert top candidates to BacktestTrade objects
        trades = []
        for candidate in candidates[:max_new_trades]:
            trade = BacktestTrade(
                trade_id=f"{candidate.underlying}_{current_date.isoformat()}_{candidate.strategy.value}",
                entry_date=current_date,
                exit_date=None,
                strategy=candidate.strategy,
                symbol=candidate.underlying,
                entry_price=candidate.net_credit,
                exit_price=Decimal("0"),
                max_loss=candidate.max_loss,
                dte_at_entry=candidate.dte,
                dte_at_exit=None,
                iv_rank_at_entry=candidate.iv_rank,
                iv_rank_at_exit=None,
                pnl=Decimal("0"),
                pnl_percent=Decimal("0"),
                is_winner=False,
                exit_reason="",
                delta_at_entry=candidate.greeks.get('delta', Decimal("0")),
                theta_at_entry=candidate.greeks.get('theta', Decimal("0")),
                kelly_score=candidate.score,
                probability_of_profit=candidate.probability_of_profit,
            )
            trades.append(trade)

        return trades

    def _evaluate_strategy(
        self,
        strategy_type: StrategyType,
        symbol: str,
        option_chain: Dict[str, Any],
        iv_rank: Decimal,
        current_date: date
    ) -> Optional[TradeCandidate]:
        """Evaluate a specific strategy"""
        puts = option_chain['puts']
        calls = option_chain['calls']
        exp_date = option_chain['expiration_date']

        # Convert string date to date object
        if isinstance(exp_date, str):
            exp_date = date.fromisoformat(exp_date)

        # Build greeks and quotes dictionaries
        put_greeks = {p['strike_price']: p for p in puts}
        call_greeks = {c['strike_price']: c for c in calls}
        put_quotes = {p['strike_price']: (p['bid'], p['ask']) for p in puts}
        call_quotes = {c['strike_price']: (c['bid'], c['ask']) for c in calls}

        # Call appropriate strategy builder
        candidate = None

        if strategy_type == StrategyType.SHORT_PUT_SPREAD:
            candidate = self.strategy_engine.build_put_spread_candidate(
                symbol, puts, put_greeks, put_quotes, iv_rank, exp_date
            )
        elif strategy_type == StrategyType.SHORT_CALL_SPREAD:
            candidate = self.strategy_engine.build_call_spread_candidate(
                symbol, calls, call_greeks, call_quotes, iv_rank, exp_date
            )
        elif strategy_type == StrategyType.IRON_CONDOR:
            candidate = self.strategy_engine.build_iron_condor_candidate(
                symbol, puts, calls, put_greeks, call_greeks,
                put_quotes, call_quotes, iv_rank, exp_date
            )
        elif strategy_type == StrategyType.SHORT_STRANGLE:
            candidate = self.strategy_engine.build_short_strangle_candidate(
                symbol, puts, calls, put_greeks, call_greeks,
                put_quotes, call_quotes, iv_rank, exp_date
            )
        elif strategy_type == StrategyType.SHORT_STRADDLE:
            candidate = self.strategy_engine.build_short_straddle_candidate(
                symbol, puts, calls, put_greeks, call_greeks,
                put_quotes, call_quotes, iv_rank, exp_date
            )
        elif strategy_type == StrategyType.CALENDAR_SPREAD:
            # Would need multiple expirations for calendar spreads
            # Skip for now in backtesting
            pass
        elif strategy_type == StrategyType.JADE_LIZARD:
            candidate = self.strategy_engine.build_jade_lizard_candidate(
                symbol, puts, calls, put_greeks, call_greeks,
                put_quotes, call_quotes, iv_rank, exp_date
            )
        elif strategy_type == StrategyType.LONG_STRANGLE:
            candidate = self.strategy_engine.build_long_strangle_candidate(
                symbol, puts, calls, put_greeks, call_greeks,
                put_quotes, call_quotes, iv_rank, exp_date
            )

        if candidate:
            candidate.calculate_kelly_score()

        return candidate

    def _check_position_exit(
        self,
        position: BacktestTrade,
        current_date: date,
        historical_data: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a position should be exited.

        Exit triggers:
        1. Profit target hit (50% of max profit)
        2. Stop loss hit (2x credit received)
        3. DTE management (21 DTE)
        4. Expiration
        """
        # Calculate current DTE
        # We need to reconstruct expiration date from entry DTE
        expiration_date = position.entry_date + timedelta(days=position.dte_at_entry)
        dte = (expiration_date - current_date).days

        # Check expiration
        if dte <= 0:
            return {
                'exit_price': Decimal("0"),  # Expired worthless (best case)
                'dte_at_exit': 0,
                'reason': 'expiration'
            }

        # Simulate current option value
        # In reality, would re-price the option based on current conditions
        symbol_data = historical_data[position.symbol]
        current_price = symbol_data['prices'].get(current_date)
        current_iv_rank = symbol_data['iv_ranks'].get(current_date)

        if not current_price or not current_iv_rank:
            return None

        # Simplified P&L estimation based on time decay and market movement
        # Assume time decay is linear (rough approximation)
        time_decay_factor = Decimal(str(1 - ((position.dte_at_entry - dte) / position.dte_at_entry)))

        # Current value of the spread (starts at entry_price, decays to 0)
        current_value = position.entry_price * time_decay_factor

        # Add some randomness for market movement
        market_factor = Decimal(str(random.gauss(1.0, 0.2)))
        current_value = current_value * market_factor

        # Ensure current value is positive
        current_value = max(Decimal("0.01"), current_value)

        # Calculate current P&L
        current_pnl = (position.entry_price - current_value) * 100
        current_pnl_percent = (current_pnl / position.max_loss) * 100

        # Check profit target (50% of max profit)
        max_profit = position.entry_price * 100
        if current_pnl >= max_profit * Decimal("0.50"):
            return {
                'exit_price': current_value,
                'dte_at_exit': dte,
                'iv_rank': current_iv_rank,
                'reason': 'profit_target'
            }

        # Check stop loss (losing 2x the credit received)
        if current_pnl <= -(position.entry_price * 100 * 2):
            return {
                'exit_price': current_value,
                'dte_at_exit': dte,
                'iv_rank': current_iv_rank,
                'reason': 'stop_loss'
            }

        # Check DTE management (close at 21 DTE)
        if dte <= 21:
            return {
                'exit_price': current_value,
                'dte_at_exit': dte,
                'iv_rank': current_iv_rank,
                'reason': 'dte_management'
            }

        return None

    def _calculate_results(
        self,
        closed_trades: List[BacktestTrade],
        equity_curve: List[Tuple[date, Decimal]],
        start_date: date,
        end_date: date,
        initial_capital: Decimal,
        final_capital: Decimal
    ) -> BacktestResult:
        """Calculate performance metrics from backtest results"""

        if not closed_trades:
            logger.warning("No trades executed during backtest")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=Decimal("0"),
                total_pnl=Decimal("0"),
                total_return_percent=Decimal("0"),
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                largest_win=Decimal("0"),
                largest_loss=Decimal("0"),
                profit_factor=Decimal("0"),
                max_drawdown=Decimal("0"),
                max_drawdown_percent=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                sortino_ratio=Decimal("0"),
            )

        # Basic statistics
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.is_winner)
        losing_trades = total_trades - winning_trades
        win_rate = Decimal(winning_trades) / Decimal(total_trades) * 100

        # P&L metrics
        total_pnl = sum(t.pnl for t in closed_trades)
        total_return_percent = (total_pnl / initial_capital) * 100

        wins = [t.pnl for t in closed_trades if t.is_winner]
        losses = [t.pnl for t in closed_trades if not t.is_winner]

        avg_win = sum(wins) / len(wins) if wins else Decimal("0")
        avg_loss = sum(losses) / len(losses) if losses else Decimal("0")
        largest_win = max(wins) if wins else Decimal("0")
        largest_loss = min(losses) if losses else Decimal("0")

        total_wins = sum(wins) if wins else Decimal("0")
        total_losses = abs(sum(losses)) if losses else Decimal("0.01")
        profit_factor = total_wins / total_losses

        # Calculate max drawdown
        max_drawdown = Decimal("0")
        max_drawdown_percent = Decimal("0")
        peak = initial_capital

        for _, equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_percent = (drawdown / peak) * 100 if peak > 0 else Decimal("0")

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = drawdown_percent

        # Calculate Sharpe ratio
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1][1]
            curr_equity = equity_curve[i][1]
            daily_return = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else Decimal("0")
            daily_returns.append(float(daily_return))

        if daily_returns:
            import statistics
            avg_return = Decimal(str(statistics.mean(daily_returns)))
            std_return = Decimal(str(statistics.stdev(daily_returns))) if len(daily_returns) > 1 else Decimal("0.0001")
            # Annualized Sharpe (assuming 252 trading days)
            sharpe_ratio = (avg_return / std_return) * Decimal(str(252 ** 0.5)) if std_return > 0 else Decimal("0")
        else:
            sharpe_ratio = Decimal("0")

        # Calculate Sortino ratio (only considers downside volatility)
        negative_returns = [r for r in daily_returns if r < 0]
        if len(negative_returns) > 1:
            import statistics
            downside_std = Decimal(str(statistics.stdev(negative_returns)))
            avg_return = Decimal(str(statistics.mean(daily_returns)))
            sortino_ratio = (avg_return / downside_std) * Decimal(str(252 ** 0.5)) if downside_std > 0 else Decimal("0")
        else:
            # Not enough downside data to calculate Sortino ratio
            sortino_ratio = Decimal("0")

        # Strategy breakdown
        trades_by_strategy = defaultdict(int)
        pnl_by_strategy = defaultdict(lambda: Decimal("0"))
        wins_by_strategy = defaultdict(int)

        for trade in closed_trades:
            strategy_name = trade.strategy.value
            trades_by_strategy[strategy_name] += 1
            pnl_by_strategy[strategy_name] += trade.pnl
            if trade.is_winner:
                wins_by_strategy[strategy_name] += 1

        win_rate_by_strategy = {
            strategy: (Decimal(wins_by_strategy[strategy]) / Decimal(trades_by_strategy[strategy]) * 100)
            for strategy in trades_by_strategy
        }

        # IV environment breakdown
        trades_by_iv = defaultdict(int)
        pnl_by_iv = defaultdict(lambda: Decimal("0"))
        wins_by_iv = defaultdict(int)

        for trade in closed_trades:
            # Classify IV environment at entry
            iv_rank = trade.iv_rank_at_entry
            if iv_rank < 20:
                iv_env = "very_low"
            elif iv_rank < 40:
                iv_env = "low"
            elif iv_rank < 60:
                iv_env = "moderate"
            elif iv_rank < 80:
                iv_env = "high"
            else:
                iv_env = "very_high"

            trades_by_iv[iv_env] += 1
            pnl_by_iv[iv_env] += trade.pnl
            if trade.is_winner:
                wins_by_iv[iv_env] += 1

        win_rate_by_iv = {
            iv_env: (Decimal(wins_by_iv[iv_env]) / Decimal(trades_by_iv[iv_env]) * 100)
            for iv_env in trades_by_iv
        }

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_percent=total_return_percent,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            trades_by_strategy=dict(trades_by_strategy),
            pnl_by_strategy=dict(pnl_by_strategy),
            win_rate_by_strategy=dict(win_rate_by_strategy),
            trades_by_iv_environment=dict(trades_by_iv),
            pnl_by_iv_environment=dict(pnl_by_iv),
            win_rate_by_iv_environment=dict(win_rate_by_iv),
            all_trades=closed_trades,
        )


class BacktestAnalyzer:
    """
    Analyzes backtest results to identify optimization opportunities.

    Suggests improvements while respecting all risk constraints.
    """

    def analyze_results(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Analyze backtest results and generate recommendations.

        Args:
            result: BacktestResult object

        Returns:
            Dict with analysis and recommendations
        """
        recommendations = []
        observations = []

        # Overall performance analysis
        if result.total_return_percent > 10:
            observations.append(
                f"Strong overall performance: {result.total_return_percent:.2f}% return"
            )
        elif result.total_return_percent < 0:
            observations.append(
                f"Negative overall performance: {result.total_return_percent:.2f}% return"
            )
            recommendations.append(
                "Consider reducing position sizes or being more selective with entries"
            )

        # Win rate analysis
        if result.win_rate < 50:
            observations.append(f"Below-average win rate: {result.win_rate:.2f}%")
            recommendations.append(
                "Consider tightening entry criteria (higher min IV rank, better probability of profit)"
            )
        elif result.win_rate > 70:
            observations.append(f"Strong win rate: {result.win_rate:.2f}%")

        # Risk-adjusted returns
        if result.sharpe_ratio < 1:
            observations.append(f"Low Sharpe ratio: {result.sharpe_ratio:.2f}")
            recommendations.append(
                "Returns not adequately compensating for risk. Consider more selective entries or better position sizing."
            )
        elif result.sharpe_ratio > 2:
            observations.append(f"Excellent Sharpe ratio: {result.sharpe_ratio:.2f}")

        # Drawdown analysis
        if result.max_drawdown_percent > 20:
            observations.append(
                f"Large drawdown: {result.max_drawdown_percent:.2f}%"
            )
            recommendations.append(
                "Drawdown exceeds 20%. Consider tighter stop losses or reduced position sizing."
            )

        # Strategy-specific analysis
        if result.trades_by_strategy:
            best_strategy = max(
                result.pnl_by_strategy.items(),
                key=lambda x: x[1]
            )
            worst_strategy = min(
                result.pnl_by_strategy.items(),
                key=lambda x: x[1]
            )

            observations.append(
                f"Best performing strategy: {best_strategy[0]} (${best_strategy[1]:.2f} total P&L)"
            )
            observations.append(
                f"Worst performing strategy: {worst_strategy[0]} (${worst_strategy[1]:.2f} total P&L)"
            )

            if worst_strategy[1] < 0:
                recommendations.append(
                    f"Consider reducing allocation to {worst_strategy[0]} strategy or avoiding it in certain market conditions"
                )

        # IV environment analysis
        if result.trades_by_iv_environment:
            best_iv_env = max(
                result.pnl_by_iv_environment.items(),
                key=lambda x: x[1]
            )
            worst_iv_env = min(
                result.pnl_by_iv_environment.items(),
                key=lambda x: x[1]
            )

            observations.append(
                f"Best IV environment: {best_iv_env[0]} (${best_iv_env[1]:.2f} total P&L)"
            )
            observations.append(
                f"Worst IV environment: {worst_iv_env[0]} (${worst_iv_env[1]:.2f} total P&L)"
            )

            if worst_iv_env[1] < 0:
                recommendations.append(
                    f"Avoid or reduce trading in {worst_iv_env[0]} IV environments"
                )

        # Profit factor analysis
        if result.profit_factor < 1:
            observations.append(
                f"Profit factor below 1: {result.profit_factor:.2f} (losing more than winning)"
            )
            recommendations.append(
                "System is not profitable. Consider major changes to entry criteria or strategy selection."
            )
        elif result.profit_factor > 2:
            observations.append(f"Strong profit factor: {result.profit_factor:.2f}")

        # Average win vs average loss
        if abs(result.avg_loss) > result.avg_win * 2:
            observations.append(
                f"Losses are significantly larger than wins (Avg Win: ${result.avg_win:.2f}, Avg Loss: ${result.avg_loss:.2f})"
            )
            recommendations.append(
                "Implement tighter stop losses or adjust position sizing to limit loss severity"
            )

        return {
            'observations': observations,
            'recommendations': recommendations,
            'summary': {
                'total_return_percent': float(result.total_return_percent),
                'win_rate': float(result.win_rate),
                'sharpe_ratio': float(result.sharpe_ratio),
                'max_drawdown_percent': float(result.max_drawdown_percent),
                'profit_factor': float(result.profit_factor),
            }
        }

    def generate_report(self, result: BacktestResult) -> str:
        """
        Generate a human-readable backtest report.

        Args:
            result: BacktestResult object

        Returns:
            Formatted report string
        """
        analysis = self.analyze_results(result)

        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS REPORT")
        report.append("=" * 80)
        report.append("")

        # Period and capital
        report.append(f"Period: {result.start_date} to {result.end_date}")
        report.append(f"Initial Capital: ${result.initial_capital:,.2f}")
        report.append(f"Final Capital: ${result.final_capital:,.2f}")
        report.append("")

        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 80)
        report.append(f"Total P&L: ${result.total_pnl:,.2f}")
        report.append(f"Total Return: {result.total_return_percent:.2f}%")
        report.append(f"Total Trades: {result.total_trades}")
        report.append(f"Winning Trades: {result.winning_trades}")
        report.append(f"Losing Trades: {result.losing_trades}")
        report.append(f"Win Rate: {result.win_rate:.2f}%")
        report.append("")

        # Risk metrics
        report.append("RISK METRICS")
        report.append("-" * 80)
        report.append(f"Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_percent:.2f}%)")
        report.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        report.append(f"Profit Factor: {result.profit_factor:.2f}")
        report.append("")

        # Win/Loss analysis
        report.append("WIN/LOSS ANALYSIS")
        report.append("-" * 80)
        report.append(f"Average Win: ${result.avg_win:,.2f}")
        report.append(f"Average Loss: ${result.avg_loss:,.2f}")
        report.append(f"Largest Win: ${result.largest_win:,.2f}")
        report.append(f"Largest Loss: ${result.largest_loss:,.2f}")
        report.append("")

        # Strategy breakdown
        if result.trades_by_strategy:
            report.append("STRATEGY BREAKDOWN")
            report.append("-" * 80)
            for strategy in sorted(result.trades_by_strategy.keys()):
                trades = result.trades_by_strategy[strategy]
                pnl = result.pnl_by_strategy[strategy]
                win_rate = result.win_rate_by_strategy[strategy]
                report.append(
                    f"{strategy:30s} | Trades: {trades:3d} | P&L: ${pnl:10,.2f} | Win Rate: {win_rate:6.2f}%"
                )
            report.append("")

        # IV environment breakdown
        if result.trades_by_iv_environment:
            report.append("IV ENVIRONMENT BREAKDOWN")
            report.append("-" * 80)
            for iv_env in sorted(result.trades_by_iv_environment.keys()):
                trades = result.trades_by_iv_environment[iv_env]
                pnl = result.pnl_by_iv_environment[iv_env]
                win_rate = result.win_rate_by_iv_environment[iv_env]
                report.append(
                    f"{iv_env:15s} | Trades: {trades:3d} | P&L: ${pnl:10,.2f} | Win Rate: {win_rate:6.2f}%"
                )
            report.append("")

        # Key observations
        report.append("KEY OBSERVATIONS")
        report.append("-" * 80)
        for obs in analysis['observations']:
            report.append(f"• {obs}")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 80)
        if analysis['recommendations']:
            for rec in analysis['recommendations']:
                report.append(f"• {rec}")
        else:
            report.append("• No major issues identified. Continue current strategy.")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)
