#!/usr/bin/env python3
"""
Test script for the backtesting system.

This demonstrates how to run backtests using both synthetic data
(for sandbox environment) and real historical data (via Tastytrade API).

Usage:
    python test_backtest.py --mode synthetic
    python test_backtest.py --mode api
"""

import asyncio
import logging
from datetime import date, timedelta
from decimal import Decimal
import argparse

from backtesting import (
    BacktestEngine,
    BacktestAnalyzer,
    TastytradeBacktestClient
)
from models import RiskParameters, StrategyType
from strategy_engine import StrategyEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_synthetic_backtest():
    """
    Run a backtest using synthetic historical data.

    This is useful in the sandbox environment where real historical
    data may not be available.
    """
    logger.info("=" * 80)
    logger.info("Running Synthetic Backtest")
    logger.info("=" * 80)

    # Initialize components
    risk_params = RiskParameters()
    strategy_engine = StrategyEngine(risk_params)

    # Create backtest engine (no API client = synthetic mode)
    backtest_engine = BacktestEngine(
        strategy_engine=strategy_engine,
        initial_capital=Decimal("50000"),
        max_positions=10
    )

    # Define backtest parameters
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA']
    end_date = date.today()
    start_date = end_date - timedelta(days=180)  # 6 months

    logger.info(f"Backtesting {len(symbols)} symbols over 6 months")
    logger.info(f"Period: {start_date} to {end_date}")

    # Run backtest
    result = backtest_engine.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        scan_frequency_days=1  # Scan daily
    )

    # Analyze results
    analyzer = BacktestAnalyzer()
    report = analyzer.generate_report(result)

    print("\n" + report)

    # Return result for further analysis if needed
    return result


async def run_api_backtest(session_token: str):
    """
    Run a backtest using Tastytrade's API with real historical data.

    This requires a valid session token and connection to the live API.

    Args:
        session_token: Tastytrade API session token
    """
    logger.info("=" * 80)
    logger.info("Running API-Based Backtest")
    logger.info("=" * 80)

    # Initialize components
    risk_params = RiskParameters()
    strategy_engine = StrategyEngine(risk_params)

    # Create API client
    api_client = TastytradeBacktestClient(session_token)

    # Create backtest engine with API support
    backtest_engine = BacktestEngine(
        strategy_engine=strategy_engine,
        initial_capital=Decimal("50000"),
        max_positions=10,
        api_client=api_client
    )

    # Define backtest parameters
    symbol = 'SPY'
    strategy_type = StrategyType.IRON_CONDOR
    end_date = date.today()
    start_date = end_date - timedelta(days=180)  # 6 months

    logger.info(f"Backtesting {strategy_type.value} on {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")

    # Check available dates first
    available_dates = await api_client.get_available_dates(symbol)
    if available_dates:
        logger.info(f"Historical data available for {symbol}")
    else:
        logger.warning(f"No historical data available for {symbol}. Falling back to synthetic.")
        return None

    # Run API backtest
    result = await backtest_engine.run_api_backtest(
        symbol=symbol,
        strategy_type=strategy_type,
        start_date=start_date,
        end_date=end_date,
        min_iv_rank=risk_params.min_iv_rank
    )

    if not result:
        logger.error("API backtest failed")
        return None

    # Analyze results
    analyzer = BacktestAnalyzer()
    report = analyzer.generate_report(result)

    print("\n" + report)

    return result


async def run_multi_strategy_comparison():
    """
    Compare multiple strategies using synthetic backtesting.

    This analyzes which strategies perform best under different conditions.
    """
    logger.info("=" * 80)
    logger.info("Multi-Strategy Comparison Backtest")
    logger.info("=" * 80)

    # Initialize components
    risk_params = RiskParameters()
    strategy_engine = StrategyEngine(risk_params)

    backtest_engine = BacktestEngine(
        strategy_engine=strategy_engine,
        initial_capital=Decimal("50000"),
        max_positions=10
    )

    # Test period
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year

    # Symbols to test
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'AMD']

    logger.info("Running backtest with all available strategies")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Symbols: {symbols}")

    # Run comprehensive backtest
    result = backtest_engine.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        scan_frequency_days=1
    )

    # Generate detailed analysis
    analyzer = BacktestAnalyzer()
    analysis = analyzer.analyze_results(result)

    # Print full report
    report = analyzer.generate_report(result)
    print("\n" + report)

    # Print additional insights
    print("\nDETAILED ANALYSIS")
    print("=" * 80)

    print("\nKey Observations:")
    for obs in analysis['observations']:
        print(f"  • {obs}")

    print("\nRecommendations for Improvement:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")

    print("\n" + "=" * 80)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run backtests on trading strategies")
    parser.add_argument(
        '--mode',
        choices=['synthetic', 'api', 'comparison'],
        default='synthetic',
        help='Backtest mode: synthetic (local simulation), api (real data), or comparison (multi-strategy)'
    )
    parser.add_argument(
        '--session-token',
        type=str,
        help='Tastytrade API session token (required for api mode)'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'synthetic':
            result = run_synthetic_backtest()
            print("\n✓ Synthetic backtest completed successfully")

        elif args.mode == 'api':
            if not args.session_token:
                print("Error: --session-token is required for api mode")
                print("Usage: python test_backtest.py --mode api --session-token YOUR_TOKEN")
                return

            result = asyncio.run(run_api_backtest(args.session_token))
            if result:
                print("\n✓ API backtest completed successfully")
            else:
                print("\n✗ API backtest failed")

        elif args.mode == 'comparison':
            result = asyncio.run(run_multi_strategy_comparison())
            print("\n✓ Multi-strategy comparison completed successfully")

    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user")
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)
        print(f"\n✗ Backtest failed: {e}")


if __name__ == "__main__":
    main()
