#!/usr/bin/env python3
"""
Test script for natural language trade opportunity requests.

Tests the chatbot's ability to process natural language requests like:
"find trade opportunities in FIG"

Uses Brave Search API for dynamic market data in testing mode.
When switching to live, the tastytrade API will be queried directly.
"""

import asyncio
import json
import logging
import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from decimal import Decimal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_nl_request(user_message: str, skip_claude_filter: bool = False):
    """
    Test a natural language request through the chatbot.

    Args:
        user_message: The natural language query to test
        skip_claude_filter: If True, disable Claude AI filtering to show raw opportunities
    """
    print("\n" + "=" * 60)
    print("NATURAL LANGUAGE REQUEST TEST")
    print("=" * 60)
    print(f"\nTest Input: \"{user_message}\"")
    print("-" * 60)

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Build risk parameters
    from trading_bot import TastytradeBot
    from models import RiskParameters

    risk_config = config.get('risk_parameters', {})
    loss_limits = risk_config.get('loss_limits', {})
    position_sizing = risk_config.get('position_sizing', {})
    entry_criteria = config.get('entry_criteria', {})

    risk_params = RiskParameters(
        max_daily_loss=Decimal(str(loss_limits.get('max_daily_loss', 500))),
        max_weekly_loss=Decimal(str(loss_limits.get('max_weekly_loss', 1500))),
        max_position_loss=Decimal(str(loss_limits.get('max_position_loss', 300))),
        max_position_size_percent=Decimal(str(position_sizing.get('max_position_size_percent', 0.05))),
        max_total_positions=position_sizing.get('max_total_positions', 15),
        min_iv_rank=Decimal(str(entry_criteria.get('iv_requirements', {}).get('min_iv_rank', 30))),
        target_dte=entry_criteria.get('dte_requirements', {}).get('target_dte', 45),
    )

    # Initialize bot in sandbox + mock data mode
    claude_config = config.get('claude_advisor', {})

    bot = TastytradeBot(
        risk_params=risk_params,
        sandbox_mode=True,  # Always use sandbox for testing
        claude_config=claude_config,
        config=config
    )

    # Initialize mock portfolio state with reasonable values for testing
    # This simulates having a funded account without needing real API connection
    bot.portfolio_state.net_liquidating_value = Decimal("50000")
    bot.portfolio_state.buying_power = Decimal("25000")
    bot.portfolio_state.account_value = Decimal("50000")
    bot.portfolio_state.cash_balance = Decimal("10000")
    bot.portfolio_state.open_positions = 0
    logger.info(f"Mock portfolio initialized: NLV=${bot.portfolio_state.net_liquidating_value}, BP=${bot.portfolio_state.buying_power}")

    # Check data sources
    print("\n[Data Sources]")
    print(f"  Mock Data: {'ENABLED' if bot.use_mock_data else 'DISABLED'}")

    if bot.mock_provider:
        brave_available = bot.mock_provider.brave_client.is_available
        print(f"  Brave Search API: {'AVAILABLE' if brave_available else 'NOT CONFIGURED'}")
        if not brave_available:
            print("    (Set BRAVE_API_KEY env var for real-time market data)")

    if bot.claude_advisor:
        claude_available = bot.claude_advisor.is_available
        print(f"  Claude AI: {'AVAILABLE' if claude_available else 'NOT CONFIGURED'}")
    else:
        print("  Claude AI: NOT CONFIGURED")

    # Create chatbot interface
    from chatbot import ChatbotInterface
    chatbot = ChatbotInterface(bot)

    # Test intent parsing first
    from intent_router import IntentRouter
    router = IntentRouter()
    parsed = router.parse_quick(user_message)

    print("\n[Intent Parsing]")
    print(f"  Intent: {parsed.intent.value}")
    print(f"  Confidence: {parsed.confidence:.2f}")
    print(f"  Extracted Symbols: {parsed.symbols or 'None'}")
    print(f"  Strategy: {parsed.strategy or 'Default (short_put_spread)'}")

    # If Brave Search is available, test fetching data for the symbol
    if parsed.symbols and bot.mock_provider:
        print("\n[Market Data Fetch]")
        for symbol in parsed.symbols:
            print(f"  Fetching market data for {symbol}...")

            if bot.mock_provider.brave_client.is_available:
                # Fetch fresh data via Brave Search
                data = await bot.mock_provider.refresh_symbol_data(symbol)
            else:
                # Use static mock data
                data = bot.mock_provider.get_symbol_data(symbol)

            print(f"    Price: ${data.get('price', 'N/A')}")
            print(f"    IV Rank: {data.get('iv_rank', 'N/A')}%")
            print(f"    Beta: {data.get('beta', 'N/A')}")
            print(f"    Strike Interval: ${data.get('option_strike_interval', 'N/A')}")
            print(f"    Source: {data.get('source', 'unknown')}")

            # Check if symbol is known or dynamic
            is_known = symbol in bot.mock_provider.DEFAULT_SYMBOLS
            print(f"    Known Symbol: {'Yes' if is_known else 'No (using Brave Search or defaults)'}")

    # Process the full message through the chatbot
    print("\n[Chatbot Response]")
    print("-" * 40)

    response = await chatbot.process_message(user_message)
    print(response)

    print("-" * 40)

    # Show pending trades if any were created
    pending = bot.get_pending_trades()
    if pending:
        print(f"\n[Pending Trades Created: {len(pending)}]")
        for trade in pending:
            print(f"  [{trade.id[:8]}] {trade.underlying_symbol} {trade.strategy.value}")
            print(f"    IV Rank: {trade.iv_rank}% | Delta: {trade.delta:.2f} | DTE: {trade.dte}")
            print(f"    Credit: ${trade.expected_credit:.2f} | Max Loss: ${trade.max_loss:.2f}")

    # Cleanup
    if bot.mock_provider:
        await bot.mock_provider.close()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60 + "\n")


async def test_direct_scan(symbol: str):
    """
    Test the scan function directly to show raw strategy evaluation.

    This bypasses the chatbot to show the raw strategy candidates
    before Claude AI filtering.
    """
    print("\n" + "=" * 60)
    print("DIRECT SCAN TEST (Raw Strategy Evaluation)")
    print("=" * 60)
    print(f"\nSymbol: {symbol}")
    print("-" * 60)

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Disable Claude for this test to see raw opportunities
    config['claude_advisor'] = {'enabled': False}

    from trading_bot import TastytradeBot
    from models import RiskParameters

    risk_config = config.get('risk_parameters', {})
    loss_limits = risk_config.get('loss_limits', {})
    position_sizing = risk_config.get('position_sizing', {})
    entry_criteria = config.get('entry_criteria', {})

    risk_params = RiskParameters(
        max_daily_loss=Decimal(str(loss_limits.get('max_daily_loss', 500))),
        max_weekly_loss=Decimal(str(loss_limits.get('max_weekly_loss', 1500))),
        max_position_loss=Decimal(str(loss_limits.get('max_position_loss', 300))),
        max_position_size_percent=Decimal(str(position_sizing.get('max_position_size_percent', 0.05))),
        max_total_positions=position_sizing.get('max_total_positions', 15),
        min_iv_rank=Decimal(str(entry_criteria.get('iv_requirements', {}).get('min_iv_rank', 30))),
        target_dte=entry_criteria.get('dte_requirements', {}).get('target_dte', 45),
    )

    bot = TastytradeBot(
        risk_params=risk_params,
        sandbox_mode=True,
        claude_config={},  # Disable Claude
        config=config
    )

    # Initialize mock portfolio
    bot.portfolio_state.net_liquidating_value = Decimal("50000")
    bot.portfolio_state.buying_power = Decimal("25000")
    bot.portfolio_state.account_value = Decimal("50000")
    bot.portfolio_state.cash_balance = Decimal("10000")

    print("\n[Scanning without Claude AI filtering...]")

    # Fetch fresh market data if Brave Search is available
    if bot.mock_provider:
        if bot.mock_provider.brave_client.is_available:
            print(f"\n[Brave Search] Fetching real-time data for {symbol}...")
            data = await bot.mock_provider.refresh_symbol_data(symbol)
            print(f"  Price: ${data.get('price', 'N/A')}")
            print(f"  IV Rank: {data.get('iv_rank', 'N/A')}%")
            print(f"  Source: {data.get('source', 'unknown')}")

    # Run scan
    opportunities = await bot.scan_for_opportunities([symbol])

    if opportunities:
        print(f"\n[Raw Opportunities Found: {len(opportunities)}]")
        for opp in opportunities:
            print(f"\n  Trade ID: {opp.id[:8]}")
            print(f"  Symbol: {opp.underlying_symbol}")
            print(f"  Strategy: {opp.strategy.value}")
            print(f"  IV Rank: {opp.iv_rank}%")
            print(f"  Delta: {opp.delta:.2f}")
            print(f"  DTE: {opp.dte} days")
            print(f"  Expected Credit: ${opp.expected_credit:.2f}")
            print(f"  Max Loss: ${opp.max_loss:.2f}")
            print(f"  Probability of Profit: {opp.probability_of_profit:.0%}")
    else:
        print("\n[No opportunities passed risk filters]")
        print("  This could be due to:")
        print("  - IV Rank below minimum threshold (30%)")
        print("  - Poor risk-adjusted returns (negative Kelly score)")
        print("  - Strategy constraints not met")

    # Cleanup
    if bot.mock_provider:
        await bot.mock_provider.close()

    print("\n" + "=" * 60 + "\n")


async def main():
    """Main entry point."""
    # Default test message - can be overridden via command line
    test_message = "find trade opportunities in FIG"

    if len(sys.argv) > 1:
        if sys.argv[1] == "--direct":
            # Direct scan mode (bypasses chatbot and Claude)
            symbol = sys.argv[2] if len(sys.argv) > 2 else "FIG"
            await test_direct_scan(symbol)
            return
        test_message = " ".join(sys.argv[1:])

    await test_nl_request(test_message)


if __name__ == "__main__":
    asyncio.run(main())
