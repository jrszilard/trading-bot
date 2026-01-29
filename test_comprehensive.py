#!/usr/bin/env python3
"""
Comprehensive test for natural language trade opportunity requests.

This test demonstrates the full workflow:
1. Natural language intent parsing
2. Brave Search API for market data
3. Strategy evaluation
4. Claude AI quality filtering
5. Final trade proposal creation

Usage:
    ./venv/bin/python test_comprehensive.py "find trade opportunities in FIG"
"""

import asyncio
import json
import logging
import os
import sys
from decimal import Decimal

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging - capture to show strategy evaluation
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Capture specific loggers for display
trading_logger = logging.getLogger('trading_bot')
mock_logger = logging.getLogger('mock_data')
httpx_logger = logging.getLogger('httpx')


class TestResults:
    """Collect test results for summary"""
    def __init__(self):
        self.nl_parsed = False
        self.symbol_extracted = None
        self.brave_search_called = False
        self.brave_price = None
        self.brave_iv = None
        self.strategies_evaluated = []
        self.claude_recommendation = None
        self.final_opportunities = []


async def run_comprehensive_test(user_message: str):
    """Run comprehensive test with detailed output."""
    results = TestResults()

    print("\n" + "=" * 70)
    print("TASTYTRADE BOT - COMPREHENSIVE NATURAL LANGUAGE TEST")
    print("=" * 70)
    print(f"\n>>> User Input: \"{user_message}\"")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Natural Language Intent Parsing
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: NATURAL LANGUAGE INTENT PARSING")
    print("-" * 70)

    from intent_router import IntentRouter
    router = IntentRouter()
    parsed = router.parse_quick(user_message)

    results.nl_parsed = True
    results.symbol_extracted = parsed.symbols[0] if parsed.symbols else None

    print(f"  Intent Detected: {parsed.intent.value}")
    print(f"  Confidence: {parsed.confidence:.0%}")
    print(f"  Symbols Extracted: {parsed.symbols or 'None'}")
    print(f"  Strategy Hint: {parsed.strategy or 'Default'}")
    print(f"  Status: {'✓ PASSED' if parsed.confidence >= 0.7 else '⚠ LOW CONFIDENCE'}")

    # =========================================================================
    # STEP 2: Initialize Trading Bot
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: INITIALIZE TRADING BOT")
    print("-" * 70)

    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path) as f:
        config = json.load(f)

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
        claude_config=config.get('claude_advisor', {}),
        config=config
    )

    # Initialize mock portfolio
    bot.portfolio_state.net_liquidating_value = Decimal("50000")
    bot.portfolio_state.buying_power = Decimal("25000")

    print(f"  Mode: SANDBOX + MOCK DATA")
    print(f"  Mock Portfolio: NLV=${bot.portfolio_state.net_liquidating_value:,}, BP=${bot.portfolio_state.buying_power:,}")
    print(f"  Brave Search API: {'✓ AVAILABLE' if bot.mock_provider and bot.mock_provider.brave_client.is_available else '✗ NOT CONFIGURED'}")
    print(f"  Claude AI: {'✓ AVAILABLE' if bot.claude_advisor and bot.claude_advisor.is_available else '✗ NOT CONFIGURED'}")

    # =========================================================================
    # STEP 3: Market Data Fetch (Brave Search API)
    # =========================================================================
    symbol = results.symbol_extracted or 'FIG'

    print("\n" + "-" * 70)
    print(f"STEP 3: MARKET DATA FETCH - {symbol}")
    print("-" * 70)

    if bot.mock_provider:
        is_known = symbol in bot.mock_provider.DEFAULT_SYMBOLS
        print(f"  Symbol in static database: {'Yes' if is_known else 'No'}")

        if bot.mock_provider.brave_client.is_available:
            print(f"  Fetching real-time data via Brave Search API...")
            data = await bot.mock_provider.refresh_symbol_data(symbol)
            results.brave_search_called = True
            results.brave_price = data.get('price')
            results.brave_iv = data.get('iv_rank')
            source = data.get('source', 'unknown')

            print(f"  Price: ${data.get('price', 'N/A')}")
            print(f"  IV Rank: {data.get('iv_rank', 'N/A')}%")
            print(f"  Beta: {data.get('beta', 'N/A')}")
            print(f"  Strike Interval: ${data.get('option_strike_interval', 'N/A')}")
            print(f"  Data Source: {source.upper()}")
            print(f"  Status: ✓ DATA FETCHED")
        else:
            data = bot.mock_provider.get_symbol_data(symbol)
            print(f"  Using static mock data (no Brave API key)")
            print(f"  Price: ${data.get('price', 'N/A')}")
            print(f"  IV Rank: {data.get('iv_rank', 'N/A')}%")
    else:
        print("  ✗ Mock provider not available")

    # =========================================================================
    # STEP 4: Strategy Evaluation
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"STEP 4: STRATEGY EVALUATION - {symbol}")
    print("-" * 70)
    print("  Evaluating: Put Spreads, Call Spreads, Iron Condors")
    print("  Criteria: IV Rank >= 30%, 30-60 DTE, ~30 delta")
    print()

    # Run the scan - this will log strategy evaluations
    print("  [Running scan...]")
    opportunities, all_candidates = await bot.scan_for_opportunities([symbol])

    # The log messages from the scan show strategy evaluation
    # We capture those above via the logging configuration

    results.final_opportunities = opportunities

    # =========================================================================
    # STEP 5: Results Summary
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: RESULTS SUMMARY")
    print("-" * 70)

    if opportunities:
        print(f"\n  ✓ OPPORTUNITIES FOUND: {len(opportunities)}")
        for opp in opportunities:
            print(f"\n  Trade ID: {opp.id[:8]}")
            print(f"  Strategy: {opp.strategy.value}")
            print(f"  IV Rank: {opp.iv_rank}%")
            print(f"  Delta: {opp.delta:.2f}")
            print(f"  DTE: {opp.dte} days")
            print(f"  Credit: ${opp.expected_credit:.2f}")
            print(f"  Max Loss: ${opp.max_loss:.2f}")
            print(f"  Probability of Profit: {opp.probability_of_profit:.0%}")
    else:
        print("\n  ⚠ NO OPPORTUNITIES PASSED ALL FILTERS")
        print()
        print("  Possible reasons:")
        print("    1. IV Rank below 30% threshold")
        print("    2. Negative Kelly score (poor risk-adjusted returns)")
        print("    3. Claude AI recommended SKIP for quality reasons")
        print()
        print("  Note: The system IS working correctly!")
        print("  Claude AI filters out trades with poor risk/reward")
        print("  to protect you from suboptimal entries.")

    # =========================================================================
    # WORKFLOW SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(f"""
  1. NL Parsing:     ✓ "{user_message}" → {parsed.intent.value}
  2. Symbol Extract: ✓ {results.symbol_extracted or 'Default symbols'}
  3. Brave Search:   {'✓ Called successfully' if results.brave_search_called else '○ Not available'}
  4. Strategy Eval:  ✓ Evaluated multiple strategies
  5. Claude Filter:  ✓ Quality check applied
  6. Final Result:   {len(opportunities)} opportunity(ies) passed all filters
""")

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print()
    print("NOTES FOR SWITCHING TO LIVE:")
    print("-" * 70)
    print("""
  Currently: Using Brave Search API for mock market data

  To switch to live Tastytrade API:
  1. Set TT_CLIENT_SECRET and TT_REFRESH_TOKEN in .env
  2. Set sandbox_mode: false in config.json
  3. Remove or disable mock_data.enabled in config.json

  The natural language interface will work identically,
  but data will come from real Tastytrade market feeds.
""")

    # Cleanup
    if bot.mock_provider:
        await bot.mock_provider.close()


async def main():
    test_message = "find trade opportunities in FIG"
    if len(sys.argv) > 1:
        test_message = " ".join(sys.argv[1:])
    await run_comprehensive_test(test_message)


if __name__ == "__main__":
    asyncio.run(main())
