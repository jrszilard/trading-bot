#!/usr/bin/env python3
"""
Example usage of the Tastytrade Trading Bot

This script demonstrates how to:
1. Configure the bot with custom risk parameters
2. Connect to the Tastytrade API
3. Scan for opportunities
4. Review and approve trades
5. Execute trades with safety checks

IMPORTANT: Always test in sandbox mode first!
"""

import asyncio
from decimal import Decimal
from trading_bot import (
    TastytradeBot,
    RiskParameters,
    StrategyType,
    TradeStatus
)


# Custom approval function - integrate with your workflow
def my_approval_callback(proposal) -> bool:
    """
    Custom approval logic - integrate with your preferred method:
    - Slack notification
    - Email approval
    - Mobile push notification
    - Web dashboard
    
    For this example, we'll auto-approve small defined-risk trades
    in sandbox mode only.
    """
    # NEVER auto-approve in production without careful consideration!
    
    # Example: Only auto-approve if:
    # 1. It's a defined risk strategy
    # 2. Max loss is under $100
    # 3. It's a single contract
    
    defined_risk_strategies = {
        StrategyType.SHORT_PUT_SPREAD,
        StrategyType.SHORT_CALL_SPREAD,
        StrategyType.IRON_CONDOR
    }
    
    is_defined_risk = proposal.strategy in defined_risk_strategies
    is_small_position = proposal.max_loss <= Decimal("100")
    is_single_contract = len(proposal.legs) <= 4  # Iron condor has 4 legs
    
    if is_defined_risk and is_small_position and is_single_contract:
        print(f"Auto-approving small defined-risk trade: {proposal.id}")
        return True
    
    # Otherwise, require manual approval
    print(f"\nManual approval required for: {proposal.id}")
    print(f"  Strategy: {proposal.strategy.value}")
    print(f"  Max Loss: ${proposal.max_loss}")
    response = input("Approve? (yes/no): ").strip().lower()
    return response in ('yes', 'y')


async def example_basic_usage():
    """
    Basic example: Scan, review, and execute trades
    """
    print("=" * 60)
    print("BASIC USAGE EXAMPLE")
    print("=" * 60)
    
    # Create bot with conservative risk parameters
    bot = TastytradeBot(
        risk_params=RiskParameters(
            # Conservative loss limits
            max_daily_loss=Decimal("300"),
            max_weekly_loss=Decimal("1000"),
            max_position_loss=Decimal("200"),
            
            # Tastylive standard entry criteria
            min_iv_rank=Decimal("30"),
            target_dte=45,
            
            # Conservative position sizing
            max_position_size_percent=Decimal("0.03"),  # 3%
            max_total_positions=10,
        ),
        sandbox_mode=True  # ALWAYS start in sandbox!
    )
    
    # Set watchlist
    watchlist = ["SPY", "QQQ", "IWM"]
    
    print(f"\nScanning watchlist: {watchlist}")
    
    # In production, connect first:
    # await bot.connect('your_client_secret', 'your_refresh_token')
    
    # Scan for opportunities
    opportunities = await bot.scan_for_opportunities(watchlist)
    
    print(f"Found {len(opportunities)} opportunities")
    
    # Review each opportunity
    for proposal in opportunities:
        print(f"\n{proposal.rationale}")
        
        # Request approval
        approved = await bot.request_approval(proposal)
        
        if approved:
            # Execute the trade
            success = await bot.execute_trade(proposal)
            if success:
                print(f"✓ Trade {proposal.id} executed successfully")
            else:
                print(f"✗ Trade {proposal.id} failed to execute")
    
    # Show final state
    print("\n" + "=" * 60)
    print("TRADING SESSION SUMMARY")
    print("=" * 60)
    
    history = bot.get_trade_history()
    executed = [t for t in history if t.status == TradeStatus.EXECUTED]
    rejected = [t for t in history if t.status == TradeStatus.REJECTED]
    
    print(f"Trades executed: {len(executed)}")
    print(f"Trades rejected: {len(rejected)}")


async def example_with_custom_approval():
    """
    Example with custom approval workflow
    """
    print("=" * 60)
    print("CUSTOM APPROVAL WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Create bot with custom approval callback
    bot = TastytradeBot(
        risk_params=RiskParameters(
            max_position_loss=Decimal("150"),
            min_iv_rank=Decimal("35"),
        ),
        approval_callback=my_approval_callback,
        sandbox_mode=True
    )
    
    # Simulate a trade proposal
    legs = [{
        'symbol': 'SPY_012524P480',
        'option_type': 'PUT',
        'strike': '480',
        'expiration': '2024-01-25',
        'action': 'SELL_TO_OPEN',
        'quantity': 1,
        'credit': '1.25',
        'max_loss': '75',
        'dte': 45
    }]
    
    greeks = {
        'delta': -0.16,
        'theta': 0.03,
        'vega': -0.08,
        'pop': 0.84
    }
    
    proposal = await bot.propose_trade(
        strategy=StrategyType.SHORT_PUT_SPREAD,
        underlying='SPY',
        legs=legs,
        greeks=greeks,
        iv_rank=Decimal("42")
    )
    
    if proposal:
        # The approval callback will be called automatically
        approved = await bot.request_approval(proposal)
        
        if approved:
            await bot.execute_trade(proposal)


async def example_portfolio_monitoring():
    """
    Example: Monitor portfolio and manage positions
    """
    print("=" * 60)
    print("PORTFOLIO MONITORING EXAMPLE")
    print("=" * 60)
    
    bot = TastytradeBot(sandbox_mode=True)
    
    # Update portfolio state
    await bot.update_portfolio_state()
    
    state = bot.portfolio_state
    print(f"\nPortfolio Overview:")
    print(f"  Net Liq Value: ${state.net_liquidating_value}")
    print(f"  Buying Power:  ${state.buying_power}")
    print(f"  Daily P&L:     ${state.daily_pnl}")
    print(f"  Positions:     {state.open_positions}")
    
    # Check for positions needing management
    print(f"\nChecking positions for management...")
    await bot.manage_positions()
    
    # Export state for record keeping
    bot.export_state('portfolio_state.json')
    print("State exported to portfolio_state.json")


async def example_risk_check():
    """
    Example: Demonstrate risk checking
    """
    print("=" * 60)
    print("RISK CHECK EXAMPLE")
    print("=" * 60)
    
    # Create bot with strict risk limits
    bot = TastytradeBot(
        risk_params=RiskParameters(
            max_daily_loss=Decimal("100"),
            max_position_loss=Decimal("50"),
            max_total_positions=5,
            min_iv_rank=Decimal("40"),
        ),
        sandbox_mode=True
    )
    
    # Simulate a trade that violates risk limits
    legs = [{
        'symbol': 'TSLA_012524P200',
        'option_type': 'PUT',
        'strike': '200',
        'expiration': '2024-01-25',
        'action': 'SELL_TO_OPEN',
        'quantity': 1,
        'credit': '5.00',
        'max_loss': '1000',  # Exceeds max_position_loss
        'dte': 45
    }]
    
    greeks = {'delta': -0.30, 'theta': 0.10, 'vega': -0.15, 'pop': 0.70}
    
    print("\nAttempting to create high-risk trade...")
    
    proposal = await bot.propose_trade(
        strategy=StrategyType.SHORT_PUT,
        underlying='TSLA',
        legs=legs,
        greeks=greeks,
        iv_rank=Decimal("55")
    )
    
    if proposal is None:
        print("✓ Trade correctly rejected by risk checks")
    else:
        print("✗ Trade was allowed (unexpected)")
    
    # Show what trades were rejected
    rejected = [t for t in bot.trade_history if t.status == TradeStatus.REJECTED]
    for t in rejected:
        print(f"  Rejected: {t.id} - {t.rejection_reason}")


async def main():
    """
    Run all examples
    """
    print("\n" + "=" * 60)
    print("TASTYTRADE TRADING BOT EXAMPLES")
    print("=" * 60)
    print("\nThese examples demonstrate bot functionality.")
    print("Always test in sandbox mode before live trading!")
    print("\n")
    
    # Run examples
    await example_risk_check()
    print("\n")
    
    await example_portfolio_monitoring()
    print("\n")
    
    # Uncomment to run interactive examples:
    # await example_basic_usage()
    # await example_with_custom_approval()


if __name__ == "__main__":
    asyncio.run(main())
