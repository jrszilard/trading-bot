# Tastytrade Trading Bot

A Python trading bot that implements **tastylive methodology** using the Tastytrade API. Features comprehensive risk management, trade approval workflow, and position management following proven tastylive best practices.

## ⚠️ Important Disclaimer

This bot is for **educational purposes only**. Trading options involves significant risk of loss. Past performance does not guarantee future results. Always understand the risks before trading.

**Key Safety Features:**
- 🔒 **Approval Required**: No trades execute without explicit user approval
- 🏖️ **Sandbox Mode**: Always test in sandbox before live trading
- 📊 **Risk Limits**: Configurable loss limits and position sizing
- 🛑 **Kill Switch**: Easy to pause/stop all trading activity

## Tastylive Best Practices Implemented

This bot follows the tastylive trading methodology developed by Tom Sosnoff and the tastylive team:

### Entry Criteria

| Parameter | Target | Rationale |
|-----------|--------|-----------|
| **IV Rank** | > 30% | Enter when implied volatility is elevated (options are "expensive") |
| **DTE** | ~45 days | Optimal theta decay without excessive gamma risk |
| **Delta** | ~30 (or 16 conservative) | ~70% probability of profit |
| **Probability OTM** | > 65% | Statistical edge in favor of the trade |

### Position Management

| Rule | Action | Rationale |
|------|--------|-----------|
| **50% Profit** | Close position | Capture most of potential profit, reduce risk |
| **21 DTE** | Roll or close | Avoid gamma risk acceleration |
| **2x Credit Loss** | Stop loss | Defined risk management |
| **Tested Position** | Roll same strike | Maintain original thesis |

### Portfolio Management

- **Trade Small**: Max 5% of portfolio per position
- **Trade Often**: Diversify across time and underlyings
- **Stay Neutral**: Beta-weight portfolio to SPY, keep delta near zero
- **Maintain Reserve**: Keep 30% buying power in reserve

## Installation

```bash
# Clone or download the bot
cd tastytrade-bot

# Install dependencies
pip install tastytrade

# Or install the official SDK
pip install tastytrade-sdk
```

## Configuration

Edit `config.json` to customize:

```json
{
  "risk_parameters": {
    "loss_limits": {
      "max_daily_loss": 500.00,
      "max_position_loss": 300.00
    },
    "position_sizing": {
      "max_position_size_percent": 0.05,
      "max_total_positions": 15
    }
  },
  "entry_criteria": {
    "iv_requirements": {
      "min_iv_rank": 30
    },
    "dte_requirements": {
      "target_dte": 45
    }
  }
}
```

## Usage

### Starting the Bot

```bash
python trading_bot.py
```

### CLI Commands

```
scan <symbols>  - Scan for opportunities (e.g., scan SPY,QQQ,IWM)
pending         - Show pending trades awaiting approval
approve <id>    - Approve a trade for execution
reject <id>     - Reject a trade
execute <id>    - Execute an approved trade
portfolio       - Show current portfolio state
risk            - Show risk parameters
history         - Show trade history
export          - Export state to file
quit            - Exit
```

### Example Session

```
> scan SPY,QQQ,IWM
Scanning: ['SPY', 'QQQ', 'IWM']
Found 2 opportunities

> pending
  [a1b2c3d4] SPY short_put - $1.50
  [e5f6g7h8] QQQ short_put - $2.25

> approve a1b2c3d4

============================================================
TRADE APPROVAL REQUEST
============================================================

Proposal ID: a1b2c3d4
Timestamp: 2024-01-15 10:30:00

Strategy: short_put
Underlying: SPY

Tastylive Criteria Met:
  ✓ IV Rank: 42% (target: >30%)
  ✓ DTE: 45 days (target: ~45)
  ✓ Probability OTM: 70% (target: >65%)

Trade Details:
  Expected Credit: $1.50
  Max Loss: $300.00
  Delta: -0.30
  Theta: 0.05 (daily decay)

Management Plan:
  • Take profit at 50% ($0.75)
  • Roll or close at 21 DTE
  • Stop loss at 2x credit ($3.00 debit)
============================================================

Approve this trade? (yes/no): yes
Trade a1b2c3d4 APPROVED

> execute a1b2c3d4
SANDBOX MODE: Would execute trade a1b2c3d4
Trade a1b2c3d4 EXECUTED
```

## API Authentication

### Option 1: OAuth (Recommended)

```python
from tastytrade import Session

# Set up OAuth application at developer.tastytrade.com
session = Session('client_secret', 'refresh_token')
```

### Option 2: Username/Password (Dev only)

```python
from tastytrade_sdk import Tastytrade

tasty = Tastytrade()
tasty.login(login='your_email', password='your_password')
```

## Risk Management

### Loss Limits

```python
risk_params = RiskParameters(
    max_daily_loss=Decimal("500"),      # Stop trading after $500 daily loss
    max_weekly_loss=Decimal("1500"),    # Stop trading after $1500 weekly loss
    max_position_loss=Decimal("300"),   # Max loss per position
)
```

### Position Sizing

The bot enforces tastylive's "trade small" philosophy:

- Maximum 5% of portfolio per position
- Maximum 15-20 total positions
- Maximum 2 positions per underlying
- 30% buying power held in reserve

### Approval Workflow

**CRITICAL**: The bot NEVER executes trades without explicit approval.

```python
# Trade flow:
# 1. Bot proposes trade
# 2. User reviews proposal
# 3. User approves or rejects
# 4. Only then can trade be executed

proposal = await bot.propose_trade(...)  # Creates proposal
await bot.request_approval(proposal)      # Asks for approval
await bot.execute_trade(proposal)         # Executes if approved
```

## Supported Strategies

| Strategy | Risk Profile | Best When |
|----------|--------------|-----------|
| Short Put | Undefined risk | Bullish, high IV |
| Short Put Spread | Defined risk | Bullish, any IV |
| Short Call | Undefined risk | Bearish, high IV |
| Short Call Spread | Defined risk | Bearish, any IV |
| Iron Condor | Defined risk | Neutral, high IV |
| Short Strangle | Undefined risk | Neutral, high IV |
| Covered Call | Stock ownership | Long stock, lower IV |

## File Structure

```
tastytrade-bot/
├── trading_bot.py      # Main bot implementation
├── config.json         # Configuration parameters
├── README.md          # This documentation
└── requirements.txt    # Python dependencies
```

## API Reference

### Tastytrade API Documentation
- [Developer Portal](https://developer.tastytrade.com/)
- [API Overview](https://developer.tastytrade.com/api-overview/)
- [Order Submission](https://developer.tastytrade.com/order-submission/)
- [Streaming Data](https://developer.tastytrade.com/streaming-market-data/)

### Python SDKs
- [Official SDK](https://github.com/tastytrade/tastytrade-sdk-python) (`pip install tastytrade-sdk`)
- [Community SDK](https://github.com/tastyware/tastytrade) (`pip install tastytrade`)

## Tastylive Resources

Learn more about the tastylive trading methodology:

- [Tastylive Learn Center](https://www.tastylive.com/learn-courses)
- [Trader Resources](https://www.tastylive.com/trader-resources)
- [Options Strategies](https://www.tastylive.com/concepts-strategies)
- [Tastytrade Platform](https://tastytrade.com)

## Key Tastylive Principles

1. **Sell Premium**: Collect theta decay by selling options
2. **Trade Mechanical**: Follow rules consistently, remove emotion
3. **Manage Winners**: Take 50% profit, don't hold to expiration
4. **Trade Small**: 1-5% of portfolio per trade
5. **Trade Often**: Diversify across time and positions
6. **Stay Neutral**: Beta-weight portfolio, avoid directional bias
7. **High IV = Opportunity**: Enter when options are expensive

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

## Support

- [Tastytrade Support](https://support.tastytrade.com)
- [API Documentation](https://developer.tastytrade.com)
- [Community Forum](https://github.com/tastyware/tastytrade/discussions)

---

**Remember**: Always start in sandbox mode, understand the risks, and never risk more than you can afford to lose.
