# Backtesting System

The trading bot now includes a comprehensive backtesting framework to test historical performance of strategies and identify optimization opportunities.

## Features

### 1. Dual-Mode Operation
- **Synthetic Mode**: Generates realistic historical option data for testing in sandbox environment
- **API Mode**: Uses Tastytrade's backtesting API with real historical data (when connected to live API)

### 2. Comprehensive Metrics
- Win rate, profit factor, Sharpe ratio, Sortino ratio
- Maximum drawdown analysis
- Strategy-specific performance breakdown
- IV environment performance analysis

### 3. Intelligent Analysis
- Identifies underperforming strategies
- Suggests parameter optimizations
- Provides actionable recommendations
- Respects all existing risk constraints

## Usage

### Via Natural Language (Chatbot)

Simply ask the bot to run a backtest:

```
"Run a backtest"
"Backtest my strategies over the past 6 months"
"Test iron condors historically on SPY"
"How would my strategies have performed last year?"
"Backtest SPY, QQQ, and TSLA over 1 year"
```

### Via Python Script

```bash
# Synthetic backtest (sandbox environment)
python test_backtest.py --mode synthetic

# API-based backtest (requires live API connection)
python test_backtest.py --mode api --session-token YOUR_TOKEN

# Multi-strategy comparison
python test_backtest.py --mode comparison
```

### Programmatic Usage

```python
from datetime import date, timedelta
from decimal import Decimal
from backtesting import BacktestEngine, BacktestAnalyzer
from models import RiskParameters
from strategy_engine import StrategyEngine
from risk_manager import RiskManager

# Initialize components
risk_params = RiskParameters()
strategy_engine = StrategyEngine(risk_params)
risk_manager = RiskManager(risk_params)

# Create backtest engine
backtest_engine = BacktestEngine(
    strategy_engine=strategy_engine,
    risk_manager=risk_manager,
    initial_capital=Decimal("50000"),
    max_positions=10
)

# Define parameters
symbols = ['SPY', 'QQQ', 'IWM']
end_date = date.today()
start_date = end_date - timedelta(days=180)

# Run backtest
result = backtest_engine.run_backtest(
    symbols=symbols,
    start_date=start_date,
    end_date=end_date,
    scan_frequency_days=1
)

# Analyze results
analyzer = BacktestAnalyzer()
report = analyzer.generate_report(result)
print(report)

# Get recommendations
analysis = analyzer.analyze_results(result)
for rec in analysis['recommendations']:
    print(f"• {rec}")
```

## How It Works

### Synthetic Mode

1. **Historical Data Generation**
   - Generates realistic price series using geometric Brownian motion
   - Creates IV rank time series with mean reversion and occasional spikes
   - Builds complete option chains with proper Greeks and pricing

2. **Trade Simulation**
   - Scans for opportunities based on your strategy rules
   - Simulates realistic trade entry, management, and exit
   - Respects all risk management rules (max loss, position sizing, etc.)

3. **Performance Tracking**
   - Tracks every trade from entry to exit
   - Calculates daily P&L and equity curve
   - Monitors drawdowns and risk metrics

### API Mode

1. **Historical Data Fetching**
   - Checks available date ranges via Tastytrade API
   - Retrieves real historical option prices and Greeks

2. **Backtest Execution**
   - Creates backtest job via API with your strategy parameters
   - Monitors backtest progress
   - Retrieves detailed results including all trades

3. **Results Analysis**
   - Converts API results to standardized format
   - Performs same analysis as synthetic mode

## Output

### Backtest Report Includes:

```
================================================================================
BACKTEST RESULTS REPORT
================================================================================

Period: 2025-07-28 to 2026-01-27
Initial Capital: $50,000.00
Final Capital: $57,350.00

OVERALL PERFORMANCE
--------------------------------------------------------------------------------
Total P&L: $7,350.00
Total Return: 14.70%
Total Trades: 145
Winning Trades: 102
Losing Trades: 43
Win Rate: 70.34%

RISK METRICS
--------------------------------------------------------------------------------
Max Drawdown: $2,150.00 (4.30%)
Sharpe Ratio: 1.85
Sortino Ratio: 2.42
Profit Factor: 2.15

WIN/LOSS ANALYSIS
--------------------------------------------------------------------------------
Average Win: $125.50
Average Loss: $-98.25
Largest Win: $450.00
Largest Loss: $-380.00

STRATEGY BREAKDOWN
--------------------------------------------------------------------------------
iron_condor                 | Trades:  45 | P&L: $3,250.00 | Win Rate: 75.56%
short_put_spread            | Trades:  38 | P&L: $2,100.00 | Win Rate: 68.42%
short_strangle              | Trades:  32 | P&L: $1,850.00 | Win Rate: 65.63%
...

IV ENVIRONMENT BREAKDOWN
--------------------------------------------------------------------------------
high           | Trades:  52 | P&L: $4,200.00 | Win Rate: 76.92%
moderate       | Trades:  58 | P&L: $2,450.00 | Win Rate: 67.24%
low            | Trades:  35 | P&L:   $700.00 | Win Rate: 62.86%
...

KEY OBSERVATIONS
--------------------------------------------------------------------------------
• Strong overall performance: 14.70% return
• Excellent Sharpe ratio: 1.85
• Best performing strategy: iron_condor ($3,250.00 total P&L)
• Best IV environment: high ($4,200.00 total P&L)

RECOMMENDATIONS
--------------------------------------------------------------------------------
• Continue current strategy
• Consider increasing allocation to iron_condor strategy
• Focus on high IV environments for best results
```

## Analysis Features

The analyzer identifies:

1. **Performance Issues**
   - Low win rates (< 50%)
   - Poor risk-adjusted returns (Sharpe < 1)
   - Excessive drawdowns (> 20%)
   - Negative profit factors

2. **Strategy Insights**
   - Best and worst performing strategies
   - Strategy-specific win rates and P&L
   - Recommendations for strategy selection

3. **Market Condition Analysis**
   - Performance across different IV environments
   - Optimal conditions for each strategy
   - Suggestions for IV rank filtering

4. **Risk Management**
   - Average win vs average loss ratios
   - Stop loss effectiveness
   - Position sizing recommendations

## Limitations

### Sandbox Environment
- Historical data is limited
- Synthetic data is modeled but not real
- Best used for relative performance comparison

### When Connected to Live API
- Full historical data available
- Real market prices and Greeks
- Accurate backtest results

## Important Notes

1. **Respects Risk Rules**: All backtests respect your configured risk parameters (max loss limits, position sizing, etc.)

2. **No Guarantee**: Past performance does not guarantee future results. Use backtesting as one tool in your analysis.

3. **Suggestions Only**: The analyzer will suggest improvements but NEVER automatically change core risk rules without your approval.

4. **Learning System**: The backtesting system helps you understand which strategies work best under different conditions, allowing you to refine your approach over time.

## Integration with Trading Bot

The backtesting system is fully integrated with your existing trading bot:

- Uses the same `StrategyEngine` for consistent strategy evaluation
- Respects all `RiskParameters` configured in your bot
- Uses the same `RiskManager` for position sizing and limits
- Compatible with all implemented strategies

This ensures that backtest results accurately reflect how your actual trading bot would behave.
