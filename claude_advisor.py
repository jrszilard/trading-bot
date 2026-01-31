#!/usr/bin/env python3
"""
Claude AI Trade Advisor for Tastytrade Trading Bot

An institutional-grade AI advisor that combines quantitative analysis with
market intelligence to identify high-probability trading opportunities.

Key Capabilities:
- Market regime identification and trend analysis
- Options flow analysis (Unusual Whales integration ready)
- Portfolio optimization with risk-adjusted position sizing
- Multi-timeframe technical and volatility analysis
- Institutional-style trade evaluation and execution timing

Author: Trading Bot
License: MIT
"""

import json
import logging
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification for strategy selection."""
    RISK_ON = "risk_on"           # Bullish, low VIX, trending up
    RISK_OFF = "risk_off"         # Bearish, high VIX, flight to safety
    NEUTRAL = "neutral"           # Sideways, mean-reverting
    HIGH_VOLATILITY = "high_vol"  # Elevated VIX, large swings
    LOW_VOLATILITY = "low_vol"    # Compressed VIX, grinding moves
    TRANSITIONAL = "transitional" # Regime change in progress


class FlowSentiment(Enum):
    """Options flow sentiment from institutional activity."""
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"
    MIXED = "mixed"  # Conflicting signals


class TrendStrength(Enum):
    """Trend strength classification."""
    STRONG_UPTREND = "strong_uptrend"
    MODERATE_UPTREND = "moderate_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    MODERATE_DOWNTREND = "moderate_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


# Complete list of strategies the engine can evaluate
SUPPORTED_STRATEGIES = [
    # Premium Selling - Defined Risk
    "short_put_spread",
    "short_call_spread",
    "iron_condor",
    # Premium Selling - Undefined Risk
    "short_strangle",
    "short_straddle",
    "jade_lizard",
    "short_put",
    "short_call",
    # Volatility Expansion
    "long_strangle",
    "long_straddle",
    "calendar_spread",
    "diagonal_spread",
    # Stock Strategies
    "long_stock",
    "short_stock",
    # Income Strategies
    "covered_call",
    "cash_secured_put",
    # Synthetic Positions
    "synthetic_long",
    "synthetic_short",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketIntelligence:
    """
    Aggregated market intelligence from multiple data sources.
    Designed for Unusual Whales API integration.
    """
    # Options Flow Data
    flow_sentiment: FlowSentiment = FlowSentiment.NEUTRAL
    call_volume: int = 0
    put_volume: int = 0
    put_call_ratio: float = 1.0
    unusual_activity_score: float = 0.0  # 0-100, higher = more unusual
    large_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Dark Pool / Institutional Activity
    dark_pool_volume: int = 0
    dark_pool_sentiment: str = "neutral"
    block_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Sector / Market Context
    sector_performance: float = 0.0  # Sector relative strength
    market_correlation: float = 0.0  # Correlation to SPY
    beta: float = 1.0

    # News / Catalyst Tracking
    upcoming_catalysts: List[str] = field(default_factory=list)
    recent_news_sentiment: float = 0.0  # -1 to 1
    analyst_rating_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Insider Activity
    insider_buying: float = 0.0
    insider_selling: float = 0.0
    insider_net_activity: str = "neutral"

    # Technical Signals
    trend_strength: TrendStrength = TrendStrength.SIDEWAYS
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

    # Data freshness
    last_updated: Optional[datetime] = None
    data_source: str = "tastytrade"  # or "unusual_whales" when integrated


@dataclass
class ClaudeAnalysis:
    """
    Comprehensive trade analysis from the AI advisor.
    Includes institutional-grade metrics and risk assessment.
    """
    # Core Recommendation
    recommendation: str  # TRADE, WAIT, SKIP, HEDGE
    confidence: int  # 1-10

    # Strategy Details
    suggested_strategy: str
    suggested_delta: float
    position_size_recommendation: float = 1.0  # Multiplier (0.5 = half size, 2.0 = double)

    # Market Context
    market_regime: str = "neutral"
    trend_alignment: str = "neutral"  # with_trend, counter_trend, neutral

    # Risk Assessment
    key_risks: List[str] = field(default_factory=list)
    risk_reward_ratio: float = 0.0
    max_drawdown_estimate: float = 0.0

    # Timing and Execution
    entry_timing: str = "immediate"  # immediate, wait_for_dip, scale_in
    optimal_entry_conditions: List[str] = field(default_factory=list)
    exit_strategy: str = ""

    # Conviction Metrics
    thesis: str = ""  # Core investment thesis
    rationale: str = ""
    supporting_factors: List[str] = field(default_factory=list)
    opposing_factors: List[str] = field(default_factory=list)

    # Flow Analysis (Unusual Whales integration)
    flow_alignment: str = "unknown"  # aligned, contrary, neutral, unknown
    institutional_activity: str = ""

    # Raw response for debugging
    raw_response: Optional[str] = None


@dataclass
class ManagementAnalysis:
    """Enhanced position management analysis with institutional perspective."""
    action: str  # CLOSE, ROLL, HOLD, MONITOR, HEDGE, ADD
    confidence: int  # 1-10
    reasoning: str

    # Specific guidance
    roll_suggestion: Optional[str] = None
    hedge_suggestion: Optional[str] = None
    target_adjustment: Optional[str] = None

    # Urgency and timing
    urgency: str = "normal"  # low, normal, high, critical
    time_horizon: str = ""  # immediate, today, this_week, expiration

    # Risk factors
    current_risk_level: str = "moderate"  # low, moderate, high, critical
    risk_trend: str = "stable"  # improving, stable, deteriorating


@dataclass
class PortfolioInsight:
    """Strategic portfolio-level insights and recommendations."""
    # Overall Assessment
    health_score: int = 0  # 1-100
    risk_score: int = 0  # 1-100
    efficiency_score: int = 0  # 1-100 (risk-adjusted returns)

    # Directional Exposure
    net_delta_assessment: str = ""
    recommended_delta_adjustment: float = 0.0

    # Volatility Exposure
    vega_assessment: str = ""
    volatility_outlook: str = ""

    # Diversification
    concentration_warnings: List[str] = field(default_factory=list)
    sector_imbalances: List[str] = field(default_factory=list)
    correlation_risks: List[str] = field(default_factory=list)

    # Opportunities
    rebalancing_suggestions: List[str] = field(default_factory=list)
    hedging_opportunities: List[str] = field(default_factory=list)
    alpha_opportunities: List[str] = field(default_factory=list)

    # Action Items
    immediate_actions: List[str] = field(default_factory=list)
    watchlist_additions: List[str] = field(default_factory=list)


# =============================================================================
# MAIN ADVISOR CLASS
# =============================================================================

class ClaudeTradeAdvisor:
    """
    Institutional-Grade AI Trade Advisor

    Combines quantitative analysis with market intelligence to identify
    high-probability trading opportunities while managing risk.

    Design Philosophy:
    - Think like an investment banker evaluating deal flow
    - Act like a hedge fund trader timing entries
    - Risk manage like an institutional portfolio manager
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        temperature: float = 0.2
    ):
        """
        Initialize the Trade Advisor.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Response temperature (lower = more focused/consistent)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

        # Cache for market intelligence
        self._market_intel_cache: Dict[str, MarketIntelligence] = {}
        self._cache_ttl_minutes = 15

        if not self.api_key:
            logger.warning("No Anthropic API key found. Claude advisor will be disabled.")

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None and self.api_key:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                return None
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if Claude advisor is available."""
        return self.client is not None

    def _call_claude(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Make a call to Claude API."""
        if not self.is_available:
            return None

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from Claude's response."""
        if not response:
            return None

        # Try to extract JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

        logger.warning("Could not parse JSON from Claude response")
        return None

    # =========================================================================
    # MARKET INTELLIGENCE
    # =========================================================================

    async def get_market_intelligence(
        self,
        symbol: str,
        tastytrade_data: Optional[Dict[str, Any]] = None,
        unusual_whales_data: Optional[Dict[str, Any]] = None
    ) -> MarketIntelligence:
        """
        Aggregate market intelligence from multiple sources.

        Args:
            symbol: Ticker symbol
            tastytrade_data: Data from Tastytrade API (IV, options chains, etc.)
            unusual_whales_data: Data from Unusual Whales API (flow, dark pool, etc.)

        Returns:
            MarketIntelligence object with aggregated data
        """
        intel = MarketIntelligence()
        intel.last_updated = datetime.now()

        # Process Tastytrade data
        if tastytrade_data:
            intel.data_source = "tastytrade"
            # Extract relevant metrics from tastytrade
            if 'put_call_ratio' in tastytrade_data:
                intel.put_call_ratio = tastytrade_data['put_call_ratio']
            if 'call_volume' in tastytrade_data:
                intel.call_volume = tastytrade_data['call_volume']
            if 'put_volume' in tastytrade_data:
                intel.put_volume = tastytrade_data['put_volume']

            # Derive flow sentiment from put/call ratio
            pcr = intel.put_call_ratio
            if pcr < 0.5:
                intel.flow_sentiment = FlowSentiment.STRONGLY_BULLISH
            elif pcr < 0.7:
                intel.flow_sentiment = FlowSentiment.BULLISH
            elif pcr < 1.0:
                intel.flow_sentiment = FlowSentiment.NEUTRAL
            elif pcr < 1.3:
                intel.flow_sentiment = FlowSentiment.BEARISH
            else:
                intel.flow_sentiment = FlowSentiment.STRONGLY_BEARISH

        # Process Unusual Whales data (future integration)
        if unusual_whales_data:
            intel.data_source = "unusual_whales"

            # Options flow
            if 'flow_sentiment' in unusual_whales_data:
                sentiment_map = {
                    'bullish': FlowSentiment.BULLISH,
                    'strongly_bullish': FlowSentiment.STRONGLY_BULLISH,
                    'bearish': FlowSentiment.BEARISH,
                    'strongly_bearish': FlowSentiment.STRONGLY_BEARISH,
                    'neutral': FlowSentiment.NEUTRAL,
                    'mixed': FlowSentiment.MIXED
                }
                intel.flow_sentiment = sentiment_map.get(
                    unusual_whales_data['flow_sentiment'],
                    FlowSentiment.NEUTRAL
                )

            # Unusual activity score
            if 'unusual_score' in unusual_whales_data:
                intel.unusual_activity_score = unusual_whales_data['unusual_score']

            # Large trades / sweeps
            if 'large_trades' in unusual_whales_data:
                intel.large_trades = unusual_whales_data['large_trades']

            # Dark pool data
            if 'dark_pool' in unusual_whales_data:
                dp = unusual_whales_data['dark_pool']
                intel.dark_pool_volume = dp.get('volume', 0)
                intel.dark_pool_sentiment = dp.get('sentiment', 'neutral')
                intel.block_trades = dp.get('blocks', [])

            # Sector and market context
            if 'sector_performance' in unusual_whales_data:
                intel.sector_performance = unusual_whales_data['sector_performance']
            if 'beta' in unusual_whales_data:
                intel.beta = unusual_whales_data['beta']

            # Catalysts and news
            if 'catalysts' in unusual_whales_data:
                intel.upcoming_catalysts = unusual_whales_data['catalysts']
            if 'news_sentiment' in unusual_whales_data:
                intel.recent_news_sentiment = unusual_whales_data['news_sentiment']

            # Insider activity
            if 'insider' in unusual_whales_data:
                insider = unusual_whales_data['insider']
                intel.insider_buying = insider.get('buying', 0)
                intel.insider_selling = insider.get('selling', 0)
                if intel.insider_buying > intel.insider_selling * 1.5:
                    intel.insider_net_activity = "bullish"
                elif intel.insider_selling > intel.insider_buying * 1.5:
                    intel.insider_net_activity = "bearish"
                else:
                    intel.insider_net_activity = "neutral"

            # Technical levels
            if 'technicals' in unusual_whales_data:
                tech = unusual_whales_data['technicals']
                intel.support_levels = tech.get('support', [])
                intel.resistance_levels = tech.get('resistance', [])

                # Trend classification
                trend = tech.get('trend', 'sideways')
                trend_map = {
                    'strong_up': TrendStrength.STRONG_UPTREND,
                    'up': TrendStrength.MODERATE_UPTREND,
                    'weak_up': TrendStrength.WEAK_UPTREND,
                    'sideways': TrendStrength.SIDEWAYS,
                    'weak_down': TrendStrength.WEAK_DOWNTREND,
                    'down': TrendStrength.MODERATE_DOWNTREND,
                    'strong_down': TrendStrength.STRONG_DOWNTREND
                }
                intel.trend_strength = trend_map.get(trend, TrendStrength.SIDEWAYS)

        # Cache the intelligence
        self._market_intel_cache[symbol] = intel

        return intel

    # =========================================================================
    # OPPORTUNITY ANALYSIS
    # =========================================================================

    async def analyze_opportunity(
        self,
        symbol: str,
        iv_rank: float,
        current_price: float,
        option_chain_summary: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_intel: Optional[MarketIntelligence] = None,
        market_context: Optional[str] = None
    ) -> Optional[ClaudeAnalysis]:
        """
        Analyze a potential trade opportunity with institutional rigor.

        Evaluates the opportunity through multiple lenses:
        - Quantitative: IV rank, Greeks, probability of profit
        - Technical: Trend alignment, support/resistance
        - Flow: Institutional activity, unusual options volume
        - Fundamental: Catalysts, sector dynamics
        - Portfolio: Fit with existing positions, correlation

        Args:
            symbol: Underlying symbol
            iv_rank: Current IV Rank percentage
            current_price: Current underlying price
            option_chain_summary: Summary of relevant options
            portfolio_state: Current portfolio metrics
            market_intel: Aggregated market intelligence
            market_context: Additional context string

        Returns:
            ClaudeAnalysis with comprehensive recommendation
        """
        if not self.is_available:
            logger.debug("Claude not available, skipping opportunity analysis")
            return None

        # Build market intelligence context
        intel_context = ""
        if market_intel:
            intel_context = f"""
## Market Intelligence
- Options Flow Sentiment: {market_intel.flow_sentiment.value}
- Put/Call Ratio: {market_intel.put_call_ratio:.2f}
- Unusual Activity Score: {market_intel.unusual_activity_score}/100
- Trend Strength: {market_intel.trend_strength.value}
- Dark Pool Sentiment: {market_intel.dark_pool_sentiment}
- Insider Activity: {market_intel.insider_net_activity}
- News Sentiment: {market_intel.recent_news_sentiment:.2f} (-1 to 1 scale)
- Upcoming Catalysts: {', '.join(market_intel.upcoming_catalysts) if market_intel.upcoming_catalysts else 'None identified'}
- Key Support Levels: {', '.join(f'${x:.2f}' for x in market_intel.support_levels[:3]) if market_intel.support_levels else 'N/A'}
- Key Resistance Levels: {', '.join(f'${x:.2f}' for x in market_intel.resistance_levels[:3]) if market_intel.resistance_levels else 'N/A'}
- Data Source: {market_intel.data_source}
"""

        system_prompt = """You are a senior portfolio manager at a quantitative hedge fund, combining
the analytical rigor of an investment banker with the market timing skills of an experienced trader.

Your role is to evaluate trade opportunities through multiple lenses:

1. QUANTITATIVE FRAMEWORK
   - IV Rank analysis: Premium selling favored >30, debit strategies <25
   - Probability of profit assessment based on delta and strategy structure
   - Risk/reward optimization with Kelly Criterion principles
   - Position sizing based on portfolio heat and correlation

2. MARKET REGIME IDENTIFICATION
   - Risk-on vs Risk-off environment
   - Volatility regime (expansion, contraction, elevated, compressed)
   - Trend strength and direction
   - Mean reversion vs momentum conditions

3. FLOW ANALYSIS (When Available)
   - Institutional vs retail flow patterns
   - Unusual options activity signals
   - Dark pool positioning
   - Put/call ratio context

4. CATALYST ASSESSMENT
   - Upcoming earnings, dividends, events
   - Sector rotation dynamics
   - Macro factors affecting the position

5. PORTFOLIO INTEGRATION
   - Correlation with existing positions
   - Delta/vega/theta budget management
   - Concentration risk assessment
   - Hedging considerations

STRATEGY UNIVERSE (evaluate the most appropriate):
Premium Selling (High IV): short_put_spread, short_call_spread, iron_condor,
                          short_strangle, short_straddle, jade_lizard,
                          short_put, short_call
Volatility Expansion (Low IV): long_strangle, long_straddle, calendar_spread,
                               diagonal_spread
Directional: long_stock, short_stock, synthetic_long, synthetic_short
Income: covered_call, cash_secured_put

Output your analysis as structured JSON for systematic processing."""

        user_prompt = f"""Analyze this trade opportunity with institutional rigor:

## Underlying Analysis
- Symbol: {symbol}
- Current Price: ${current_price:.2f}
- IV Rank: {iv_rank:.1f}%
- IV Environment: {"Elevated (favor premium selling)" if iv_rank > 30 else "Low (consider debit strategies or wait)"}

## Portfolio Context
- Net Liquidating Value: ${portfolio_state.get('net_liquidating_value', 0):.2f}
- Available Buying Power: ${portfolio_state.get('buying_power', 0):.2f}
- Portfolio Delta: {portfolio_state.get('portfolio_delta', 0):.1f}
- Portfolio Theta: {portfolio_state.get('portfolio_theta', 0):.2f}
- Open Positions: {portfolio_state.get('open_positions', 0)}
- Daily P&L: ${portfolio_state.get('daily_pnl', 0):.2f}
- Buying Power Utilization: {portfolio_state.get('buying_power_used_pct', 0):.1f}%

## Option Chain Data
{json.dumps(option_chain_summary, indent=2, default=str)}
{intel_context}
{f"## Additional Context{chr(10)}{market_context}" if market_context else ""}

## Required Analysis Output
Provide your institutional-grade analysis as JSON:
{{
  "recommendation": "TRADE" | "WAIT" | "SKIP" | "HEDGE",
  "confidence": 1-10,
  "suggested_strategy": "<strategy_from_universe>",
  "suggested_delta": 0.10-0.50,
  "position_size_recommendation": 0.5-2.0,
  "market_regime": "risk_on" | "risk_off" | "neutral" | "high_vol" | "low_vol" | "transitional",
  "trend_alignment": "with_trend" | "counter_trend" | "neutral",
  "risk_reward_ratio": <float>,
  "key_risks": ["risk1", "risk2", "risk3"],
  "entry_timing": "immediate" | "wait_for_dip" | "scale_in" | "wait_for_confirmation",
  "optimal_entry_conditions": ["condition1", "condition2"],
  "exit_strategy": "Take profit at X%, stop at Y%",
  "thesis": "1-2 sentence core investment thesis",
  "rationale": "2-3 sentence detailed explanation",
  "supporting_factors": ["factor1", "factor2"],
  "opposing_factors": ["factor1", "factor2"],
  "flow_alignment": "aligned" | "contrary" | "neutral" | "unknown",
  "institutional_activity": "Brief description of notable flow"
}}

Consider:
1. Does the IV environment support this strategy type?
2. How does this trade fit the current market regime?
3. What is the flow telling us about institutional sentiment?
4. How does this impact overall portfolio Greeks and risk?
5. What catalysts could accelerate or derail this thesis?
6. Is the timing optimal or should we wait for better entry?"""

        response = self._call_claude(system_prompt, user_prompt)
        parsed = self._parse_json_response(response)

        if not parsed:
            return None

        try:
            return ClaudeAnalysis(
                recommendation=parsed.get("recommendation", "SKIP"),
                confidence=int(parsed.get("confidence", 5)),
                suggested_strategy=parsed.get("suggested_strategy", "iron_condor"),
                suggested_delta=float(parsed.get("suggested_delta", 0.30)),
                position_size_recommendation=float(parsed.get("position_size_recommendation", 1.0)),
                market_regime=parsed.get("market_regime", "neutral"),
                trend_alignment=parsed.get("trend_alignment", "neutral"),
                risk_reward_ratio=float(parsed.get("risk_reward_ratio", 0.0)),
                key_risks=parsed.get("key_risks", []),
                entry_timing=parsed.get("entry_timing", "immediate"),
                optimal_entry_conditions=parsed.get("optimal_entry_conditions", []),
                exit_strategy=parsed.get("exit_strategy", ""),
                thesis=parsed.get("thesis", ""),
                rationale=parsed.get("rationale", ""),
                supporting_factors=parsed.get("supporting_factors", []),
                opposing_factors=parsed.get("opposing_factors", []),
                flow_alignment=parsed.get("flow_alignment", "unknown"),
                institutional_activity=parsed.get("institutional_activity", ""),
                raw_response=response
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing Claude analysis: {e}")
            return None

    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================

    async def evaluate_management_action(
        self,
        position: Dict[str, Any],
        current_pnl_pct: float,
        dte_remaining: int,
        is_tested: bool,
        portfolio_context: Dict[str, Any],
        market_intel: Optional[MarketIntelligence] = None,
        market_conditions: Optional[str] = None
    ) -> Optional[ManagementAnalysis]:
        """
        Evaluate position management with institutional discipline.

        Framework:
        - Profit taking: Mechanical at 50%, but adjust for trend/flow
        - Loss management: 2x credit max, but consider roll opportunities
        - Time management: 21 DTE threshold for gamma risk
        - Defense: When to hedge vs close vs roll

        Args:
            position: Position details
            current_pnl_pct: Current P&L as percentage of max profit
            dte_remaining: Days to expiration
            is_tested: Whether strike has been breached
            portfolio_context: Current portfolio state
            market_intel: Market intelligence data
            market_conditions: Additional context

        Returns:
            ManagementAnalysis with action recommendation
        """
        if not self.is_available:
            return None

        intel_context = ""
        if market_intel:
            intel_context = f"""
## Current Market Intelligence for {position.get('underlying', 'Unknown')}
- Flow Sentiment: {market_intel.flow_sentiment.value}
- Trend: {market_intel.trend_strength.value}
- Unusual Activity: {market_intel.unusual_activity_score}/100
"""

        system_prompt = """You are a risk manager at a systematic trading firm, responsible for
position management decisions that protect capital while maximizing risk-adjusted returns.

Position Management Framework:

1. PROFIT TAKING DISCIPLINE
   - Standard target: 50% of max profit
   - Accelerate exit if: Trend reversing, flow turning negative, catalyst approaching
   - Extend hold if: Strong trend continuation, supportive flow, theta still rich

2. LOSS MANAGEMENT PROTOCOL
   - Hard stop: 200% of credit received (2x loss)
   - Consider defense at 100% of credit
   - Evaluate roll feasibility before closing for loss

3. TIME-BASED MANAGEMENT
   - 21 DTE: Gamma inflection point, consider rolling or closing
   - Theta decay accelerates <14 DTE but gamma risk spikes
   - Never hold undefined risk into expiration week

4. DEFENSE STRATEGIES
   - Roll: Same strategy, new expiration (for credit if possible)
   - Invert: Close tested side, open new position opposite side
   - Hedge: Add protection (buy wing, stock hedge)
   - Reduce: Close partial position

5. CONTEXT INTEGRATION
   - Market regime shifts may warrant early exit
   - Institutional flow may signal impending move
   - Correlation events (market selloff) may require portfolio-level action

Output structured JSON for systematic execution."""

        user_prompt = f"""Evaluate this position for management action:

## Position Details
- Symbol: {position.get('symbol', 'Unknown')}
- Underlying: {position.get('underlying', 'Unknown')}
- Strategy: {position.get('strategy', 'Unknown')}
- Type: {position.get('option_type', 'Unknown')}
- Strike(s): {position.get('strikes', position.get('strike', 'N/A'))}
- Quantity: {position.get('quantity', 0)}
- Entry Credit: ${position.get('entry_credit', 0):.2f}
- Current Value: ${position.get('current_value', 0):.2f}

## Performance Metrics
- P&L: {current_pnl_pct:.1%} of max profit
- DTE Remaining: {dte_remaining} days
- Position Tested: {"⚠️ YES - STRIKE BREACHED" if is_tested else "No - Position safe"}
- Days in Trade: {position.get('days_held', 'N/A')}

## Portfolio Context
- Total Positions: {portfolio_context.get('open_positions', 0)}
- Portfolio Delta: {portfolio_context.get('portfolio_delta', 0):.1f}
- Portfolio Theta: {portfolio_context.get('portfolio_theta', 0):.2f}
- Daily P&L: ${portfolio_context.get('daily_pnl', 0):.2f}
- Buying Power Available: ${portfolio_context.get('buying_power', 0):.2f}
{intel_context}
{f"## Market Conditions{chr(10)}{market_conditions}" if market_conditions else ""}

## Management Decision Required
Provide your recommendation as JSON:
{{
  "action": "CLOSE" | "ROLL" | "HOLD" | "MONITOR" | "HEDGE" | "ADD",
  "confidence": 1-10,
  "reasoning": "2-3 sentence explanation of the decision",
  "roll_suggestion": "If rolling, specify: direction, expiration, strikes" or null,
  "hedge_suggestion": "If hedging, specify the hedge structure" or null,
  "target_adjustment": "If adjusting profit target, specify new target" or null,
  "urgency": "low" | "normal" | "high" | "critical",
  "time_horizon": "immediate" | "today" | "this_week" | "expiration",
  "current_risk_level": "low" | "moderate" | "high" | "critical",
  "risk_trend": "improving" | "stable" | "deteriorating"
}}

Decision Framework:
1. Is this at profit target (≥50%)? → Consider CLOSE
2. Is this at loss threshold (≥200% credit)? → CLOSE or ROLL
3. Is DTE < 21? → Evaluate ROLL for gamma management
4. Is position tested? → CLOSE, ROLL, or HEDGE based on outlook
5. Is flow/trend supportive? → May extend hold or ADD
6. Is there correlation risk? → May need portfolio-level HEDGE"""

        response = self._call_claude(system_prompt, user_prompt)
        parsed = self._parse_json_response(response)

        if not parsed:
            return None

        try:
            return ManagementAnalysis(
                action=parsed.get("action", "MONITOR"),
                confidence=int(parsed.get("confidence", 5)),
                reasoning=parsed.get("reasoning", ""),
                roll_suggestion=parsed.get("roll_suggestion"),
                hedge_suggestion=parsed.get("hedge_suggestion"),
                target_adjustment=parsed.get("target_adjustment"),
                urgency=parsed.get("urgency", "normal"),
                time_horizon=parsed.get("time_horizon", ""),
                current_risk_level=parsed.get("current_risk_level", "moderate"),
                risk_trend=parsed.get("risk_trend", "stable")
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing management analysis: {e}")
            return None

    # =========================================================================
    # TRADE APPROVAL SUMMARY
    # =========================================================================

    async def generate_approval_summary(
        self,
        proposal: Any,  # TradeProposal
        portfolio_context: Dict[str, Any],
        market_intel: Optional[MarketIntelligence] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate an institutional-quality trade summary for approval.

        Format designed for rapid decision-making by presenting:
        - Trade structure and key metrics
        - Risk/reward profile
        - Portfolio impact
        - Market context alignment

        Args:
            proposal: TradeProposal object
            portfolio_context: Current portfolio state
            market_intel: Market intelligence data
            additional_context: Additional context

        Returns:
            Formatted summary string for review
        """
        if not self.is_available:
            return self._generate_basic_summary(proposal)

        intel_section = ""
        if market_intel:
            intel_section = f"""
Flow Analysis:
  Sentiment: {market_intel.flow_sentiment.value}
  Trend: {market_intel.trend_strength.value}
  Unusual Activity: {market_intel.unusual_activity_score}/100"""

        system_prompt = """You are presenting a trade proposal to a portfolio committee.
Your summary must be:
- Concise but complete (suitable for terminal display)
- Structured for rapid decision-making
- Honest about both opportunity and risk
- Quantitative where possible

Format for monospace terminal display:
- No markdown headers (use CAPS or === dividers)
- Aligned columns where appropriate
- Bullet points for lists
- Keep total length under 40 lines"""

        legs_desc = []
        for leg in proposal.legs:
            action = leg.get('action', 'SELL')
            qty = leg.get('quantity', 1)
            opt_type = leg.get('option_type', 'PUT')
            strike = leg.get('strike', 0)
            exp = leg.get('expiration', 'Unknown')
            legs_desc.append(f"  {action} {qty}x {opt_type} ${strike} exp {exp}")

        user_prompt = f"""Create a trade approval summary:

## Trade Structure
- Strategy: {proposal.strategy.value}
- Underlying: {proposal.underlying_symbol} @ ${getattr(proposal, 'underlying_price', 'N/A')}
- IV Rank: {proposal.iv_rank:.1f}%
- DTE: {proposal.dte} days

## Economics
- Expected Credit: ${proposal.expected_credit:.2f}
- Max Loss: ${proposal.max_loss:.2f}
- Risk/Reward: {(float(proposal.expected_credit) / float(proposal.max_loss) * 100):.1f}%
- Probability of Profit: {proposal.probability_of_profit:.1%}

## Greeks
- Delta: {proposal.delta}
- Theta: {proposal.theta}
- Vega: {getattr(proposal, 'vega', 'N/A')}

## Legs
{chr(10).join(legs_desc)}

## Portfolio Impact
- Current Positions: {portfolio_context.get('open_positions', 0)}
- Portfolio Delta Before: {portfolio_context.get('portfolio_delta', 0):.1f}
- Portfolio Delta After: {float(portfolio_context.get('portfolio_delta', 0)) + float(proposal.delta):.1f}
- Buying Power Required: ~${float(proposal.max_loss):.2f}
- Buying Power Available: ${portfolio_context.get('buying_power', 0):.2f}
{intel_section}
{f"{chr(10)}Additional Context: {additional_context}" if additional_context else ""}

Create a terminal-formatted summary with:
1. TRADE OVERVIEW (2 lines max)
2. KEY METRICS (aligned, easy to scan)
3. RISK FACTORS (bullet points)
4. RECOMMENDATION (1 line verdict)"""

        response = self._call_claude(system_prompt, user_prompt)

        if response:
            return response

        return self._generate_basic_summary(proposal)

    def _generate_basic_summary(self, proposal: Any) -> str:
        """Generate a basic summary when Claude is unavailable."""
        return f"""═══ TRADE SUMMARY (AI Analysis Unavailable) ═══
Strategy:  {proposal.strategy.value}
Symbol:    {proposal.underlying_symbol}
Credit:    ${proposal.expected_credit:.2f}
Max Loss:  ${proposal.max_loss:.2f}
IV Rank:   {proposal.iv_rank:.1f}%
DTE:       {proposal.dte} days
P.O.P:     {proposal.probability_of_profit:.1%}
Delta:     {proposal.delta}
═══════════════════════════════════════════════"""

    # =========================================================================
    # PORTFOLIO ANALYSIS
    # =========================================================================

    async def analyze_portfolio_health(
        self,
        portfolio_state: Dict[str, Any],
        positions: List[Dict[str, Any]],
        recent_trades: List[Dict[str, Any]],
        market_regime: Optional[str] = None
    ) -> PortfolioInsight:
        """
        Comprehensive portfolio health analysis with actionable insights.

        Evaluates:
        - Risk metrics and exposure analysis
        - Diversification and concentration
        - Performance attribution
        - Optimization opportunities

        Args:
            portfolio_state: Current portfolio metrics
            positions: List of open positions
            recent_trades: Recent trade history
            market_regime: Current market regime assessment

        Returns:
            PortfolioInsight with comprehensive analysis
        """
        if not self.is_available:
            return PortfolioInsight(
                health_score=50,
                immediate_actions=["Enable Claude advisor for detailed analysis"]
            )

        system_prompt = """You are a portfolio risk analyst at an institutional asset manager.
Provide a comprehensive portfolio health assessment covering:

1. RISK METRICS
   - Delta exposure relative to account size
   - Theta/day as percentage of account
   - Concentration risk (single name, sector)
   - Correlation risk

2. PERFORMANCE ANALYSIS
   - Win rate and average win/loss
   - Risk-adjusted returns
   - Drawdown analysis

3. OPTIMIZATION OPPORTUNITIES
   - Rebalancing recommendations
   - Hedging opportunities
   - Alpha generation ideas

4. ACTION ITEMS
   - Immediate priorities
   - Watchlist candidates
   - Risk reduction needs

Output as structured JSON for systematic processing."""

        positions_summary = []
        for pos in positions[:15]:
            positions_summary.append(
                f"- {pos.get('symbol', 'Unknown')}: {pos.get('quantity', 0)} contracts, "
                f"P&L: {pos.get('pnl_pct', 0):.1%}, DTE: {pos.get('dte', 0)}, "
                f"Delta: {pos.get('delta', 0)}"
            )

        win_count = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        total_trades = max(len(recent_trades), 1)
        win_rate = win_count / total_trades

        user_prompt = f"""Analyze this options portfolio:

## Account Metrics
- Net Liquidating Value: ${portfolio_state.get('net_liquidating_value', 0):.2f}
- Buying Power: ${portfolio_state.get('buying_power', 0):.2f}
- Buying Power Utilization: {portfolio_state.get('buying_power_used_pct', 0):.1f}%
- Maintenance Requirement: ${portfolio_state.get('maintenance_requirement', 0):.2f}

## Greek Exposure
- Portfolio Delta: {portfolio_state.get('portfolio_delta', 0):.1f}
- Portfolio Theta: ${portfolio_state.get('portfolio_theta', 0):.2f}/day
- Portfolio Vega: {portfolio_state.get('portfolio_vega', 'N/A')}
- Portfolio Gamma: {portfolio_state.get('portfolio_gamma', 'N/A')}

## Performance
- Daily P&L: ${portfolio_state.get('daily_pnl', 0):.2f}
- Weekly P&L: ${portfolio_state.get('weekly_pnl', 0):.2f}
- YTD P&L: ${portfolio_state.get('ytd_pnl', 0):.2f}

## Position Summary ({len(positions)} open)
{chr(10).join(positions_summary) if positions_summary else "No open positions"}

## Recent Activity
- Trades (7 days): {len(recent_trades)}
- Win Rate: {win_rate:.1%}
- Avg Winner: ${sum(t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) > 0) / max(win_count, 1):.2f}
- Avg Loser: ${sum(t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) < 0) / max(total_trades - win_count, 1):.2f}

{f"## Market Regime: {market_regime}" if market_regime else ""}

Provide comprehensive analysis as JSON:
{{
  "health_score": 1-100,
  "risk_score": 1-100,
  "efficiency_score": 1-100,
  "net_delta_assessment": "description of delta exposure",
  "recommended_delta_adjustment": <float>,
  "vega_assessment": "description of vega exposure",
  "volatility_outlook": "description",
  "concentration_warnings": ["warning1", "warning2"],
  "sector_imbalances": ["imbalance1"],
  "correlation_risks": ["risk1"],
  "rebalancing_suggestions": ["suggestion1", "suggestion2"],
  "hedging_opportunities": ["opportunity1"],
  "alpha_opportunities": ["opportunity1"],
  "immediate_actions": ["action1", "action2"],
  "watchlist_additions": ["symbol1", "symbol2"]
}}"""

        response = self._call_claude(system_prompt, user_prompt)
        parsed = self._parse_json_response(response)

        if not parsed:
            return PortfolioInsight(health_score=50)

        try:
            return PortfolioInsight(
                health_score=int(parsed.get("health_score", 50)),
                risk_score=int(parsed.get("risk_score", 50)),
                efficiency_score=int(parsed.get("efficiency_score", 50)),
                net_delta_assessment=parsed.get("net_delta_assessment", ""),
                recommended_delta_adjustment=float(parsed.get("recommended_delta_adjustment", 0)),
                vega_assessment=parsed.get("vega_assessment", ""),
                volatility_outlook=parsed.get("volatility_outlook", ""),
                concentration_warnings=parsed.get("concentration_warnings", []),
                sector_imbalances=parsed.get("sector_imbalances", []),
                correlation_risks=parsed.get("correlation_risks", []),
                rebalancing_suggestions=parsed.get("rebalancing_suggestions", []),
                hedging_opportunities=parsed.get("hedging_opportunities", []),
                alpha_opportunities=parsed.get("alpha_opportunities", []),
                immediate_actions=parsed.get("immediate_actions", []),
                watchlist_additions=parsed.get("watchlist_additions", [])
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing portfolio analysis: {e}")
            return PortfolioInsight(health_score=50)

    # =========================================================================
    # INTENT PARSING
    # =========================================================================

    async def parse_intent(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context_summary: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse user intent from natural language.

        Uses Claude for sophisticated NLU to extract:
        - Intent type and confidence
        - Entities (symbols, strategies, trade IDs)
        - Contextual understanding from conversation

        Args:
            user_message: The user's natural language input
            conversation_history: Recent conversation for context
            context_summary: Current context state

        Returns:
            Dict with parsed intent information
        """
        if not self.is_available:
            return None

        system_prompt = """You are an intent parser for an institutional trading system.
Extract precise intents and entities from natural language trading commands.

INTENT TYPES:
- scan_opportunities: Find new trading opportunities
- show_pending: View pending trades awaiting approval
- approve_trade: Approve a specific trade (needs trade_id)
- reject_trade: Reject a specific trade (needs trade_id)
- execute_trade: Execute an approved trade
- create_trade: User wants to create a specific trade (extract symbol, strategy)
- select_option: User is selecting from presented options (extract selection)
- get_portfolio: Portfolio overview (delta, theta, P&L)
- get_positions: View current positions
- get_buying_power: Buying power information
- get_pnl: P&L information
- manage_positions: Check positions for management actions
- close_position: Close a specific position
- roll_position: Roll a position to new expiration
- position_analysis: Analyze a specific position
- get_iv_rank: IV rank for a symbol
- research_trade: Research a trade/strategy
- market_analysis: Market overview and regime
- get_risk_params: View risk parameters
- confirm_yes: Confirming/approving something
- confirm_no: Declining/rejecting something
- help: User needs help
- general_chat: General trading discussion

STRATEGIES (normalize to these):
Premium Selling: short_put_spread, short_call_spread, iron_condor,
                short_strangle, short_straddle, jade_lizard,
                short_put, short_call
Volatility: long_strangle, long_straddle, calendar_spread, diagonal_spread
Directional: long_stock, short_stock, synthetic_long, synthetic_short
Income: covered_call, cash_secured_put

Output ONLY valid JSON."""

        context_parts = []
        if conversation_history:
            context_parts.append("Recent conversation:")
            for turn in conversation_history[-5:]:
                role = turn.get('role', 'user')
                content = turn.get('content', '')[:200]
                context_parts.append(f"  {role}: {content}")

        if context_summary:
            context_parts.append(f"\nCurrent context:")
            if context_summary.get('current_symbol'):
                context_parts.append(f"  Active symbol: {context_summary['current_symbol']}")
            if context_summary.get('active_trade_id'):
                context_parts.append(f"  Active trade ID: {context_summary['active_trade_id']}")
            if context_summary.get('pending_action'):
                context_parts.append(f"  Pending action: {context_summary['pending_action']}")
            if context_summary.get('state'):
                context_parts.append(f"  Conversation state: {context_summary['state']}")
            if context_summary.get('presented_options'):
                context_parts.append(f"  Presented options: {context_summary['presented_options']}")

        user_prompt = f"""Parse this trading command:

"{user_message}"

{chr(10).join(context_parts) if context_parts else "No additional context."}

Extract as JSON:
{{
  "intent": "<intent_type>",
  "confidence": 0.0-1.0,
  "symbols": ["SYM1", "SYM2"],
  "trade_id": "abc123" or null,
  "strategy": "<strategy_type>" or null,
  "action": "approve/reject/execute/close/roll" or null,
  "quantity": <int> or null,
  "selection": <int> or null (if selecting from numbered options),
  "reasoning": "brief explanation"
}}"""

        response = self._call_claude(system_prompt, user_prompt)
        parsed = self._parse_json_response(response)

        if not parsed:
            return None

        try:
            return {
                "intent": parsed.get("intent", "general_chat"),
                "confidence": float(parsed.get("confidence", 0.5)),
                "symbols": parsed.get("symbols", []),
                "trade_id": parsed.get("trade_id"),
                "strategy": parsed.get("strategy"),
                "action": parsed.get("action"),
                "quantity": parsed.get("quantity"),
                "selection": parsed.get("selection"),
                "reasoning": parsed.get("reasoning", "")
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Error normalizing parsed intent: {e}")
            return None

    # =========================================================================
    # CONVERSATIONAL RESPONSE
    # =========================================================================

    async def generate_conversational_response(
        self,
        user_message: str,
        intent: str,
        context_data: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate natural conversational response.

        Maintains professional but approachable tone suitable for
        a CLI trading interface.

        Args:
            user_message: The user's message
            intent: The classified intent
            context_data: Data relevant to the response
            conversation_history: Recent conversation

        Returns:
            Natural language response
        """
        if not self.is_available:
            return self._generate_fallback_response(intent, context_data)

        system_prompt = """You are an AI trading assistant at a quantitative trading firm.

Your communication style:
- Professional but conversational (not stuffy)
- Concise (this is a CLI interface)
- Data-driven with specific numbers
- Proactive about risk awareness
- Confident but not arrogant

Trading philosophy (tastylive methodology):
- Sell premium when IV is elevated (>30 IV rank)
- 45 DTE optimal, manage at 21 DTE
- 30 delta standard, 16 delta conservative
- Take profits at 50% of max profit
- Position size: 1-5% of portfolio per trade
- Delta neutral to slightly directional overall

Response guidelines:
- Keep responses brief (2-4 sentences usually)
- Use bullet points for data
- Include relevant metrics
- Remind about approval workflow for trades
- Never make guarantees about outcomes

Format for terminal (no markdown headers, clean formatting)."""

        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{t['role']}: {t['content'][:150]}"
                for t in conversation_history[-5:]
            ])

        user_prompt = f"""Generate a response:

User: "{user_message}"
Intent: {intent}

Data:
{json.dumps(context_data, indent=2, default=str)[:2000]}

{f"Recent conversation:{chr(10)}{history_text}" if history_text else ""}

Respond naturally and helpfully."""

        response = self._call_claude(system_prompt, user_prompt)
        return response or self._generate_fallback_response(intent, context_data)

    def _generate_fallback_response(self, intent: str, context_data: Dict[str, Any]) -> str:
        """Generate a basic response when Claude is unavailable."""
        responses = {
            "get_portfolio": f"Portfolio: NLV ${context_data.get('net_liquidating_value', 'N/A')}, "
                           f"Delta {context_data.get('portfolio_delta', 'N/A')}, "
                           f"Theta ${context_data.get('portfolio_theta', 'N/A')}/day",
            "show_pending": f"{len(context_data.get('pending_trades', []))} pending trade(s).",
            "get_positions": f"{context_data.get('position_count', 0)} open position(s).",
            "get_buying_power": f"Buying Power: ${context_data.get('buying_power', 'N/A')}",
            "help": "Commands: scan, pending, approve, reject, execute, portfolio, "
                   "positions, manage, research, market",
            "scan_opportunities": "Scanning for opportunities...",
        }
        return responses.get(intent, "Processing request... (Claude unavailable for enhanced response)")

    # =========================================================================
    # MARKET CONTEXT
    # =========================================================================

    async def get_market_context(
        self,
        symbols: List[str],
        include_regime: bool = True
    ) -> str:
        """
        Generate market context summary.

        Note: This is a foundation for integration with external data sources
        like Unusual Whales, news APIs, and market data feeds.

        Args:
            symbols: List of symbols to get context for
            include_regime: Whether to include regime analysis

        Returns:
            Market context summary
        """
        # Placeholder for external data integration
        # In production, this would fetch from:
        # - Unusual Whales API (options flow, dark pool)
        # - News aggregators
        # - Economic calendar
        # - Sector ETF performance

        context = f"Market context for {', '.join(symbols[:5])}"

        if include_regime:
            context += "\n[Market regime analysis requires external data integration]"

        return context

    # =========================================================================
    # TREND AND EARLY MOVE DETECTION
    # =========================================================================

    async def detect_early_moves(
        self,
        watchlist: List[str],
        market_data: Dict[str, Any],
        flow_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect early moves and potential opportunities.

        Screens for:
        - Unusual options activity
        - Volume spikes
        - IV rank changes
        - Technical breakouts
        - Flow imbalances

        Args:
            watchlist: Symbols to monitor
            market_data: Current market data (prices, IV, volume)
            flow_data: Options flow data (from Unusual Whales when integrated)

        Returns:
            List of detected opportunities with scores and reasoning
        """
        opportunities = []

        for symbol in watchlist:
            sym_data = market_data.get(symbol, {})
            sym_flow = flow_data.get(symbol, {}) if flow_data else {}

            score = 0
            signals = []

            # IV Rank spike detection
            iv_rank = sym_data.get('iv_rank', 0)
            iv_rank_change = sym_data.get('iv_rank_change_1d', 0)
            if iv_rank > 50 and iv_rank_change > 10:
                score += 20
                signals.append(f"IV spike: {iv_rank:.0f}% (+{iv_rank_change:.0f}%)")

            # Volume anomaly
            volume_ratio = sym_data.get('volume_vs_avg', 1.0)
            if volume_ratio > 2.0:
                score += 15
                signals.append(f"Volume {volume_ratio:.1f}x average")

            # Options flow signals (Unusual Whales integration)
            if sym_flow:
                unusual_score = sym_flow.get('unusual_score', 0)
                if unusual_score > 70:
                    score += 25
                    signals.append(f"Unusual activity: {unusual_score}/100")

                large_trades = sym_flow.get('large_trades', [])
                if large_trades:
                    bullish_premium = sum(t.get('premium', 0) for t in large_trades
                                         if t.get('sentiment') == 'bullish')
                    bearish_premium = sum(t.get('premium', 0) for t in large_trades
                                         if t.get('sentiment') == 'bearish')
                    if bullish_premium > bearish_premium * 2:
                        score += 20
                        signals.append(f"Bullish flow: ${bullish_premium/1000:.0f}K premium")
                    elif bearish_premium > bullish_premium * 2:
                        score += 20
                        signals.append(f"Bearish flow: ${bearish_premium/1000:.0f}K premium")

            # Price momentum
            price_change = sym_data.get('price_change_pct', 0)
            if abs(price_change) > 3:
                score += 10
                direction = "up" if price_change > 0 else "down"
                signals.append(f"Price {direction} {abs(price_change):.1f}%")

            if score >= 30 and signals:
                opportunities.append({
                    'symbol': symbol,
                    'score': score,
                    'signals': signals,
                    'iv_rank': iv_rank,
                    'current_price': sym_data.get('price', 0),
                    'suggested_action': 'EVALUATE' if score >= 50 else 'WATCH'
                })

        # Sort by score descending
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        return opportunities

    async def analyze_trend(
        self,
        symbol: str,
        price_history: List[float],
        volume_history: List[int],
        timeframe: str = "daily"
    ) -> Dict[str, Any]:
        """
        Analyze trend strength and direction.

        Args:
            symbol: Ticker symbol
            price_history: Recent prices (oldest to newest)
            volume_history: Recent volumes
            timeframe: Analysis timeframe

        Returns:
            Trend analysis with strength, direction, and key levels
        """
        if len(price_history) < 20:
            return {
                'symbol': symbol,
                'trend': 'insufficient_data',
                'strength': 0,
                'direction': 'unknown'
            }

        # Simple trend analysis (in production, would use more sophisticated methods)
        current = price_history[-1]
        sma_20 = sum(price_history[-20:]) / 20
        sma_50 = sum(price_history[-50:]) / 50 if len(price_history) >= 50 else sma_20

        # Direction
        if current > sma_20 > sma_50:
            direction = 'bullish'
            strength = min((current / sma_20 - 1) * 100, 100)
        elif current < sma_20 < sma_50:
            direction = 'bearish'
            strength = min((sma_20 / current - 1) * 100, 100)
        else:
            direction = 'neutral'
            strength = 0

        # Classify trend strength
        if strength > 5:
            trend = f"strong_{direction.replace('ish', '')}"
        elif strength > 2:
            trend = f"moderate_{direction.replace('ish', '')}"
        elif strength > 0.5:
            trend = f"weak_{direction.replace('ish', '')}"
        else:
            trend = 'sideways'

        # Key levels (simple pivot calculation)
        recent_high = max(price_history[-20:])
        recent_low = min(price_history[-20:])
        pivot = (recent_high + recent_low + current) / 3

        return {
            'symbol': symbol,
            'trend': trend,
            'strength': round(strength, 1),
            'direction': direction,
            'current_price': current,
            'sma_20': round(sma_20, 2),
            'sma_50': round(sma_50, 2),
            'resistance': round(recent_high, 2),
            'support': round(recent_low, 2),
            'pivot': round(pivot, 2),
            'timeframe': timeframe
        }
