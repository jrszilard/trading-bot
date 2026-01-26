#!/usr/bin/env python3
"""
Research Engine for Tastytrade Chatbot

Provides market research, strategy analysis, and position
evaluation without committing to trades.

Author: Trading Bot
License: MIT
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot import TastytradeBot
    from claude_advisor import ClaudeTradeAdvisor

logger = logging.getLogger(__name__)


@dataclass
class StrategyResearch:
    """Result of strategy research"""
    symbol: str
    strategy: str
    iv_rank: float
    current_price: float

    # Strategy structure
    put_side: Optional[Dict[str, Any]] = None  # short_strike, long_strike
    call_side: Optional[Dict[str, Any]] = None

    # Metrics
    net_credit: float = 0.0
    max_loss: float = 0.0
    probability_of_profit: float = 0.0

    # Greeks
    delta: float = 0.0
    theta: float = 0.0

    # Analysis
    rationale: str = ""
    recommendation: str = ""  # TRADE, WAIT, SKIP


class ResearchEngine:
    """
    Research engine for exploring trading opportunities without commitment.

    Provides:
    - Symbol research (IV rank, price, metrics)
    - Strategy analysis (what would a trade look like?)
    - Position evaluation (should I close/roll?)
    - Market overview
    """

    def __init__(
        self,
        bot: 'TastytradeBot',
        claude_advisor: Optional['ClaudeTradeAdvisor'] = None
    ):
        self.bot = bot
        self.claude = claude_advisor or bot.claude_advisor

    async def research_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Research a single symbol.

        Returns IV rank, current price, and basic metrics.
        """
        result = {
            "symbol": symbol,
            "iv_rank": 0.0,
            "current_price": 0.0,
            "beta": 1.0,
            "suitable_for_premium": False,
            "recommendation": ""
        }

        # Use mock data if available
        if self.bot.use_mock_data and self.bot.mock_provider:
            mock_data = self.bot.mock_provider.get_symbol_data(symbol)
            if mock_data:
                result["iv_rank"] = float(mock_data.get('iv_rank', 0))
                result["current_price"] = float(mock_data.get('price', 0))
                result["beta"] = float(mock_data.get('beta', 1.0))
                result["suitable_for_premium"] = result["iv_rank"] >= 30
                result["data_source"] = "mock"
        else:
            # Try to get real data
            try:
                iv_rank = await self.bot.get_iv_rank(symbol)
                if iv_rank is not None:
                    result["iv_rank"] = float(iv_rank)
                    result["suitable_for_premium"] = result["iv_rank"] >= 30
                    result["data_source"] = "live"
            except Exception as e:
                logger.warning(f"Could not get IV rank for {symbol}: {e}")

        # Add recommendation
        if result["iv_rank"] >= 50:
            result["recommendation"] = "IV rank is elevated - good candidate for selling premium"
        elif result["iv_rank"] >= 30:
            result["recommendation"] = "IV rank above minimum threshold - acceptable for premium selling"
        else:
            result["recommendation"] = "IV rank is low - consider waiting for higher volatility"

        return result

    async def research_strategy(
        self,
        symbol: str,
        strategy: str = "short_put_spread",
        target_delta: float = 0.30
    ) -> Dict[str, Any]:
        """
        Research a specific strategy on a symbol.

        Analyzes what the trade would look like without creating it.
        """
        result = {
            "symbol": symbol,
            "strategy": strategy,
            "iv_rank": 0.0,
            "current_price": 0.0,
            "put_side": None,
            "call_side": None,
            "metrics": {},
            "rationale": "",
            "would_meet_criteria": False
        }

        # Get symbol data first
        symbol_data = await self.research_symbol(symbol)
        result["iv_rank"] = symbol_data["iv_rank"]
        result["current_price"] = symbol_data.get("current_price", 0)

        # Check if IV rank meets criteria
        if result["iv_rank"] < 30:
            result["rationale"] = f"IV Rank ({result['iv_rank']:.1f}%) is below 30% threshold. " \
                                  "Not recommended for selling premium."
            return result

        # Build strategy structure based on mock/live data
        if self.bot.use_mock_data and self.bot.mock_provider:
            result = await self._research_with_mock_data(symbol, strategy, target_delta, result)
        else:
            result = await self._research_with_live_data(symbol, strategy, target_delta, result)

        # Get Claude's analysis if available
        if self.claude and self.claude.is_available:
            try:
                analysis = await self._get_claude_analysis(result)
                if analysis:
                    result["rationale"] = analysis.get("rationale", result.get("rationale", ""))
                    result["recommendation"] = analysis.get("recommendation", "WAIT")
            except Exception as e:
                logger.warning(f"Claude analysis failed: {e}")

        return result

    async def _research_with_mock_data(
        self,
        symbol: str,
        strategy: str,
        target_delta: float,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build research result using mock data"""
        mock_data = self.bot.mock_provider.get_symbol_data(symbol)
        if not mock_data:
            result["rationale"] = f"No mock data available for {symbol}"
            return result

        price = float(mock_data.get('price', 100))
        strike_interval = float(mock_data.get('option_strike_interval', 1.0))

        # Calculate strikes based on delta
        # Approximate: 30 delta put is roughly 5-7% OTM
        otm_pct = 0.05 + (0.30 - target_delta) * 0.1  # Adjust OTM % based on delta

        if strategy in ('short_put', 'short_put_spread', 'iron_condor'):
            short_put_strike = round(price * (1 - otm_pct) / strike_interval) * strike_interval
            long_put_strike = short_put_strike - (strike_interval * 5)  # $5 wide spread

            result["put_side"] = {
                "short_strike": short_put_strike,
                "long_strike": long_put_strike,
                "width": short_put_strike - long_put_strike
            }

        if strategy in ('short_call', 'short_call_spread', 'iron_condor'):
            short_call_strike = round(price * (1 + otm_pct) / strike_interval) * strike_interval
            long_call_strike = short_call_strike + (strike_interval * 5)

            result["call_side"] = {
                "short_strike": short_call_strike,
                "long_strike": long_call_strike,
                "width": long_call_strike - short_call_strike
            }

        # Estimate metrics
        spread_width = strike_interval * 5
        estimated_credit = spread_width * 0.33  # Roughly 1/3 of width
        max_loss = (spread_width * 100) - (estimated_credit * 100)

        if strategy == 'iron_condor':
            estimated_credit *= 2  # Both sides
            max_loss = (spread_width * 100) - (estimated_credit * 100 / 2)  # Max loss is one side

        result["metrics"] = {
            "net_credit": estimated_credit,
            "max_loss": max_loss,
            "pop": 0.65 + (0.30 - target_delta) * 0.2,  # Approximate POP
            "dte": 45
        }

        result["would_meet_criteria"] = True
        result["rationale"] = (
            f"Based on current IV Rank of {result['iv_rank']:.1f}%, "
            f"a {strategy.replace('_', ' ')} at {target_delta:.0%} delta would offer "
            f"approximately ${estimated_credit:.2f} credit with ${max_loss:.2f} max risk."
        )

        return result

    async def _research_with_live_data(
        self,
        symbol: str,
        strategy: str,
        target_delta: float,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build research result using live data"""
        # Try to use the bot's research_strategy method if available
        try:
            bot_research = await self.bot.research_strategy(symbol, strategy)
            if bot_research:
                result.update(bot_research)
                result["would_meet_criteria"] = True
        except Exception as e:
            logger.warning(f"Live research failed for {symbol}: {e}")
            result["rationale"] = f"Could not get live data for {symbol}. Try again or use mock data."

        return result

    async def _get_claude_analysis(self, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get Claude's analysis of the research"""
        if not self.claude or not self.claude.is_available:
            return None

        try:
            analysis = await self.claude.analyze_opportunity(
                symbol=research_data["symbol"],
                iv_rank=research_data["iv_rank"],
                current_price=research_data.get("current_price", 0),
                option_chain_summary={
                    "strategy": research_data["strategy"],
                    "put_side": research_data.get("put_side"),
                    "call_side": research_data.get("call_side"),
                    "metrics": research_data.get("metrics", {})
                },
                portfolio_state={
                    "portfolio_delta": float(self.bot.portfolio_state.portfolio_delta),
                    "open_positions": self.bot.portfolio_state.open_positions,
                    "buying_power": float(self.bot.portfolio_state.buying_power)
                }
            )

            if analysis:
                return {
                    "recommendation": analysis.recommendation,
                    "confidence": analysis.confidence,
                    "rationale": analysis.rationale,
                    "key_risks": analysis.key_risks
                }
        except Exception as e:
            logger.warning(f"Claude analysis error: {e}")

        return None

    async def analyze_position_action(
        self,
        symbol: str,
        position_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze whether to close, roll, or hold a position.
        """
        result = {
            "symbol": symbol,
            "action": "MONITOR",
            "reasoning": "",
            "urgency": "normal"
        }

        # Get position if not provided
        if not position_data:
            position_data = await self.bot.get_position_by_symbol(symbol)

        if not position_data:
            result["reasoning"] = f"No position found for {symbol}"
            return result

        # Extract key metrics
        pnl_pct = position_data.get('pnl_pct', 0)
        dte = position_data.get('dte', 99)
        is_tested = position_data.get('is_tested', False)

        # Apply tastylive rules
        if pnl_pct >= 0.50:
            result["action"] = "CLOSE"
            result["reasoning"] = f"Position at {pnl_pct:.0%} profit - at 50% profit target"
            result["urgency"] = "normal"
        elif dte <= 7:
            result["action"] = "CLOSE"
            result["reasoning"] = f"Only {dte} DTE remaining - close to avoid assignment risk"
            result["urgency"] = "high"
        elif dte <= 21:
            result["action"] = "ROLL"
            result["reasoning"] = f"{dte} DTE - consider rolling to next cycle per 21 DTE rule"
            result["urgency"] = "normal"
        elif is_tested:
            result["action"] = "ROLL"
            result["reasoning"] = "Position is tested (strike breached) - consider defensive roll"
            result["urgency"] = "high"
        elif pnl_pct <= -1.0:  # Losing 100%+ of credit
            result["action"] = "CLOSE"
            result["reasoning"] = f"Position at {pnl_pct:.0%} loss - at 2x credit stop loss"
            result["urgency"] = "high"
        else:
            result["action"] = "HOLD"
            result["reasoning"] = f"Position healthy: {pnl_pct:.0%} P&L, {dte} DTE"

        # Get Claude's analysis if available
        if self.claude and self.claude.is_available:
            try:
                analysis = await self.claude.evaluate_management_action(
                    position=position_data,
                    current_pnl_pct=pnl_pct,
                    dte_remaining=dte,
                    is_tested=is_tested,
                    portfolio_context={
                        "portfolio_delta": float(self.bot.portfolio_state.portfolio_delta),
                        "open_positions": self.bot.portfolio_state.open_positions,
                        "daily_pnl": float(self.bot.portfolio_state.daily_pnl)
                    }
                )

                if analysis:
                    result["action"] = analysis.action
                    result["reasoning"] = analysis.reasoning
                    result["urgency"] = analysis.urgency
                    if analysis.roll_suggestion:
                        result["roll_suggestion"] = analysis.roll_suggestion
            except Exception as e:
                logger.warning(f"Claude management analysis error: {e}")

        return result

    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get a high-level market overview.
        """
        result = {
            "market_indices": [],
            "overall_sentiment": "neutral",
            "iv_environment": "normal",
            "recommendation": ""
        }

        # Check major indices
        indices = ['SPY', 'QQQ', 'IWM']

        for symbol in indices:
            symbol_data = await self.research_symbol(symbol)
            result["market_indices"].append({
                "symbol": symbol,
                "iv_rank": symbol_data["iv_rank"],
                "suitable": symbol_data["suitable_for_premium"]
            })

        # Determine overall IV environment
        avg_iv = sum(idx["iv_rank"] for idx in result["market_indices"]) / len(result["market_indices"])

        if avg_iv >= 50:
            result["iv_environment"] = "elevated"
            result["overall_sentiment"] = "favorable for premium selling"
            result["recommendation"] = "Good environment for selling premium. Consider iron condors and strangles."
        elif avg_iv >= 30:
            result["iv_environment"] = "moderate"
            result["overall_sentiment"] = "acceptable for premium selling"
            result["recommendation"] = "Acceptable IV levels. Focus on defined-risk strategies like put spreads."
        else:
            result["iv_environment"] = "low"
            result["overall_sentiment"] = "not ideal for premium selling"
            result["recommendation"] = "Low IV environment. Consider waiting or using directional strategies."

        return result
