#!/usr/bin/env python3
"""
Claude AI Trade Advisor for Tastytrade Trading Bot

Integrates Claude for intelligent trade reasoning:
- Opportunity analysis with market context
- Dynamic delta/strategy recommendations
- Position management decisions
- Human-readable trade summaries

Author: Trading Bot
License: MIT
"""

import json
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ClaudeAnalysis:
    """Result of Claude's trade analysis"""
    recommendation: str  # TRADE, WAIT, SKIP
    confidence: int  # 1-10
    suggested_strategy: str
    suggested_delta: float
    key_risks: List[str]
    rationale: str
    raw_response: Optional[str] = None


@dataclass
class ManagementAnalysis:
    """Result of Claude's position management analysis"""
    action: str  # CLOSE, ROLL, HOLD, MONITOR
    confidence: int  # 1-10
    reasoning: str
    roll_suggestion: Optional[str] = None
    urgency: str = "normal"  # low, normal, high, critical


class ClaudeTradeAdvisor:
    """
    AI-powered trade advisor using Claude for reasoning

    Provides intelligent analysis while maintaining human oversight.
    All recommendations require user approval before execution.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        """
        Initialize the Claude Trade Advisor

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Response temperature (lower = more focused)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

        if not self.api_key:
            logger.warning("No Anthropic API key found. Claude advisor will be disabled.")

    @property
    def client(self):
        """Lazy-load the Anthropic client"""
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
        """Check if Claude advisor is available"""
        return self.client is not None

    def _call_claude(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Make a call to Claude API"""
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
        """Parse JSON from Claude's response"""
        if not response:
            return None

        # Try to extract JSON from response
        try:
            # First try direct parse
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

    async def analyze_opportunity(
        self,
        symbol: str,
        iv_rank: float,
        current_price: float,
        option_chain_summary: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_context: Optional[str] = None
    ) -> Optional[ClaudeAnalysis]:
        """
        Analyze a potential trade opportunity

        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            iv_rank: Current IV Rank percentage
            current_price: Current underlying price
            option_chain_summary: Summary of relevant options (strikes, deltas, credits)
            portfolio_state: Current portfolio metrics
            market_context: Optional market context/news

        Returns:
            ClaudeAnalysis with recommendation and reasoning
        """
        if not self.is_available:
            logger.debug("Claude not available, skipping opportunity analysis")
            return None

        system_prompt = """You are an expert options trading advisor following the tastylive methodology.

Core tastylive principles:
- Sell premium when IV is elevated (IV Rank > 30)
- Target 45 DTE for optimal theta decay
- Use ~30 delta for standard positions, 16 delta for conservative
- Take profits at 50% of max profit
- Manage positions at 21 DTE
- Trade small (1-5% of portfolio per position)
- Stay delta neutral to slightly directional

You analyze trade opportunities and provide structured recommendations.
Be concise and focus on actionable insights."""

        user_prompt = f"""Analyze this potential options trade opportunity:

## Underlying Information
- Symbol: {symbol}
- Current Price: ${current_price:.2f}
- IV Rank: {iv_rank:.1f}%

## Portfolio Context
- Net Delta: {portfolio_state.get('portfolio_delta', 0)}
- Open Positions: {portfolio_state.get('open_positions', 0)}
- Buying Power Used: {portfolio_state.get('buying_power_used_pct', 0):.1f}%
- Daily P&L: ${portfolio_state.get('daily_pnl', 0):.2f}

## Option Chain Summary
{json.dumps(option_chain_summary, indent=2, default=str)}

{f"## Market Context{chr(10)}{market_context}" if market_context else ""}

## Analysis Required
Provide your analysis as JSON with these fields:
{{
  "recommendation": "TRADE" | "WAIT" | "SKIP",
  "confidence": 1-10,
  "suggested_strategy": "short_put" | "short_put_spread" | "iron_condor" | "short_strangle" | etc,
  "suggested_delta": 0.16 to 0.40,
  "key_risks": ["risk1", "risk2", "risk3"],
  "rationale": "2-3 sentence explanation"
}}

Consider:
1. Is IV Rank favorable for selling premium?
2. How does this fit the current portfolio delta?
3. What strategy best fits current conditions?
4. What's the optimal delta given IV and market context?
5. What are the key risks to monitor?"""

        response = self._call_claude(system_prompt, user_prompt)
        parsed = self._parse_json_response(response)

        if not parsed:
            return None

        try:
            return ClaudeAnalysis(
                recommendation=parsed.get("recommendation", "SKIP"),
                confidence=int(parsed.get("confidence", 5)),
                suggested_strategy=parsed.get("suggested_strategy", "short_put"),
                suggested_delta=float(parsed.get("suggested_delta", 0.30)),
                key_risks=parsed.get("key_risks", []),
                rationale=parsed.get("rationale", ""),
                raw_response=response
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing Claude analysis: {e}")
            return None

    async def evaluate_management_action(
        self,
        position: Dict[str, Any],
        current_pnl_pct: float,
        dte_remaining: int,
        is_tested: bool,
        portfolio_context: Dict[str, Any],
        market_conditions: Optional[str] = None
    ) -> Optional[ManagementAnalysis]:
        """
        Evaluate whether to close, roll, or hold an existing position

        Args:
            position: Position details (symbol, strike, type, quantity)
            current_pnl_pct: Current P&L as percentage of max profit
            dte_remaining: Days to expiration
            is_tested: Whether the strike has been breached
            portfolio_context: Current portfolio state
            market_conditions: Optional market context

        Returns:
            ManagementAnalysis with action recommendation
        """
        if not self.is_available:
            return None

        system_prompt = """You are an expert options position manager following tastylive methodology.

Management principles:
- Take profits at 50% of max profit (don't be greedy)
- Manage/roll at 21 DTE to avoid gamma risk
- Close losing positions at 2x credit received
- Roll tested positions when possible for credit
- Never hold to expiration
- When in doubt, reduce risk

Provide clear, actionable management recommendations."""

        user_prompt = f"""Evaluate this options position for management action:

## Position Details
- Symbol: {position.get('symbol', 'Unknown')}
- Underlying: {position.get('underlying', 'Unknown')}
- Type: {position.get('option_type', 'Unknown')}
- Strike: ${position.get('strike', 0)}
- Quantity: {position.get('quantity', 0)}
- Entry Credit: ${position.get('entry_credit', 0):.2f}

## Current Status
- P&L: {current_pnl_pct:.1%} of max profit
- DTE Remaining: {dte_remaining} days
- Position Tested: {"YES - STRIKE BREACHED" if is_tested else "No"}

## Portfolio Context
- Total Positions: {portfolio_context.get('open_positions', 0)}
- Portfolio Delta: {portfolio_context.get('portfolio_delta', 0)}
- Daily P&L: ${portfolio_context.get('daily_pnl', 0):.2f}

{f"## Market Conditions{chr(10)}{market_conditions}" if market_conditions else ""}

## Analysis Required
Provide your recommendation as JSON:
{{
  "action": "CLOSE" | "ROLL" | "HOLD" | "MONITOR",
  "confidence": 1-10,
  "reasoning": "2-3 sentence explanation",
  "roll_suggestion": "description of roll if applicable, null otherwise",
  "urgency": "low" | "normal" | "high" | "critical"
}}

Consider:
1. Is this at profit target (50%+)?
2. Is gamma risk increasing (DTE < 21)?
3. Is the position being tested?
4. Can we roll for a credit?
5. What's the overall portfolio impact?"""

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
                urgency=parsed.get("urgency", "normal")
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing management analysis: {e}")
            return None

    async def generate_approval_summary(
        self,
        proposal: Any,  # TradeProposal
        portfolio_context: Dict[str, Any],
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable summary for trade approval

        Args:
            proposal: TradeProposal object
            portfolio_context: Current portfolio state
            additional_context: Any additional context to consider

        Returns:
            Formatted summary string for human review
        """
        if not self.is_available:
            return self._generate_basic_summary(proposal)

        system_prompt = """You are a trading assistant helping a human review trade proposals.

Your job is to summarize the trade in clear, simple language and highlight:
1. What the trade is (strategy, underlying, expected outcome)
2. Why it might be a good trade (tastylive criteria)
3. Key risks to be aware of
4. How it fits the portfolio

Be concise but thorough. Use bullet points. The human makes the final decision."""

        # Build proposal details
        legs_desc = []
        for leg in proposal.legs:
            legs_desc.append(f"  - {leg.get('action', 'SELL')} {leg.get('quantity', 1)} "
                           f"{leg.get('option_type', 'PUT')} @ ${leg.get('strike', 0)} "
                           f"exp {leg.get('expiration', 'Unknown')}")

        user_prompt = f"""Summarize this trade proposal for human approval:

## Trade Details
- Strategy: {proposal.strategy.value}
- Underlying: {proposal.underlying_symbol}
- IV Rank: {proposal.iv_rank:.1f}%
- DTE: {proposal.dte} days
- Expected Credit: ${proposal.expected_credit:.2f}
- Max Loss: ${proposal.max_loss:.2f}
- Probability of Profit: {proposal.probability_of_profit:.1%}
- Position Delta: {proposal.delta}
- Position Theta: {proposal.theta}

## Legs
{chr(10).join(legs_desc)}

## Portfolio Impact
- Current Positions: {portfolio_context.get('open_positions', 0)}
- Portfolio Delta (before): {portfolio_context.get('portfolio_delta', 0)}
- Portfolio Delta (after): {float(portfolio_context.get('portfolio_delta', 0)) + float(proposal.delta):.1f}
- Buying Power Available: ${portfolio_context.get('buying_power', 0):.2f}

{f"## Additional Context{chr(10)}{additional_context}" if additional_context else ""}

Provide a clear summary with:
1. Trade Overview (1-2 sentences)
2. Tastylive Alignment (bullet points)
3. Risk Considerations (bullet points)
4. Portfolio Fit (1-2 sentences)

Format for terminal display (no markdown headers, use simple formatting)."""

        response = self._call_claude(system_prompt, user_prompt)

        if response:
            return response

        return self._generate_basic_summary(proposal)

    def _generate_basic_summary(self, proposal: Any) -> str:
        """Generate a basic summary when Claude is unavailable"""
        return f"""Trade Summary (AI analysis unavailable)
Strategy: {proposal.strategy.value}
Underlying: {proposal.underlying_symbol}
Credit: ${proposal.expected_credit:.2f}
Max Loss: ${proposal.max_loss:.2f}
IV Rank: {proposal.iv_rank:.1f}%
DTE: {proposal.dte} days
P.O.P: {proposal.probability_of_profit:.1%}"""

    async def analyze_portfolio_health(
        self,
        portfolio_state: Dict[str, Any],
        positions: List[Dict[str, Any]],
        recent_trades: List[Dict[str, Any]]
    ) -> str:
        """
        Analyze overall portfolio health and provide recommendations

        Args:
            portfolio_state: Current portfolio metrics
            positions: List of open positions
            recent_trades: Recent trade history

        Returns:
            Portfolio health analysis as formatted string
        """
        if not self.is_available:
            return "Portfolio analysis unavailable (Claude not configured)"

        system_prompt = """You are a portfolio risk analyst following tastylive methodology.

Analyze the portfolio and provide:
1. Overall health assessment
2. Risk warnings if any
3. Diversification analysis
4. Actionable recommendations

Be direct and practical. Focus on risk management."""

        positions_summary = []
        for pos in positions[:10]:  # Limit to first 10 for context
            positions_summary.append(
                f"- {pos.get('symbol', 'Unknown')}: {pos.get('quantity', 0)} contracts, "
                f"P&L: {pos.get('pnl_pct', 0):.1%}, DTE: {pos.get('dte', 0)}"
            )

        user_prompt = f"""Analyze this options portfolio:

## Portfolio Metrics
- Net Liquidating Value: ${portfolio_state.get('net_liquidating_value', 0):.2f}
- Buying Power: ${portfolio_state.get('buying_power', 0):.2f}
- Daily P&L: ${portfolio_state.get('daily_pnl', 0):.2f}
- Weekly P&L: ${portfolio_state.get('weekly_pnl', 0):.2f}
- Portfolio Delta: {portfolio_state.get('portfolio_delta', 0)}
- Portfolio Theta: {portfolio_state.get('portfolio_theta', 0)}
- Open Positions: {portfolio_state.get('open_positions', 0)}

## Current Positions
{chr(10).join(positions_summary) if positions_summary else "No open positions"}

## Recent Activity
- Trades in last 7 days: {len(recent_trades)}
- Win rate: {sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / max(len(recent_trades), 1):.1%}

Provide a brief portfolio health report (terminal-friendly format)."""

        response = self._call_claude(system_prompt, user_prompt)
        return response or "Portfolio analysis unavailable"

    async def get_market_context(self, symbols: List[str]) -> str:
        """
        Generate market context summary for trading decisions

        Note: This is a placeholder. In production, you would integrate
        with news APIs, market data feeds, or other sources.

        Args:
            symbols: List of symbols to get context for

        Returns:
            Market context summary
        """
        # This would typically fetch real market data/news
        # For now, return a placeholder
        return f"Market context for {', '.join(symbols[:5])} - Use external data integration for real context"
