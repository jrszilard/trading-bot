#!/usr/bin/env python3
"""
Conversation Management for Tastytrade Chatbot

Handles conversation context, history, and reference resolution
for natural multi-turn interactions.

Author: Trading Bot
License: MIT
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class ConversationState(Enum):
    """Current state of the conversation flow"""
    IDLE = "idle"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_SYMBOL = "awaiting_symbol"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_EXECUTION = "awaiting_execution"
    RESEARCHING = "researching"
    # Position management states
    AWAITING_CLOSE_APPROVAL = "awaiting_close_approval"
    AWAITING_ROLL_APPROVAL = "awaiting_roll_approval"


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None


@dataclass
class ActiveEntities:
    """
    Entities currently active in the conversation context.
    Used for reference resolution (e.g., "it", "that trade", "the position")
    """
    # Most recently discussed symbol(s)
    symbols: List[str] = field(default_factory=list)

    # Most recently discussed trade ID
    trade_id: Optional[str] = None

    # Most recently discussed strategy
    strategy: Optional[str] = None

    # Most recently discussed position
    position_symbol: Optional[str] = None

    # Pending action context
    pending_action: Optional[str] = None  # e.g., "approve", "execute", "research"
    pending_target: Optional[str] = None  # The target of the pending action

    # Research context
    last_research_symbol: Optional[str] = None
    last_research_strategy: Optional[str] = None

    # Position management context
    pending_close_position: Optional[Dict[str, Any]] = None
    pending_roll_position: Optional[Dict[str, Any]] = None

    # Scan results context (for option selection)
    last_scan_candidates: List[Dict[str, Any]] = field(default_factory=list)
    last_scan_symbol: Optional[str] = None

    def update_from_entities(self, entities: Dict[str, Any]) -> None:
        """Update active entities from parsed intent entities"""
        if entities.get('symbols'):
            self.symbols = entities['symbols']
        if entities.get('trade_id'):
            self.trade_id = entities['trade_id']
        if entities.get('strategy'):
            self.strategy = entities['strategy']
        if entities.get('position_symbol'):
            self.position_symbol = entities['position_symbol']

    def get_current_symbol(self) -> Optional[str]:
        """Get the most relevant current symbol"""
        if self.symbols:
            return self.symbols[0]
        if self.position_symbol:
            return self.position_symbol
        if self.last_research_symbol:
            return self.last_research_symbol
        return None

    def clear_pending(self) -> None:
        """Clear pending action context"""
        self.pending_action = None
        self.pending_target = None

    def set_pending_close(self, position_data: Dict[str, Any]) -> None:
        """Set pending close position data"""
        self.pending_close_position = position_data
        self.pending_roll_position = None  # Clear roll if setting close

    def set_pending_roll(self, position_data: Dict[str, Any]) -> None:
        """Set pending roll position data"""
        self.pending_roll_position = position_data
        self.pending_close_position = None  # Clear close if setting roll

    def clear_pending_management(self) -> None:
        """Clear all pending position management data"""
        self.pending_close_position = None
        self.pending_roll_position = None


@dataclass
class ConversationContext:
    """
    Complete context for a conversation session.
    Maintains history and state for natural multi-turn interactions.
    """
    # Conversation history (limited for context window management)
    history: List[ConversationTurn] = field(default_factory=list)
    max_history: int = 20

    # Current conversation state
    state: ConversationState = ConversationState.IDLE

    # Active entities for reference resolution
    entities: ActiveEntities = field(default_factory=ActiveEntities)

    # Session metadata
    session_start: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_turn(
        self,
        role: str,
        content: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Add a new turn to the conversation"""
        turn = ConversationTurn(
            role=role,
            content=content,
            intent=intent,
            entities=entities
        )
        self.history.append(turn)
        self.last_activity = datetime.now()

        # Update active entities if provided
        if entities:
            self.entities.update_from_entities(entities)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return turn

    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get the most recent turns"""
        return self.history[-count:] if self.history else []

    def get_history_for_llm(self, count: int = 10) -> List[Dict[str, str]]:
        """Get history formatted for LLM context"""
        turns = self.get_recent_turns(count)
        return [
            {"role": turn.role, "content": turn.content}
            for turn in turns
        ]

    def set_state(self, state: ConversationState) -> None:
        """Update conversation state"""
        self.state = state

    def is_awaiting_input(self) -> bool:
        """Check if we're waiting for specific user input"""
        return self.state in (
            ConversationState.AWAITING_CONFIRMATION,
            ConversationState.AWAITING_SYMBOL,
            ConversationState.AWAITING_APPROVAL,
            ConversationState.AWAITING_EXECUTION,
            ConversationState.AWAITING_CLOSE_APPROVAL,
            ConversationState.AWAITING_ROLL_APPROVAL
        )

    def clear_state(self) -> None:
        """Reset to idle state"""
        self.state = ConversationState.IDLE
        self.entities.clear_pending()
        self.entities.clear_pending_management()


class ConversationManager:
    """
    Manages conversation context and provides reference resolution.

    Responsibilities:
    - Maintain conversation history
    - Resolve pronouns and references ("it", "that trade", etc.)
    - Track conversation state for multi-turn flows
    - Provide context to LLM for informed responses
    """

    def __init__(self):
        self.context = ConversationContext()

    def add_user_message(
        self,
        message: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a user message to the conversation"""
        self.context.add_turn("user", message, intent, entities)

    def add_assistant_message(
        self,
        message: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an assistant message to the conversation"""
        self.context.add_turn("assistant", message, intent, entities)

    def resolve_symbol_reference(self, text: str) -> Optional[str]:
        """
        Resolve symbol references in user input.

        Handles:
        - "it" -> most recent symbol
        - "that stock" -> most recent symbol
        - "the position" -> most recent position symbol
        """
        text_lower = text.lower()

        # Check for pronoun references
        pronoun_patterns = [
            "on it", "for it", "about it",
            "that stock", "that symbol", "the stock", "the symbol",
            "that one", "the same",
            "that position", "the position"
        ]

        for pattern in pronoun_patterns:
            if pattern in text_lower:
                symbol = self.context.entities.get_current_symbol()
                if symbol:
                    return symbol

        return None

    def resolve_trade_reference(self, text: str) -> Optional[str]:
        """
        Resolve trade ID references in user input.

        Handles:
        - "it" (when awaiting trade action)
        - "that trade"
        - "the trade"
        """
        text_lower = text.lower()

        trade_patterns = [
            "that trade", "the trade", "this trade",
            "approve it", "reject it", "execute it",
            "approve that", "reject that", "execute that"
        ]

        for pattern in trade_patterns:
            if pattern in text_lower:
                if self.context.entities.trade_id:
                    return self.context.entities.trade_id

        # Also check for bare affirmatives when awaiting approval
        if self.context.state in (
            ConversationState.AWAITING_APPROVAL,
            ConversationState.AWAITING_EXECUTION
        ):
            affirmatives = ["yes", "yeah", "yep", "sure", "ok", "okay", "do it", "go ahead"]
            if text_lower.strip() in affirmatives:
                return self.context.entities.trade_id

        return None

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for decision making"""
        return {
            "state": self.context.state.value,
            "current_symbol": self.context.entities.get_current_symbol(),
            "active_symbols": self.context.entities.symbols,
            "active_trade_id": self.context.entities.trade_id,
            "pending_action": self.context.entities.pending_action,
            "is_awaiting_input": self.context.is_awaiting_input(),
            "recent_history_count": len(self.context.history),
            "last_activity": self.context.last_activity.isoformat()
        }

    def set_pending_action(self, action: str, target: Optional[str] = None) -> None:
        """Set a pending action that may need confirmation"""
        self.context.entities.pending_action = action
        self.context.entities.pending_target = target

    def clear_pending_action(self) -> None:
        """Clear any pending action"""
        self.context.entities.clear_pending()

    def set_research_context(self, symbol: str, strategy: Optional[str] = None) -> None:
        """Set research context for follow-up questions"""
        self.context.entities.last_research_symbol = symbol
        if strategy:
            self.context.entities.last_research_strategy = strategy

    def get_llm_context(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM"""
        return self.context.get_history_for_llm(max_turns)

    def reset(self) -> None:
        """Reset the conversation to initial state"""
        self.context = ConversationContext()
