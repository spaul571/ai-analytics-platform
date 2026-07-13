"""Conversational context (Task B4).

Holds the last N question/answer turns and replays them into the code-generation
prompt so follow-up questions resolve pronouns and ellipsis ("what about the
West?", "and by month?") against what was just asked.

Only the question and the generated code are replayed, not the full result
tables. Replaying result tables would consume most of the context window within
three turns and, on a 4B model, actively degrades code generation by burying the
schema under numbers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

MAX_TURNS = 5  # the assignment requires the last 5 interactions


@dataclass
class Turn:
    """One completed question/answer exchange."""

    question: str
    code: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    row_count: int = 0
    elapsed_seconds: float = 0.0
    retried: bool = False


class ConversationMemory:
    """A bounded, replayable history of the session's interactions."""

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, turn: Turn) -> None:
        self._turns.append(turn)

    def reset(self) -> None:
        """Clear all context. Backs the session reset button required by B4."""
        self._turns.clear()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def as_messages(self) -> list[dict[str, str]]:
        """Replay history as chat messages for the code-generation call.

        Each past turn becomes a user/assistant pair carrying the question and
        the code that answered it. The model therefore sees its own prior
        working, which is what lets "and by month?" compile into a variation on
        the previous snippet rather than a guess.
        """
        messages: list[dict[str, str]] = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.code})
        return messages

    def summary(self) -> str:
        """One-line-per-turn view for the UI's context panel."""
        if not self._turns:
            return "No prior context in this session."
        return "\n".join(
            f"{i}. {t.question}" for i, t in enumerate(self._turns, start=1)
        )
