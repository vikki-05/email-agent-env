from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EmailState:
    """Tracks the internal state of an email triage episode."""

    email_text: str = ""
    true_intent: str = ""
    priority: str = "medium"
    time_waiting: int = 0
    history: List[str] = field(default_factory=list)
    resolved: bool = False
    steps: int = 0
    escalated: bool = False
    waiting_penalty_applied: bool = False
    customer_type: str = "new"

    def is_done(self) -> bool:
        """Episode is over when resolved or step limit reached (5 max)."""
        return self.resolved or self.steps >= 5
