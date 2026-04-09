from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

class EmailState:
    def __init__(
        self,
        email_text="",
        true_intent="",
        priority="low",
        time_waiting=0,
        history=None,
        resolved=False,
        steps=0,
        escalated=False,
        waiting_penalty_applied=False,
        customer_type="new",
    ):
        self.email_text = email_text
        self.true_intent = true_intent
        self.priority = priority
        self.time_waiting = time_waiting
        self.history = history or []
        self.resolved = resolved
        self.steps = steps
        self.escalated = escalated
        self.waiting_penalty_applied = waiting_penalty_applied
        self.customer_type = customer_type

    def is_done(self):
        return self.resolved or self.steps >= 5
