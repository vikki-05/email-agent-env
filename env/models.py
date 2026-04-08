from dataclasses import dataclass
from typing import Optional

VALID_ACTIONS = ("classify", "reply", "escalate", "close")

@dataclass
class Action:
    action_type: str
    content: Optional[str] = None

@dataclass
class Observation:
    email_text: str
    customer_type: str
    priority: str
    time_waiting: int
    history: list[str]
