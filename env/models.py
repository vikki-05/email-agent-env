from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# Valid action types
VALID_ACTIONS = ("classify", "reply", "escalate", "close")

# Intent constants used by the environment
INTENTS = (
    "refund_request",
    "delivery_issue",
    "technical_issue",
    "billing_inquiry",
    "account_access",
)

# Priority constants
PRIORITIES = ("low", "medium", "high")

# Customer types
CUSTOMER_TYPES = ("new", "returning", "premium")


@dataclass
class Observation:
    """What the agent sees at each step."""

    email_text: str
    customer_type: str
    priority: str
    time_waiting: int
    history: List[str] = field(default_factory=list)


@dataclass
class Action:
    """An action the agent can take."""

    action_type: str  # classify | reply | escalate | close
    content: Optional[str] = None  # e.g. predicted intent or reply text
