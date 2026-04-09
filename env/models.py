from pydantic import BaseModel
from typing import Optional

VALID_ACTIONS = ("classify", "reply", "escalate", "close")

class Action(BaseModel):
    action_type: str
    content: Optional[str] = None

class Observation(BaseModel):
    email_text: str
    customer_type: str
    priority: str
    time_waiting: int
    history: list[str]
