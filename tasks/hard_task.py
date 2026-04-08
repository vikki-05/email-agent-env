"""
Task 3 (Hard): Full Workflow Decision

Goal: Classify intent, generate reply, decide escalation, and determine resolution.
Input: email dict with "text" and "priority" fields
Output: dict with "predicted_intent", "reply", "escalate", "resolved"
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.agent import SupportAgent


def run(email: dict) -> dict:
    """Run the full support workflow on the given email.

    Args:
        email: Dict containing:
            - "text": the email body
            - "priority": "low", "medium", or "high"

    Returns:
        Dict with keys:
            - "predicted_intent": the classified intent string
            - "reply": the generated reply text
            - "escalate": bool, whether the case should be escalated
            - "resolved": bool, whether the case is fully resolved
    """
    agent = SupportAgent()
    return agent.run_hard(email)
