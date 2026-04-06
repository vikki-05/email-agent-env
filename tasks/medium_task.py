"""
Task 2 (Medium): Intent Classification + Reply Generation

Goal: Classify intent AND generate an appropriate reply.
Input: email dict with "text" field
Output: dict with "predicted_intent" and "reply"
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.agent import SupportAgent


def run(email: dict) -> dict:
    """Classify intent and generate a reply for the given email.

    Args:
        email: Dict containing at least "text" key with the email body.

    Returns:
        Dict with keys:
            - "predicted_intent": the classified intent string
            - "reply": the generated reply text
    """
    agent = SupportAgent()
    return agent.run_medium(email)
