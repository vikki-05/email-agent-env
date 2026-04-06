"""
Task 1 (Easy): Intent Classification

Goal: Classify the intent of a customer support email.
Input: email dict with "text" field
Output: dict with "predicted_intent"
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.agent import SupportAgent


def run(email: dict) -> dict:
    """Classify the intent of the given email.

    Args:
        email: Dict containing at least "text" key with the email body.

    Returns:
        Dict with key "predicted_intent" mapping to the classified intent string.
    """
    agent = SupportAgent()
    return agent.run_easy(email)
