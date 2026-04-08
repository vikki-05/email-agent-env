from __future__ import annotations

import os
import sys
import random
from typing import Dict, List, Optional, Tuple

# Ensure the project root is on sys.path so `env.*` imports resolve
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env.models import VALID_ACTIONS, Action, Observation
from env.state import EmailState


# ---------------------------------------------------------------------------
# Realistic email templates mapped to their true intent
# ---------------------------------------------------------------------------
EMAIL_TEMPLATES: Dict[str, List[str]] = {
    "refund_request": [
        "I ordered a product last week but it never arrived. I want a full refund.",
        "The item I received was damaged. Please process my refund immediately.",
        "I was charged twice for the same order. I need a refund for the duplicate charge.",
    ],
    "delivery_issue": [
        "My tracking number hasn't updated in 5 days. Where is my package?",
        "The delivery address was wrong on the confirmation. Can we fix this?",
        "My order was marked delivered but nothing arrived at my doorstep.",
    ],
    "technical_issue": [
        "The app keeps crashing every time I try to open the settings page.",
        "I can't log in to my account even though I reset my password twice.",
        "The checkout page throws a 500 error whenever I apply a coupon code.",
    ],
    "billing_inquiry": [
        "I see an extra charge of $9.99 on my statement I don't recognise.",
        "Can you explain the annual plan pricing? It changed since last month.",
        "My invoice shows a different amount than what was displayed at checkout.",
    ],
    "account_access": [
        "My account was locked after too many failed login attempts. Please help.",
        "I no longer have access to the email on my account and can't reset my password.",
        "Two-factor authentication is not sending the code to my phone.",
    ],
}

# Keyword sets used to judge reply quality per intent
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "refund_request": ["refund", "money back", "charged", "return"],
    "delivery_issue": ["delivery", "tracking", "package", "ship", "arrive"],
    "technical_issue": ["crash", "error", "bug", "login", "password", "500"],
    "billing_inquiry": ["charge", "invoice", "billing", "price", "statement"],
    "account_access": ["account", "locked", "2fa", "two-factor", "access"],
}

# Derive INTENTS from keyword dict keys — single source of truth
INTENTS = tuple(INTENT_KEYWORDS.keys())

# Priority mapping: some intents are inherently higher priority
INTENT_DEFAULT_PRIORITY: Dict[str, str] = {
    "refund_request": "high",
    "delivery_issue": "medium",
    "technical_issue": "medium",
    "billing_inquiry": "low",
    "account_access": "high",
}

# Customer type keywords (simplified heuristic)
CUSTOMER_KEYWORDS: Dict[str, List[str]] = {
    "premium": ["premium", "pro", "enterprise", "vip"],
    "returning": ["again", "second time", "loyal", "previous"],
}


def _resolve_customer_type(email_text: str) -> str:
    """Determine customer type from email keywords. Cached at reset."""
    for ctype, keywords in CUSTOMER_KEYWORDS.items():
        if any(kw in email_text.lower() for kw in keywords):
            return ctype
    return "new"


class EmailEnv:
    """Gym-style RL environment for customer support email triage."""

    # Reward values — single source of truth
    REWARD_CORRECT_CLASSIFY = 0.3
    REWARD_WRONG_CLASSIFY = -0.2
    REWARD_GOOD_REPLY = 0.4
    REWARD_BAD_REPLY = -0.2
    REWARD_CORRECT_ESCALATE = 0.3
    REWARD_UNNECESSARY_ESCALATE = -0.1
    REWARD_PROPER_CLOSE = 0.5
    REWARD_EARLY_CLOSE = -0.3
    REWARD_WAITING_PENALTY = -0.2
    REWARD_STEP_LIMIT_PENALTY = -0.5
    REWARD_INVALID_ACTION = -0.1

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.state: Optional[EmailState] = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        """Start a new episode with a random email."""
        intent = self.rng.choice(INTENTS)
        template = self.rng.choice(EMAIL_TEMPLATES[intent])
        priority = INTENT_DEFAULT_PRIORITY[intent]
        time_waiting = self.rng.randint(1, 48)  # hours

        customer_type = _resolve_customer_type(template)

        self.state = EmailState(
            email_text=template,
            true_intent=intent,
            priority=priority,
            time_waiting=time_waiting,
            history=[],
            resolved=False,
            steps=0,
            escalated=False,
            waiting_penalty_applied=False,
            customer_type=customer_type,
        )

        return self._make_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute one action and return (obs, reward, done, info)."""
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        s = self.state
        s.steps += 1
        reward = 0.0
        info: dict = {}

        if action.action_type not in VALID_ACTIONS:
            reward += self.REWARD_INVALID_ACTION
            s.history.append(f"invalid_action:{action.action_type}")
            obs = self._make_observation()
            done = s.is_done()
            if done:
                info["resolved"] = s.resolved
            return obs, reward, done, info

        if action.action_type == "classify":
            reward += self._reward_classify(action, s)
            s.history.append(f"classify:{action.content}")

        elif action.action_type == "reply":
            reward += self._reward_reply(action, s)
            s.history.append("reply")

        elif action.action_type == "escalate":
            reward += self._reward_escalate(action, s)
            s.escalated = True
            s.history.append("escalate")

        elif action.action_type == "close":
            reward += self._reward_close(action, s)
            s.history.append("close")

        obs = self._make_observation()
        done = s.is_done()

        if done:
            info["resolved"] = s.resolved
            if not s.resolved and s.steps >= 5:
                info["step_limit_exceeded"] = True
            return obs, reward, done, info

        if s.time_waiting > 24 and not s.escalated and not s.waiting_penalty_applied:
            reward += self.REWARD_WAITING_PENALTY
            s.waiting_penalty_applied = True
            info["waiting_penalty"] = True

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------
    def _reward_classify(self, action: Action, s: EmailState) -> float:
        predicted = (action.content or "").strip().lower()
        if predicted == s.true_intent:
            return self.REWARD_CORRECT_CLASSIFY
        return self.REWARD_WRONG_CLASSIFY

    @staticmethod
    def _reward_reply(action: Action, s: EmailState) -> float:
        keywords = INTENT_KEYWORDS.get(s.true_intent, [])
        content = (action.content or "").lower()
        if not content:
            return EmailEnv.REWARD_BAD_REPLY
        if any(kw in content for kw in keywords):
            return EmailEnv.REWARD_GOOD_REPLY
        return EmailEnv.REWARD_BAD_REPLY

    def _reward_escalate(self, action: Action, s: EmailState) -> float:
        if s.priority == "high":
            return self.REWARD_CORRECT_ESCALATE
        return self.REWARD_UNNECESSARY_ESCALATE

    def _reward_close(self, action: Action, s: EmailState) -> float:
        has_replied = "reply" in s.history or s.escalated
        classifications = [
            h.split(":", 1)[1]
            for h in s.history
            if h.startswith("classify:")
        ]
        if not classifications:
            return self.REWARD_EARLY_CLOSE

        any_correct = any(pred == s.true_intent for pred in classifications)
        if any_correct and has_replied:
            s.resolved = True
            return self.REWARD_PROPER_CLOSE

        return self.REWARD_EARLY_CLOSE

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------
    def _make_observation(self) -> Observation:
        assert self.state is not None
        s = self.state
        return Observation(
            email_text=s.email_text,
            customer_type=s.customer_type,
            priority=s.priority,
            time_waiting=s.time_waiting,
            history=list(s.history),
        )

    # ------------------------------------------------------------------
    # Render helper (for debugging)
    # ------------------------------------------------------------------
    def render(self) -> None:
        if self.state is None:
            print("Environment not initialized.")
            return
        s = self.state
        print(f"Email : {s.email_text}")
        print(f"Intent: {s.true_intent} | Priority: {s.priority}")
        print(f"Wait  : {s.time_waiting}h | Steps : {s.steps}")
        print(f"Resolved: {s.resolved} | Escalated: {s.escalated}")
        print(f"History: {s.history}")


# ======================================================================
#  Test block
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Customer Support Email Triage — Environment Test")
    print("=" * 60)

    env = EmailEnv(seed=42)
    obs = env.reset()

    print(f"\n[RESET] Observation:")
    print(f"  Email:        {obs.email_text}")
    print(f"  Customer:     {obs.customer_type}")
    print(f"  Priority:     {obs.priority}")
    print(f"  Time waiting: {obs.time_waiting}h")

    actions = [
        Action("classify", "refund_request"),
        Action("reply", "I can help process your refund for the charged amount."),
        Action("close"),
    ]

    total_reward = 0.0
    for i, act in enumerate(actions, 1):
        obs, rew, done, info = env.step(act)
        total_reward += rew
        print(f"\n[Step {i}] Action: {act.action_type} | content={act.content}")
        print(f"  Reward: {rew:+.2f} | Done: {done} | Info: {info}")
        if done:
            break

    print(f"\nTotal reward: {total_reward:+.2f}")
