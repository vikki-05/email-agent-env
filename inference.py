import os
import json
from models.agent import SupportAgent
from env.models import Action
from openai import OpenAI
from env.environment import EmailEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

INTENT_RULES = [
    ("refund_request",  ["refund", "money back", "charge", "charged", "overcharged", "reimburs", "payment", "cancel order", "chargeback"]),
    ("delivery_issue",  ["not delivered", "not arrived", "late", "shipping", "shipment", "delivery", "courier", "tracking", "lost package", "missing package", "hasn't arrived", "where is my order"]),
    ("technical_issue", ["error", "crash", "bug", "not working", "broken", "fails", "failure", "can't login", "cannot login", "won't load", "issue with", "glitch", "technical", "reset password", "password"]),
    ("complaint",       ["bad", "terrible", "awful", "horrible", "worst", "unacceptable", "disappointed", "disgusting", "rude", "poor service", "complaint", "complain", "unhappy", "dissatisfied"]),
]
DEFAULT_INTENT = "general_inquiry"


def classify_intent(observation) -> str:
    if isinstance(observation, dict):
        text = " ".join(str(v) for v in observation.values())
    else:
        text = str(observation)
    lower = text.lower()
    for intent, keywords in INTENT_RULES:
        for kw in keywords:
            if kw in lower:
                return intent
    return DEFAULT_INTENT

def detect_intent(text: str) -> str:
    text = text.lower()

    if any(word in text for word in ["refund", "charge", "money back"]):
        return "refund_request"
    elif any(word in text for word in ["delivery", "not delivered", "late", "shipping"]):
        return "delivery_issue"
    elif any(word in text for word in ["error", "crash", "bug", "issue"]):
        return "technical_issue"
    elif any(word in text for word in ["bill", "payment", "invoice"]):
        return "billing_inquiry"
    elif any(word in text for word in ["login", "password", "account", "access"]):
        return "account_access"
    else:
        return "general_inquiry"

def generate_action(observation, step_count, agent):
    from env.environment import INTENT_KEYWORDS  # import here to avoid circular
    
    email = {
        "text": observation.email_text,
        "priority": observation.priority,
        "id": "test",
        "expected_keywords": []
    }

    intent = agent.classify(email)
    email["expected_keywords"] = INTENT_KEYWORDS.get(intent, [])

    history = observation.history

    has_classified = any(h.startswith("classify:") for h in history)
    has_replied = "reply" in history
    has_escalated = "escalate" in history
    has_closed = "close" in history

    if not has_classified:
        return Action("classify", intent)

    if not has_replied:
        return Action("reply", agent.reply(email, intent))

    if agent.decide_escalation(email, intent) and not has_escalated:
        return Action("escalate", None)

    if has_escalated or agent.decide_resolution(email, intent, has_escalated):
        return Action("close", None)

    # If not resolved and can't escalate more, reply again
    return Action("reply", "We are continuing to work on your issue.")

def main():
    env = EmailEnv()
    obs = env.reset()

    agent = SupportAgent()

    print("[START]")
    print(f"environment: EmailEnv")
    print(f"model: {MODEL_NAME}")
    print(f"api_base: {API_BASE_URL}")

    total_reward = 0.0
    step_count = 1   # 🔥 START FROM 1 (IMPORTANT)
    done = False

    while not done:
        action = generate_action(obs, step_count, agent)

        result = env.step(action)

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
        elif isinstance(result, tuple) and len(result) == 3:
            obs, reward, done = result
        else:
            obs = result
            reward = 0.0
            done = True

        total_reward += float(reward)

        print("[STEP]")
        print(f"step: {step_count}")
        print(f"action_type: {action.action_type}")
        print(f"content: {action.content}")
        print(f"reward: {reward}")
        print(f"done: {done}")

        step_count += 1   #  increment AFTER step

    print("[END]")
    print(f"final_score: {total_reward}")
    
if __name__ == "__main__":
    main()