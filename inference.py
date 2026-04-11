#!/usr/bin/env python3
"""
inference.py — Scaler OpenEnv Hackathon submission script.
Emits [START] / [STEP] / [END] structured logs for each task.
Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN env vars.
"""

import os
import json
from openai import OpenAI

# ── Environment variables (required by platform) ─────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# ── OpenAI client (required by platform) ──────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

# ── Local environment imports ────────────────────────────────────────────────
from env.environment import EmailEnv
from env.models import Action
from models.agent import SupportAgent

ENV_NAME = "email-agent-env"
TASKS    = ["easy", "medium", "hard"]
MAX_STEPS = 5


def get_action_from_model(task: str, observation: dict, step_num: int) -> dict:
    """Use the OpenAI client to decide the next action."""
    prompt = (
        f"You are an email support agent. Task: {task}. Step: {step_num}.\n"
        f"Observation: {json.dumps(observation)}\n"
        f"Respond with a JSON object representing your action. "
        f"For 'easy': classify the email intent. "
        f"For 'medium': classify and write a reply. "
        f"For 'hard': classify, reply, decide escalation, and resolve.\n"
        f"Return ONLY a JSON object, no explanation."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        # Deterministic fallback so baseline always produces scores
        return _deterministic_action(task, observation)


def _deterministic_action(task: str, email_text: str) -> dict:
    """
    Rule-based fallback agent — ensures baseline scores are always reproducible
    even when no LLM API key is provided.
    """
    text_lower = email_text.lower()

    # Intent classification (keyword-based)
    intent = "general_inquiry"
    if any(w in text_lower for w in ["refund", "money back", "return"]):
        intent = "refund_request"
    elif any(w in text_lower for w in ["deliver", "shipping", "package", "arrive"]):
        intent = "delivery_issue"
    elif any(w in text_lower for w in ["billing", "invoice", "charge", "payment"]):
        intent = "billing_inquiry"
    elif any(w in text_lower for w in ["login", "password", "account", "access"]):
        intent = "account_access"
    elif any(w in text_lower for w in ["broken", "error", "not working", "issue", "bug"]):
        intent = "technical_issue"
    elif any(w in text_lower for w in ["complaint", "unhappy", "disappointed", "frustrated"]):
        intent = "complaint"

    reply_map = {
        "refund_request":   "Thank you for contacting us. We have initiated your refund and it will be processed within 5-7 business days.",
        "delivery_issue":   "We apologize for the inconvenience. We are looking into your delivery issue and will update you shortly.",
        "billing_inquiry":  "Thank you for reaching out. We have reviewed your billing query and will resolve it promptly.",
        "account_access":   "We understand you are having trouble accessing your account. Please reset your password or contact support.",
        "technical_issue":  "We are sorry to hear about the technical issue. Our team is working on a fix and will follow up soon.",
        "complaint":        "We sincerely apologize for your experience. We take all feedback seriously and will address your concern.",
        "general_inquiry":  "Thank you for contacting us. We will get back to you with the information you need shortly.",
    }

    should_escalate = intent in ("complaint", "billing_inquiry", "technical_issue")

    if task == "easy":
        return {"intent": intent}
    elif task == "medium":
        return {"intent": intent, "reply": reply_map[intent]}
    else:  # hard
        return {
            "intent": intent,
            "reply": reply_map[intent],
            "escalate": should_escalate,
            "resolved": True,
        }


def run_task(task: str) -> dict:
    """Run one full task episode against the local environment and return result."""
    env = EmailEnv()
    agent = SupportAgent()
    obs = env.reset()

    email_text = obs.email_text

    # Emit [START]
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    rewards     = []
    final_score = 0.0
    success     = False
    done        = False
    step_num    = 0
    error_val   = "null"

    for step_num in range(1, MAX_STEPS + 1):
        # Build observation dict for the LLM
        obs_dict = {
            "email_text": obs.email_text,
            "customer_type": obs.customer_type,
            "priority": obs.priority,
            "time_waiting": obs.time_waiting,
            "history": obs.history,
        }

        try:
            action_data = get_action_from_model(task, obs_dict, step_num)
        except Exception as exc:
            action_data = _deterministic_action(task, email_text)

        action_str = json.dumps(action_data, separators=(",", ":"))

        try:
            # Convert action_data to env Action
            intent = action_data.get("intent", "")
            reply = action_data.get("reply", "")
            escalate = action_data.get("escalate", False)
            resolved = action_data.get("resolved", False)

            # Determine env action based on history
            has_classified = any(h.startswith("classify:") for h in obs.history)
            has_replied = "reply" in obs.history
            has_escalated = "escalate" in obs.history

            if not has_classified:
                env_action = Action(action_type="classify", content=intent)
            elif not has_replied:
                env_action = Action(action_type="reply", content=reply)
            elif not has_escalated and escalate:
                env_action = Action(action_type="escalate", content=None)
            else:
                env_action = Action(action_type="close", content=None)

            result = env.step(env_action)

            if isinstance(result, tuple) and len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif isinstance(result, tuple) and len(result) == 4:
                obs, reward, done, info = result
            elif isinstance(result, tuple) and len(result) == 3:
                obs, reward, done = result
                info = {}
            else:
                obs = result
                reward = 0.0
                done = True

            final_score = float(info.get("score", max(0.0, min(1.0, (reward + 1.0) / 2.0))))
            error_val = "null"
        except Exception as exc:
            reward = 0.0
            done = True
            error_val = str(exc).replace(" ", "_")[:80]

        rewards.append(reward)

        # Emit [STEP] — exact field order required
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
            flush=True,
        )

        if done:
            success = reward >= 0.5
            break

    # If env never returned done=true, compute success from rewards
    if not done and rewards:
        final_score = sum(rewards) / len(rewards)
        success = final_score >= 0.5

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # Emit [END] — exact field order required
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"score={final_score:.3f} rewards={rewards_str}",
        flush=True,
    )

    return {"task": task, "score": final_score, "success": success, "steps": step_num}


def main():
    results = []
    for task in TASKS:
        result = run_task(task)
        results.append(result)

    # Summary (plain text, not parsed by platform)
    print("\n--- Summary ---", flush=True)
    for r in results:
        print(f"  {r['task']}: score={r['score']:.3f} success={r['success']}", flush=True)


if __name__ == "__main__":
    main()
