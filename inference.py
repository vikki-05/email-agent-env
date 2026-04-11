import os
from models.agent import SupportAgent
from env.models import Action
from env.environment import EmailEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def generate_action(observation, step_count, agent):
    from env.environment import INTENT_KEYWORDS

    email = {
        "text": observation.email_text,
        "priority": observation.priority,
        "id": "test",
        "expected_keywords": [],
    }

    intent = agent.classify(email)
    email["expected_keywords"] = INTENT_KEYWORDS.get(intent, [])

    history = observation.history
    has_classified = any(h.startswith("classify:") for h in history)
    has_replied = "reply" in history
    has_escalated = "escalate" in history

    if not has_classified:
        return Action(action_type="classify", content=intent)

    if not has_replied:
        return Action(action_type="reply", content=agent.reply(email, intent))

    if agent.decide_escalation(email, intent) and not has_escalated:
        return Action(action_type="escalate", content=None)

    if has_escalated or agent.decide_resolution(email, intent, has_escalated):
        return Action(action_type="close", content=None)

    return Action(action_type="reply", content="We are continuing to work on your issue.")


def main():
    env = EmailEnv()
    obs = env.reset()

    agent = SupportAgent()

    print("[START]")

    total_reward = 0.0
    step_count = 1
    done = False

    while not done:
        action = generate_action(obs, step_count, agent)
        result = env.step(action)

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

        reward = round(float(reward), 2)
        safe_content = str(action.content).replace("\n", " ")

        print("[STEP]")
        print(f"step: {step_count}")
        print(f"action_type: {action.action_type}")
        print(f"content: {safe_content}")
        print(f"reward: {reward}")
        print(f"done: {done}")

        step_count += 1

    print("[END]")
    print(f"final_score: {total_reward}")


if __name__ == "__main__":
    main()
