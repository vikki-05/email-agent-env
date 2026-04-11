"""
API Server for Hugging Face Spaces deployment.
Supports episode-based reset/step protocol used by inference.py.
"""

import os
import uuid
from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action

app = FastAPI()

# Single-env instance (HF Spaces runs one worker)
env = EmailEnv()
current_episode = None


@app.get("/")
def root():
    return {"message": "Email Agent Env API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(body: dict = None):
    """Reset the environment and return an episode_id + observation."""
    global current_episode
    obs = env.reset()
    current_episode = {
        "episode_id": str(uuid.uuid4()),
        "observation": obs,
    }
    return {
        "episode_id": current_episode["episode_id"],
        "observation": obs.dict(),
    }


@app.post("/step")
def step(body: dict = None):
    """Execute an action and return (observation, reward, done, score)."""
    global current_episode
    if current_episode is None:
        return {"error": "No active episode. Call /reset first."}

    action_data = body.get("action", body)
    # Support both action_type/content and intent-based formats
    if "action_type" in action_data:
        action_obj = Action(
            action_type=action_data["action_type"],
            content=action_data.get("content"),
        )
    else:
        # intent-based action from LLM
        intent = action_data.get("intent", "general_inquiry")
        reply = action_data.get("reply", "")
        escalate = action_data.get("escalate", False)
        resolved = action_data.get("resolved", False)

        # Convert to env action format
        if not any(h.startswith("classify:") for h in current_episode["observation"].history):
            action_obj = Action(action_type="classify", content=intent)
        elif "reply" not in current_episode["observation"].history:
            action_obj = Action(action_type="reply", content=reply)
        elif not current_episode["observation"].escalated and escalate:
            action_obj = Action(action_type="escalate", content=None)
        else:
            action_obj = Action(action_type="close", content=None)

    obs, reward, done, info = env.step(action_obj)
    current_episode["observation"] = obs

    # Compute a score based on reward accumulation
    score = max(0.0, min(1.0, (reward + 1.0) / 2.0))  # normalize to [0, 1]

    return {
        "observation": obs.dict(),
        "reward": float(reward),
        "done": done,
        "score": score,
        "info": info,
    }
