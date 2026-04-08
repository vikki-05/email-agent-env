from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action, Observation
from pydantic import BaseModel

app = FastAPI()
env = EmailEnv()


class ActionRequest(BaseModel):
    action_type: str
    content: str | None = None


# ✅ FIXED RESET
@app.post("/reset", response_model=Observation)
def reset():
    obs = env.reset()
    return obs


# ✅ FIXED STEP
@app.post("/step")
def step(action: ActionRequest):
    action_obj = Action(**action.dict())
    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": obs.dict(),
        "reward": float(reward),
        "done": done,
        "info": info
    }