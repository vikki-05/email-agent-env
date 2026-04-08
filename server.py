from fastapi import FastAPI
from env.environment import EmailEnv
from env.models import Action

app = FastAPI()

env = EmailEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }