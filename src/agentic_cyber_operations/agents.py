from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from models import PPO


class RedAgent(BaseAgent):
    def __init__(self, action_size=None, state_size=None, model=None):
        self.model = PPO(state_dim=state_size, action_dim=action_size)
        if model is not None:
            self.model.load(model)

    def get_action(self, observation, action_space):
        action = self.model.select_action(observation)
        return action

    def train(self, results):
        self.model.update()


class BlueAgent(BaseAgent):
    def __init__(self, action_size=None, state_size=None, model=None):
        self.model = PPO(state_dim=state_size, action_dim=action_size)
        if model is not None:
            self.model.load(model)

    def get_action(self, observation, action_space):
        action = self.model.select_action(observation)
        return action

    def train(self, results):
        self.model.update()
