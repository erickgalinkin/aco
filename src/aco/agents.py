from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from aco.models import PPO, DynamicStatePPO


class RedAgent(BaseAgent):
    def __init__(self, action_size=None, state_size=None, model=None, embedding=False):
        if embedding:
            self.model = DynamicStatePPO(
                max_input_len=1024, embedding_dim=1024, action_dim=action_size
            )
        else:
            self.model = PPO(state_dim=state_size, action_dim=action_size)
        if model is not None:
            self.model.load(model)

    def get_action(self, observation, action_space):
        action = self.model.select_action(observation)
        return action

    def train(self, results):
        self.model.update()

    def end(self, results):
        pass


class BlueAgent(BaseAgent):
    def __init__(self, action_size=None, state_size=None, model=None, embedding=False):
        if embedding:
            self.model = DynamicStatePPO(
                max_input_len=1024, embedding_dim=1024, action_dim=action_size
            )
        else:
            self.model = PPO(state_dim=state_size, action_dim=action_size)
        if model is not None:
            self.model.load(model)

    def get_action(self, observation, action_space):
        action = self.model.select_action(observation)
        return action

    def train(self, results):
        self.model.update()

    def end_episode(self):
        pass


def load_red_agent(load_path: str, scenario: str):
    if scenario != "Scenario2":
        raise NotImplementedError("load_red_agent only supports Scenario2")
    red_agent = RedAgent(action_size=888, state_size=40, model=load_path)
    return red_agent


def load_blue_agent(load_path: str, scenario: str):
    if scenario != "Scenario2":
        raise NotImplementedError("load_blue_agent only supports Scenario2")
    blue_agent = BlueAgent(action_size=145, state_size=52, model=load_path)
    return blue_agent
