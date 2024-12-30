import inspect
from typing import Optional, Union, List
import numpy as np
from tqdm import tqdm
from gymnasium import spaces, Env
from CybORG import CybORG
from CybORG.Agents.SimpleAgents.GreenAgent import GreenAgent
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.IntListToAction import IntListToActionWrapper
from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from agents import RedAgent, BlueAgent
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4

MAX_STEPS_PER_GAME = 500
MAX_EPS = 10000


class MultiAgentGymWrapper(Env, BaseWrapper):
    def __init__(self, env: BaseWrapper = None):
        super().__init__(env)
        red_action_space = spaces.MultiDiscrete(self.get_action_space("Red"))
        red_box_len = len(self.observation_change(self.env.reset("Red").observation))
        red_observation_space = spaces.Box(
            -1.0, 1.0, shape=(red_box_len,), dtype=np.float32
        )
        self.red_agent = RedAgent(
            action_size=len(red_action_space), state_size=red_box_len
        )
        blue_action_space = spaces.MultiDiscrete(self.get_action_space("Blue"))
        blue_box_len = len(self.observation_change(self.env.reset("Blue").observation))
        blue_observation_space = spaces.Box(
            -1.0, 1.0, shape=(blue_box_len,), dtype=np.float32
        )
        self.blue_agent = BlueAgent(
            action_size=len(blue_action_space), state_size=blue_box_len
        )
        self.green_agent = GreenAgent()
        self.reward_range = (float("-inf"), float("inf"))
        self.metadata = dict()
        self.agents = {
            "Red": self.red_agent,
            "Blue": self.blue_agent,
            "Green": self.green_agent,
        }
        self.observation_spaces = {
            "Red": red_observation_space,
            "Blue": blue_observation_space,
        }
        self.action_spaces = {"Red": red_action_space, "Blue": blue_action_space}
        self.uuid = str(uuid4())
        self.writer = SummaryWriter(log_dir=f"./logs/{self.uuid}")
        self.action_space = None
        self.observation_space = None
        self.action = None

    def step(self, agent: str = None, action: Union[int, List[int]] = None):
        if agent in ["Red", "Blue"]:
            self.action_space = self.action_spaces[agent]
            self.observation_space = self.observation_spaces[agent]
        self.action = action
        result = self.env.step(agent, action)
        result.observation = self.observation_change(result.observation)
        result.action_space = self.action_space_change(result.action_space)
        terminated = result.done
        truncated = False
        info = vars(result)
        return np.array(result.observation), result.reward, terminated, truncated, info

    def reset(self, agent=None, **kwargs):
        result = self.env.reset(agent=agent, **kwargs)
        result.action_space = self.action_space_change(result.action_space)
        result.observation = self.observation_change(result.observation)
        return np.array(result.observation), vars(result)

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.get_attr("get_agent_state")(agent)

    def get_action_space(self, agent: str):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent: str):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()


def run_training_example(scenario="Scenario1b"):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = MultiAgentGymWrapper(
        env=IntListToActionWrapper(FixedFlatWrapper(CybORG(path, "sim")))
    )

    for i in tqdm(range(MAX_EPS), position=0):  # playing multiple games
        rewards = {"Red": 0, "Blue": 0}
        for j in tqdm(range(MAX_STEPS_PER_GAME), position=1):  # step in 1 game
            for player in ["Blue", "Red", "Green"]:
                observation = cyborg.get_observation(player)
                action_space = cyborg.get_action_space(player)
                action = cyborg.agents[player].get_action(observation, action_space)
                next_observation, r, terminated, truncated, info = cyborg.step(
                    agent=player, action=[action]
                )
                done = terminated or truncated
                if player in rewards.keys():
                    rewards[player] += r
                    cyborg.agents[player].model.buffer.rewards.append(r)
                    cyborg.agents[player].model.buffer.is_terminals.append(done)

                cyborg.agents[player].train(observation)  # training the agent
                if done or j == MAX_STEPS_PER_GAME - 1:
                    cyborg.writer.add_scalar("Red Episode Reward", rewards["Red"], i)
                    cyborg.writer.add_scalar("Blue Episode Reward", rewards["Blue"], i)
                    cyborg.writer.add_scalar("Episode Length", j, i)
                    break


if __name__ == "__main__":
    run_training_example("Scenario1b")
