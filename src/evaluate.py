import inspect
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from aco.wrappers import MultiAgentChallengeWrapper
from aco.agents import RedAgent, BlueAgent
from gymnasium import spaces
from typing import Union, Tuple
from argparse import ArgumentParser

MAX_EPS = 100

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="./logs/evaluate.log",
    encoding="utf-8",
)

parser = ArgumentParser()
parser.add_argument(
    "--model_path", type=str, help="Path to saved model files", required=True
)
parser.add_argument(
    "--ppo_only", action="store_true", help="Run evaluation on red/blue PPO only."
)
parser.add_argument("--scenario", type=str, help="Scenario to evaluate")


def get_sizes(scenario: str) -> Tuple[int, int, int, int]:
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = MultiAgentChallengeWrapper(env=CybORG(path, "sim"))
    action_spaces = cyborg.action_spaces
    observation_spaces = cyborg.observation_spaces
    if isinstance(action_spaces["Red"], spaces.MultiDiscrete):
        red_action_size = len(action_spaces["Red"])
    elif isinstance(action_spaces["Red"], spaces.Discrete):
        red_action_size = action_spaces["Red"].n
    else:
        logger.critical("No valid red action space!")
        raise ValueError("No valid red action space!")
    red_observation_size = len(
        cyborg.observation_change(observation=cyborg.env.env.reset("Red").observation)
    )
    if isinstance(action_spaces["Blue"], spaces.MultiDiscrete):
        blue_action_size = len(action_spaces["Blue"])
    elif isinstance(action_spaces["Blue"], spaces.Discrete):
        blue_action_size = action_spaces["Blue"].n
    else:
        logger.critical("No valid blue action space!")
        raise ValueError("No valid blue action space!")
    blue_observation_size = len(
        cyborg.observation_change(observation=cyborg.env.env.reset("Blue").observation)
    )
    return (
        red_action_size,
        red_observation_size,
        blue_action_size,
        blue_observation_size,
    )


def load_agents(red_params: dict, blue_params: dict) -> Tuple[BlueAgent, RedAgent]:
    red_model_path = red_params["model_path"]
    red_agent = RedAgent(
        action_size=red_params["action_size"],
        state_size=red_params["state_size"],
        model=red_model_path,
    )

    blue_agent = BlueAgent(
        action_size=blue_params["action_size"],
        state_size=blue_params["state_size"],
        model=blue_params["model_path"],
    )

    return red_agent, blue_agent


def evaluate_models(
    red_model_path: str, blue_model_path: str, scenario: str = "Scenario2"
):
    logging.info(f"Starting evaluation for models on {scenario}")
    logging.info(f"red model: {red_model_path}, blue_model: {blue_model_path}")
    red_action_size, red_state_size, blue_action_size, blue_state_size = get_sizes(
        scenario
    )
    red_params = {
        "model_path": red_model_path,
        "action_size": red_action_size,
        "state_size": red_state_size,
    }
    blue_params = {
        "model_path": blue_model_path,
        "action_size": blue_action_size,
        "state_size": blue_state_size,
    }
    red_agent, blue_agent = load_agents(red_params, blue_params)
    if not isinstance(red_agent, RedAgent):
        msg = (
            "`evaluate_models` is intended to evaluate trained agents. "
            "To evaluate bline or meander, use `evaluate_blue`."
        )
        logger.critical(msg)
        print(msg)
        exit(0)

    model_id = blue_model_path.split("/")[-2].split("_")[0]

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    for num_steps in [30, 50, 100]:
        logging.info(f"Starting evaluation for {num_steps} steps")
        cyborg = MultiAgentChallengeWrapper(env=CybORG(path, "sim"))
        overall_rewards = {"Red": list(), "Blue": list()}
        overall_actions = {"Red": list(), "Blue": list()}
        for i in tqdm(range(MAX_EPS), position=0, leave=False):
            rewards = {"Red": list(), "Blue": list()}
            actions = {"Red": list(), "Blue": list()}
            for j in tqdm(range(num_steps), position=1, leave=False):
                for player in ["Red", "Blue"]:
                    observation = cyborg.get_observation(player)
                    action_space = cyborg.get_action_space(player)
                    action = cyborg.agents[player].get_action(observation, action_space)
                    next_observation, r, terminated, truncated, info = cyborg.step(
                        agent=player, action=action
                    )
                    done = terminated or truncated
                    rewards[player].append(r)
                    actions[player].append(str(cyborg.get_last_action(player)))
                    if done:
                        _ = cyborg.reset("Blue")
                        break
            overall_rewards["Red"].append(rewards["Red"])
            overall_rewards["Blue"].append(rewards["Blue"])
            overall_actions["Red"].append(actions["Red"])
            overall_actions["Blue"].append(actions["Blue"])
        data = {"rewards": overall_rewards, "actions": overall_actions}
        Path(f"./logs/evaluation/{model_id}/").mkdir(parents=True, exist_ok=True)
        try:
            data_path = (
                f"./logs/evaluation/{model_id}/{model_id}_{num_steps}_{scenario}.json"
            )
            logging.info(f"Writing data to {data_path}")
            with open(data_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            msg = f"Exception while writing data: {e}"
            logging.error(msg)
            print(msg)


def evaluate_blue(blue_model_path: str, scenario: str = "Scenario2"):
    logging.info(f"Starting evaluation for blue model on {scenario}")
    logging.info(f"blue model: {blue_model_path}")
    model_id = blue_model_path.split("/")[-2].split("_")[0]

    red_action_size, red_state_size, blue_action_size, blue_state_size = get_sizes(
        scenario
    )
    blue_params = {
        "model_path": blue_model_path,
        "action_size": blue_action_size,
        "state_size": blue_state_size,
    }
    blue_agent = BlueAgent(
        action_size=blue_params["action_size"],
        state_size=blue_params["state_size"],
        model=blue_params["model_path"],
    )

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    total_rewards = dict()
    total_actions = dict()
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:
            cyborg = ChallengeWrapper(
                env=CybORG(path, "sim", agents={"Red": red_agent}), agent_name="Blue"
            )
            setting_rewards = list()
            setting_actions = list()
            observation = cyborg.reset()
            action_space = cyborg.get_action_space("Blue")
            for i in tqdm(range(MAX_EPS), position=0, leave=False):
                rewards = list()
                actions = {"Blue": list(), "Red": list()}
                for j in tqdm(range(num_steps), position=1, leave=False):
                    action = blue_agent.get_action(observation, action_space)
                    observation, r, terminated, truncated, info = cyborg.step(action)
                    done = terminated or truncated
                    rewards.append(r)
                    actions["Blue"].append(str(cyborg.get_last_action("Blue")))
                    actions["Red"].append(str(cyborg.get_last_action("Red")))
                    if done:
                        _ = cyborg.reset("Blue")
                        break
                blue_agent.end_episode()
                setting_rewards.append(sum(rewards))
                setting_actions.append(actions)
            print(
                f"Average {num_steps} reward for red agent {red_agent.__name__}: {np.mean(setting_rewards)}"
            )
            total_rewards[f"{red_agent.__name__}_{num_steps}"] = np.mean(
                setting_rewards
            )
            total_actions[f"{red_agent.__name__}_{num_steps}"] = setting_actions
    data = {"rewards": total_rewards, "actions": total_actions}
    Path(f"./logs/evaluation/{model_id}/").mkdir(parents=True, exist_ok=True)
    try:
        data_path = f"./logs/evaluation/{model_id}/{scenario}.json"
        logging.info(f"Writing data to {data_path}")
        with open(data_path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        msg = f"Exception while writing data: {e}"
        logging.error(msg)
        print(msg)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.scenario is not None:
        scenario = args.scenario
    else:
        scenario = "Scenario2"
    if args.ppo_only:
        red_model_path = f"{args.model_path}/red.ckpt"
        blue_model_path = f"{args.model_path}/blue.ckpt"
        evaluate_models(red_model_path, blue_model_path, scenario)
    else:
        red_model_path = f"{args.model_path}/red.ckpt"
        blue_model_path = f"{args.model_path}/blue.ckpt"
        evaluate_blue(blue_model_path, scenario)
        evaluate_models(red_model_path, blue_model_path, scenario)
