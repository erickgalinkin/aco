import inspect
import logging
import json
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent
from wrappers import MultiAgentChallengeWrapper
from agents import RedAgent, BlueAgent
from gymnasium import spaces
from typing import Union, Tuple
from argparse import ArgumentParser

MAX_EPS = 1000

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
    "--red_type",
    choices=["bline", "meander", "ppo"],
    type=str,
    help="Type of red agent",
    required=True,
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


def load_agents(
    red_params: dict, blue_params: dict
) -> Tuple[BlueAgent, Union[RedAgent, BlueAgent, B_lineAgent, RedMeanderAgent]]:
    red_model_path = red_params["model_path"]
    match red_model_path.lower():
        case "bline":
            red_agent = B_lineAgent()
        case "meander":
            red_agent = RedMeanderAgent()
        case _:
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
    red_model_path: str, blue_model_path: str, scenario: str = "Scenario1b"
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
        try:
            data_path = f"./logs/{cyborg.env.uuid}_{num_steps}_{scenario}.json"
            logging.info(f"Writing data to {data_path}")
            with open(data_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            msg = f"Exception while writing data: {e}"
            logging.error(msg)
            print(msg)


def evaluate_blue(blue_model_path: str, red_type: str, scenario: str = "Scenario1b"):
    logging.info(f"Starting evaluation for blue model on {scenario}")
    logging.info(f"red type: {red_type}, blue model: {blue_model_path}")

    red_action_size, red_state_size, blue_action_size, blue_state_size = get_sizes(
        scenario
    )
    red_params = {
        "model_path": red_type,
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

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    for num_steps in [30, 50, 100]:
        logging.info(f"Starting evaluation for {num_steps} steps")
        cyborg = MultiAgentChallengeWrapper(
            env=CybORG(path, "sim", agents={"Red": red_agent})
        )
        overall_rewards = list()
        overall_actions = list()
        for i in tqdm(range(MAX_EPS), position=0, leave=False):
            rewards = list()
            actions = list()
            for j in tqdm(range(num_steps), position=1, leave=False):
                observation = cyborg.reset("Blue")
                action_space = cyborg.get_action_space("Blue")
                action = blue_agent.get_action(observation, action_space)
                next_observation, r, terminated, truncated, info = cyborg.step(
                    agent="Blue", action=action
                )
                done = terminated or truncated
                rewards.append(r)
                actions.append(str(cyborg.get_last_action("Blue")))
                if done:
                    _ = cyborg.reset("Blue")
                    break
            overall_rewards.append(rewards)
            overall_actions.append(actions)
        data = {"rewards": overall_rewards, "actions": overall_actions}
        try:
            data_path = f"./logs/{red_type}_{num_steps}_{scenario}.json"
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
        scenario = "Scenario1b"
    if args.red_type == "ppo":
        red_model_path = f"{args.model_path}/red.ckpt"
        blue_model_path = f"{args.model_path}/blue.ckpt"
        evaluate_models(red_model_path, blue_model_path, scenario)
    else:
        red_model_type = args.red_type
        blue_model_path = f"{args.model_path}/blue.ckpt"
        evaluate_blue(blue_model_path, red_model_type, scenario)
