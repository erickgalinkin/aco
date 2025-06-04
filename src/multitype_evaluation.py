import inspect
import logging
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Shared.Actions import Sleep, InvalidAction
from CybORG.Agents import B_lineAgent
from aco.wrappers import MultiAgentChallengeWrapper
from aco.agents import load_red_agent, load_blue_agent, load_hippo_agent
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
    "--model_path", type=str, help="Path to saved defender model", required=True
)
parser.add_argument(
    "--ransomware",
    type=str,
    default=None,
    help="Path (.ckpt) to load ransomware attacker.",
)
parser.add_argument(
    "--cryptominer",
    type=str,
    default=None,
    help="Path (.ckpt) to load cryptominer attacker.",
)
parser.add_argument(
    "--apt", type=str, default=None, help="Path (.ckpt) to load apt attacker."
)
parser.add_argument(
    "--rw_defender",
    type=str,
    default=None,
    help="Path (.ckpt) to load ransomware defender agent. (Only used if --hierarchical)",
)
parser.add_argument(
    "--apt_defender",
    type=str,
    default=None,
    help="Path (.ckpt) to load apt defender agent. (Only used if --hierarchical)",
)
parser.add_argument(
    "--cm_defender",
    type=str,
    default=None,
    help="Path (.ckpt) to load cryptominer defender agent. (Only used if --hierarchical)",
)
parser.add_argument(
    "--patience", type=int, default=30, help="Maximum InvalidActions to ignore."
)
parser.add_argument(
    "--hierarchical",
    action="store_true",
    default=False,
    help="Use HiPPO for defending agent",
)


def load_agents(
    ransomware_path: str,
    cryptominer_path: str,
    apt_path: str,
    blue_agent_path: str,
    rw_defender: str = None,
    cm_defender: str = None,
    apt_defender: str = None,
    hierarchical: bool = False,
    use_embedding_agents: bool = False,
    max_invalid: int = 30,
):
    ransomware_agent = load_red_agent(
        load_path=ransomware_path,
        scenario="Scenario2_ransomware",
        embedding=use_embedding_agents,
    )
    cryptominer_agent = load_red_agent(
        load_path=cryptominer_path,
        scenario="Scenario2_cryptominer",
        embedding=use_embedding_agents,
    )
    apt_agent = load_red_agent(
        load_path=apt_path, scenario="Scenario2", embedding=use_embedding_agents
    )
    b_line = B_lineAgent()
    if hierarchical:
        if rw_defender is None or apt_defender is None or cm_defender is None:
            raise TypeError(
                "Must provide rw_defender, apt_defender, and cm_defender for hierarchical."
            )
        defending_agent = load_hippo_agent(
            load_path=blue_agent_path,
            ransomware=rw_defender,
            apt=apt_defender,
            cryptominer=cm_defender,
            scenario="Scenario2",
            embedding=use_embedding_agents,
        )
    else:
        defending_agent = load_blue_agent(
            load_path=blue_agent_path,
            scenario="Scenario2",
            embedding=use_embedding_agents,
        )

    return ransomware_agent, cryptominer_agent, apt_agent, b_line, defending_agent


def evaluate_models(
    ransomware_path: str,
    cryptominer_path: str,
    apt_path: str,
    blue_agent_path: str,
    rw_defender: str = None,
    cm_defender: str = None,
    apt_defender: str = None,
    hierarchical: bool = False,
    use_embedding_agents: bool = False,
    max_invalid: int = 30,
):
    model_id = blue_agent_path.split("/")[-2].split("_")[0]
    logging.info("Starting evaluation...")
    scenarios = ["B_line", "Scenario2", "Scenario2_ransomware", "Scenario2_cryptominer"]

    cyborgs = dict()
    for scenario in scenarios:
        logger.info(f"Setting up scenario {scenario}...")
        if scenario == "B_line":
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f"/Shared/Scenarios/Scenario2.yaml"
            agents = {"Red": B_lineAgent}
            cyborg = MultiAgentChallengeWrapper(
                env=CybORG(path, "sim", agents=agents),
                use_embedding_agents=use_embedding_agents,
            )
            cyborgs[scenario] = cyborg
        else:
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

            cyborg = MultiAgentChallengeWrapper(
                env=CybORG(path, "sim"), use_embedding_agents=use_embedding_agents
            )
            cyborgs[scenario] = cyborg
        logger.info(
            f"Scenario {scenario} loaded. Cyborg UUID: {cyborgs[scenario].uuid}"
        )

    ransomware_agent, cryptominer_agent, apt_agent, b_line, defending_agent = (
        load_agents(
            ransomware_path,
            cryptominer_path,
            apt_path,
            blue_agent_path,
            rw_defender,
            cm_defender,
            apt_defender,
            hierarchical,
            use_embedding_agents,
        )
    )

    cyborgs["Scenario2_ransomware"].agents["Red"] = ransomware_agent
    cyborgs["Scenario2_cryptominer"].agents["Red"] = cryptominer_agent
    cyborgs["Scenario2"].agents["Red"] = apt_agent
    cyborgs["B_line"].agents["Red"] = b_line

    meta_rewards = {30: dict(), 50: dict(), 100: dict()}
    for num_steps in [30, 50, 100]:
        for scenario in scenarios:
            logging.info(f"Evaluating scenario {scenario} with {num_steps} steps...")
            overall_rewards = {"Red": dict(), "Blue": dict()}
            overall_actions = {"Red": dict(), "Blue": dict()}
            cyborg = cyborgs[scenario]
            for i in tqdm(range(MAX_EPS), position=0, leave=False):
                rewards = {"Red": list(), "Blue": list()}
                actions = {"Red": list(), "Blue": list()}
                for j in tqdm(range(num_steps), position=1, leave=False):
                    for player in ["Red", "Blue"]:
                        if player == "Red" and scenario == "B_line":
                            observation = cyborg.env.env.env.env.env.get_observation(
                                player
                            )
                            action_space = cyborg.env.env.env.env.env.get_action_space(
                                player
                            )
                            action = cyborg.agents[player].get_action(
                                observation, action_space
                            )
                            # B_line never Sleeps except when it encounters an exception.
                            if isinstance(action, Sleep):
                                prior_observation = (
                                    cyborg.env.env.env.env.env.get_agent_state("Red")
                                )
                                if ["User0"] in prior_observation:
                                    cyborg.agents[player].initial_ip = (
                                        prior_observation["User0"]["Interface"][0][
                                            "IP Address"
                                        ]
                                    )
                                    cyborg.agents[player].last_subnet = (
                                        prior_observation["User0"]["Interface"][0][
                                            "Subnet"
                                        ]
                                    )
                                action = cyborg.agents[player].get_action(
                                    observation, action_space
                                )
                                if isinstance(action, Sleep):
                                    logger.warning("B_line unlikely to run properly!")
                            results = cyborg.env.env.env.env.env.step(
                                agent=player, action=action
                            )
                            r = results.reward
                        else:
                            observation = cyborg.get_observation(player)
                            action_space = cyborg.get_action_space(player)
                            valid_action = False
                            attempts = 0
                            while not (valid_action or attempts > max_invalid):
                                attempts += 1
                                if player == "Blue":
                                    action = defending_agent.get_action(
                                        observation, action_space
                                    )
                                else:
                                    action = cyborg.agents[player].get_action(
                                        observation, action_space
                                    )
                                next_observation, r, terminated, truncated, info = (
                                    cyborg.step(agent=player, action=action)
                                )
                                if not isinstance(
                                    cyborg.get_last_action(player), InvalidAction
                                ):
                                    valid_action = True
                                elif attempts <= max_invalid:
                                    cyborg.agents[player].model.buffer.states.pop()
                                    cyborg.agents[player].model.buffer.actions.pop()
                                    cyborg.agents[player].model.buffer.logprobs.pop()
                                    cyborg.agents[
                                        player
                                    ].model.buffer.state_values.pop()
                            if isinstance(
                                cyborg.get_last_action(player), InvalidAction
                            ):
                                r = -1.0
                            if j < num_steps - 1:
                                done = terminated or truncated
                            else:
                                done = True
                            rewards[player].append(r)
                            actions[player].append(action)

                    if done:
                        overall_rewards["Red"][i] = rewards["Red"]
                        overall_rewards["Blue"][i] = rewards["Blue"]
                        overall_actions["Red"][i] = actions["Red"]
                        overall_actions["Blue"][i] = actions["Blue"]

            logger.info(
                f"Recording episode rewards for scenario {scenario} with {num_steps} steps..."
            )
            data = {"rewards": overall_rewards, "actions": overall_actions}
            blue_reward = np.average(
                [np.sum(x) for x in overall_rewards["Blue"].values()]
            )
            blue_std = np.std([np.sum(x) for x in overall_rewards["Blue"].values()])
            red_reward = np.average(
                [np.sum(x) for x in overall_rewards["Red"].values()]
            )
            red_std = np.std([np.sum(x) for x in overall_rewards["Red"].values()])
            meta_rewards[num_steps][scenario] = {
                "Blue": {"reward": blue_reward, "std": blue_std},
                "Red": {"reward": red_reward, "std": red_std},
            }
            Path(f"./logs/evaluation/{model_id}/").mkdir(parents=True, exist_ok=True)
            try:
                data_path = f"./logs/evaluation/{model_id}/{scenario}_{num_steps}.json"
                logging.info(f"Writing data to {data_path}")
                with open(data_path, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                msg = f"Exception while writing data: {e}"
                logging.error(msg)
                print(msg)

    for k, v in meta_rewards.items():
        for scenario in scenarios:
            blue_reward = v[scenario]["Blue"]["reward"]
            blue_std = v[scenario]["Blue"]["std"]
            red_reward = v[scenario]["Red"]["reward"]
            red_std = v[scenario]["Red"]["std"]
            print(
                f"Defender reward for scenario {scenario} {k} steps: {blue_reward} +- {blue_std}"
            )
            print(
                f"Attacker reward for scenario {scenario} {k} steps: {red_reward} +- {red_std}"
            )


if __name__ == "__main__":
    args = parser.parse_args()
    evaluate_models(
        ransomware_path=args.ransomware,
        cryptominer_path=args.cryptominer,
        apt_path=args.apt,
        blue_agent_path=args.model_path,
        rw_defender=args.rw_defender,
        cm_defender=args.cm_defender,
        apt_defender=args.apt_defender,
        hierarchical=args.hierarchical,
        max_invalid=args.patience,
    )
