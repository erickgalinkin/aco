import inspect
import logging

from CybORG.Shared.Actions import ExecuteRansomware
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Shared.Actions.Action import InvalidAction
from CybORG.Agents import B_lineAgent
import json
from argparse import ArgumentParser
import numpy as np
from aco.agents import load_red_agent, load_blue_agent, load_hippo_agent
from aco.wrappers import MultiAgentChallengeWrapper

logger = logging.getLogger(__name__)
handler = logging.FileHandler("./logs/training.log")

logging.basicConfig(
    level=logging.DEBUG,
    encoding="utf-8",
    handlers=[handler],
)

parser = ArgumentParser()
parser.add_argument("--max_steps", type=int, default=100, help="Max steps per game")
parser.add_argument("--max_eps", type=int, default=30000, help="Max episodes per game")
parser.add_argument(
    "--ransomware",
    type=str,
    default="./checkpoints/7aa5ca3a-5f7f-4919-bc9d-3195cb0eb0a2/Red.ckpt",
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
    "--blue_agent", type=str, default=None, help="Path (.ckpt) to load blue agent."
)
parser.add_argument(
    "--reduce_rewards",
    action="store_true",
    default=False,
    help="Reduce rewards at each step by a factor of the opponent's reward.",
)
parser.add_argument(
    "--embedding_agent", action="store_true", default=False, help="Use embedding agent"
)
parser.add_argument(
    "--hierarchical",
    action="store_true",
    default=False,
    help="Use HiPPO for defending agent",
)
parser.add_argument(
    "--patience", type=int, default=30, help="Maximum InvalidActions to ignore."
)


def run_training_example(
    max_steps=100,
    max_eps=10000,
    ransomware_path=None,
    cryptominer_path=None,
    apt_path=None,
    blue_agent=None,
    use_embedding_agents=False,
    reduce_rewards=False,
    max_invalid=30,
    hierarchical=False,
):
    print("Setting up environments...")
    scenarios = ["B_line", "Scenario2", "Scenario2_ransomware", "Scenario2_cryptominer"]
    scenario_mapping = {
        "B_line": "B_line",
        "Scenario2": "APT",
        "Scenario2_ransomware": "Ransomware",
        "Scenario2_cryptominer": "Cryptominer",
    }
    instance_counts = {
        "B_line": 0,
        "Scenario2": 0,
        "Scenario2_ransomware": 0,
        "Scenario2_cryptominer": 0,
    }
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
            cyborg.agents["Red"] = agents["Red"]()
            cyborgs[scenario] = cyborg
            continue
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

        cyborg = MultiAgentChallengeWrapper(
            env=CybORG(path, "sim"), use_embedding_agents=use_embedding_agents
        )
        cyborgs[scenario] = cyborg
        logger.info(
            f"Scenario {scenario} loaded. Cyborg UUID: {cyborgs[scenario].uuid}"
        )

    # Tweak the formatter
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s")
    )

    logging.info(f"Loading agent {ransomware_path}")
    ransomware_agent = load_red_agent(
        load_path=ransomware_path,
        scenario="Scenario2_ransomware",
        embedding=use_embedding_agents,
    )
    logging.info(f"Loading agent {cryptominer_path}")
    cryptominer_agent = load_red_agent(
        load_path=cryptominer_path,
        scenario="Scenario2_cryptominer",
        embedding=use_embedding_agents,
    )
    logging.info(f"Loading agent {apt_path}")
    apt_agent = load_red_agent(
        load_path=apt_path, scenario="Scenario2", embedding=use_embedding_agents
    )
    if blue_agent is not None:
        logging.info(f"Loading agent {blue_agent}")
        if hierarchical:
            defending_agent = load_hippo_agent(
                load_path=blue_agent,
                scenario="Scenario2",
                embedding=use_embedding_agents,
            )
        else:
            defending_agent = load_blue_agent(
                load_path=blue_agent,
                scenario="Scenario2",
                embedding=use_embedding_agents,
            )
    elif hierarchical:
        defending_agent = load_hippo_agent(
            load_path=None, scenario=None, embedding=use_embedding_agents
        )
    else:
        defending_agent = cyborgs["Scenario2"].agents["Blue"]

    cyborgs["Scenario2"].agents["Red"] = apt_agent
    cyborgs["Scenario2_ransomware"].agents["Red"] = ransomware_agent
    cyborgs["Scenario2_cryptominer"].agents["Red"] = cryptominer_agent

    print(
        f"Starting multitype training. Results logged at ./logs/multitype_{cyborgs['Scenario2'].uuid}"
    )

    writer = SummaryWriter(log_dir=f"./logs/multitype_{cyborgs['Scenario2'].uuid}")

    for i in tqdm(range(max_eps), position=0):
        scenario = np.random.choice(scenarios)
        instance_counts[scenario] = instance_counts[scenario] + 1
        cyborg = cyborgs[scenario]
        _ = cyborg.reset("Blue")
        _ = cyborg.reset("Red")
        # last_red_reward = 0
        # last_blue_reward = 0
        rewards = {"Red": 0, "Blue": 0}
        for j in tqdm(range(max_steps), position=1, leave=False):
            for player in ["Red", "Blue"]:
                # B_line requires different observation and action_space
                if player == "Red" and scenario == "B_line":
                    observation = cyborg.env.env.env.env.env.get_observation(player)
                    action_space = cyborg.env.env.env.env.env.get_action_space(player)
                    action = cyborg.agents[player].get_action(observation, action_space)
                    results = cyborg.env.env.env.env.env.step(
                        agent=player, action=action
                    )
                    reward = results.reward
                    done = results.done
                    rewards[player] += reward
                    continue

                observation = cyborg.get_observation(player)
                action_space = cyborg.get_action_space(player)
                valid_action = False
                attempts = 0
                while not (valid_action or attempts > max_invalid):
                    attempts += 1
                    if player == "Blue":
                        action = defending_agent.get_action(observation, action_space)
                    else:
                        action = cyborg.agents[player].get_action(
                            observation, action_space
                        )
                    next_observation, r, terminated, truncated, info = cyborg.step(
                        agent=player, action=action
                    )
                    if not isinstance(cyborg.get_last_action(player), InvalidAction):
                        valid_action = True
                    elif attempts <= max_invalid:
                        cyborg.agents[player].model.buffer.states.pop()
                        cyborg.agents[player].model.buffer.actions.pop()
                        cyborg.agents[player].model.buffer.logprobs.pop()
                        cyborg.agents[player].model.buffer.state_values.pop()
                # Penalize invalid actions
                if isinstance(cyborg.get_last_action(player), InvalidAction):
                    r = -1.0
                if j < max_steps - 1:
                    done = terminated or truncated
                else:
                    done = True
                if player in rewards.keys():
                    if reduce_rewards:
                        # Implement this in the future maybe?
                        raise NotImplementedError(
                            "Reduced rewards not implemented for multitype."
                        )
                        # if player == "Red":
                        #     last_red_reward = r
                        #     r -= last_blue_reward
                        # rewards[player] += r
                        # if player == "Blue":
                        #     last_blue_reward = r
                        #     r -= last_red_reward
                    # Specialized reward function for ransomware agent
                    if scenario == "Scenario2_ransomware" and player == "Red":
                        if isinstance(
                            cyborg.get_last_action(player), ExecuteRansomware
                        ):
                            r = r + j
                        else:
                            r = r / (j + 1)
                    rewards[player] += r
                    if player == "Red":
                        cyborg.agents[player].model.buffer.rewards.append(r)
                        cyborg.agents[player].model.buffer.is_terminals.append(done)
                        cyborg.agents[player].train(observation)
                    if player == "Blue":
                        defending_agent.model.buffer.rewards.append(r)
                        defending_agent.model.buffer.is_terminals.append(done)
                        defending_agent.train(observation)

            if done:
                red_type = scenario_mapping[scenario]
                writer.add_scalar(f"{red_type} Episode Reward", rewards["Red"], i)
                writer.add_scalar("Blue Episode Reward", rewards["Blue"], i)
                writer.add_scalar("Episode Length", j + 1, i)

            if done and j < max_steps:
                break

    defender_type = "hierarchical" if hierarchical else "multitype"
    logging.info(f"Finished {defender_type} training.")
    print(f"{defender_type} training complete! Saving results...")
    model_subdir = f"training_run_{max_eps}_{max_steps}_{defender_type}_{cyborgs['Scenario2'].uuid}"

    logging.info(f"Writing models to ./checkpoints/{model_subdir}")
    model_path = Path(f"./checkpoints/{model_subdir}")
    model_path.mkdir(parents=True, exist_ok=True)
    for scenario_name, cyborg in cyborgs.items():
        red_type = scenario_mapping[scenario]
        cyborg.agents["Red"].model.save(
            f"./checkpoints/{model_subdir}/{red_type}_red.ckpt"
        )
        action_record_path = f"./logs/{model_subdir}/{red_type}_action_record.json"
        logging.info(f"Writing action record to {action_record_path}")
        try:
            with open(action_record_path, "w") as f:
                json.dump(cyborg.action_record, f)
        except Exception as e:
            logging.critical(
                f"Failed to write action record to {action_record_path}: {e}"
            )
            print(f"Failed to write action record for {scenario_name}!")
    defending_agent.model.save(f"./checkpoints/{model_subdir}/defender.ckpt")

    print("Instance counts:")
    for k, v in instance_counts.items():
        red_type = scenario_mapping[k]
        print(f"{red_type}: {v} instances \t {(v / max_eps) * 100:.2f}%")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_eps <= 0:
        raise ValueError("Max steps and max_eps must be greater than zero.")
    run_training_example(
        max_steps=args.max_steps,
        max_eps=args.max_eps,
        ransomware_path=args.ransomware,
        cryptominer_path=args.cryptominer,
        apt_path=args.apt,
        blue_agent=args.blue_agent,
        use_embedding_agents=args.embedding_agent,
        reduce_rewards=args.reduce_rewards,
        hierarchical=args.hierarchical,
        max_invalid=args.patience,
    )
