import inspect
import logging
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
    randomize=False,
    ransomware_path=None,
    cryptominer_path=None,
    apt_path=None,
    blue_agent=None,
    use_embedding_agents=False,
    reduce_rewards=False,
    max_invalid=30,
    hierarchical=False,
):
    scenarios = ["B_line", "Scenario2", "Scenario2_ransomware", "Scenario2_cryptominer"]
    cyborgs = dict()
    for scenario in scenarios:
        if scenario == "B_line":
            path = str(inspect.getfile(CybORG))
            path = path[:-10] + f"/Shared/Scenarios/Scenario2.yaml"
            agents = {"Red": B_lineAgent}
            cyborg = MultiAgentChallengeWrapper(
                env=CybORG(path, "sim", agents=agents),
                use_embedding_agents=use_embedding_agents,
            )
            cyborgs[scenario] = cyborg
            continue
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

        cyborg = MultiAgentChallengeWrapper(
            env=CybORG(path, "sim"), use_embedding_agents=use_embedding_agents
        )
        cyborgs[scenario] = cyborg

    writer = SummaryWriter(log_dir=f"./logs/multitype/{cyborgs['Scenario2'].uuid}")

    # Tweak the formatter
    handler.setFormatter(
        logging.Formatter(f"%(asctime)s - %(levelname)s - %(message)s")
    )

    logging.info(f"Loading agent {ransomware_path}")
    ransomware_agent = load_red_agent(
        load_path=ransomware_path, scenario=scenario, embedding=use_embedding_agents
    )
    logging.info(f"Loading agent {cryptominer_path}")
    cryptominer_agent = load_red_agent(
        load_path=cryptominer_path, scenario=scenario, embedding=use_embedding_agents
    )
    logging.info(f"Loading agent {apt_path}")
    apt_agent = load_red_agent(
        load_path=apt_path, scenario=scenario, embedding=use_embedding_agents
    )
    if blue_agent is not None:
        logging.info(f"Loading agent {blue_agent}")
        if hierarchical:
            defending_agent = load_hippo_agent(
                load_path=blue_agent, scenario=scenario, embedding=use_embedding_agents
            )
        else:
            defending_agent = load_blue_agent(
                load_path=blue_agent, scenario=scenario, embedding=use_embedding_agents
            )
    else:
        defending_agent = cyborgs["Scenario2"].agents["Blue"]

    cyborgs["Scenario2"].agents["Red"] = apt_agent
    cyborgs["Scenario2_ransomware"].agents["Red"] = ransomware_agent
    cyborgs["Scenario2_cryptominer"].agents["Red"] = cryptominer_agent

    print(
        f"Starting multitype training. Results logged at ./logs/multitype/{cyborgs['Scenario2'].uuid}"
    )

    for i in tqdm(range(max_eps), position=0):
        scenario = np.random.choice(scenarios)
        cyborg = cyborgs[scenario]
        _ = cyborg.reset("Blue")
        _ = cyborg.reset("Red")
        last_red_reward = 0
        last_blue_reward = 0
        rewards = {"Red": 0, "Blue": 0}
        losses = {"Red": list(), "Blue": list()}
        if randomize:
            max_steps = np.random.choice([30, 50, 100])
        for j in tqdm(range(max_steps), position=1, leave=False):
            for player in ["Red", "Blue"]:
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
                        if player == "Red":
                            last_red_reward = r
                            r -= last_blue_reward
                        rewards[player] += r
                        if player == "Blue":
                            last_blue_reward = r
                            r -= last_red_reward
                        rewards[player] += r
                    if player == "Red":
                        cyborg.agents[player].model.buffer.rewards.append(r)
                        cyborg.agents[player].model.buffer.is_terminals.append(done)
                    if player == "Blue":
                        defending_agent.model.buffer.rewards.append(r)
                        defending_agent.model.buffer.is_terminals.append(done)

            if done:
                writer.add_scalar("Red Episode Reward", rewards["Red"], i)
                writer.add_scalar("Blue Episode Reward", rewards["Blue"], i)
                writer.add_scalar("Episode Length", j + 1, i)

            if done and j < max_steps:
                break

    logging.info(f"Finished multitype training.")
    model_subdir = (
        f"training_run_{max_eps}_{max_steps}_multitype_{cyborgs['Scenario2'].uuid}"
    )

    logging.info(f"Writing models to ./checkpoints/{model_subdir}")
    model_path = Path(f"./checkpoints/{model_subdir}")
    model_path.mkdir(parents=True, exist_ok=True)
    for id, cyborg in cyborgs.items():
        cyborg.agents["Red"].model.save(f"./checkpoints/{model_subdir}/{id}_red.ckpt")
    defending_agent.model.save(f"./checkpoints/{model_subdir}/defender.ckpt")
    action_record_path = f"./logs/{model_subdir}_action_record.json"
    logging.info(f"Writing action record to {action_record_path}")
    try:
        with open(action_record_path, "w") as f:
            json.dump(cyborg.action_record, f)
    except Exception as e:
        logging.critical(f"Failed to write action record to {action_record_path}: {e}")
        print("Failed to write action record!")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.max_steps <= 0 or args.max_eps <= 0:
        raise ValueError("Max steps and max_eps must be greater than zero.")
    run_training_example(
        max_steps=args.max_steps,
        max_eps=args.max_eps,
        randomize=args.randomize,
        ransomware_path=args.ransomware_path,
        cryptominer_path=args.cryptominer_path,
        apt_path=args.apt_path,
        blue_agent=args.blue_agent,
        use_embedding_agents=args.embedding_agent,
        reduce_rewards=args.reduce_rewards,
        hierarchical=args.hierarchical,
        max_invalid=args.patience,
    )
