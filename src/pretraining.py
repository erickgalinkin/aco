import inspect
import logging
from pathlib import Path

from CybORG.Agents import SleepAgent, B_lineAgent
from tqdm import tqdm
from CybORG import CybORG
from aco.wrappers import MultiAgentChallengeWrapper
import json
from argparse import ArgumentParser
import numpy as np

MAX_STEPS_PER_GAME = 100
MAX_EPISODES = 10000

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="./logs/training.log",
    encoding="utf-8",
)

parser = ArgumentParser()
parser.add_argument(
    "--player",
    type=str,
    choices=["Red", "Blue"],
    required=True,
    help="Red or Blue player pretraining",
)
parser.add_argument(
    "--max_steps", type=int, default=MAX_STEPS_PER_GAME, help="Max steps per game"
)
parser.add_argument(
    "--max_eps", type=int, default=MAX_EPISODES, help="Max episodes per game"
)
parser.add_argument(
    "--randomize", action="store_true", default=False, help="Randomize steps"
)
parser.add_argument("--scenario", type=str, default="Scenario2", help="Scenario name")


def run_training_example(
    player,
    scenario="Scenario2",
    max_steps=MAX_STEPS_PER_GAME,
    max_eps=MAX_EPISODES,
    randomize=False,
):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    if player == "Red":
        agents = {"Blue": SleepAgent}
    elif player == "Blue":
        agents = {"Red": B_lineAgent}
    else:
        raise ValueError("Only 'Red' or 'Blue' player values are supported")

    cyborg = MultiAgentChallengeWrapper(env=CybORG(path, "sim", agents=agents))

    logging.info(f"Starting {player} player pretraining for {scenario}")
    for i in tqdm(range(max_eps), position=0):
        _ = cyborg.reset(player)
        rewards = {player: 0}
        if randomize:
            max_steps = np.random.choice([30, 50, 100])
        for j in tqdm(range(max_steps), position=1, leave=False):
            observation = cyborg.get_observation(player)
            action_space = cyborg.get_action_space(player)
            action = cyborg.agents[player].get_action(observation, action_space)
            next_observation, r, terminated, truncated, info = cyborg.step(
                agent=player, action=action
            )
            if j < max_steps - 1:
                done = terminated or truncated
            else:
                done = True
            rewards[player] += r
            cyborg.agents[player].model.buffer.rewards.append(r)
            cyborg.agents[player].model.buffer.is_terminals.append(done)

            cyborg.agents[player].train(observation)  # training the agent

            if done:
                cyborg.writer.add_scalar(f"{player} Episode Reward", rewards[player], i)
                cyborg.writer.add_scalar("Episode Length", j, i)

            if done and j < max_steps:
                break

    logging.info(f"Finished training for {scenario}.")
    if hasattr(cyborg, "uuid"):
        model_subdir = f"{cyborg.uuid}"
    else:
        if randomize:
            max_steps = "random"
        model_subdir = f"training_run_{max_eps}_{max_steps}_{scenario}"

    logging.info(f"Writing models to ./checkpoints/{model_subdir}")
    model_path = Path(f"./checkpoints/{model_subdir}")
    model_path.mkdir(parents=True, exist_ok=True)
    cyborg.agents[player].model.save(f"./checkpoints/{model_subdir}/{player}.ckpt")
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
        player=args.player,
        scenario=args.scenario,
        max_steps=args.max_steps,
        max_eps=args.max_eps,
        randomize=args.randomize,
    )
