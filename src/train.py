import inspect
import logging
from pathlib import Path
from tqdm import tqdm
from CybORG import CybORG
import json
from argparse import ArgumentParser
import numpy as np
from aco.agents import load_red_agent, load_blue_agent
from aco.wrappers import MultiAgentChallengeWrapper

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
    "--max_steps", type=int, default=MAX_STEPS_PER_GAME, help="Max steps per game"
)
parser.add_argument(
    "--max_eps", type=int, default=MAX_EPISODES, help="Max episodes per game"
)
parser.add_argument(
    "--randomize", action="store_true", default=False, help="Randomize steps"
)
parser.add_argument("--scenario", type=str, default="Scenario2", help="Scenario name")
parser.add_argument(
    "--red_agent", type=str, default=None, help="Path (.ckpt) to load red agent."
)
parser.add_argument(
    "--blue_agent", type=str, default=None, help="Path (.ckpt) to load blue agent."
)


def run_training_example(
    scenario="Scenario2",
    max_steps=MAX_STEPS_PER_GAME,
    max_eps=MAX_EPISODES,
    randomize=False,
    red_agent=None,
    blue_agent=None,
):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = MultiAgentChallengeWrapper(env=CybORG(path, "sim"))
    if red_agent is not None:
        cyborg.agents["Red"] = load_red_agent(load_path=red_agent, scenario=scenario)
    if blue_agent is not None:
        cyborg.agents["Blue"] = load_blue_agent(load_path=blue_agent, scenario=scenario)

    logging.info(f"Starting training for {scenario}")
    for i in tqdm(range(max_eps), position=0):
        _ = cyborg.reset("Blue")
        _ = cyborg.reset("Red")
        last_red_reward = 0
        last_blue_reward = 0
        rewards = {"Red": 0, "Blue": 0}
        if randomize:
            max_steps = np.random.choice([30, 50, 100])
        for j in tqdm(range(max_steps), position=1, leave=False):
            for player in ["Red", "Blue"]:
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
                if player in rewards.keys():
                    if player == "Red":
                        last_red_reward = r / 10
                        r -= last_blue_reward
                        rewards[player] += r
                    if player == "Blue":
                        last_blue_reward = r / 10
                        r -= last_red_reward
                        rewards[player] += r
                    cyborg.agents[player].model.buffer.rewards.append(r)
                    cyborg.agents[player].model.buffer.is_terminals.append(done)

                cyborg.agents[player].train(observation)  # training the agent

            if done:
                cyborg.writer.add_scalar("Red Episode Reward", rewards["Red"], i)
                cyborg.writer.add_scalar("Blue Episode Reward", rewards["Blue"], i)
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
    cyborg.agents["Red"].model.save(f"./checkpoints/{model_subdir}/red.ckpt")
    cyborg.agents["Blue"].model.save(f"./checkpoints/{model_subdir}/blue.ckpt")
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
        scenario=args.scenario,
        max_steps=args.max_steps,
        max_eps=args.max_eps,
        randomize=args.randomize,
        red_agent=args.red_agent,
        blue_agent=args.blue_agent,
    )
