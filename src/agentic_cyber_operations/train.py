import inspect
import logging
from pathlib import Path
from tqdm import tqdm
from CybORG import CybORG
from wrappers import MultiAgentChallengeWrapper
import json

MAX_STEPS_PER_GAME = 200
MAX_EPS = 10000

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="./logs/training.log",
    encoding="utf-8",
)


def run_training_example(scenario="Scenario1b"):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = MultiAgentChallengeWrapper(env=CybORG(path, "sim"))

    logging.info(f"Starting training for {scenario}")
    for i in tqdm(range(MAX_EPS), position=0):
        rewards = {"Red": 0, "Blue": 0}
        for j in tqdm(range(MAX_STEPS_PER_GAME), position=1, leave=False):
            for player in ["Red", "Blue"]:
                observation = cyborg.get_observation(player)
                action_space = cyborg.get_action_space(player)
                action = cyborg.agents[player].get_action(observation, action_space)
                next_observation, r, terminated, truncated, info = cyborg.step(
                    agent=player, action=action
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

    logging.info(f"Finished training for {scenario}.")
    if hasattr(cyborg.env, "uuid"):
        model_subdir = f"{cyborg.env.uuid}_{MAX_EPS}_{MAX_STEPS_PER_GAME}"
    else:
        model_subdir = f"training_run_{MAX_EPS}_{MAX_STEPS_PER_GAME}"

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
    run_training_example("Scenario1b")
