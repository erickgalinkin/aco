import inspect
import logging
from pathlib import Path

from CybORG.Agents import SleepAgent, B_lineAgent, BlueReactRestoreAgent
from CybORG.Shared.Actions import ExecuteRansomware
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Shared.Actions.Action import InvalidAction
from aco.wrappers import MultiAgentChallengeWrapper
import json
from argparse import ArgumentParser
import numpy as np

MAX_STEPS_PER_GAME = 100
MAX_EPISODES = 10000

logger = logging.getLogger(__name__)
handler = logging.FileHandler("./logs/training.log")

logging.basicConfig(
    level=logging.DEBUG,
    encoding="utf-8",
    handlers=[handler],
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
parser.add_argument(
    "--embedding_agent", action="store_true", default=False, help="Use embedding agent"
)
parser.add_argument(
    "--patience", type=int, default=30, help="Maximum InvalidActions to ignore."
)


def run_training_example(
    player,
    scenario="Scenario2",
    max_steps=MAX_STEPS_PER_GAME,
    max_eps=MAX_EPISODES,
    randomize=False,
    use_embedding_agents=False,
    max_invalid=30,
):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    if player == "Red":
        agents = {"Blue": SleepAgent}
        opponent = "Blue"
    elif player == "Blue":
        agents = {"Red": B_lineAgent}
        opponent = "Red"
    else:
        raise ValueError("Only 'Red' or 'Blue' player values are supported")

    cyborg = MultiAgentChallengeWrapper(
        env=CybORG(path, "sim", agents=agents),
        use_embedding_agents=use_embedding_agents,
    )
    agents = agents.popitem()
    cyborg.agents[agents[0]] = agents[1]()

    writer = SummaryWriter(log_dir=f"./logs/{cyborg.uuid}")

    # Tweak the formatter
    handler.setFormatter(
        logging.Formatter(
            f"{cyborg.uuid}: " "%(asctime)s - %(levelname)s - %(message)s"
        )
    )

    msg = f"Starting {player} player pretraining for {scenario}. cyborg uuid: {cyborg.uuid}"
    print(msg)
    logging.info(msg)
    for i in tqdm(range(max_eps), position=0):
        _ = cyborg.reset(player)
        _ = cyborg.reset(opponent)

        rewards = {player: 0}
        if randomize:
            max_steps = np.random.choice([30, 50, 100])
        for j in tqdm(range(max_steps), position=1, leave=False):
            if isinstance(cyborg.agents[opponent], B_lineAgent):
                try:
                    print(f"Agent status on entry: {cyborg.agents[opponent].action}")
                    observation = cyborg.env.env.env.env.env.get_observation("Red")
                    action_space = cyborg.env.env.env.env.env.get_action_space("Red")
                    action = cyborg.agents[opponent].get_action(
                        observation, action_space
                    )
                    _ = cyborg.env.env.env.env.env.step(agent=opponent, action=action)
                except Exception as e:
                    print(
                        f"Agent status on exception: {cyborg.agents[opponent].action}"
                    )
                    print(f"Exception: {e} at episode {i} step {j}")
                    exit(1)
            else:
                observation = cyborg.get_observation(opponent)
                action_space = cyborg.get_action_space(opponent)
                action = cyborg.agents[opponent].get_action(observation, action_space)
                _ = cyborg.step(agent=opponent, action=action)

            observation = cyborg.get_observation(player)
            action_space = cyborg.get_action_space(player)
            valid_action = False
            attempts = 0
            while not (valid_action or attempts > max_invalid):
                attempts += 1
                action = cyborg.agents[player].get_action(observation, action_space)
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
                if scenario == "Scenario2_ransomware":
                    if (
                        isinstance(cyborg.get_last_action(player), ExecuteRansomware)
                        and player == "Red"
                    ):
                        r = r + j
                    elif player == "Red":
                        r = r / (j + 1)
            if j < max_steps - 1:
                done = (
                    terminated
                    or truncated
                    or isinstance(cyborg.get_last_action("Red"), ExecuteRansomware)
                )
            else:
                done = True
            rewards[player] += r
            cyborg.agents[player].model.buffer.rewards.append(r)
            cyborg.agents[player].model.buffer.is_terminals.append(done)

            cyborg.agents[player].train(observation)

            if done:
                writer.add_scalar(f"{player} Episode Reward", rewards[player], i)
                writer.add_scalar("Episode Length", j + 1, i)
                cyborg.agents["Red"].end_episode()
                cyborg.agents["Blue"].end_episode()

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
        use_embedding_agents=args.embedding_agent,
        max_invalid=args.patience,
    )
