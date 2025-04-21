import inspect
import logging
from pathlib import Path
from tqdm import tqdm
from CybORG import CybORG
from CybORG.Shared.Actions.Action import InvalidAction
import json
from argparse import ArgumentParser
import numpy as np
from aco.agents import load_red_agent, load_blue_agent, Cardiff
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
parser.add_argument("--max_eps", type=int, default=10000, help="Max episodes per game")
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
parser.add_argument(
    "--reduce_rewards",
    action="store_true",
    default=False,
    help="Reduce rewards at each step by a factor of the opponent's reward.",
)
parser.add_argument(
    "--embedding_agent", action="store_true", default=False, help="Use embedding agent"
)


def run_training_example(
    scenario="Scenario2",
    max_steps=100,
    max_eps=10000,
    randomize=False,
    red_agent=None,
    blue_agent=None,
    use_embedding_agents=False,
    reduce_rewards=False,
    max_invalid=30,
):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    cyborg = MultiAgentChallengeWrapper(
        env=CybORG(path, "sim"), use_embedding_agents=use_embedding_agents
    )
    # Tweak the formatter
    handler.setFormatter(
        logging.Formatter(
            f"{cyborg.uuid}: " "%(asctime)s - %(levelname)s - %(message)s"
        )
    )
    if red_agent is not None:
        logging.info(f"Loading agent {red_agent}")
        cyborg.agents["Red"] = load_red_agent(load_path=red_agent, scenario=scenario)
    if blue_agent is not None:
        logging.info(f"Loading agent {blue_agent}")
        if blue_agent.lower() == "cardiff":
            cyborg.agents["Blue"] = Cardiff()
        else:
            cyborg.agents["Blue"] = load_blue_agent(
                load_path=blue_agent, scenario=scenario
            )
    else:
        blue_agent = ""

    param_groups = cyborg.agents["Red"].model.optimizer.param_groups
    actor_lr = param_groups[0]["lr"]
    critic_lr = param_groups[1]["lr"]
    msg = f"Starting training for {scenario}, cyborg uuid: {cyborg.uuid}; actor lr: {actor_lr}; critic lr: {critic_lr}"
    print(msg)
    logging.info(msg)

    for i in tqdm(range(max_eps), position=0):
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
                # Penalize invalid actions
                if isinstance(cyborg.get_last_action(player), InvalidAction):
                    r = -1.0
                if j < max_steps - 1:
                    done = terminated or truncated
                else:
                    done = True
                if player in rewards.keys():
                    if player == "Red":
                        if reduce_rewards:
                            last_red_reward = r
                            r -= last_blue_reward
                        rewards[player] += r
                    if player == "Blue":
                        if reduce_rewards:
                            last_blue_reward = r
                            r -= last_red_reward
                        rewards[player] += r
                    if blue_agent.lower() != "cardiff" and player == "Blue":
                        cyborg.agents[player].model.buffer.rewards.append(r)
                        cyborg.agents[player].model.buffer.is_terminals.append(done)
                if player == "Blue" and blue_agent.lower() != "cardiff":
                    mean_loss = cyborg.agents[player].train(observation)
                else:
                    mean_loss = 0

                if mean_loss is not None:
                    losses[player].append(mean_loss)

            if done:
                cyborg.writer.add_scalar("Red Episode Reward", rewards["Red"], i)
                cyborg.writer.add_scalar("Blue Episode Reward", rewards["Blue"], i)
                cyborg.writer.add_scalar(
                    "Red Episode Mean Loss", np.mean(losses["Red"]), i
                )
                if blue_agent.lower() != "cardiff":
                    cyborg.writer.add_scalar(
                        "Blue Episode Mean Loss", np.mean(losses["Blue"]), i
                    )
                cyborg.writer.add_scalar("Episode Length", j + 1, i)
                if blue_agent.lower() == "cardiff":
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
    cyborg.agents["Red"].model.save(f"./checkpoints/{model_subdir}/red.ckpt")
    if blue_agent.lower() != "cardiff":
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
        use_embedding_agents=args.embedding_agent,
        reduce_rewards=args.reduce_rewards,
    )
