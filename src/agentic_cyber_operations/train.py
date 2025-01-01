import inspect
import logging
from tqdm import tqdm
from CybORG import CybORG
from wrappers import MultiAgentChallengeWrapper

MAX_STEPS_PER_GAME = 2000
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

    for i in tqdm(range(MAX_EPS), position=0):  # playing multiple games
        rewards = {"Red": 0, "Blue": 0}
        for j in tqdm(
            range(MAX_STEPS_PER_GAME), position=1, leave=False
        ):  # step in 1 game
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


if __name__ == "__main__":
    run_training_example("Scenario1b")
