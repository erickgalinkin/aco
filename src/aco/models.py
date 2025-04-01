import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pad(array, max_len):
    """
    Assume zero padding on the right side of the array.
    :param array: np.array to be padded
    :param max_len: length to pad array to
    :return: padded array
    """
    diff = max_len - array.shape[0]
    padding_array = np.array([0] * diff)
    padded_array = np.hstack([array, padding_array])
    return padded_array


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        has_continuous_action_space,
        action_std_init,
    ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init
            ).to(DEVICE)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1),
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std
            ).to(DEVICE)
        else:
            logging.warning(
                "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy"
            )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(DEVICE)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        k_epochs=4,
        lr_actor=0.0001,
        lr_critic=0.0003,
        gamma=0.99,
        eps_clip=0.1,
        has_continuous_action_space=False,
        action_std_init=0.6,
        batch_size=32,
    ):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim,
            action_dim,
            hidden_dim,
            has_continuous_action_space,
            action_std_init,
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim,
            action_dim,
            hidden_dim,
            has_continuous_action_space,
            action_std_init,
        ).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.batch_size = batch_size
        logging.info(f"Initialized PPO on {DEVICE}")

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            logging.warning(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                logging.warning(
                    f"setting actor output action_std to min_action_std : {self.action_std}"
                )
            else:
                logging.info(f"setting actor output action_std to {self.action_std}")
            self.set_action_std(self.action_std)

        else:
            logging.warning(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
            )

    def select_action(self, state):
        if len(state) != self.policy_old.state_dim:
            if len(state[0]) == self.policy_old.state_dim:
                state = state[0]
            else:
                raise ValueError(
                    f"Expected {self.policy_old.state_dim} dimensions but got a {type(state)} of size {len(state)}!"
                )
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(DEVICE)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(DEVICE)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(DEVICE)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        mean_loss = 0
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )
            mean_loss += loss.mean().detach().cpu().numpy().item()

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return mean_loss

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


class ProjectionActor(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        max_input_len=128,
    ):
        super().__init__()
        self.current_state_dim = state_dim
        self.projection = nn.Linear(state_dim, max_input_len)
        self.input_layer = nn.Linear(max_input_len, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_input_len = max_input_len

    def forward(self, x):
        if x.shape[0] != self.current_state_dim:
            self.current_state_dim = x.shape[0]
            self.projection = nn.Linear(self.current_state_dim, self.hidden_dim)
        x = self.projection(x)
        x = self.input_layer(x)
        x = F.tanh(self.fc(x))
        x = F.tanh(self.fc(x))
        x = F.softmax(self.output_layer(x), -1)
        return x.squeeze()


class ProjectionCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        max_input_len=128,
    ):
        super().__init__()
        self.current_state_dim = state_dim
        self.projection = nn.Linear(state_dim, max_input_len)
        self.input_layer = nn.Linear(max_input_len, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.max_input_len = max_input_len

    def forward(self, x):
        if x.shape[0] != self.current_state_dim:
            self.current_state_dim = x.shape[0]
            self.projection = nn.Linear(self.current_state_dim, self.hidden_dim)
        x = self.projection(x)
        x = self.input_layer(x)
        x = F.tanh(self.fc(x))
        x = F.tanh(self.fc(x))
        x = self.output_layer(x)
        return x.squeeze()


class ProjectionActorCritic(ActorCritic):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        max_input_len=128,
        has_continuous_action_space=False,
        action_std_init=0.6,
    ):
        super().__init__(
            state_dim,
            hidden_dim,
            action_dim,
            has_continuous_action_space,
            action_std_init,
        )
        # Actor
        self.actor = ProjectionActor(state_dim, hidden_dim, action_dim, max_input_len)
        # Critic
        self.critic = ProjectionActor(state_dim, hidden_dim, max_input_len)
        self.actor.to(DEVICE)
        self.critic.to(DEVICE)
        self.state_dim = state_dim
        self.max_input_len = max_input_len

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class DynamicStatePPO(PPO):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_input_len=128,
        hidden_dim=64,
        k_epochs=6,
        lr_actor=0.0005,
        lr_critic=0.0008,
        gamma=0.999,
        eps_clip=0.3,
        has_continuous_action_space=False,
        action_std_init=0.6,
        batch_size=32,
    ):
        if has_continuous_action_space:
            raise NotImplementedError(
                "DynamicStatePPO does not support continuous action space"
            )
        super().__init__(
            state_dim,
            action_dim,
            hidden_dim,
            k_epochs,
            lr_actor,
            lr_critic,
            gamma,
            eps_clip,
            has_continuous_action_space,
            action_std_init,
            batch_size,
        )

        self.max_input_len = max_input_len
        self.policy = ProjectionActorCritic(
            state_dim,
            hidden_dim,
            action_dim,
            max_input_len,
            has_continuous_action_space,
            action_std_init,
        ).to(DEVICE)
        self.policy_old = ProjectionActorCritic(
            state_dim,
            hidden_dim,
            action_dim,
            max_input_len,
            has_continuous_action_space,
            action_std_init,
        ).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        logging.info(f"Initialized DynamicStatePPO on {DEVICE}")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
