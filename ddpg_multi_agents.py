import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512       # minibatch size
GAMMA = 0.99            # 0.99 discount factor
TAU = 1e-3              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 2        # Update the networks 10 times after every 20 timesteps
LEARN_NUMBER = 3       # Update the networks 10 times after every 20 timesteps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgents():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.agents = [Agent(state_size, action_size, i, n_agents, random_seed) for i in range(n_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, rewards, next_states, dones, timestamp):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            # Learn only LEARN_EVERY steps
            if timestamp % LEARN_EVERY == 0:
                for _ in range(LEARN_NUMBER):
                    for agent in self.agents:
                        experiences = self.memory.sample()
                        self.learn(experiences, agent, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        # TODO make this look better
        actions = [np.squeeze(agent.act(np.expand_dims(state, axis=0), add_noise), axis=0) for agent, state in zip(self.agents, states)]
        return np.stack(actions)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, agent, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #TODO makes this look better
        actions_target =[agent_j.actor_target(states.index_select(1, torch.tensor([j]).to(device)).squeeze(1)) for j, agent_j in enumerate(self.agents)]
        
        agent_action_pred = agent.actor_local(states.index_select(1, agent.index).squeeze(1))
        actions_pred = [agent_action_pred if j==agent.index.numpy()[0] else actions.index_select(1, torch.tensor([j]).to(device)).squeeze(1) for j, agent_j in enumerate(self.agents)]
        
        agent.learn(experiences,
                    gamma,
                    actions_target,
                    actions_pred)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)