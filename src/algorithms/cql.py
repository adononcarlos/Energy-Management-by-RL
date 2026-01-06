"""
Conservative Q-Learning (CQL) - Offline RL Algorithm.

Reference: Kumar et al., 2020
https://arxiv.org/abs/2006.04779
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from collections import deque
import os


class OfflineReplayBuffer:
    """Offline replay buffer (no sampling with replacement)."""
    
    def __init__(self, max_size: int = 1e6):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.max_size = int(max_size)
    
    def add(self, state, action, reward, next_state, done):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.actions[i] for i in indices]),
            np.array([self.rewards[i] for i in indices]).reshape(-1, 1),
            np.array([self.next_states[i] for i in indices]),
            np.array([self.dones[i] for i in indices]).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.states)


class CriticQ(nn.Module):
    """Q-network for CQL."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class CQL:
    """Conservative Q-Learning (Offline RL)."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        cql_weight: float = 1.0,
        cql_temp: float = 1.0,
        num_random_actions: int = 10,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_weight = cql_weight
        self.cql_temp = cql_temp
        self.num_random_actions = num_random_actions
        self.device = torch.device(device)
        
        # Networks
        self.q_network = CriticQ(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_network = CriticQ(state_dim, action_dim, hidden_dim).to(self.device)
        self._copy_weights(self.q_network, self.target_q_network)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
    
    def _copy_weights(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _soft_update(self, source, target, tau: float = 0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select action (CQL is offline, so mostly deterministic)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Greedy action selection
        action = np.random.uniform(-1, 1, self.action_dim)  # Placeholder
        return action
    
    def train_step(self, replay_buffer: OfflineReplayBuffer, batch_size: int = 256):
        """One training step with CQL penalty."""
        if len(replay_buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute target Q-value
        with torch.no_grad():
            next_q = self.target_q_network(next_states, actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        
        # Q-learning loss
        q_pred = self.q_network(states, actions)
        td_loss = ((q_pred - target_q) ** 2).mean()
        
        # CQL penalty: penalize Q-values on random actions
        random_actions = torch.FloatTensor(
            np.random.uniform(-1, 1, (batch_size * self.num_random_actions, self.action_dim))
        ).to(self.device)
        
        repeated_states = states.repeat(self.num_random_actions, 1)
        q_random = self.q_network(repeated_states, random_actions)
        
        cql_penalty = torch.logsumexp(q_random / self.cql_temp, dim=0).mean()
        cql_penalty -= q_pred.mean()
        
        # Combined loss
        loss = td_loss + self.cql_weight * cql_penalty
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target
        self._soft_update(self.q_network, self.target_q_network)
        
        return {
            'td_loss': td_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'total_loss': loss.item(),
        }
    
    def save(self, path: str):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path: str):
        """Load checkpoint."""
        self.q_network.load_state_dict(torch.load(path))
        self._copy_weights(self.q_network, self.target_q_network)