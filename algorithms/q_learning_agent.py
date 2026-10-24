"""
Tabular Q-learning agent for stochastic problems.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple
from core.agent import Agent
from core.game_state import GameState
from core.stochastic_problem import StochasticProblem


class QLearningAgent(Agent):
    """
    Tabular Q-learning.

    The agent does not know the probabilities of the problem: it learns only from
    the transitions (state, action, reward, next_state) it experiences through
    problem.step. It must NOT call problem.get_chance_outcomes.

    The Q table is a dictionary {(state, action): value}. Pairs that are not in the
    dictionary are worth 0.
    """

    def __init__(self, problem: StochasticProblem, alpha: float = 0.1, gamma: float = 1.0,
                 epsilon: float = 0.1, seed: Optional[int] = 42):
        """
        Initialize the agent.

        Args:
            problem: The game to play
            alpha: Learning rate (how far each update moves towards the target)
            gamma: Discount factor
            epsilon: Exploration probability of the epsilon-greedy policy
            seed: Seed of the agent's random number generator (used for exploration
                  and for the environment during training)
        """
        super().__init__(problem)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q: Dict[Tuple[GameState, str], float] = {}

    def get_q(self, state: GameState, action: str) -> float:
        """Current estimate Q(state, action); 0 for pairs never updated."""
        return self.q.get((state, action), 0.0)

    def best_action(self, state: GameState) -> str:
        """Legal action with the largest Q value (the first one in case of ties)."""
        actions = self.problem.get_actions(state)
        return max(actions, key=lambda action: self.get_q(state, action))

    def choose_action(self, state: GameState) -> str:
        """Greedy policy, used to play once the agent has been trained."""
        return self.best_action(state)

    def epsilon_greedy_action(self, state: GameState) -> str:
        """
        Action used while training.
        With probability epsilon, a legal action chosen uniformly at random;
        otherwise, the best action according to Q.
        """

        #   Use self.rng for every random choice (self.rng.random(), self.rng.choice(...)),
        #   so that results are reproducible.

        actions = self.problem.get_actions(state)

        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)

        return self.best_action(state)

    def update(self, state: GameState, action: str, reward: float,
               next_state: GameState, done: bool) -> None:
        """
        Q-learning update for one observed transition.

        Args:
            state: State where the action was taken
            action: Action taken
            reward: Reward obtained
            next_state: Resulting state
            done: Whether next_state is terminal
        """

        #     Q(s, a) <- Q(s, a) + alpha * (y - Q(s, a))
        #   where the target y is
        #     - reward, if next_state is terminal;
        #     - reward + gamma * (maximum of Q(next_state, a') over the LEGAL actions a'
        #       of next_state), otherwise.

        y = reward
        if (not self.problem.is_terminal(next_state)):
            best_next_action = self.best_action(next_state)
            y += self.gamma * self.get_q(next_state, best_next_action)

        q_value = self.get_q(state, action)
        new_value = q_value + self.alpha * (y - q_value)
        self.q[(state, action)] = new_value

    def train(self, episodes: int, report_every: int = 0,
              report: Optional[Callable[[int, "QLearningAgent"], None]] = None) -> List[float]:
        """
        Train the agent by playing complete episodes with the epsilon-greedy policy.

        Args:
            episodes: Number of episodes (games) to play
            report_every: If positive, call report(episode, self) every that many episodes
            report: Function called to report progress

        Returns:
            The total reward of every training episode
        """
        scores = []
        for episode in range(1, episodes + 1):
            state = self.problem.get_initial_state(self.rng)
            score = 0.0
            while not self.problem.is_terminal(state):
                action = self.epsilon_greedy_action(state)
                reward, next_state, done = self.problem.step(state, action, self.rng)
                self.update(state, action, reward, next_state, done)
                score += reward
                state = next_state
            scores.append(score)
            if report is not None and report_every > 0 and episode % report_every == 0:
                report(episode, self)
        return scores
