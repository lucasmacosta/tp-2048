"""
Depth-limited minimax agent for stochastic games.

The game tree alternates two kinds of nodes:
  - MAX nodes: states where the player chooses an action (max_value).
  - chance nodes: afterstates where the environment places the new tile (chance_value).

Minimax treats the environment as an adversary that always places the tile in the
worst possible way for the player.
"""

from core.agent import Agent
from core.game_state import GameState
from core.stochastic_problem import StochasticProblem


class MinimaxAgent(Agent):
    """
    Depth-limited minimax agent.

    The value of a state counts the rewards that will be obtained from it on:
      - a terminal state is worth 0 (no more rewards);
      - when the depth runs out, a non-terminal state is worth problem.heuristic(state);
      - otherwise, the player picks the action that maximizes
            reward(action) + value of the resulting afterstate;
      - and the value of an afterstate is the value of its WORST possible outcome,
        looking one move less ahead.

    The depth counts player moves: with depth = 1 the agent looks at its own move and
    at where the new tile may appear, and then evaluates the resulting boards with
    the heuristic.
    """

    def __init__(self, problem: StochasticProblem, depth: int = 2):
        """
        Initialize the agent.

        Args:
            problem: The game to play
            depth: Number of player moves to look ahead (at least 1)
        """
        super().__init__(problem)
        assert depth >= 1, "Depth must be at least 1"
        self.depth = depth
        self.expanded_nodes = 0     # number of calls to max_value (do not remove)

    def choose_action(self, state: GameState) -> str:
        """Choose the action with the best value, looking self.depth moves ahead."""
        best_action, best_value = None, float("-inf")
        for action in self.problem.get_actions(state):
            afterstate, reward = self.problem.apply_action(state, action)
            value = reward + self.chance_value(afterstate, self.depth)
            if value > best_value:
                best_action, best_value = action, value
        return best_action

    def max_value(self, state: GameState, depth: int) -> float:
        """
        Value of a state in which the player moves, looking `depth` moves ahead.

        Args:
            state: State where the player has to move (it may be terminal)
            depth: Remaining player moves to look ahead (0 = use the heuristic)
        """
        self.expanded_nodes += 1

        if (self.problem.is_terminal(state)):
            return 0

        if (depth == 0):
            return self.problem.heuristic(state)

        max_score = float('-inf')
        for action in self.problem.get_actions(state):
            afterstate, reward = self.problem.apply_action(state, action)
            score = reward + self.chance_value(afterstate, depth)
            max_score = max(max_score, score)

        return max_score
        
        # TODO (Parte 1): implement the MAX node.
        #   - terminal state                -> 0
        #   - depth == 0                    -> self.problem.heuristic(state)
        #   - otherwise, the maximum over the legal actions of
        #         reward + self.chance_value(afterstate, depth)
        #     where (afterstate, reward) = self.problem.apply_action(state, action)
        # raise NotImplementedError("Parte 1: implementar MinimaxAgent.max_value")

    def chance_value(self, afterstate: GameState, depth: int) -> float:
        """
        Value of an afterstate, before the new tile appears. Minimax assumes the tile
        appears in the worst possible place for the player.

        Args:
            afterstate: Board after the player's move, without the new tile
            depth: Depth of the move that produced this afterstate
        """

        min_score = float('inf')
        for next_state, _score in self.problem.get_chance_outcomes(afterstate):
            max_value = self.max_value(next_state, depth - 1)
            min_score = min(min_score, max_value)
        return min_score

