"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver, TOL
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        for restart in range(self.max_restarts + 1):
            if restart == 0:
                current = self.problem.generate_initial_state()
            else:
                current = random_state(self.problem, self.rng)

            current_cost = self.evaluate(current)
            self.consider(current, current_cost)
            self.log_step()
            sideways_count = 0
            visited = {current}

            while True:
                neighbors = get_neighbors(current, self.problem)
                if not neighbors:
                    break
