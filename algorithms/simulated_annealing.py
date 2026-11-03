"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

from typing import Tuple
import math
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Guidelines:
        - Start from a high temperature where worse moves are accepted with significant probability.
        - Lower temperature over iterations according to a cooling schedule.
        - Downhill moves (improvements) must always be accepted.
        - Uphill moves (worse costs) should be accepted probabilistically according to temperature.
        - Record objective values throughout the run in self.history.

        Prohibited:
        - Ignoring the temperature parameter or turning this into pure hill climbing.
        """
        current = self.problem.generate_initial_state()
        current_cost = self.evaluate(current)
        self.consider(current, current_cost)
        self.log_step()
        temperature = self.initial_temp

        for _ in range(self.max_iterations):
            if temperature <= self.min_temp:
                break

            neighbors = get_neighbors(current, self.problem)
            if not neighbors:
                break

            candidate = self.rng.choice(neighbors)
            candidate_cost = self.evaluate(candidate)
            difference = candidate_cost - current_cost

            if difference < 0:
                accept = True
            else:
                probability = math.exp(-difference / temperature)
                accept = self.rng.random() < probability

            if accept:
                current = candidate
                current_cost = candidate_cost

            self.consider(current, current_cost)
            self.log_step()
            temperature = temperature * self.cooling_rate

        return self.best_state, self.best_cost
