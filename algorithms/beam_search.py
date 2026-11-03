"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver, TOL
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = max(1, self.params.get("beam_width", 5))
        self.max_iterations: int = max(0, self.params.get("max_iterations", 150))
        self.same_route_limit: int = max(1, self.params.get("same_route_limit", 2))
        self.patience: int = max(1, self.params.get("patience", 10))

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        first_state = self.problem.generate_initial_state()
        beam = [first_state]
        self.consider(first_state, self.evaluate(first_state))
                attempts = 0
        while len(beam) < self.beam_width and attempts < self.beam_width * 20:
            candidate = random_state(self.problem, self.rng)
            if candidate not in beam:
                beam.append(candidate)
            attempts += 1

        for state in beam:
            self.consider(state, self.evaluate(state))
        self.log_step()

        no_progress = 0
        for _ in range(self.max_iterations):
            candidates = set()

        # Generate neighbors from every state in the current beam.
            for state in beam:
                for neighbor in get_neighbors(state, self.problem):
                    candidates.add(neighbor)

            if not candidates:
                break

        # Keep the best K candidates from the combined list.
            ranked = sorted(candidates, key=lambda item: (self.evaluate(item),
                                                            item.departure_time, item.path))

        # Limit how many states with the same route enter the beam.
            beam = []
            route_counts = {}
            for candidate in ranked:
                count = route_counts.get(candidate.path, 0)
                if count >= self.same_route_limit:
                    continue
                beam.append(candidate)
                route_counts[candidate.path] = count + 1
                if len(beam) >= self.beam_width:
                    break

            if not beam:
                break

            old_best = self.best_cost
            self.consider(beam[0], self.evaluate(beam[0]))
            if self.best_cost < old_best - TOL:
                no_progress = 0
            else:
                no_progress += 1

            self.log_step()
            if no_progress >= self.patience:
                break

        return self.best_state, self.best_cost
        