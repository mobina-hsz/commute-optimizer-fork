"""
algorithms/base.py
Abstract base class for local search solvers.
"""

from abc import ABC, abstractmethod
import math
import random
from typing import Tuple, List, Any, Dict, Optional
from core.state import State
from core.problem import CommuteProblem


TOL = 1e-9


class LocalSearchSolver(ABC):
    def __init__(self, problem: CommuteProblem, **kwargs):
        self.problem = problem
        self.params: Dict[str, Any] = kwargs
        self.history: List[float] = []
        self.rng = random.Random(kwargs.get("seed", random.getrandbits(64)))
        self.best_state: Optional[State] = None
        self.best_cost = math.inf
        self.best_step = 0
        self._cost_cache: Dict[State, float] = {}

    def evaluate(self, state: State) -> float:
        """Evaluate each state only once during a solver run."""
        if state not in self._cost_cache:
            self._cost_cache[state] = self.problem.evaluate(state)
        return self._cost_cache[state]

    def consider(self, state: State, cost: float) -> None:
        """Keep the best state found so far."""
        if cost < self.best_cost - TOL:
            self.best_state = state
            self.best_cost = cost
            self.best_step = len(self.history)

    def log_step(self) -> None:
        self.history.append(self.best_cost)

    @abstractmethod
    def solve(self) -> Tuple[State, float]:
        """
        Executes the search procedure.

        Returns:
            Tuple[State, float]: The best State found and its evaluated objective cost.
        """
        pass
