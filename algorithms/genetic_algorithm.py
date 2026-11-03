from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)
        self.tournament_size: int = self.params.get("tournament_size", 3)

    def select_parent(self, population):
        """Choose the best member from a small random tournament."""
        size = max(1, min(self.tournament_size, len(population)))
        group = self.rng.sample(population, size)
        return min(group, key=self.evaluate)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        # Create the first population of valid commute plans.
        population = [self.problem.generate_initial_state()]
        attempts = 0
        while len(population) < self.population_size and attempts < self.population_size * 30:
            candidate = random_state(self.problem, self.rng)
            if candidate not in population:
                population.append(candidate)
            attempts += 1

        # If the graph has few unique states, duplicates are better than an undersized population.
        while len(population) < self.population_size:
            population.append(random_state(self.problem, self.rng))

        first_best = min(population, key=self.evaluate)
        self.consider(first_best, self.evaluate(first_best))
        self.log_step()

        for _ in range(self.generations):
            population.sort(key=self.evaluate)
        # Keep the best individuals unchanged (elitism).
            new_population = population[:min(self.elitism, self.population_size)]
            while len(new_population) < self.population_size:
                # Tournament selection: choose the better of two random plans.
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)