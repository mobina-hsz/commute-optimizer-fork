import random
from typing import List, Optional, Tuple
from core.state import State
from core.problem import CommuteProblem


MAX_DETOUR_EDGES = 4
MAX_REPLACED_EDGES = 3


def random_state(problem: CommuteProblem, rng: Optional[random.Random] = None) -> State:
    rng = rng or random

    def build_path(path: List[str]):
        if path[-1] == problem.destination:
            return tuple(path)

        choices = [node for node in problem.network.get_neighbors(path[-1])
                   if node not in path]
        rng.shuffle(choices)
        for node in choices:
            path.append(node)
            result = build_path(path)
            if result is not None:
                return result
            path.pop()
        return None

    path = build_path([problem.origin])
    if path is not None and problem.is_valid_path(path):
        times = problem.get_valid_time_domain()
        return State(path, rng.choice(times))

    return problem.generate_initial_state()


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:

    path = state.path
    new_paths = set()

    for start_index in range(len(path) - 1):
        last_end = min(len(path) - 1, start_index + MAX_REPLACED_EDGES)
        for end_index in range(start_index + 1, last_end + 1):
            start_node = path[start_index]
            end_node = path[end_index]
            blocked = set(path[:start_index]) | set(path[end_index + 1:])

            def extend(partial):
                current = partial[-1]
                if current == end_node:
                    new_path = path[:start_index] + tuple(partial) + path[end_index + 1:]
                    if new_path != path and len(new_path) == len(set(new_path)):
                        new_paths.add(new_path)
                    return
                if len(partial) - 1 >= MAX_DETOUR_EDGES:
                    return
                for next_node in problem.network.get_neighbors(current):
                    if next_node in blocked or next_node in partial:
                        continue
                    partial.append(next_node)
                    extend(partial)
                    partial.pop()

            extend([start_node])

    times = [state.departure_time,
             state.departure_time - problem.t_step,
             state.departure_time + problem.t_step]
    candidates = []

    for new_path in new_paths:
        for new_time in times:
            candidate = State(new_path, new_time)
            if problem.is_valid_state(candidate):
                candidates.append(candidate)

    for new_time in times[1:]:
        candidate = State(path, new_time)
        if problem.is_valid_state(candidate):
            candidates.append(candidate)

    neighbors = []
    seen = {state}
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            neighbors.append(candidate)
    return neighbors

def crossover(parent1: State, parent2: State, problem: CommuteProblem,
              rng: Optional[random.Random] = None) -> State:

    rng = rng or random
    path1 = parent1.path
    path2 = parent2.path

    valid_times = problem.get_valid_time_domain()
    average_time = (parent1.departure_time + parent2.departure_time) / 2
    new_time = min(valid_times, key=lambda value: abs(value - average_time))

    common_nodes = []
    for node in path1[1:-1]:
        if node in path2[1:-1]:
            common_nodes.append(node)

    rng.shuffle(common_nodes)
    for junction in common_nodes:
        index1 = path1.index(junction)
        index2 = path2.index(junction)
        new_path = path1[:index1] + path2[index2:]
        child = State(new_path, new_time)
        if problem.is_valid_state(child):
            return child

    if problem.is_valid_state(parent1):
        return State(path1, new_time)
    return State(path2, new_time)
