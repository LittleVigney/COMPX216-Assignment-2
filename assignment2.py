from search import *
from random import randint
from assignment2aux import *
import numpy as np


def read_tiles_from_file(filename):
    # Task 1
    # Return a tile board constructed using a configuration in a file.

    # Store tile board
    tile_map = list()

    f = open(filename, 'r')
    while f:
        line = f.readline()
        if not line:
            break
        temp_list = list()
        for ele in line:
            match ele:
                case " ":
                    temp_list.append(tuple(()))
                case "i":
                    temp_list.append(tuple((0,)))
                case "L":
                    temp_list.append(tuple((0, 1)))
                case "I":
                    temp_list.append(tuple((0, 2)))
                case "T":
                    temp_list.append(tuple((0, 1, 2)))
        tile_map.append(tuple(temp_list))
    f.close()

    return tuple(tile_map)


class KNetWalk(Problem):
    def __init__(self, tiles):
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles
        height = len(self.tiles)
        width = len(self.tiles[0])

        # max fitness: number of when all tile can connect with their all neighbor
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)
        super().__init__(self.generate_random_state())

    def generate_random_state(self):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]

    def actions(self, state):
        height = len(self.tiles)
        width = len(self.tiles[0])

        # For every tile, three possible orientation (except current orientation)
        # were added to action list
        return [(i, j, k) for i in range(height) for j in range(width) for k in [0, 1, 2, 3] if
                state[i * width + j] != k]

    def result(self, state, action):
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]

    def goal_test(self, state):
        return self.value(state) == self.max_fitness

    def value(self, state):
        # Task 2
        # Return an integer fitness value of a given state.

        dx = [0, -1, 0, 1]
        dy = [1, 0, -1, 0]

        # In a correct connection, 0 connect with 2, 1 connect with 3,
        # 2 connect with 0, 3 connect with 1
        correct_conn = [2, 3, 0, 1]

        height = len(self.tiles)
        width = len(self.tiles[0])

        # fitness of current state
        current_fitness = 0

        # Check all tile
        for i in range(height):
            for j in range(width):
                if type(state) is not list:
                    break
                # Get real orientation from state and self.tiles
                current_ori = state[i * width + j]
                true_ori = [(ele + current_ori) % 4 for ele in self.tiles[i][j]]

                # Check 4 direction
                for dir in range(4):
                    xx = i + dx[dir]
                    yy = j + dy[dir]

                    # If this direction exist
                    if dir in true_ori and 0 <= xx < height and 0 <= yy < width:
                        # Get the real orientation of neighbor in this direction
                        neighbor_ori = state[xx * width + yy]
                        true_neighbor_ori = tuple((con + neighbor_ori) % 4 for con in self.tiles[xx][yy])
                        # true_neighbor_ori = [(ele + neighbor_ori) % 4 for ele in self.tiles[xx][yy]]

                        # Update fitness if there is a correct connection
                        if correct_conn[dir] in true_neighbor_ori:
                            current_fitness += 1

        return current_fitness


# Task 3
# Configure an exponential schedule for simulated annealing.
sa_schedule = exp_schedule(k=75, lam=0.08, limit=200)

# Task 4
# Configure parameters for the genetic algorithm.
pop_size = 15
num_gen = 1000
mutation_prob = 0.1


def local_beam_search(problem, population):
    # Task 5
    # Implement local beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the next population contains no fitter state.
    # Replace the line below with your code.

    # get beam width
    beam_width = len(population)

    # store population in priority queue
    # sort population by their value
    curr_q = []
    for peo in population:
        curr_q.append((problem.value(peo), peo))
    curr_q.sort(reverse=True)

    while True:
        temp_q = []

        # get all possible child state and store in temp_q
        for item in curr_q:
            actions = problem.actions(item[1])

            for act in actions:
                child = problem.result(item[1], act)

                temp_q.append((problem.value(child), child))

        temp_q.sort(reverse=True)

        # if reach goal fitness, return that state
        if temp_q[0][0] == problem.max_fitness:
            return temp_q[0][1]

        # if no better instance, return the best one
        elif temp_q[0][0] < curr_q[0][0]:
            return temp_q[0][1]

        # else select some best children as next population, number is beam_width
        else:
            curr_q.clear()
            for every_child in range(min(beam_width, len(temp_q))):
                curr_q.append(temp_q[every_child])


def stochastic_beam_search(problem, population, limit=1000):
    # Task 6
    # Implement stochastic beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the generation limit is reached.
    # Replace the line below with your code.

    beam_width = len(population)

    curr_q = []
    for peo in population:
        curr_q.append((problem.value(peo), peo))
    curr_q.sort(reverse=True)

    for i in range(limit):
        temp_q = []

        # get all possible child state and store in temp_q
        for item in curr_q:
            if not isinstance(item[1], list):
                continue

            actions = problem.actions(item[1])

            # print(actions)
            for act in actions:
                child = problem.result(item[1], act)

                # print(child)
                temp_q.append((problem.value(child), child))

        if not temp_q:
            break

        temp_q.sort(reverse=True)

        # if best child reach goal fitness, finish search
        if temp_q[0][0] == problem.max_fitness:
            return temp_q[0][1]
        # else find the best child of number beam width as next population
        else:
            # calculate probability of every child
            min_value = temp_q[-1][0]

            value_diff = [(child[0] - min_value) ** 2 for child in temp_q]
            diff_sum = sum(value_diff)

            temp_q_array = np.array([child[1] for child in temp_q])

            # if all child has same fitness
            if diff_sum != 0:
                child_prob = [prob / diff_sum for prob in value_diff]

                select_index = list(np.random.choice([a for a in range(len(temp_q_array))],
                                                     min(beam_width, len(temp_q_array)),
                                                     False, child_prob))
            # if all child has no same fitness
            else:
                select_index = list(np.random.choice([a for a in range(len(temp_q_array))],
                                                     min(beam_width, len(temp_q_array)),
                                                     False))

            # take child from temp_q as next population
            curr_q = [temp_q[idx] for idx in select_index]

    # return the best state after search
    return curr_q[0][1]


if __name__ == '__main__':
    # Task 1 test code
    # network = KNetWalk('assignment2config.txt')
    # visualise(network.tiles, network.initial)

    # Task 2 test code

    # run = 0
    # method = 'hill climbing'
    # while True:
    #     network = KNetWalk('assignment2config.txt')
    #     state = hill_climbing(network)
    #     if network.goal_test(state):
    #         break
    #     else:
    #         print(f'{method} run {run}: no solution found')
    #         print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
    #         visualise(network.tiles, state)
    #     run += 1
    # print(f'{method} run {run}: solution found')
    # visualise(network.tiles, state)

    # Task 3 test code

    # run = 0
    # method = 'simulated annealing'
    # while True:
    #     network = KNetWalk('assignment2config.txt')
    #     state = simulated_annealing(network, schedule=sa_schedule)
    #     if network.goal_test(state):
    #         break
    #     else:
    #         print(f'{method} run {run}: no solution found')
    #         print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
    #         visualise(network.tiles, state)
    #     run += 1
    # print(f'{method} run {run}: solution found')
    # visualise(network.tiles, state)

    # Task 4 test code

    # run = 0
    # method = 'genetic algorithm'
    # while True:
    #     network = KNetWalk('assignment2config.txt')
    #     height = len(network.tiles)
    #     width = len(network.tiles[0])
    #     state = genetic_algorithm([network.generate_random_state() for _ in range(pop_size)], network.value, [0, 1, 2, 3], network.max_fitness, num_gen, mutation_prob)
    #     if network.goal_test(state):
    #         break
    #     else:
    #         print(f'{method} run {run}: no solution found')
    #         print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
    #         visualise(network.tiles, state)
    #     run += 1
    # print(f'{method} run {run}: solution found')
    # visualise(network.tiles, state)

    # Task 5 test code

    # run = 0
    # method = 'local beam search'
    # while True:
    #     network = KNetWalk('assignment2config.txt')
    #     height = len(network.tiles)
    #     width = len(network.tiles[0])
    #     state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
    #     if network.goal_test(state):
    #         break
    #     else:
    #         print(f'{method} run {run}: no solution found')
    #         print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
    #         visualise(network.tiles, state)
    #     run += 1
    # print(f'{method} run {run}: solution found')
    # visualise(network.tiles, state)

    # Task 6 test code

    # run = 0
    # method = 'stochastic beam search'
    # while True:
    #     network = KNetWalk('assignment2config.txt')
    #     height = len(network.tiles)
    #     width = len(network.tiles[0])
    #     state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
    #     if network.goal_test(state):
    #         break
    #     else:
    #         print(f'{method} run {run}: no solution found')
    #         print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
    #         visualise(network.tiles, state)
    #     run += 1
    # print(f'{method} run {run}: solution found')
    # visualise(network.tiles, state)
