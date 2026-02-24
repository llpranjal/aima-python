'''
Part III: Hill Climbing, Simulated Annealing, Genetic Algorithms
'''

import sys
import numpy as np
import random
import math
import time

#same helper as part1
def tour_cost(mat, tour):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += mat[tour[i]][tour[i + 1]]
    # return to start
    cost += mat[tour[-1]][tour[0]]
    return cost

#get a random permutaion of cities from 0 to n-1
def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour


#Hill Climbing (random restart one)
def hill_climbing(mat, num_restarts=50):
    #random restarting hill climbing for tsp
    #for each restart,
    # - generate a random tour
    # - pick two random cities and swap them
    # - if it makes the tour better, keep it
    # - if no improvement after a bunch of times then stop this restart
    #return the best tour after all of the restarts, returns (best_tour, best_cost, cost_history)

    n = len(mat)
    best_tour = None
    best_cost = float('inf')
    cost_history = []

    for restart in range(num_restarts):
        #start with random tour
        current_tour = random_tour(n)
        current_cost = tour_cost(mat, current_tour)

        #try swapping pairs to improve
        #do up to n*n attempts without improvement before giving up
        no_improve_count = 0
        max_no_improve = n * n

        while no_improve_count < max_no_improve:
            #pick two random positions to swap
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue

            #swap cities at positions i and j
            new_tour = current_tour[:]
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            new_cost = tour_cost(mat, new_tour)

            if new_cost < current_cost:
                current_tour = new_tour
                current_cost = new_cost
                #reset counter
                no_improve_count = 0
            else:
                no_improve_count += 1

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour

        cost_history.append(best_cost)

    return best_tour, best_cost, cost_history


#Simulated Annealing
def simulated_annealing(mat, alpha=0.995, initial_temp=1000, max_iterations=100000):
    #same idea as hill climbing but sometimes accepts worse solutions on purpose
    #start with random tour
    #get a neighbor by swapping two random cities
    #if neighbor is better accept other wise accept with probability e^((score' - score) / t)
    #                 (where score' is the new cost, score is the old cost, t is temp)
    #decrease temp returns (best_tour, best_cost, cost_history)

    n = len(mat)
    current_tour = random_tour(n)
    current_cost = tour_cost(mat, current_tour)
    best_tour = current_tour[:]
    best_cost = current_cost
    cost_history = []

    temp = initial_temp

    for iteration in range(max_iterations):
        #generate neighbor: swap two random positions
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue

        new_tour = current_tour[:]
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_cost = tour_cost(mat, new_tour)

        #decide whether to accept, positive means worse
        delta = new_cost - current_cost

        if delta < 0:
            #better solution, always accept
            current_tour = new_tour
            current_cost = new_cost
        elif temp > 0:
            #worse solution, accept with the probability
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                current_tour = new_tour
                current_cost = new_cost

        #track best
        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour[:]

        #cool down
        temp *= alpha

        #record history every 1000 iterations
        if iteration % 1000 == 0:
            cost_history.append(best_cost)

    cost_history.append(best_cost)
    return best_tour, best_cost, cost_history


#Genetic Algorithm
def order_crossover(parent1, parent2):
    #pick two random cut points, copy the segment from parent1 to the child
    #fill the remaining positions with cities from parent2 in order

    n = len(parent1)
    start = random.randint(0, n - 1)
    end = random.randint(0, n - 1)
    if start > end:
        start, end = end, start

    #child gets the segment from parent1
    child = [None] * n
    child[start:end + 1] = parent1[start:end + 1]

    #fill remaining from parent2, keeping the order
    used = set(child[start:end + 1])
    fill_pos = (end + 1) % n
    for city in parent2:
        if city not in used:
            child[fill_pos] = city
            fill_pos = (fill_pos + 1) % n

    return child


def mutate(tour, mutation_chance):
    #swap two random cities with mutation_chance probability
    if random.random() < mutation_chance:
        n = len(tour)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        tour[i], tour[j] = tour[j], tour[i]
    return tour


def genetic_algorithm(mat, mutation_chance=0.1, population_size=100, num_generations=500):
    #make the initial population of random tours
    #each generation, select parents, make children, mutate the children, combine parents
    #and choldren, and keep the best
    #return the best tour after all generations, returns (best_tour, best_cost, cost_history)

    n = len(mat)

    #initialize the population
    population = []
    for _ in range(population_size):
        t = random_tour(n)
        c = tour_cost(mat, t)
        population.append((t, c))

    cost_history = []

    for gen in range(num_generations):
        #sort by cost (best first)
        population.sort(key=lambda x: x[1])
        cost_history.append(population[0][1])

        #create children (2 per parent pair)
        children = []
        for _ in range(population_size // 2):
            #pick 3 random, take best
            candidates = random.sample(population, min(3, len(population)))
            parent1 = min(candidates, key=lambda x: x[1])[0]

            candidates = random.sample(population, min(3, len(population)))
            parent2 = min(candidates, key=lambda x: x[1])[0]

            #generate 2 children from each pair
            child1 = order_crossover(parent1, parent2)
            child2 = order_crossover(parent2, parent1)

            #mutate
            child1 = mutate(child1, mutation_chance)
            child2 = mutate(child2, mutation_chance)

            children.append((child1, tour_cost(mat, child1)))
            children.append((child2, tour_cost(mat, child2)))

        #combine parents and children, keep best (elitism)
        combined = population + children
        combined.sort(key=lambda x: x[1])
        population = combined[:population_size]

    #final sort
    population.sort(key=lambda x: x[1])
    cost_history.append(population[0][1])
    best_tour, best_cost = population[0]

    return best_tour, best_cost, cost_history


#for output from main
def run_local(name, func, mat, **kwargs):
    t0_wall = time.time_ns()
    t0_cpu = time.process_time_ns()
    tour, cost, history = func(mat, **kwargs)
    t1_cpu = time.process_time_ns()
    t1_wall = time.time_ns()

    wall = (t1_wall - t0_wall) / 1e9
    cpu = (t1_cpu - t0_cpu) / 1e9

    print(f"{name}:")
    print(f"  Tour: {tour}")
    print(f"  Cost: {cost:.4f}")
    print(f"  Wall time: {wall:.6f} s")
    print(f"  CPU time:  {cpu:.6f} s")
    print()


#main: run all 3 algorithms on the given matrix file
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use in this format: python3 part3_local.py <matrix_file_name>")
        sys.exit(1)

    fname = sys.argv[1]
    mat = np.loadtxt(fname)

    run_local("Hill Climbing (restarts=50)", hill_climbing, mat, num_restarts=50)
    run_local("Simulated Annealing (alpha=0.995, temp=1000, iters=100000)",
              simulated_annealing, mat, alpha=0.995, initial_temp=1000, max_iterations=100000)
    run_local("Genetic Algorithm (mutation=0.1, pop=100, gens=500)",
              genetic_algorithm, mat, mutation_chance=0.1, population_size=100, num_generations=500)