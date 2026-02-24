'''
Part I: Nearest Neighbor, Nearest Neighbor 2-Opt, Repeated Randomness, Nearest Neighbor
'''

import sys
import numpy as np
import random
import time


#helper that computes total tour cost (round trip)
def tour_cost(mat, tour):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += mat[tour[i]][tour[i + 1]]
    # return to start
    cost += mat[tour[-1]][tour[0]]
    return cost


#Nearest Neighbor Algorithm
def nearest_neighbor(mat, start=0):
    #start at the 'start' city, goes to the closest unvisitied city then returns
    #to start when done, returns (tour, cost)

    n = len(mat)
    visited = [False] * n
    tour = [start]
    visited[start] = True

    current = start
    for _ in range(n - 1):
        #find the nearest unvisited city
        best_next = -1
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and mat[current][j] < best_dist:
                best_dist = mat[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True
        current = best_next

    cost = tour_cost(mat, tour)
    return tour, cost




#Nearest Neighbor + 2-Opt
def two_opt_swap(tour, i, j):
    #reverses the segment between i and j ex.
    #[0, 1, 2, 3, 4, 5], two_opt_swap(tour, 1, 4) gives [0, 4, 3, 2, 1, 5]
    new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
    return new_tour


def nearest_neighbor_2opt(mat, start=0):
    #run nearest neighbor then improve with 2-opt, checks all pairs (i, j) to see
    #if reversing the segment improves the tour, if there is one then do it then restart
    #from the beginning, returns (tour, cost)

    tour, cost = nearest_neighbor(mat, start)
    n = len(tour)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_tour = two_opt_swap(tour, i, j)
                new_cost = tour_cost(mat, new_tour)
                if new_cost < cost:
                    tour = new_tour
                    cost = new_cost
                    improved = True
                    #restart from beginning because reversing changes tour,
                    #earlier pairs need to be re-evaluated
                    break  
            if improved:
                break

    return tour, cost




#Repeated Random Nearest Neighbor (RRNN)
def random_nearest_neighbor(mat, k=3):
    #like nearest neighbor but randomly picks from the k closest unvisited cities
    #uses 2-opt improvement along the way, returns (tour, cost)

    n = len(mat)
    # pick a random starting city
    start = random.randint(0, n - 1)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        #find all unvisited cities and their distances
        unvisited = []
        for j in range(n):
            if not visited[j]:
                unvisited.append((mat[current][j], j))
        # sort by distance
        unvisited.sort()

        #pick randomly from the k closest (or less if not enough)
        choices = min(k, len(unvisited))
        pick = random.randint(0, choices - 1)
        next_city = unvisited[pick][1]

        tour.append(next_city)
        visited[next_city] = True
        current = next_city

    #do the 2-opt improvement
    cost = tour_cost(mat, tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_tour = two_opt_swap(tour, i, j)
                new_cost = tour_cost(mat, new_tour)
                if new_cost < cost:
                    tour = new_tour
                    cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return tour, cost


def repeated_random_nearest_neighbor(mat, k=3, num_repeats=20):
    #runs random_nearest_neighbor, num_repeats amount of times,
    #we then keep the best, returns(best_tour, best_cost)

    best_tour = None
    best_cost = float('inf')

    for _ in range(num_repeats):
        tour, cost = random_nearest_neighbor(mat, k)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost



#for output from main
def run_algorithm(name, func, mat, **kwargs):
    t0_wall = time.time_ns()
    t0_cpu = time.process_time_ns()
    tour, cost = func(mat, **kwargs)
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
        print("Use in this format: python3 part1_nn.py <matrix_file_name>")
        sys.exit(1)

    fname = sys.argv[1]
    mat = np.loadtxt(fname)

    run_algorithm("Nearest Neighbor", nearest_neighbor, mat, start=0)
    run_algorithm("Nearest Neighbor + 2 Opt", nearest_neighbor_2opt, mat, start=0)
    run_algorithm("Repeated Random Nearest Neighbor",repeated_random_nearest_neighbor, mat, k=3, num_repeats=20)