'''
Part II: A* with MST Heuristic
'''

import sys
import heapq
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree


#same helper as part1
def tour_cost(mat, tour):
    cost = 0
    n = len(tour)
    for i in range(n - 1):
        cost += mat[tour[i]][tour[i + 1]]
    # return to start
    cost += mat[tour[-1]][tour[0]]
    return cost


#MST Heuristic using scipy
def mst_cost(mat, cities):
    #finds the cost of a MST over the given cities, uses scipy's minimum spanning tree
    #mat is the full adj matrix and cities is the list of city indicies to include in the MST
    #returns the total MST edge weight

    if len(cities) <= 1:
        return 0
    
    #build a submatrix for just these cities
    idx = list(cities)
    submat = mat[np.ix_(idx, idx)]

    #finds MST of given graph and returns total edge weight
    tree = minimum_spanning_tree(submat)
    return tree.toarray().sum()


def heuristic(mat, current_city, visited, start_city, n):
    #our heuristic must be admissible,
    #h(n) = MST of univisited cities + cheapest edge from curr city to any unvisited city 
    #                                + cheapest edge from any unvisited city to start city
    #if all are visited then h = cost to return back to start

    unvisited = []
    for c in range(n):
        if c not in visited:
            unvisited.append(c)

    if len(unvisited) == 0:
        #all visited, need to return to start
        return mat[current_city][start_city]

    #MST cost over unvisited cities
    mst = mst_cost(mat, unvisited)

    #minedge from current city to any unvisited
    min_to_unvisited = min(mat[current_city][c] for c in unvisited)

    #min edge from any unvisited city back to the start city
    min_from_unvisited = min(mat[c][start_city] for c in unvisited)

    return mst + min_to_unvisited + min_from_unvisited


#A* Search for TSP
def astar_tsp(mat):

    #state is (curr_city, set of visited cities)
    #frozenset is just like an immutable set (no modifying after creating)
    #start from city 0, go to any unvisited city and visit all cities and return back to city 0
    #heap/priority q has (f_cost, g_cost, curr_city, visited_set, path)
    #returns (best_tour, best_cost, nodes_expanded)

    n = len(mat)
    start = 0

    #initial state, at city 0 with only city 0 visited
    visited_init = frozenset([start])
    h0 = heuristic(mat, start, visited_init, start, n)
    g0 = 0
    f0 = g0 + h0

    #use a counter to break ties
    counter = 0
    pq = [(f0, counter, g0, start, visited_init, [start])]
    counter += 1

    #keep track of best g-cost for each state
    best_g = {}
    best_g[(start, visited_init)] = 0

    nodes_expanded = 0
    all_visited = frozenset(range(n))

    while pq:
        #for the '-', dont need counter after popping
        f, _, g, current, visited, path = heapq.heappop(pq)

        #check if this state has been reached already with a better cost
        state_key = (current, visited)
        if g > best_g.get(state_key, float('inf')):
            continue

        nodes_expanded += 1

        #goal test, all cities visited?
        if visited == all_visited:
            #complete the tour by returning to start
            total_cost = g + mat[current][start]
            return path, total_cost, nodes_expanded

        #expand by trying to go to each unvisited city
        for next_city in range(n):
            if next_city not in visited:
                new_g = g + mat[current][next_city]
                new_visited = visited | frozenset([next_city])
                new_state = (next_city, new_visited)

                #only add if this is better than what we've seen
                if new_g < best_g.get(new_state, float('inf')):
                    best_g[new_state] = new_g
                    h = heuristic(mat, next_city, new_visited, start, n)
                    new_f = new_g + h
                    new_path = path + [next_city]
                    heapq.heappush(pq, (new_f, counter, new_g, next_city, new_visited, new_path))
                    counter += 1

    #should not get here for a valid TSP
    return None, float('inf'), nodes_expanded


#for output from main
def run_astar(mat):
    t0_wall = time.time_ns()
    t0_cpu = time.process_time_ns()
    tour, cost, nodes_expanded = astar_tsp(mat)
    t1_cpu = time.process_time_ns()
    t1_wall = time.time_ns()

    wall = (t1_wall - t0_wall) / 1e9
    cpu = (t1_cpu - t0_cpu) / 1e9

    if tour is not None:
        print("A* Search:")
        print(f"  Tour: {tour}")
        print(f"  Cost: {cost:.4f}")
        print(f"  Nodes expanded: {nodes_expanded}")
        print(f"  Wall time: {wall:.6f} s")
        print(f"  CPU time:  {cpu:.6f} s")
    else:
        print("A* didn't find a solution")


#main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use in this format: python3 part2_astar.py <matrix_file_name>")
        sys.exit(1)

    fname = sys.argv[1]
    mat = np.loadtxt(fname)
    n = len(mat)

    if n > 15:
        print(f"A* with {n} cities can run very slow, might have to wait for results")

    run_astar(mat)