"""
Demo for A* with EightPuzzle
For use with the AIMA codebase: https://github.com/aimacode/aima-python
CMSC 421 - Fall 2025
"""
from time import time
from search import EightPuzzle, astar_search

# INITIAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0) # already solved
INITIAL_STATE = (1, 2, 3, 4, 5, 6, 7, 0, 8) # just one swap to solve
# INITIAL_STATE = (1, 2, 3, 4, 0, 6, 7, 5, 8) # just two swaps to solve
# INITIAL_STATE = (1, 2, 3, 0, 4, 6, 7, 5, 8) # just three swaps to solve
# INITIAL_STATE = (8, 3, 2, 4, 5, 7, 1, 6, 0) # more complex initial state

EP = EightPuzzle(INITIAL_STATE)
print('Initial State: ' + str(INITIAL_STATE))
print('Initial State solvable: ' + str(EP.check_solvability(INITIAL_STATE)))
print('Running A*...')
t0 = time()
astar_search(EP, display=True)
print('Solved in %f seconds'%(time()-t0))