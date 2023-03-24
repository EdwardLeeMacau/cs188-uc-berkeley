# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    frontier = util.Stack()
    backtrack = dict()
    explored_set = set()

    # Graph search initialization
    start_state = problem.getStartState()
    frontier.push(start_state)
    backtrack[start_state] = (None, None)

    while not frontier.isEmpty():
        # Obtain next to-be-evaluated state
        state = frontier.pop()

        # If the current state is explored, skip it
        if state in explored_set:
            continue

        # If the state is goal state, backtrack and learn how to obtain it
        if problem.isGoalState(state):
            actions = []

            while True:
                # Query the parent state from game
                parent, action = backtrack[state]

                # If the parent state is not none, keep running
                if parent is None:
                    break

                actions.append(action)
                state = parent

            actions.reverse()
            return actions

        # Otherwise, keep exploring
        explored_set.add(state)
        for child, action, _ in problem.getSuccessors(state):
            frontier.push(child)
            if child not in explored_set:
                backtrack[child] = (state, action)

    # Error case: return empty list
    return []

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    frontier = util.Queue()
    backtrack = dict()
    explored_set = set()

    # Graph search initialization
    start_state = problem.getStartState()
    frontier.push(start_state)
    backtrack[start_state] = (None, None)

    while not frontier.isEmpty():
        # Obtain next to-be-evaluated state
        state = frontier.pop()

        # If the current state is explored, skip it
        if state in explored_set:
            continue

        # If the state is goal state, backtrack and learn how to obtain it
        if problem.isGoalState(state):
            actions = []

            while True:
                # Query the parent state from game
                parent, action = backtrack[state]

                # If the parent state is not none, keep running
                if parent is None:
                    break

                actions.append(action)
                state = parent

            actions.reverse()
            return actions

        # Otherwise, keep exploring
        explored_set.add(state)
        for child, action, _ in problem.getSuccessors(state):
            frontier.push(child)
            if child not in backtrack:
                backtrack[child] = (state, action)

    # Error case: return empty list
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    # A* search can be reduced as uniformCostSearch by h(x) = 0.
    return aStarSearch(problem, heuristic=nullHeuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic) -> list:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # A* search evaluates game state with extra information.
    #
    # f(n) = g(n) + h(n)
    # - f(n) is the priority (estimate cost)
    # - g(n) is the cost spent to arrive node n.
    # - h(n) is the estimated remain cost to goal.
    #
    # Recall property admissible and consistent when specifying heuristics.
    #
    # A* is problem generic, but cost function f(n) is problem specific.
    # heuristic functions accept game problem and state as argument, returns a real number.
    # See function prototype of heuristic() for details.

    # Notes that util.PriorityQueue() implies an linear iteration to do heap element updating,
    # which is time consuming when user calls PriorityQueue.update().
    frontier = util.PriorityQueue()

    costs = dict()
    backtrack = dict()
    explored_set = set()

    # Graph search initialization
    start_state = problem.getStartState()

    costs[start_state] = 0
    frontier.update(start_state, 0)
    backtrack[start_state] = (None, None)

    while not frontier.isEmpty():
        # Check whether the current state is the goal
        state = frontier.pop()
        if state in explored_set:
            continue

        # If the state is goal state, backtrack and learn how to obtain it
        if problem.isGoalState(state):
            actions = []

            while True:
                # Query the parent state from game
                parent, action = backtrack[state]

                # If the parent state is not none, keep running
                if parent is None:
                    break

                actions.append(action)
                state = parent

            actions.reverse()
            return actions

        # Otherwise, keep exploring children nodes. Evaluate them!
        # FIXME: The updating strategy is unsafe when using not admissible / consistent heuristics.
        cost_from_root = costs[state]
        explored_set.add(state)

        for child, action, action_consumption in problem.getSuccessors(state):
            # Evaluate children nodes.
            g = cost_from_root + action_consumption
            h = heuristic(child, problem)
            f = g + h

            frontier.update(child, f)
            if (child not in costs) or costs[child] > f:
                costs[child] = g
                backtrack[child] = (state, action)

    # Error case: return empty list
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch