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
import pdb

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return graphSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return graphSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    import heapq

    # To reuse to function graphSearch
    # Redifine the method push.
    # Add the parameter list and point to heap.
    def push(self, node):
        entry = (node.cost, self.count, node)
        heapq.heappush(self.heap, entry)
        self.count += 1

    util.PriorityQueue.push = push
    # util.PriorityQueue.list = self.heap

    priorityQueue = util.PriorityQueue()
    priorityQueue.list = priorityQueue.heap

    return graphSearch(problem, priorityQueue)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def MyHeuristic(presentState, problem):
    """
    The idea of heuristic is ManhattanDistance.
    I have try some of the heuristic, but the Manhattan performs best QQ.
    """
    goalState = problem.goal
    estimateCost = abs(presentState[0] - goalState[0]) + abs(presentState[1] - goalState[1])

    return estimateCost

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #TODO: a good heuristic function that calculate the searching priority,
    #      f(n) = g(n) + h(n)
    #      f(n) is the priority(estimate cost)
    #      g(n) is the cost spent to arrive node n
    #      h(n) is the heuristic, estimate cost.
    import heapq

    def push(self, node):
        # f(n) = g(n) + h(n)
        priority = node.cost + 2*heuristic(node.state, problem)
        entry = (priority, self.count, node)
        heapq.heappush(self.heap, entry)
        self.count += 1

    util.PriorityQueue.push = push
    # util.PriorityQueue.list = self.heap

    priorityQueue = util.PriorityQueue()
    priorityQueue.list = priorityQueue.heap

    return graphSearch(problem, priorityQueue)

class Node:
    def __init__(self, parent, state, action, cost):
        self.parent = parent
        self.state = state
        self.action = action
        self.cost = cost

    def __str__(self):
        return str((self.parent, self.state, self.action, self.cost))

    def getLeaf(self, successor):
        return Node(self, successor[0], successor[1], self.cost + successor[2])

def graphSearch(problem, search):
    frontier = search
    exploredSet = {}

    startNode = Node(None, problem.getStartState(), None, 0)
    frontier.push(startNode)

    while not frontier.isEmpty():
        # Check the node wheter it is the goal
        node = frontier.pop()
        exploredSet[node.state] = node

        # print "Now state: " + str(node.state) + ", is Goal? " + str(problem.isGoalState(node.state))

        # If the node contains the goal state, get the path and return actions.
        if problem.isGoalState(node.state):
            path = util.Stack()

            while (node.parent != None):
                path.push(exploredSet[node.state].action)
                node = node.parent

            path.list.reverse()

            return path.list

        # If the node doesn't contain the goal state, then save and explore the leaves.
        exploredSet[node.state] = node

        for successor in problem.getSuccessors(node.state):
            leaf = node.getLeaf(successor)

            # Condition: Not (ready to search or have searched)
            if not ((leaf in frontier.list) or (leaf.state in exploredSet)):
                frontier.push(leaf)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch